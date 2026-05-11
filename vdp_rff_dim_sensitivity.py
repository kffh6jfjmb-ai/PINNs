import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp


# =========================================================
# 0. Settings
# =========================================================
torch.set_default_dtype(torch.float32)

SEED = 0


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =========================================================
# 1. Reference solution
# =========================================================
def get_reference_solution(mu, t_min, t_max, x0, v0, n_eval=1000):
    t_ref = np.linspace(t_min, t_max, n_eval)

    sol = solve_ivp(
        lambda t, z: [
            z[1],
            mu * (1.0 - z[0] ** 2) * z[1] - z[0]
        ],
        [t_min, t_max],
        [x0, v0],
        t_eval=t_ref,
        rtol=1e-10,
        atol=1e-12
    )

    return t_ref, sol.y[0]


def compute_l2_errors(x_pred, x_ref):
    abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
    return abs_l2, rel_l2


# =========================================================
# 2. Networks
# =========================================================
class SoftPINN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, t):
        return self.net(t)


class RFFLayer(nn.Module):
    def __init__(self, out_features=64, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(1, out_features // 2) * sigma,
            requires_grad=False
        )

    def forward(self, t):
        proj = 2.0 * np.pi * torch.matmul(t, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SoftRFFPINN(nn.Module):
    def __init__(self, rff_features=64, sigma=1.0, width=64):
        super().__init__()
        self.rff = RFFLayer(out_features=rff_features, sigma=sigma)
        self.net = nn.Sequential(
            nn.Linear(rff_features, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, t):
        return self.net(self.rff(t))


class HardPINN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, t):
        return self.net(t)


class HardRFFPINN(nn.Module):
    def __init__(self, rff_features=64, sigma=1.0, width=64):
        super().__init__()
        self.rff = RFFLayer(out_features=rff_features, sigma=sigma)
        self.net = nn.Sequential(
            nn.Linear(rff_features, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, t):
        return self.net(self.rff(t))


# =========================================================
# 3. Losses and hard constraint
# =========================================================
def residual_loss_soft(model, t, mu):
    t = t.clone().detach().requires_grad_(True)

    x = model(t)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]

    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


def initial_loss_soft(model, x0, v0):
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

    x_pred = model(t0)
    x_t_pred = torch.autograd.grad(
        x_pred, t0, torch.ones_like(x_pred), create_graph=True
    )[0]

    return torch.mean((x_pred - x0) ** 2 + (x_t_pred - v0) ** 2)


def hard_output(model, t, x0, v0, t0=0.0):
    """
    Hard ansatz for x(t0)=x0 and x'(t0)=v0.
    """
    x_hat_t = model(t)

    t0_tensor = torch.tensor(
        [[t0]], dtype=t.dtype, device=t.device, requires_grad=True
    )
    x_hat_t0 = model(t0_tensor)

    dx_hat_t0 = torch.autograd.grad(
        x_hat_t0,
        t0_tensor,
        torch.ones_like(x_hat_t0),
        create_graph=True
    )[0]

    return x_hat_t + x0 - x_hat_t0 + (t - t0_tensor) * (v0 - dx_hat_t0)


def hard_predict(model, t, x0, v0, t0=0.0):
    with torch.enable_grad():
        t_req = t.clone().detach().requires_grad_(True)
        x = hard_output(model, t_req, x0, v0, t0)
    return x.detach()


def residual_loss_hard(model, t, mu, x0, v0, t0=0.0):
    t = t.clone().detach().requires_grad_(True)

    x = hard_output(model, t, x0, v0, t0)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]

    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


# =========================================================
# 4. Training routines
# =========================================================
def train_soft_model(
    model,
    model_name,
    mu,
    t_min,
    t_max,
    x0,
    v0,
    t_ref,
    x_ref,
    epochs=15000,
    n_col=200,
    lr=3e-3,
    w_ic=1.0,
    eval_every=50,
    verbose=False
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_domain = torch.linspace(t_min, t_max, n_col).reshape(-1, 1)
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    l2_epochs = []
    abs_l2_history = []
    rel_l2_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_ode = residual_loss_soft(model, t_domain, mu)
        loss_ic = initial_loss_soft(model, x0, v0)
        loss = loss_ode + w_ic * loss_ic

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            with torch.no_grad():
                x_pred = model(t_eval_tensor).cpu().numpy().flatten()
            abs_l2, rel_l2 = compute_l2_errors(x_pred, x_ref)
            l2_epochs.append(epoch + 1)
            abs_l2_history.append(abs_l2)
            rel_l2_history.append(rel_l2)

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"[{model_name}] Epoch {epoch+1:5d} | Loss = {loss.item():.3e}")

    return {
        "model": model,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history
    }


def train_hard_model(
    model,
    model_name,
    mu,
    t_min,
    t_max,
    x0,
    v0,
    t_ref,
    x_ref,
    epochs=15000,
    n_col=200,
    lr=3e-3,
    eval_every=50,
    verbose=False
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_domain = torch.linspace(t_min, t_max, n_col).reshape(-1, 1)
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    l2_epochs = []
    abs_l2_history = []
    rel_l2_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = residual_loss_hard(model, t_domain, mu, x0, v0, t0=0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            x_pred = hard_predict(model, t_eval_tensor, x0, v0, t0=0.0).cpu().numpy().flatten()
            abs_l2, rel_l2 = compute_l2_errors(x_pred, x_ref)
            l2_epochs.append(epoch + 1)
            abs_l2_history.append(abs_l2)
            rel_l2_history.append(rel_l2)

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"[{model_name}] Epoch {epoch+1:5d} | Loss = {loss.item():.3e}")

    return {
        "model": model,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history
    }


# =========================================================
# 5. RFF input-dimension sensitivity study
# =========================================================
def run_vdp_rff_dim_study():

    # Problem setup: same as compare_vdp.py
    mu = 1.0
    t_min, t_max = 0.0, 10.0
    x0, v0 = 1.0, 0.0

    # Training setup: same as compare_vdp.py
    epochs = 15000
    n_col = 200
    lr = 3e-3
    w_ic = 1.0
    eval_every = 50
    width = 64

    # Only the Fourier input dimension is varied.
    # rff_features is the dimension after concatenating cos and sin features.
    fixed_sigma = 1.0
    rff_feature_values = [16, 32, 64, 128, 256]

    t_ref, x_ref = get_reference_solution(
        mu=mu,
        t_min=t_min,
        t_max=t_max,
        x0=x0,
        v0=v0,
        n_eval=1000
    )

    histories = {"Soft RFF-PINN": {}, "Hard RFF-PINN": {}}
    rows = []

    for rff_features in rff_feature_values:
        print(f"\n================ RFF input dimension = {rff_features} ================")

        # Soft RFF-PINN
        set_seed(SEED)
        soft_rff = SoftRFFPINN(
            rff_features=rff_features,
            sigma=fixed_sigma,
            width=width
        )
        soft_result = train_soft_model(
            model=soft_rff,
            model_name=f"Soft RFF dim={rff_features}",
            mu=mu,
            t_min=t_min,
            t_max=t_max,
            x0=x0,
            v0=v0,
            t_ref=t_ref,
            x_ref=x_ref,
            epochs=epochs,
            n_col=n_col,
            lr=lr,
            w_ic=w_ic,
            eval_every=eval_every,
            verbose=False
        )
        histories["Soft RFF-PINN"][rff_features] = soft_result
        rows.append({
            "problem": "Van der Pol",
            "model": "Soft RFF-PINN",
            "rff_features": rff_features,
            "num_frequencies": rff_features // 2,
            "sigma": fixed_sigma,
            "final_abs_l2": soft_result["abs_l2"][-1],
            "final_rel_l2": soft_result["rel_l2"][-1],
            "best_rel_l2": min(soft_result["rel_l2"])
        })
        print(f"Soft RFF-PINN | final Rel L2 = {soft_result['rel_l2'][-1]:.6e}")

        # Hard RFF-PINN
        set_seed(SEED)
        hard_rff = HardRFFPINN(
            rff_features=rff_features,
            sigma=fixed_sigma,
            width=width
        )
        hard_result = train_hard_model(
            model=hard_rff,
            model_name=f"Hard RFF dim={rff_features}",
            mu=mu,
            t_min=t_min,
            t_max=t_max,
            x0=x0,
            v0=v0,
            t_ref=t_ref,
            x_ref=x_ref,
            epochs=epochs,
            n_col=n_col,
            lr=lr,
            eval_every=eval_every,
            verbose=False
        )
        histories["Hard RFF-PINN"][rff_features] = hard_result
        rows.append({
            "problem": "Van der Pol",
            "model": "Hard RFF-PINN",
            "rff_features": rff_features,
            "num_frequencies": rff_features // 2,
            "sigma": fixed_sigma,
            "final_abs_l2": hard_result["abs_l2"][-1],
            "final_rel_l2": hard_result["rel_l2"][-1],
            "best_rel_l2": min(hard_result["rel_l2"])
        })
        print(f"Hard RFF-PINN | final Rel L2 = {hard_result['rel_l2'][-1]:.6e}")


    plot_final_errors(rows, rff_feature_values)
    plot_histories(histories)


def plot_final_errors(rows, rff_feature_values):
    soft = [r["final_rel_l2"] for r in rows if r["model"] == "Soft RFF-PINN"]
    hard = [r["final_rel_l2"] for r in rows if r["model"] == "Hard RFF-PINN"]

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(rff_feature_values, soft, marker="o", lw=2, label="Soft RFF-PINN")
    plt.semilogy(rff_feature_values, hard, marker="s", lw=2, label="Hard RFF-PINN")
    plt.xscale("log", base=2)
    plt.xticks(rff_feature_values, [str(v) for v in rff_feature_values])
    plt.xlabel(r"Fourier input dimension $d_\gamma$")
    plt.ylabel(r"Final relative $L^2$ error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_histories(histories):
    for model_name, model_histories in histories.items():
        plt.figure(figsize=(8, 4.8))
        for rff_features, result in model_histories.items():
            plt.semilogy(
                result["l2_epochs"],
                result["rel_l2"],
                lw=2,
                label=rf"$d_\gamma={rff_features}$"
            )
        plt.xlabel("Epoch")
        plt.ylabel(r"Relative $L^2$ error")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        plt.show()


if __name__ == "__main__":
    run_vdp_rff_dim_study()
