import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# =========================================================
# 0. Settings
# =========================================================
torch.set_default_dtype(torch.float32)

SEED = 0

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

# =========================================================
# 1. Exact solution
# =========================================================
def exact_solution(t, n):
    return np.sin(n * np.pi * t) / (n * np.pi * np.cos(n * np.pi))

def compute_l2_errors(x_pred, x_ref):
    abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
    return abs_l2, rel_l2

# =========================================================
# 2. Networks
# =========================================================
class RFFLayer(nn.Module):
    def __init__(self, out_features=64, sigma=3.0):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(1, out_features // 2) * sigma,
            requires_grad=False
        )

    def forward(self, t):
        proj = 2.0 * np.pi * torch.matmul(t, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SoftRFFPINN(nn.Module):
    def __init__(self, rff_features=64, sigma=3.0, width=64):
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



class HardRFFPINN(nn.Module):
    def __init__(self, rff_features=64, sigma=3.0, width=64):
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
def residual_loss_soft(model, t, n):
    t = t.clone().detach().requires_grad_(True)

    x = model(t)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]

    res = x_tt + (n * np.pi) ** 2 * x
    return torch.mean(res ** 2)


def boundary_loss_soft(model):
    # x(0)=0
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    x0 = model(t0)
    loss_x0 = (x0 - 0.0) ** 2

    # x'(1)=1
    t1 = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    x1 = model(t1)
    x1_t = torch.autograd.grad(x1, t1, torch.ones_like(x1), create_graph=True)[0]
    loss_x1_t = (x1_t - 1.0) ** 2

    return torch.mean(loss_x0 + loss_x1_t)


def hard_output(model, t):

    x_hat = model(t)

    t0 = torch.tensor([[0.0]], dtype=t.dtype, device=t.device)
    x_hat_0 = model(t0)

    t1 = torch.tensor([[1.0]], dtype=t.dtype, device=t.device, requires_grad=True)
    x_hat_1 = model(t1)

    dx_hat_1 = torch.autograd.grad(
        x_hat_1,
        t1,
        torch.ones_like(x_hat_1),
        create_graph=True
    )[0]

    return x_hat - x_hat_0 + t * (1.0 - dx_hat_1)


def residual_loss_hard(model, t, n):
    t = t.clone().detach().requires_grad_(True)

    x = hard_output(model, t)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]

    res = x_tt + (n * np.pi) ** 2 * x
    return torch.mean(res ** 2)


# =========================================================
# 4. Training routines
# =========================================================
def train_soft_model(
    model,
    model_name,
    n,
    t_min,
    t_max,
    t_ref,
    x_ref,
    epochs=15000,
    n_col=50,
    lr=1e-3,
    w_bc=10.0,
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

        loss_ode = residual_loss_soft(model, t_domain, n)
        loss_bc = boundary_loss_soft(model)
        loss = loss_ode + w_bc * loss_bc

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
    n,
    t_min,
    t_max,
    t_ref,
    x_ref,
    epochs=15000,
    n_col=50,
    lr=1e-3,
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

        loss = residual_loss_hard(model, t_domain, n)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            x_pred = hard_output(model, t_eval_tensor).detach().cpu().numpy().flatten()
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
def run_harmonic_rff_dim_study():

    # Same setup as compare_harmonic_tx.py
    t_min, t_max = 0.0, 1.0
    epochs = 15000
    n_col = 50
    lr = 1e-3
    w_bc = 10.0
    eval_every = 50
    width = 64

    # Only the Fourier input dimension is varied.
    # rff_features is the dimension after concatenating cos and sin features.
    fixed_sigma = 1.0
    rff_feature_values = [16, 32, 64, 128, 256]

    n_values = [3, 6]

    rows = []
    histories = {}

    for n in n_values:
        print(f"\n================================================")
        print(f"Harmonic oscillator: n = {n}")
        print(f"================================================")

        t_ref = np.linspace(t_min, t_max, 1000)
        x_ref = exact_solution(t_ref, n)

        histories[n] = {"Soft RFF-PINN": {}, "Hard RFF-PINN": {}}

        for rff_features in rff_feature_values:
            print(f"\n---------------- RFF input dimension = {rff_features} ----------------")

            # Soft RFF-PINN
            set_seed(SEED)
            soft_rff = SoftRFFPINN(
                rff_features=rff_features,
                sigma=fixed_sigma,
                width=width
            )
            soft_result = train_soft_model(
                model=soft_rff,
                model_name=f"Soft RFF n={n}, dim={rff_features}",
                n=n,
                t_min=t_min,
                t_max=t_max,
                t_ref=t_ref,
                x_ref=x_ref,
                epochs=epochs,
                n_col=n_col,
                lr=lr,
                w_bc=w_bc,
                eval_every=eval_every,
                verbose=False
            )
            histories[n]["Soft RFF-PINN"][rff_features] = soft_result
            rows.append({
                "problem": "Harmonic oscillator",
                "n": n,
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
                model_name=f"Hard RFF n={n}, dim={rff_features}",
                n=n,
                t_min=t_min,
                t_max=t_max,
                t_ref=t_ref,
                x_ref=x_ref,
                epochs=epochs,
                n_col=n_col,
                lr=lr,
                eval_every=eval_every,
                verbose=False
            )
            histories[n]["Hard RFF-PINN"][rff_features] = hard_result
            rows.append({
                "problem": "Harmonic oscillator",
                "n": n,
                "model": "Hard RFF-PINN",
                "rff_features": rff_features,
                "num_frequencies": rff_features // 2,
                "sigma": fixed_sigma,
                "final_abs_l2": hard_result["abs_l2"][-1],
                "final_rel_l2": hard_result["rel_l2"][-1],
                "best_rel_l2": min(hard_result["rel_l2"])
            })
            print(f"Hard RFF-PINN | final Rel L2 = {hard_result['rel_l2'][-1]:.6e}")


    plot_final_errors(rows, n_values, rff_feature_values)
    plot_histories(histories)


def plot_final_errors(rows, n_values, rff_feature_values):
    fig, axes = plt.subplots(1, len(n_values), figsize=(12, 4.5), sharey=True)

    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        soft = [
            r["final_rel_l2"] for r in rows
            if r["n"] == n and r["model"] == "Soft RFF-PINN"
        ]
        hard = [
            r["final_rel_l2"] for r in rows
            if r["n"] == n and r["model"] == "Hard RFF-PINN"
        ]

        ax.semilogy(rff_feature_values, soft, marker="o", lw=2, label="Soft RFF-PINN")
        ax.semilogy(rff_feature_values, hard, marker="s", lw=2, label="Hard RFF-PINN")
        ax.set_xscale("log", base=2)
        ax.set_xticks(rff_feature_values)
        ax.set_xticklabels([str(v) for v in rff_feature_values])
        ax.set_xlabel(r"Fourier input dimension $d_\gamma$")
        ax.set_title(rf"$n={n}$")
        ax.legend(loc="best", fontsize=9)
        ax.grid(False)

    axes[0].set_ylabel(r"Final relative $L^2$ error")
    plt.tight_layout()
    plt.show()


def plot_histories(histories):
    for n, n_histories in histories.items():
        for model_name, model_histories in n_histories.items():
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
    run_harmonic_rff_dim_study()
