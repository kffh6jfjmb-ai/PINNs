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

def set_seed(seed=0):
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

    x_ref = sol.y[0]
    return t_ref, x_ref


def compute_l2_errors(x_pred, x_ref):
    abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
    return abs_l2, rel_l2


# =========================================================
# 2. Standard Soft PINN
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


# =========================================================
# 3. Random Fourier Feature Layer
# =========================================================
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


# =========================================================
# 4. Soft RFF-PINN
# =========================================================
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
        features = self.rff(t)
        return self.net(features)


# =========================================================
# 5. Hard PINN base network
# =========================================================
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


# =========================================================
# 6. Hard RFF-PINN base network
# =========================================================
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
        features = self.rff(t)
        return self.net(features)


# =========================================================
# 7. Soft losses
# =========================================================
def residual_loss_soft(model, t, mu):
    t = t.clone().detach().requires_grad_(True)

    x = model(t)
    x_t = torch.autograd.grad(
        x, t, torch.ones_like(x), create_graph=True
    )[0]
    x_tt = torch.autograd.grad(
        x_t, t, torch.ones_like(x_t), create_graph=True
    )[0]

    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


def initial_loss_soft(model, x0, v0):
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

    x_pred = model(t0)
    x_t_pred = torch.autograd.grad(
        x_pred, t0, torch.ones_like(x_pred), create_graph=True
    )[0]

    loss_x = (x_pred - x0) ** 2
    loss_v = (x_t_pred - v0) ** 2
    return torch.mean(loss_x + loss_v)


# =========================================================
# 8. Hard output construction
# =========================================================
def hard_output(model, t, x0, v0, t0=0.0):
    """
    Hard ansatz:
        x(t) = x_hat(t) + x0 - x_hat(t0) + (t-t0) * (v0 - x_hat'(t0))
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

    x = x_hat_t + x0 - x_hat_t0 + (t - t0_tensor) * (v0 - dx_hat_t0)
    return x


def hard_predict(model, t, x0, v0, t0=0.0):
    with torch.enable_grad():
        t_req = t.clone().detach().requires_grad_(True)
        x = hard_output(model, t_req, x0, v0, t0)
    return x.detach()


def residual_loss_hard(model, t, mu, x0, v0, t0=0.0):
    t = t.clone().detach().requires_grad_(True)

    x = hard_output(model, t, x0, v0, t0)
    x_t = torch.autograd.grad(
        x, t, torch.ones_like(x), create_graph=True
    )[0]
    x_tt = torch.autograd.grad(
        x_t, t, torch.ones_like(x_t), create_graph=True
    )[0]

    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


# =========================================================
# 9. Train soft model
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
    eval_every=50
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_domain = torch.linspace(t_min, t_max, n_col).reshape(-1, 1)
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    loss_history = []
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

        loss_history.append(loss.item())

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            with torch.no_grad():
                x_pred = model(t_eval_tensor).cpu().numpy().flatten()

            abs_l2, rel_l2 = compute_l2_errors(x_pred, x_ref)
            l2_epochs.append(epoch + 1)
            abs_l2_history.append(abs_l2)
            rel_l2_history.append(rel_l2)

        if (epoch + 1) % 1000 == 0:
            print(f"[{model_name}] Epoch {epoch+1:5d} | Loss = {loss.item():.3e}")

    return {
        "model": model,
        "loss": loss_history,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history
    }


# =========================================================
# 10. Train hard model
# =========================================================
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
    eval_every=50
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_domain = torch.linspace(t_min, t_max, n_col).reshape(-1, 1)
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    loss_history = []
    l2_epochs = []
    abs_l2_history = []
    rel_l2_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = residual_loss_hard(model, t_domain, mu, x0, v0, t0=0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            x_pred = hard_predict(model, t_eval_tensor, x0, v0, t0=0.0).cpu().numpy().flatten()

            abs_l2, rel_l2 = compute_l2_errors(x_pred, x_ref)
            l2_epochs.append(epoch + 1)
            abs_l2_history.append(abs_l2)
            rel_l2_history.append(rel_l2)

        if (epoch + 1) % 1000 == 0:
            print(f"[{model_name}] Epoch {epoch+1:5d} | Loss = {loss.item():.3e}")

    return {
        "model": model,
        "loss": loss_history,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history
    }


# =========================================================
# 11. Main
# =========================================================
if __name__ == "__main__":

    # Problem setup
    mu = 1.0
    t_min, t_max = 0.0, 10.0
    x0, v0 = 1.0, 0.0

    # Training setup
    epochs = 15000
    n_col = 200
    lr = 3e-3
    w_ic = 1.0
    eval_every = 50

    # Reference solution
    t_ref, x_ref = get_reference_solution(
        mu=mu,
        t_min=t_min,
        t_max=t_max,
        x0=x0,
        v0=v0,
        n_eval=1000
    )
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    # -----------------------------------------------------
    # 1) Soft PINN
    # -----------------------------------------------------
    set_seed(0)
    soft_model = SoftPINN(width=64)
    soft_result = train_soft_model(
        model=soft_model,
        model_name="Soft PINN",
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
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 2) Hard PINN
    # -----------------------------------------------------
    set_seed(0)
    hard_model = HardPINN(width=64)
    hard_result = train_hard_model(
        model=hard_model,
        model_name="Hard PINN",
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
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 3) Soft RFF-PINN
    # -----------------------------------------------------
    set_seed(0)
    soft_rff_model = SoftRFFPINN(rff_features=64, sigma=1.0, width=64)
    soft_rff_result = train_soft_model(
        model=soft_rff_model,
        model_name="Soft RFF-PINN",
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
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 4) Hard RFF-PINN
    # -----------------------------------------------------
    set_seed(0)
    hard_rff_model = HardRFFPINN(rff_features=64, sigma=1.0, width=64)
    hard_rff_result = train_hard_model(
        model=hard_rff_model,
        model_name="Hard RFF-PINN",
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
        eval_every=eval_every
    )

    # =====================================================
    # 12. Final predictions
    # =====================================================
    with torch.no_grad():
        x_soft = soft_result["model"](t_eval_tensor).cpu().numpy().flatten()
        x_soft_rff = soft_rff_result["model"](t_eval_tensor).cpu().numpy().flatten()

    x_hard = hard_predict(hard_result["model"], t_eval_tensor, x0, v0, t0=0.0).cpu().numpy().flatten()
    x_hard_rff = hard_predict(hard_rff_result["model"], t_eval_tensor, x0, v0, t0=0.0).cpu().numpy().flatten()

    # Final errors
    soft_abs_l2, soft_rel_l2 = compute_l2_errors(x_soft, x_ref)
    hard_abs_l2, hard_rel_l2 = compute_l2_errors(x_hard, x_ref)
    soft_rff_abs_l2, soft_rff_rel_l2 = compute_l2_errors(x_soft_rff, x_ref)
    hard_rff_abs_l2, hard_rff_rel_l2 = compute_l2_errors(x_hard_rff, x_ref)

    print("\n================ Final L2 Errors ================")
    print(f"Soft PINN       | Abs L2 = {soft_abs_l2:.6e} | Rel L2 = {soft_rel_l2:.6e}")
    print(f"Hard PINN       | Abs L2 = {hard_abs_l2:.6e} | Rel L2 = {hard_rel_l2:.6e}")
    print(f"Soft RFF-PINN   | Abs L2 = {soft_rff_abs_l2:.6e} | Rel L2 = {soft_rff_rel_l2:.6e}")
    print(f"Hard RFF-PINN   | Abs L2 = {hard_rff_abs_l2:.6e} | Rel L2 = {hard_rff_rel_l2:.6e}")

    # =====================================================
    # 13. Plot solution comparison: one big figure with 4 subplots
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    methods = [
        ("Soft PINN", x_soft, "red"),
        ("Hard PINN", x_hard, "blue"),
        ("Soft RFF-PINN", x_soft_rff, "green"),
        ("Hard RFF-PINN", x_hard_rff, "purple"),
    ]

    for ax, (name, x_pred, color) in zip(axes.flat, methods):
        ax.plot(t_ref, x_ref, color="gray", lw=3, alpha=0.8, linestyle="-", label="Reference")
        ax.plot(t_ref, x_pred, color=color, lw=2, linestyle="--", label="Prediction")

        # no title; only a small in-panel label
        ax.text(
            0.03, 0.92, name,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top"
        )

        ax.legend(frameon=True, fontsize=9)
        ax.grid(False)

    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("x(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 14. Plot absolute L2 error comparison
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(soft_result["l2_epochs"], soft_result["abs_l2"], color="red", lw=2, label="Soft PINN")
    plt.semilogy(hard_result["l2_epochs"], hard_result["abs_l2"], color="blue", lw=2, label="Hard PINN")
    plt.semilogy(soft_rff_result["l2_epochs"], soft_rff_result["abs_l2"], color="green", lw=2, label="Soft RFF-PINN")
    plt.semilogy(hard_rff_result["l2_epochs"], hard_rff_result["abs_l2"], color="purple", lw=2, label="Hard RFF-PINN")
    plt.xlabel("Epoch")
    plt.ylabel("Absolute L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 15. Plot relative L2 error comparison
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(soft_result["l2_epochs"], soft_result["rel_l2"], color="red", lw=2, label="Soft PINN")
    plt.semilogy(hard_result["l2_epochs"], hard_result["rel_l2"], color="blue", lw=2, label="Hard PINN")
    plt.semilogy(soft_rff_result["l2_epochs"], soft_rff_result["rel_l2"], color="green", lw=2, label="Soft RFF-PINN")
    plt.semilogy(hard_rff_result["l2_epochs"], hard_rff_result["rel_l2"], color="purple", lw=2, label="Hard RFF-PINN")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()