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


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =========================================================
# 1. Exact solution
#    x'' + (n*pi)^2 x = 0, x(0)=0, x'(1)=1
# =========================================================
def exact_solution(t, n):

    return np.sin(n * np.pi * t) / (n * np.pi * np.cos(n * np.pi))


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
def residual_loss_soft(model, t, n):
    t = t.clone().detach().requires_grad_(True)

    x = model(t)

    x_t = torch.autograd.grad(
        x,
        t,
        torch.ones_like(x),
        create_graph=True
    )[0]

    x_tt = torch.autograd.grad(
        x_t,
        t,
        torch.ones_like(x_t),
        create_graph=True
    )[0]

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

    x1_t = torch.autograd.grad(
        x1,
        t1,
        torch.ones_like(x1),
        create_graph=True
    )[0]

    loss_x1_t = (x1_t - 1.0) ** 2

    return torch.mean(loss_x0 + loss_x1_t)


# =========================================================
# 8. Hard constraint output
# =========================================================
def hard_output(model, t):
  
    x_hat = model(t)

    t0 = torch.tensor(
        [[0.0]],
        dtype=t.dtype,
        device=t.device
    )
    x_hat_0 = model(t0)

    t1 = torch.tensor(
        [[1.0]],
        dtype=t.dtype,
        device=t.device,
        requires_grad=True
    )
    x_hat_1 = model(t1)

    dx_hat_1 = torch.autograd.grad(
        x_hat_1,
        t1,
        torch.ones_like(x_hat_1),
        create_graph=True
    )[0]

    x = x_hat - x_hat_0 + t * (1.0 - dx_hat_1)
    return x


def residual_loss_hard(model, t, n):
    t = t.clone().detach().requires_grad_(True)

    x = hard_output(model, t)

    x_t = torch.autograd.grad(
        x,
        t,
        torch.ones_like(x),
        create_graph=True
    )[0]

    x_tt = torch.autograd.grad(
        x_t,
        t,
        torch.ones_like(x_t),
        create_graph=True
    )[0]

    res = x_tt + (n * np.pi) ** 2 * x
    return torch.mean(res ** 2)


# =========================================================
# 9. Train soft models
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

        loss_ode = residual_loss_soft(model, t_domain, n)
        loss_bc = boundary_loss_soft(model)

        loss = loss_ode + w_bc * loss_bc

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
            print(
                f"[{model_name}] Epoch {epoch+1:5d} | "
                f"Loss = {loss.item():.3e} | "
                f"ODE = {loss_ode.item():.3e} | "
                f"BC = {loss_bc.item():.3e}"
            )

    return {
        "model": model,
        "loss": loss_history,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history
    }


# =========================================================
# 10. Train hard models
# =========================================================
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

        loss = residual_loss_hard(model, t_domain, n)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            # Do not wrap this in torch.no_grad(),
            # because hard_output computes x_hat'(1).
            x_pred = hard_output(model, t_eval_tensor).detach().cpu().numpy().flatten()

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
# 11. Check hard constraints
# =========================================================
def check_hard_constraints(model, name):
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    x0 = hard_output(model, t0)

    t1 = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    x1 = hard_output(model, t1)

    x1_t = torch.autograd.grad(
        x1,
        t1,
        torch.ones_like(x1),
        create_graph=False
    )[0]

    print(
        f"{name} hard BC check: "
        f"x(0)={x0.item():.8f}, target=0.00000000 | "
        f"x'(1)={x1_t.item():.8f}, target=1.00000000"
    )


# =========================================================
# 12. Main comparison
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------------------
    # Problem setup
    # -----------------------------------------------------
    n = 6
    t_min, t_max = 0.0, 1.0

    # -----------------------------------------------------
    # Training setup
    # -----------------------------------------------------
    epochs = 15000
    n_col = 50
    lr = 1e-3
    w_bc = 10.0
    eval_every = 50

    # Fourier feature scale
    sigma_rff = 1.0
    rff_features = 64

    # -----------------------------------------------------
    # Reference solution
    # -----------------------------------------------------
    t_ref = np.linspace(t_min, t_max, 1000)
    x_ref = exact_solution(t_ref, n)
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    # -----------------------------------------------------
    # 1) Soft PINN
    # -----------------------------------------------------
    set_seed(0)
    soft_model = SoftPINN(width=64)

    soft_result = train_soft_model(
        model=soft_model,
        model_name="Soft PINN",
        n=n,
        t_min=t_min,
        t_max=t_max,
        t_ref=t_ref,
        x_ref=x_ref,
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        w_bc=w_bc,
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
        n=n,
        t_min=t_min,
        t_max=t_max,
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
    soft_rff_model = SoftRFFPINN(
        rff_features=rff_features,
        sigma=sigma_rff,
        width=64
    )

    soft_rff_result = train_soft_model(
        model=soft_rff_model,
        model_name="Soft RFF-PINN",
        n=n,
        t_min=t_min,
        t_max=t_max,
        t_ref=t_ref,
        x_ref=x_ref,
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        w_bc=w_bc,
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 4) Hard RFF-PINN
    # -----------------------------------------------------
    set_seed(0)
    hard_rff_model = HardRFFPINN(
        rff_features=rff_features,
        sigma=sigma_rff,
        width=64
    )

    hard_rff_result = train_hard_model(
        model=hard_rff_model,
        model_name="Hard RFF-PINN",
        n=n,
        t_min=t_min,
        t_max=t_max,
        t_ref=t_ref,
        x_ref=x_ref,
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        eval_every=eval_every
    )

    # =====================================================
    # 13. Final predictions
    # =====================================================
    with torch.no_grad():
        x_soft = soft_result["model"](t_eval_tensor).cpu().numpy().flatten()
        x_soft_rff = soft_rff_result["model"](t_eval_tensor).cpu().numpy().flatten()

    # hard_output needs autograd to compute x_hat'(1)
    x_hard = hard_output(hard_result["model"], t_eval_tensor).detach().cpu().numpy().flatten()
    x_hard_rff = hard_output(hard_rff_result["model"], t_eval_tensor).detach().cpu().numpy().flatten()

    # Final errors
    soft_abs_l2, soft_rel_l2 = compute_l2_errors(x_soft, x_ref)
    hard_abs_l2, hard_rel_l2 = compute_l2_errors(x_hard, x_ref)
    soft_rff_abs_l2, soft_rff_rel_l2 = compute_l2_errors(x_soft_rff, x_ref)
    hard_rff_abs_l2, hard_rff_rel_l2 = compute_l2_errors(x_hard_rff, x_ref)

    print("\n================ Final L2 Errors ================")
    print(f"n = {n}")
    print(f"Soft PINN       | Abs L2 = {soft_abs_l2:.6e} | Rel L2 = {soft_rel_l2:.6e}")
    print(f"Hard PINN       | Abs L2 = {hard_abs_l2:.6e} | Rel L2 = {hard_rel_l2:.6e}")
    print(f"Soft RFF-PINN   | Abs L2 = {soft_rff_abs_l2:.6e} | Rel L2 = {soft_rff_rel_l2:.6e}")
    print(f"Hard RFF-PINN   | Abs L2 = {hard_rff_abs_l2:.6e} | Rel L2 = {hard_rff_rel_l2:.6e}")

    check_hard_constraints(hard_result["model"], "Hard PINN")
    check_hard_constraints(hard_rff_result["model"], "Hard RFF-PINN")

    # =====================================================
    # 14. Solution comparison
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    methods = [
        ("Soft PINN", x_soft, "red"),
        ("Hard PINN", x_hard, "blue"),
        ("Soft RFF-PINN", x_soft_rff, "green"),
        ("Hard RFF-PINN", x_hard_rff, "purple"),
    ]

    for ax, (name, x_pred, color) in zip(axes.flat, methods):
        ax.plot(
            t_ref,
            x_ref,
            color="gray",
            lw=3,
            alpha=0.8,
            linestyle="-",
            label="Exact"
        )

        ax.plot(
            t_ref,
            x_pred,
            color=color,
            lw=2.2,
            linestyle="--",
            label="Prediction"
        )

        ax.text(
            0.03,
            0.92,
            name,
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
    # 15. Absolute L2 error comparison
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(
        soft_result["l2_epochs"],
        soft_result["abs_l2"],
        color="red",
        lw=2,
        label="Soft PINN"
    )
    plt.semilogy(
        hard_result["l2_epochs"],
        hard_result["abs_l2"],
        color="blue",
        lw=2,
        label="Hard PINN"
    )
    plt.semilogy(
        soft_rff_result["l2_epochs"],
        soft_rff_result["abs_l2"],
        color="green",
        lw=2,
        label="Soft RFF-PINN"
    )
    plt.semilogy(
        hard_rff_result["l2_epochs"],
        hard_rff_result["abs_l2"],
        color="purple",
        lw=2,
        label="Hard RFF-PINN"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Absolute L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 16. Relative L2 error comparison
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(
        soft_result["l2_epochs"],
        soft_result["rel_l2"],
        color="red",
        lw=2,
        label="Soft PINN"
    )
    plt.semilogy(
        hard_result["l2_epochs"],
        hard_result["rel_l2"],
        color="blue",
        lw=2,
        label="Hard PINN"
    )
    plt.semilogy(
        soft_rff_result["l2_epochs"],
        soft_rff_result["rel_l2"],
        color="green",
        lw=2,
        label="Soft RFF-PINN"
    )
    plt.semilogy(
        hard_rff_result["l2_epochs"],
        hard_rff_result["rel_l2"],
        color="purple",
        lw=2,
        label="Hard RFF-PINN"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
