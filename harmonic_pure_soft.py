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
# 1. Reference solution
#    x'' + (n*pi)^2 x = 0, x(0)=0, x'(1)=1
# =========================================================
def get_reference_solution(n, t_min, t_max, n_eval=1000):
    t_ref = np.linspace(t_min, t_max, n_eval)
    x_ref = np.sin(n * np.pi * t_ref) / (n * np.pi * np.cos(n * np.pi))
    return t_ref, x_ref


def compute_l2_errors(x_pred, x_ref):
    abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
    return abs_l2, rel_l2


# =========================================================
# 2. Network
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
# 3. Soft constraint losses
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
# 4. Train soft model
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
# 5. Main
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------------------
    # Problem setup
    # -----------------------------------------------------
    n = 2
    t_min, t_max = 0.0, 1.0

    # -----------------------------------------------------
    # Training setup
    # -----------------------------------------------------
    epochs = 15000
    n_col = 50
    lr = 1e-3
    w_bc = 10.0
    eval_every = 50

    # -----------------------------------------------------
    # Reference solution
    # -----------------------------------------------------
    t_ref, x_ref = get_reference_solution(
        n=n,
        t_min=t_min,
        t_max=t_max,
        n_eval=1000
    )
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    # -----------------------------------------------------
    # Soft PINN
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

    # =====================================================
    # 6. Final prediction
    # =====================================================
    with torch.no_grad():
        x_soft = soft_result["model"](t_eval_tensor).cpu().numpy().flatten()

    soft_abs_l2, soft_rel_l2 = compute_l2_errors(x_soft, x_ref)

    print("\n================ Final L2 Errors ================")
    print(f"n = {n}")
    print(f"Soft PINN | Abs L2 = {soft_abs_l2:.6e} | Rel L2 = {soft_rel_l2:.6e}")

    # =====================================================
    # 7. Solution comparison
    # =====================================================
    plt.figure(figsize=(6, 4))
    plt.plot(
        t_ref,
        x_ref,
        color="gray",
        lw=3,
        alpha=0.8,
        linestyle="-",
        label="Exact"
    )
    plt.plot(
        t_ref,
        x_soft,
        color="red",
        lw=2.2,
        linestyle="--",
        label="Soft PINN"
    )
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend(frameon=True, fontsize=9)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("harmonic_pure_soft_solution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =====================================================
    # 8. Absolute L2 error
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(
        soft_result["l2_epochs"],
        soft_result["abs_l2"],
        color="red",
        lw=2,
        label="Soft PINN"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Absolute L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("harmonic_pure_soft_abs_l2.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =====================================================
    # 9. Relative L2 error
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(
        soft_result["l2_epochs"],
        soft_result["rel_l2"],
        color="red",
        lw=2,
        label="Soft PINN"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("harmonic_pure_soft_rel_l2.png", dpi=300, bbox_inches="tight")
    plt.show()
