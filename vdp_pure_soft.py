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
#    x'' - mu(1-x^2)x' + x = 0, x(0)=x0, x'(0)=v0
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
def residual_loss_soft(model, t, mu):
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

    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


def initial_loss_soft(model, x0, v0):
    # x(0)=x0
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    x_pred = model(t0)
    loss_x = (x_pred - x0) ** 2

    # x'(0)=v0
    x_t_pred = torch.autograd.grad(
        x_pred,
        t0,
        torch.ones_like(x_pred),
        create_graph=True
    )[0]
    loss_v = (x_t_pred - v0) ** 2

    return torch.mean(loss_x + loss_v)


# =========================================================
# 4. Train soft model
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
            print(
                f"[{model_name}] Epoch {epoch+1:5d} | "
                f"Loss = {loss.item():.3e} | "
                f"ODE = {loss_ode.item():.3e} | "
                f"IC = {loss_ic.item():.3e}"
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
    mu = 1.0
    t_min, t_max = 0.0, 10.0
    x0, v0 = 1.0, 0.0

    # -----------------------------------------------------
    # Training setup
    # -----------------------------------------------------
    epochs = 15000
    n_col = 200
    lr = 3e-3
    w_ic = 1.0
    eval_every = 50

    # -----------------------------------------------------
    # Reference solution
    # -----------------------------------------------------
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
    # Soft PINN
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

    # =====================================================
    # 6. Final prediction
    # =====================================================
    with torch.no_grad():
        x_soft = soft_result["model"](t_eval_tensor).cpu().numpy().flatten()

    soft_abs_l2, soft_rel_l2 = compute_l2_errors(x_soft, x_ref)

    print("\n================ Final L2 Errors ================")
    print(f"mu = {mu}")
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
        label="Reference"
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
    plt.savefig("vdp_pure_soft_solution.png", dpi=300, bbox_inches="tight")
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
    plt.savefig("vdp_pure_soft_abs_l2.png", dpi=300, bbox_inches="tight")
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
    plt.savefig("vdp_pure_soft_rel_l2.png", dpi=300, bbox_inches="tight")
    plt.show()
