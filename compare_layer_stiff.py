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


def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# 1. Exact solution
# =========================================================
def exact_solution(x, eps):
    
    return np.expm1(-x / eps) / np.expm1(-1.0 / eps)


# =========================================================
# 2. Meshes in the independent variable x
# =========================================================
def uniform_mesh(N, xL=0.0, xR=1.0, device="cpu", dtype=torch.float32):
    x = torch.linspace(xL, xR, N + 1, dtype=dtype, device=device).reshape(-1, 1)
    return x


def shishkin_mesh_left(
    N,
    eps,
    xL=0.0,
    xR=1.0,
    sigma_factor=8.0,
    device="cpu",
    dtype=torch.float32,
    return_info=False
):
    """
    Shishkin-type mesh refined near the left boundary layer x=0.

    The layer width is O(eps log N).  We use half of the points in [0,sigma]
    and half in [sigma,1].
    """
    if N % 2 != 0:
        N += 1

    L = xR - xL
    sigma = min(0.5 * L, sigma_factor * eps * np.log(N))
    x_mid = xL + sigma

    left = np.linspace(xL, x_mid, N // 2 + 1)
    right = np.linspace(x_mid, xR, N // 2 + 1)

    x = np.concatenate([left[:-1], right])
    x_tensor = torch.tensor(x, dtype=dtype, device=device).reshape(-1, 1)

    if return_info:
        return x_tensor, sigma, x_mid
    return x_tensor


# =========================================================
# 3. Feature map
#    Phi_eps(x) = [x, - x/eps]
# =========================================================
class FeatureMap(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):
        return torch.cat([x, - x / self.eps], dim=-1)


# =========================================================
# 4. Hard PINN
# =========================================================
class HardPINN(nn.Module):
    """
    Hard-constrained PINN:
        u_theta(x) = x + x(1-x) N_theta(x).

    This enforces u_theta(0)=0 and u_theta(1)=1 exactly.
    """
    def __init__(self, width=32):
        super().__init__()
        self.raw_net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return x + x * (1.0 - x) * self.raw_net(x)


# =========================================================
# 5. Feature + Hard PINN
# =========================================================
class FeatureHardPINN(nn.Module):

    def __init__(self, eps, width=32):
        super().__init__()
        self.feature_map = FeatureMap(eps)
        self.raw_net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        z = self.feature_map(x)
        return x + x * (1.0 - x) * self.raw_net(z)


# =========================================================
# 6. Residual loss
# =========================================================
def residual_loss(model, x, eps):
    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    u_x = torch.autograd.grad(
        u,
        x,
        torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        torch.ones_like(u_x),
        create_graph=True
    )[0]

    res = eps * u_xx + u_x
    return torch.mean(res ** 2)


# =========================================================
# 7. Error computation
# =========================================================
def compute_l2_errors(model, eps, n_test=2000):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x_test_np = np.linspace(0.0, 1.0, n_test)
    u_true_np = exact_solution(x_test_np, eps)

    x_test = torch.tensor(
        x_test_np,
        dtype=dtype,
        device=device
    ).reshape(-1, 1)

    with torch.no_grad():
        u_pred_np = model(x_test).cpu().numpy().squeeze()

    abs_l2 = np.sqrt(np.mean((u_pred_np - u_true_np) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(u_true_np ** 2)) + 1e-12)

    return abs_l2, rel_l2


# =========================================================
# 8. Training function
# =========================================================
def train_model(
    model,
    model_name,
    eps,
    mesh_type="uniform",
    epochs=10000,
    n_col=100,
    lr=1e-3,
    sigma_factor=4.0,
    eval_every=50
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if mesh_type == "uniform":
        x_domain = uniform_mesh(
            N=n_col,
            device=device,
            dtype=dtype
        )
        x_mid_used = None
        sigma_used = None

    elif mesh_type == "shishkin":
        x_domain, sigma_used, x_mid_used = shishkin_mesh_left(
            N=n_col,
            eps=eps,
            sigma_factor=sigma_factor,
            device=device,
            dtype=dtype,
            return_info=True
        )

    else:
        raise ValueError("mesh_type must be 'uniform' or 'shishkin'.")

    loss_history = []
    l2_epochs = []
    abs_l2_history = []
    rel_l2_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = residual_loss(model, x_domain, eps)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            abs_l2, rel_l2 = compute_l2_errors(model, eps, n_test=2000)
            l2_epochs.append(epoch + 1)
            abs_l2_history.append(abs_l2)
            rel_l2_history.append(rel_l2)

        if (epoch + 1) % 1000 == 0:
            print(
                f"[{model_name}] Epoch {epoch+1:5d} | "
                f"Loss = {loss.item():.3e} | "
                f"Rel L2 = {rel_l2_history[-1]:.3e}"
            )

    return {
        "model": model,
        "loss": loss_history,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history,
        "x_mid": x_mid_used,
        "sigma": sigma_used
    }


# =========================================================
# 9. Main comparison
# =========================================================
if __name__ == "__main__":

    # Problem setup
    eps = 1e-2

    # Training setup
    epochs = 10000
    n_col = 100
    lr = 1e-3
    eval_every = 50
    sigma_factor = 4.0
    width = 32

    # -----------------------------------------------------
    # 1) Hard PINN + Uniform mesh
    # -----------------------------------------------------
    set_seed(1234)
    hard_uniform = HardPINN(width=width)

    result_hard_uniform = train_model(
        model=hard_uniform,
        model_name="Hard PINN + Uniform",
        eps=eps,
        mesh_type="uniform",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 2) Hard PINN + Shishkin mesh near x=0
    # -----------------------------------------------------
    set_seed(1234)
    hard_shishkin = HardPINN(width=width)

    result_hard_shishkin = train_model(
        model=hard_shishkin,
        model_name="Hard PINN + Shishkin",
        eps=eps,
        mesh_type="shishkin",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 3) Feature + Hard PINN + Uniform mesh
    # -----------------------------------------------------
    set_seed(1234)
    feature_hard_uniform = FeatureHardPINN(eps=eps, width=width)

    result_feature_uniform = train_model(
        model=feature_hard_uniform,
        model_name="Feature + Hard PINN + Uniform",
        eps=eps,
        mesh_type="uniform",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every
    )

    # -----------------------------------------------------
    # 4) Feature + Hard PINN + Shishkin mesh near x=0
    # -----------------------------------------------------
    set_seed(1234)
    feature_hard_shishkin = FeatureHardPINN(eps=eps, width=width)

    result_feature_shishkin = train_model(
        model=feature_hard_shishkin,
        model_name="Feature + Hard PINN + Shishkin",
        eps=eps,
        mesh_type="shishkin",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every
    )

    # =====================================================
    # 10. Final prediction
    # =====================================================
    x_ref = np.linspace(0.0, 1.0, 2000)
    u_ref = exact_solution(x_ref, eps)

    x_plot = torch.tensor(
        x_ref,
        dtype=torch.float32
    ).reshape(-1, 1)

    with torch.no_grad():
        u_hard_uniform = result_hard_uniform["model"](x_plot).cpu().numpy().squeeze()
        u_hard_shishkin = result_hard_shishkin["model"](x_plot).cpu().numpy().squeeze()
        u_feature_uniform = result_feature_uniform["model"](x_plot).cpu().numpy().squeeze()
        u_feature_shishkin = result_feature_shishkin["model"](x_plot).cpu().numpy().squeeze()

    # Final errors
    methods_final = [
        ("Hard PINN + Uniform", u_hard_uniform),
        ("Hard PINN + Shishkin", u_hard_shishkin),
        ("Feature + Hard PINN + Uniform", u_feature_uniform),
        ("Feature + Hard PINN + Shishkin", u_feature_shishkin),
    ]

    print("\n================ Final L2 Errors ================")
    for name, u_pred in methods_final:
        abs_l2 = np.sqrt(np.mean((u_pred - u_ref) ** 2))
        rel_l2 = abs_l2 / (np.sqrt(np.mean(u_ref ** 2)) + 1e-12)
        print(f"{name:35s} | Abs L2 = {abs_l2:.6e} | Rel L2 = {rel_l2:.6e}")

    # =====================================================
    # 11. Solution comparison: 2x2 subplots
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    methods_plot = [
        ("Hard PINN + Uniform", u_hard_uniform, "red"),
        ("Hard PINN + Shishkin", u_hard_shishkin, "blue"),
        ("Feature + Hard + Uniform", u_feature_uniform, "green"),
        ("Feature + Hard + Shishkin", u_feature_shishkin, "purple"),
    ]

    for ax, (name, u_pred, color) in zip(axes.flat, methods_plot):
        ax.plot(
            x_ref,
            u_ref,
            color="gray",
            lw=3,
            alpha=0.8,
            linestyle="-",
            label="Exact"
        )

        ax.plot(
            x_ref,
            u_pred,
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
            fontsize=10,
            verticalalignment="top"
        )

        ax.grid(False)
        ax.legend(frameon=True, fontsize=9)

    axes[0, 0].set_ylabel("u(x)")
    axes[1, 0].set_ylabel("u(x)")
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 12. Relative L2 error comparison
    # =====================================================
    plt.figure(figsize=(11, 5))

    plt.semilogy(
        result_hard_uniform["l2_epochs"],
        result_hard_uniform["rel_l2"],
        color="red",
        lw=2,
        label="Hard PINN + Uniform"
    )

    plt.semilogy(
        result_hard_shishkin["l2_epochs"],
        result_hard_shishkin["rel_l2"],
        color="blue",
        lw=2,
        label="Hard PINN + Shishkin"
    )

    plt.semilogy(
        result_feature_uniform["l2_epochs"],
        result_feature_uniform["rel_l2"],
        color="green",
        lw=2,
        label="Feature + Hard + Uniform"
    )

    plt.semilogy(
        result_feature_shishkin["l2_epochs"],
        result_feature_shishkin["rel_l2"],
        color="purple",
        lw=2,
        label="Feature + Hard + Shishkin"
    )

    plt.xlabel("Epoch")
    plt.ylabel(r"Relative $L^2$ Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
