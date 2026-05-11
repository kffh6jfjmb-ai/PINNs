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
def exact_solution(t, eps):
    return t + (
        np.exp(-(1.0 - t) / eps) - np.exp(-1.0 / eps)
    ) / (
        np.exp(-1.0 / eps) - 1.0
    )


# =========================================================
# 2. Meshes
# =========================================================
def uniform_mesh(N, tL=0.0, tR=1.0, device="cpu", dtype=torch.float32):
    t = torch.linspace(tL, tR, N + 1, dtype=dtype, device=device).reshape(-1, 1)
    return t


def shishkin_mesh(
    N,
    eps,
    tL=0.0,
    tR=1.0,
    sigma_factor=4.0,
    device="cpu",
    dtype=torch.float32,
    return_info=False
):
    if N % 2 != 0:
        N += 1

    L = tR - tL
    sigma = min(0.5 * L, sigma_factor * eps * np.log(N))
    t_mid = tR - sigma

    left = np.linspace(tL, t_mid, N // 2 + 1)
    right = np.linspace(t_mid, tR, N // 2 + 1)

    t = np.concatenate([left[:-1], right])
    t_tensor = torch.tensor(t, dtype=dtype, device=device).reshape(-1, 1)

    if return_info:
        return t_tensor, sigma, t_mid
    return t_tensor


# =========================================================
# 3. Feature map
# =========================================================
class FeatureMap(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, t):
        return torch.cat([t, (1.0 - t) / self.eps], dim=-1)


# =========================================================
# 4. Hard PINN without feature map
# =========================================================
class HardPINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()

        self.raw_net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def x_hat(self, t):
        return self.raw_net(t)

    def forward(self, t):
        t0 = torch.tensor([[0.0]], dtype=t.dtype, device=t.device)
        t1 = torch.tensor([[1.0]], dtype=t.dtype, device=t.device)

        xh = self.x_hat(t)
        xh0 = self.x_hat(t0)
        xh1 = self.x_hat(t1)

        # hard constraint: x(0)=0, x(1)=0
        x = xh - (1.0 - t) * xh0 - t * xh1
        return x


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

    def x_hat(self, t):
        z = self.feature_map(t)
        return self.raw_net(z)

    def forward(self, t):
        t0 = torch.tensor([[0.0]], dtype=t.dtype, device=t.device)
        t1 = torch.tensor([[1.0]], dtype=t.dtype, device=t.device)

        xh = self.x_hat(t)
        xh0 = self.x_hat(t0)
        xh1 = self.x_hat(t1)

        # hard constraint: x(0)=0, x(1)=0
        x = xh - (1.0 - t) * xh0 - t * xh1
        return x


# =========================================================
# 6. Residual loss
# =========================================================
def residual_loss(model, t, eps):
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

    res = -eps * x_tt + x_t - 1.0
    return torch.mean(res ** 2)


# =========================================================
# 7. Error computation
# =========================================================
def compute_l2_errors(model, eps, n_test=2000):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_test_np = np.linspace(0.0, 1.0, n_test)
    x_true_np = exact_solution(t_test_np, eps)

    t_test = torch.tensor(
        t_test_np,
        dtype=dtype,
        device=device
    ).reshape(-1, 1)

    with torch.no_grad():
        x_pred_np = model(t_test).cpu().numpy().squeeze()

    abs_l2 = np.sqrt(np.mean((x_pred_np - x_true_np) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_true_np ** 2)) + 1e-12)

    return abs_l2, rel_l2


# =========================================================
# 8. Training function
# =========================================================
def train_model(
    model,
    model_name,
    eps,
    mesh_type="uniform",
    epochs=20000,
    n_col=100,
    lr=1e-3,
    sigma_factor=4.0,
    eval_every=50
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if mesh_type == "uniform":
        t_domain = uniform_mesh(
            N=n_col,
            tL=0.0,
            tR=1.0,
            device=device,
            dtype=dtype
        )
        t_mid_used = None

    elif mesh_type == "shishkin":
        t_domain, sigma_used, t_mid_used = shishkin_mesh(
            N=n_col,
            eps=eps,
            tL=0.0,
            tR=1.0,
            sigma_factor=sigma_factor,
            device=device,
            dtype=dtype,
            return_info=True
        )
        print(
            f"[{model_name}] Shishkin sigma = {sigma_used:.6e}, "
            f"t_mid = {t_mid_used:.6f}"
        )

    else:
        raise ValueError("mesh_type must be 'uniform' or 'shishkin'.")

    loss_history = []
    l2_epochs = []
    abs_l2_history = []
    rel_l2_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = residual_loss(model, t_domain, eps)

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
        "t_mid": t_mid_used
    }


# =========================================================
# 9. Main comparison
# =========================================================
if __name__ == "__main__":

    # Problem setup
    eps = 1e-2

    # Training setup
    epochs = 20000
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
    # 2) Hard PINN + Shishkin mesh
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
    # 4) Feature + Hard PINN + Shishkin mesh
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
    t_ref = np.linspace(0.0, 1.0, 2000)
    x_ref = exact_solution(t_ref, eps)

    t_plot = torch.tensor(
        t_ref,
        dtype=torch.float32
    ).reshape(-1, 1)

    with torch.no_grad():
        x_hard_uniform = result_hard_uniform["model"](t_plot).cpu().numpy().squeeze()
        x_hard_shishkin = result_hard_shishkin["model"](t_plot).cpu().numpy().squeeze()
        x_feature_uniform = result_feature_uniform["model"](t_plot).cpu().numpy().squeeze()
        x_feature_shishkin = result_feature_shishkin["model"](t_plot).cpu().numpy().squeeze()

    # Final errors
    methods_final = [
        ("Hard PINN + Uniform", x_hard_uniform),
        ("Hard PINN + Shishkin", x_hard_shishkin),
        ("Feature + Hard PINN + Uniform", x_feature_uniform),
        ("Feature + Hard PINN + Shishkin", x_feature_shishkin),
    ]

    print("\n================ Final L2 Errors ================")
    for name, x_pred in methods_final:
        abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
        rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
        print(f"{name:35s} | Abs L2 = {abs_l2:.6e} | Rel L2 = {rel_l2:.6e}")

    # =====================================================
    # 11. Solution comparison: 2x2 subplots
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    methods_plot = [
        ("Hard PINN + Uniform", x_hard_uniform, "red"),
        ("Hard PINN + Shishkin", x_hard_shishkin, "blue"),
        ("Feature + Hard + Uniform", x_feature_uniform, "green"),
        ("Feature + Hard + Shishkin", x_feature_shishkin, "purple"),
    ]

    for ax, (name, x_pred, color) in zip(axes.flat, methods_plot):
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
            fontsize=10,
            verticalalignment="top"
        )

        ax.grid(False)
        ax.legend(frameon=True, fontsize=9)

    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("x(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 12. Boundary-layer zoom-in near t=1
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, (name, x_pred, color) in zip(axes.flat, methods_plot):
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

        ax.set_xlim(0.97, 1.0)

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

    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("x(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 13. Relative L2 error comparison
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
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 14. Mesh visualization
    # =====================================================
    _, _, t_mid = shishkin_mesh(
        N=n_col,
        eps=eps,
        sigma_factor=sigma_factor,
        return_info=True
    )

    t_uniform_mesh = uniform_mesh(n_col).cpu().numpy().squeeze()
    t_shishkin_mesh = shishkin_mesh(n_col, eps).cpu().numpy().squeeze()

    plt.figure(figsize=(11, 2.2))
    plt.plot(
        t_uniform_mesh,
        np.zeros_like(t_uniform_mesh),
        "o",
        ms=3,
        alpha=0.6,
        label="Uniform mesh"
    )
    plt.plot(
        t_shishkin_mesh,
        0.08 * np.ones_like(t_shishkin_mesh),
        "o",
        ms=3,
        alpha=0.6,
        label="Shishkin mesh"
    )
    plt.axvline(
        t_mid,
        color="black",
        linestyle="--",
        lw=1.5,
        label=f"transition point = {t_mid:.4f}"
    )
    plt.yticks([])
    plt.xlabel("t")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
