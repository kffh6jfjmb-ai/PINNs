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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. Exact solution
#    eps u'' + t^2 u' = 0, u(0)=0, u(1)=1
# =========================================================
class ExactSolution:

    def __init__(self, eps: float, n_quad: int = 60000):
        self.eps = float(eps)
        self.grid = np.linspace(0.0, 1.0, int(n_quad))
        integrand = np.exp(-(self.grid ** 3) / (3.0 * self.eps))

        cum = np.zeros_like(self.grid)
        dx = np.diff(self.grid)
        cum[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dx)

        denom = cum[-1]
        if denom <= 0.0:
            raise RuntimeError("Degenerate quadrature denominator for exact solution.")
        self.values_grid = cum / denom

    def __call__(self, t):
        t_np = np.asarray(t, dtype=float)
        return np.interp(t_np, self.grid, self.values_grid)


# =========================================================
# 2. Meshes in the independent variable t
# =========================================================
def uniform_mesh(N, tL=0.0, tR=1.0, device="cpu", dtype=torch.float32):
    """Interior uniform collocation points.  Boundary conditions are hard-imposed."""
    t = torch.linspace(tL, tR, N + 2, dtype=dtype, device=device).reshape(-1, 1)
    return t[1:-1]


def cubic_layer_mesh(
    N,
    eps,
    tL=0.0,
    tR=1.0,
    sigma_factor=1.5,
    device="cpu",
    dtype=torch.float32,
    return_info=False,
):
    """
    Layer-adapted mesh near t=0 for exp(-t^3/(3 eps)).

    The layer width is O(eps^(1/3)).  To include the exponentially small tail,
    we use
        sigma = sigma_factor * (eps * log(N+1))^(1/3),
    truncated by half the domain length.
    """
    if N % 2 != 0:
        N += 1

    L = tR - tL
    sigma = min(0.5 * L, sigma_factor * (eps * np.log(N + 1.0)) ** (1.0 / 3.0))
    t_mid = tL + sigma

    left = np.linspace(tL, t_mid, N // 2 + 2)[1:-1]
    right = np.linspace(t_mid, tR, N // 2 + 2)[1:-1]
    t = np.concatenate([left, right])
    t_tensor = torch.tensor(t, dtype=dtype, device=device).reshape(-1, 1)

    if return_info:
        return t_tensor, sigma, t_mid
    return t_tensor


# =========================================================
# 3. Feature maps
# =========================================================
class FeatureMap(nn.Module):
    """
    Feature map Phi(t) = [t, t/scale].

    feature_type:
      - "cubic": scale = eps^(1/3), the correct boundary-layer scale;
      - "sqrt":  scale = sqrt(eps), included for comparison;
      - "eps":   scale = eps, included for comparison.
    """
    def __init__(self, eps, feature_type="cubic"):
        super().__init__()
        self.eps = float(eps)
        self.feature_type = feature_type

        if feature_type == "cubic":
            self.scale = self.eps ** (1.0 / 3.0)
        elif feature_type == "sqrt":
            self.scale = np.sqrt(self.eps)
        elif feature_type == "eps":
            self.scale = self.eps
        else:
            raise ValueError("feature_type must be 'cubic', 'sqrt', or 'eps'.")

    def forward(self, t):
        return torch.cat([t, t / self.scale], dim=-1)


# =========================================================
# 4. Hard PINN models
# =========================================================
def make_mlp(in_dim, width=32):
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, 1),
    )


class HardPINN(nn.Module):

    def __init__(self, width=32):
        super().__init__()
        self.raw_net = make_mlp(1, width)

    def forward(self, t):
        return t + t * (1.0 - t) * self.raw_net(t)


class FeatureHardPINN(nn.Module):
 
    def __init__(self, eps, width=32, feature_type="cubic"):
        super().__init__()
        self.feature_map = FeatureMap(eps, feature_type=feature_type)
        self.raw_net = make_mlp(2, width)

    def forward(self, t):
        z = self.feature_map(t)
        return t + t * (1.0 - t) * self.raw_net(z)


# =========================================================
# 5. Residual loss
# =========================================================
def residual_loss(model, t, eps):
    t = t.clone().detach().requires_grad_(True)

    u = model(t)

    u_t = torch.autograd.grad(
        u,
        t,
        torch.ones_like(u),
        create_graph=True,
    )[0]

    u_tt = torch.autograd.grad(
        u_t,
        t,
        torch.ones_like(u_t),
        create_graph=True,
    )[0]

    res = eps * u_tt + (t ** 2) * u_t
    return torch.mean(res ** 2)


# =========================================================
# 6. Error computation
# =========================================================
def compute_l2_errors(model, exact, n_test=2500):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_test_np = np.linspace(0.0, 1.0, n_test)
    u_true_np = exact(t_test_np)

    t_test = torch.tensor(t_test_np, dtype=dtype, device=device).reshape(-1, 1)
    with torch.no_grad():
        u_pred_np = model(t_test).cpu().numpy().squeeze()

    abs_l2 = np.sqrt(np.mean((u_pred_np - u_true_np) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(u_true_np ** 2)) + 1.0e-12)
    return abs_l2, rel_l2


# =========================================================
# 7. Training function
# =========================================================
def train_model(
    model,
    model_name,
    eps,
    exact,
    mesh_type="uniform",
    epochs=20000,
    n_col=100,
    lr=3.0e-3,
    sigma_factor=1.5,
    eval_every=100,
    n_test=2500,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if mesh_type == "uniform":
        t_domain = uniform_mesh(N=n_col, device=device, dtype=dtype)
        t_mid_used = None
        sigma_used = None
    elif mesh_type == "cubic_layer":
        t_domain, sigma_used, t_mid_used = cubic_layer_mesh(
            N=n_col,
            eps=eps,
            sigma_factor=sigma_factor,
            device=device,
            dtype=dtype,
            return_info=True,
        )
        print(
            f"[{model_name}] cubic-layer sigma = {sigma_used:.6e}, "
            f"transition point = {t_mid_used:.6f}"
        )
    else:
        raise ValueError("mesh_type must be 'uniform' or 'cubic_layer'.")

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
            abs_l2, rel_l2 = compute_l2_errors(model, exact, n_test=n_test)
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
        "t_mid": t_mid_used,
        "sigma": sigma_used,
    }


# =========================================================
# 8. Plot helpers
# =========================================================
def show_figure(fig, show=True):
    if show:
        plt.show()
    else:
        plt.close(fig)


def predict_on_grid(result, t_ref):
    model = result["model"]
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    t_plot = torch.tensor(t_ref, dtype=dtype, device=device).reshape(-1, 1)
    with torch.no_grad():
        return model(t_plot).cpu().numpy().squeeze()


# =========================================================
# 9. Main comparison
# =========================================================
def main():
    # Problem and training setup
    eps = 1.0e-4
    epochs = 20000
    n_col = 100
    lr = 3.0e-3
    eval_every = 100
    width = 32
    seed = 1234
    sigma_factor = 1.5
    n_test = 2500
    n_quad = 60000
    compare_scales = False
    show = True

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    exact = ExactSolution(eps, n_quad=n_quad)
    layer_scale = eps ** (1.0 / 3.0)

    print("Using device:", device)
    print(f"eps = {eps:.3e}, natural layer scale eps^(1/3) = {layer_scale:.6e}")

    # -----------------------------------------------------
    # 1) Hard PINN + Uniform mesh
    # -----------------------------------------------------
    set_seed(seed)
    hard_uniform = HardPINN(width=width).to(device=device, dtype=dtype)
    result_hard_uniform = train_model(
        hard_uniform,
        "Hard PINN + Uniform",
        eps,
        exact,
        mesh_type="uniform",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every,
        n_test=n_test,
    )

    # -----------------------------------------------------
    # 2) Hard PINN + cubic-layer mesh near t=0
    # -----------------------------------------------------
    set_seed(seed)
    hard_layer = HardPINN(width=width).to(device=device, dtype=dtype)
    result_hard_layer = train_model(
        hard_layer,
        "Hard PINN + cubic-layer mesh",
        eps,
        exact,
        mesh_type="cubic_layer",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every,
        n_test=n_test,
    )

    # -----------------------------------------------------
    # 3) Correct feature + Hard PINN + Uniform mesh
    # -----------------------------------------------------
    set_seed(seed)
    feature_uniform = FeatureHardPINN(
        eps=eps,
        width=width,
        feature_type="cubic",
    ).to(device=device, dtype=dtype)
    result_feature_uniform = train_model(
        feature_uniform,
        "Feature t/eps^(1/3) + Uniform",
        eps,
        exact,
        mesh_type="uniform",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every,
        n_test=n_test,
    )

    # -----------------------------------------------------
    # 4) Correct feature + Hard PINN + cubic-layer mesh
    # -----------------------------------------------------
    set_seed(seed)
    feature_layer = FeatureHardPINN(
        eps=eps,
        width=width,
        feature_type="cubic",
    ).to(device=device, dtype=dtype)
    result_feature_layer = train_model(
        feature_layer,
        "Feature t/eps^(1/3) + cubic-layer mesh",
        eps,
        exact,
        mesh_type="cubic_layer",
        epochs=epochs,
        n_col=n_col,
        lr=lr,
        sigma_factor=sigma_factor,
        eval_every=eval_every,
        n_test=n_test,
    )

    # =====================================================
    # 10. Final prediction and errors
    # =====================================================
    t_ref = np.linspace(0.0, 1.0, n_test)
    u_ref = exact(t_ref)

    results = [
        ("Hard PINN + Uniform", result_hard_uniform, "red"),
        ("Hard PINN + layer mesh", result_hard_layer, "blue"),
        ("Feature + Hard + Uniform", result_feature_uniform, "green"),
        ("Feature + Hard + layer mesh", result_feature_layer, "purple"),
    ]

    pred = {name: predict_on_grid(res, t_ref) for name, res, _ in results}

    print("\n================ Final L2 Errors ================")
    for name, _, _ in results:
        u_pred = pred[name]
        abs_l2 = np.sqrt(np.mean((u_pred - u_ref) ** 2))
        rel_l2 = abs_l2 / (np.sqrt(np.mean(u_ref ** 2)) + 1.0e-12)
        print(f"{name:38s} | Abs L2 = {abs_l2:.6e} | Rel L2 = {rel_l2:.6e}")

    # =====================================================
    # 11. Solution comparison: 2x2 subplots
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (name, _, color) in zip(axes.flat, results):
        ax.plot(t_ref, u_ref, color="gray", lw=3, alpha=0.8, label="Exact")
        ax.plot(t_ref, pred[name], color=color, lw=2.2, linestyle="--", label="Prediction")
        ax.text(0.03, 0.92, name, transform=ax.transAxes, fontsize=10, va="top")
        ax.grid(False)
        ax.legend(frameon=True, fontsize=9)

    axes[0, 0].set_ylabel("u(t)")
    axes[1, 0].set_ylabel("u(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.tight_layout()
    show_figure(fig, show)

    # =====================================================
    # 12. Boundary-layer zoom near t=0
    # =====================================================
    zoom_right = min(0.35, max(6.0 * layer_scale, 0.08))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (name, _, color) in zip(axes.flat, results):
        ax.plot(t_ref, u_ref, color="gray", lw=3, alpha=0.8, label="Exact")
        ax.plot(t_ref, pred[name], color=color, lw=2.2, linestyle="--", label="Prediction")
        ax.set_xlim(0.0, zoom_right)
        ax.text(0.03, 0.92, name, transform=ax.transAxes, fontsize=10, va="top")
        ax.grid(False)
        ax.legend(frameon=True, fontsize=9)

    axes[0, 0].set_ylabel("u(t)")
    axes[1, 0].set_ylabel("u(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.tight_layout()
    show_figure(fig, show)

    # =====================================================
    # 13. Relative L2 error comparison
    # =====================================================
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res, color in results:
        ax.semilogy(res["l2_epochs"], res["rel_l2"], color=color, lw=2, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    show_figure(fig, show)

    # =====================================================
    # 14. Mesh visualization
    # =====================================================
    _, sigma_used, t_mid = cubic_layer_mesh(
        N=n_col,
        eps=eps,
        sigma_factor=sigma_factor,
        return_info=True,
    )
    t_uniform_mesh = uniform_mesh(n_col).cpu().numpy().squeeze()
    t_layer_mesh = cubic_layer_mesh(n_col, eps, sigma_factor=sigma_factor).cpu().numpy().squeeze()

    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot(t_uniform_mesh, np.zeros_like(t_uniform_mesh), "o", ms=3, alpha=0.6, label="Uniform mesh")
    ax.plot(t_layer_mesh, 0.08 * np.ones_like(t_layer_mesh), "o", ms=3, alpha=0.6, label="cubic-layer mesh")
    ax.axvline(t_mid, color="black", linestyle="--", lw=1.5, label=f"transition point = {t_mid:.4f}")
    ax.axvline(layer_scale, color="gray", linestyle=":", lw=1.5, label=r"$\epsilon^{1/3}$")
    ax.set_yticks([])
    ax.set_xlabel("t")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    show_figure(fig, show)

    # =====================================================
    # 15. Optional feature-scale comparison
    # =====================================================
    if compare_scales:
        scale_results = []
        feature_types = [
            ("Feature $t/\\epsilon$", "eps", "orange"),
            ("Feature $t/\\sqrt{\\epsilon}$", "sqrt", "brown"),
            ("Feature $t/\\epsilon^{1/3}$", "cubic", "purple"),
        ]
        for label, ftype, color in feature_types:
            set_seed(seed)
            model = FeatureHardPINN(eps=eps, width=width, feature_type=ftype).to(device=device, dtype=dtype)
            res = train_model(
                model,
                label,
                eps,
                exact,
                mesh_type="uniform",
                epochs=epochs,
                n_col=n_col,
                lr=lr,
                sigma_factor=sigma_factor,
                eval_every=eval_every,
                n_test=n_test,
            )
            scale_results.append((label, res, color))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(result_hard_uniform["l2_epochs"], result_hard_uniform["rel_l2"], color="red", lw=2, label="No feature")
        for label, res, color in scale_results:
            ax.semilogy(res["l2_epochs"], res["rel_l2"], color=color, lw=2, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative L2 Error")
        ax.legend()
        ax.grid(False)
        fig.tight_layout()
        show_figure(fig, show)


if __name__ == "__main__":
    main()
