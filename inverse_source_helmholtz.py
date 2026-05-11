import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================================================
# 0. Settings
# =========================================================
USE_FLOAT64 = True
if USE_FLOAT64:
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
else:
    torch.set_default_dtype(torch.float32)
    dtype = torch.float32

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1234
K = 2.0

# Data and evaluation grids
N_DATA_SIDE = 45
N_TEST_SIDE = 110
INTERIOR_EPS_TRAIN = 0.005
INTERIOR_EPS_TEST = 0.02
USE_CHEBYSHEV_DATA = True

# Training schedule
# The two models use the same number of Adam epochs and the same evaluation interval.
EPOCHS = 5000
EVAL_EVERY = 100

# Optimizer
# A common learning rate and the same L-BFGS polishing budget are used for fairness.
LR = 5e-4
USE_LBFGS = True
LBFGS_MAX_ITER = 120

# Network parameters
# The downstream MLP width/depth are identical; the only architectural difference
# is the fixed random Fourier feature map inserted before the RFF-PINN MLP.
HIDDEN_WIDTH = 96
HIDDEN_DEPTH = 2
RFF_DIM = 128
RFF_SCALE = 0.8

# Loss options
USE_RAW_TARGET_LOSS = True
RAW_EPS = 1e-12
SOLUTION_LOSS_WEIGHT = 0.05


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. Manufactured solution and source
# =========================================================
def u_exact(x, y):
    """A smooth oscillatory solution satisfying u=0 on the boundary."""
    return (
        x * (1.0 - x) * y * (1.0 - y)
        * torch.sin(2.0 * np.pi * x)
        * torch.sin(2.0 * np.pi * y)
    )


def f_exact(x, y, k=K):
    """Exact source computed from the manufactured solution; used only for evaluation."""
    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)

    u = u_exact(x_req, y_req)
    u_x = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_req, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_req, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_req, torch.ones_like(u_y), create_graph=True)[0]

    return (-u_xx - u_yy - k**2 * u).detach()


# =========================================================
# 2. Networks
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim, width=64, depth=2):
        super().__init__()
        layers = []
        last_dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.Tanh())
            last_dim = width
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class RFFLayer(nn.Module):
    def __init__(self, in_dim=2, mapping_size=128, scale=0.8):
        super().__init__()
        B = scale * torch.randn(mapping_size, in_dim, dtype=dtype)
        self.register_buffer("B", B)

    def forward(self, xy):
        proj = 2.0 * np.pi * xy @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


def hard_factor(x, y):
    """Hard constraint factor for u=0 on the boundary of the unit square."""
    return x * (1.0 - x) * y * (1.0 - y)


class StandardHardPINN(nn.Module):
    def __init__(self, width=48, depth=2):
        super().__init__()
        self.net = MLP(in_dim=2, width=width, depth=depth)

    def raw(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

    def forward(self, x, y):
        return hard_factor(x, y) * self.raw(x, y)


class RFFHardPINN(nn.Module):
    def __init__(self, width=96, depth=2, rff_dim=128, rff_scale=0.8):
        super().__init__()
        self.rff = RFFLayer(in_dim=2, mapping_size=rff_dim, scale=rff_scale)
        self.net = MLP(in_dim=2 * rff_dim, width=width, depth=depth)

    def raw(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.net(self.rff(xy))

    def forward(self, x, y):
        return hard_factor(x, y) * self.raw(x, y)


# =========================================================
# 3. Differentiation and errors
# =========================================================
def helmholtz_source_from_u(u_model, x, y, k=K):
    """Compute -Delta u_theta - k^2 u_theta using automatic differentiation."""
    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)

    u = u_model(x_req, y_req)
    u_x = torch.autograd.grad(
        u, x_req, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u, y_req, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_req, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y_req, torch.ones_like(u_y), create_graph=True, retain_graph=True
    )[0]

    return -u_xx - u_yy - k**2 * u


def relative_l2(pred, true):
    return (
        torch.sqrt(torch.mean((pred - true) ** 2))
        / (torch.sqrt(torch.mean(true**2)) + 1e-12)
    ).item()


def make_grid(n, eps=0.0, chebyshev=False):
    if chebyshev:
        j = torch.arange(n, device=device, dtype=dtype).reshape(-1, 1)
        s = 0.5 * (1.0 - torch.cos(np.pi * j / (n - 1)))
        xs = eps + (1.0 - 2.0 * eps) * s
        ys = xs.clone()
    else:
        xs = torch.linspace(eps, 1.0 - eps, n, device=device, dtype=dtype).reshape(-1, 1)
        ys = torch.linspace(eps, 1.0 - eps, n, device=device, dtype=dtype).reshape(-1, 1)

    X, Y = torch.meshgrid(xs.squeeze(), ys.squeeze(), indexing="ij")
    return X.reshape(-1, 1), Y.reshape(-1, 1), X.detach().cpu().numpy(), Y.detach().cpu().numpy()


# =========================================================
# 4. Training
# =========================================================
def solution_loss(model, x_data, y_data, u_data):
    u_pred = model(x_data, y_data)
    loss_solution = torch.mean((u_pred - u_data) ** 2)

    if not USE_RAW_TARGET_LOSS:
        return loss_solution

    # Since u_theta = h*N_theta, fitting N_theta to u_data/h prevents the raw
    # network from becoming under-constrained near the boundary where h is small.
    h = hard_factor(x_data, y_data)
    raw_target = (u_data / (h + RAW_EPS)).detach()
    loss_raw = torch.mean((model.raw(x_data, y_data) - raw_target) ** 2)
    return loss_raw + SOLUTION_LOSS_WEIGHT * loss_solution


def train_solution_model(model, name, epochs, lr, lbfgs_max_iter=0):
    x_data, y_data, _, _ = make_grid(
        N_DATA_SIDE, eps=INTERIOR_EPS_TRAIN, chebyshev=USE_CHEBYSHEV_DATA
    )
    u_data = u_exact(x_data, y_data).detach()

    x_test, y_test, _, _ = make_grid(N_TEST_SIDE, eps=INTERIOR_EPS_TEST, chebyshev=False)
    u_test = u_exact(x_test, y_test).detach()
    f_test = f_exact(x_test, y_test).detach()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist_epoch = []
    hist_u = []
    hist_f = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = solution_loss(model, x_data, y_data, u_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            with torch.no_grad():
                u_pred_test = model(x_test, y_test)
                rel_u = relative_l2(u_pred_test, u_test)

            # The source error is only monitored, not used in the loss.
            # It needs autograd because f_theta is obtained from second derivatives of u_theta.
            f_pred_test = helmholtz_source_from_u(model, x_test, y_test).detach()
            rel_f = relative_l2(f_pred_test, f_test)

            hist_epoch.append(epoch + 1)
            hist_u.append(rel_u)
            hist_f.append(rel_f)

        if (epoch + 1) % 1000 == 0:
            print(
                f"[{name}] Epoch {epoch+1:5d} | "
                f"loss = {loss.item():.3e} | "
                f"RelL2_u = {hist_u[-1]:.3e} | RelL2_f = {hist_f[-1]:.3e}"
            )

    if USE_LBFGS and lbfgs_max_iter > 0:
        print(f"[{name}] Starting L-BFGS polishing ...")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=lbfgs_max_iter,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            loss = solution_loss(model, x_data, y_data, u_data)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)
        with torch.no_grad():
            u_pred_test = model(x_test, y_test)
            rel_u = relative_l2(u_pred_test, u_test)
        f_pred_test = helmholtz_source_from_u(model, x_test, y_test).detach()
        rel_f = relative_l2(f_pred_test, f_test)

        hist_epoch.append(epochs + lbfgs_max_iter)
        hist_u.append(rel_u)
        hist_f.append(rel_f)
        print(f"[{name}] After L-BFGS | RelL2_u = {rel_u:.3e} | RelL2_f = {rel_f:.3e}")

    return {"model": model, "epochs": hist_epoch, "rel_u": hist_u, "rel_f": hist_f}


# =========================================================
# 5. Evaluation and plotting
# =========================================================
def evaluate_model(model):
    x, y, X, Y = make_grid(N_TEST_SIDE, eps=INTERIOR_EPS_TEST, chebyshev=False)
    u_true = u_exact(x, y).detach()
    f_true = f_exact(x, y).detach()

    with torch.no_grad():
        u_pred = model(x, y)
    f_pred = helmholtz_source_from_u(model, x, y).detach()

    out = {
        "X": X,
        "Y": Y,
        "u_true": u_true.cpu().numpy().reshape(N_TEST_SIDE, N_TEST_SIDE),
        "u_pred": u_pred.cpu().numpy().reshape(N_TEST_SIDE, N_TEST_SIDE),
        "f_true": f_true.cpu().numpy().reshape(N_TEST_SIDE, N_TEST_SIDE),
        "f_pred": f_pred.cpu().numpy().reshape(N_TEST_SIDE, N_TEST_SIDE),
        "rel_u": relative_l2(u_pred, u_true),
        "rel_f": relative_l2(f_pred, f_true),
        "max_u": torch.max(torch.abs(u_pred - u_true)).item(),
        "max_f": torch.max(torch.abs(f_pred - f_true)).item(),
    }
    out["u_abs"] = np.abs(out["u_pred"] - out["u_true"])
    out["f_abs"] = np.abs(out["f_pred"] - out["f_true"])
    return out


def plot_results(std_result, rff_result):
    std = evaluate_model(std_result["model"])
    rff = evaluate_model(rff_result["model"])


    # Each heatmap uses its own scale. The panel reports its own maximum value.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    items = [
        (std["u_abs"], "Standard PINN |u error|"),
        (std["f_abs"], "Standard PINN |f error|"),
        (rff["u_abs"], "RFF-PINN |u error|"),
        (rff["f_abs"], "RFF-PINN |f error|"),
    ]

    for ax, (Z, title) in zip(axes.flat, items):
        zmax = float(np.max(Z))
        im = ax.imshow(
            Z.T,
            extent=[INTERIOR_EPS_TEST, 1 - INTERIOR_EPS_TEST,
                    INTERIOR_EPS_TEST, 1 - INTERIOR_EPS_TEST],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=zmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(
            0.02, 0.98, f"max = {zmax:.3e}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0.0, 0.25 * zmax, 0.5 * zmax, 0.75 * zmax, zmax])

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.semilogy(std_result["epochs"], std_result["rel_u"], lw=2, label="Standard PINN")
    plt.semilogy(rff_result["epochs"], rff_result["rel_u"], lw=2, label="RFF-PINN")
    plt.xlabel("Epoch")
    plt.ylabel("Relative $L^2$ error of $u$")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.semilogy(std_result["epochs"], std_result["rel_f"], lw=2, label="Standard PINN")
    plt.semilogy(rff_result["epochs"], rff_result["rel_f"], lw=2, label="RFF-PINN")
    plt.xlabel("Epoch")
    plt.ylabel("Relative $L^2$ error of $f$")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Source reconstruction comparison with a shared colour scale.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    recon_items = [
        (std["f_true"], "Exact source"),
        (std["f_pred"], "Standard reconstructed source"),
        (rff["f_pred"], "RFF reconstructed source"),
    ]
    vmin = min(float(Z.min()) for Z, _ in recon_items)
    vmax = max(float(Z.max()) for Z, _ in recon_items)
    for ax, (Z, title) in zip(axes, recon_items):
        im = ax.imshow(
            Z.T,
            extent=[INTERIOR_EPS_TEST, 1 - INTERIOR_EPS_TEST,
                    INTERIOR_EPS_TEST, 1 - INTERIOR_EPS_TEST],
            origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return std, rff


# =========================================================
# 6. Main
# =========================================================
def main(quick=False):
    global N_DATA_SIDE, N_TEST_SIDE, EPOCHS, RFF_DIM, HIDDEN_WIDTH, LBFGS_MAX_ITER

    if quick:
        N_DATA_SIDE = 20
        N_TEST_SIDE = 40
        EPOCHS = 100
        RFF_DIM = 32
        HIDDEN_WIDTH = 48
        LBFGS_MAX_ITER = 0

    print("Using device:", device)
    print("dtype:", dtype)
    print("Common setting: epochs =", EPOCHS, "lr =", LR, "hidden width =", HIDDEN_WIDTH)

    set_seed(SEED)
    std_model = StandardHardPINN(width=HIDDEN_WIDTH, depth=HIDDEN_DEPTH).to(device)
    std_result = train_solution_model(
        std_model,
        name="Standard PINN",
        epochs=EPOCHS,
        lr=LR,
        lbfgs_max_iter=LBFGS_MAX_ITER,
    )

    set_seed(SEED)
    rff_model = RFFHardPINN(
        width=HIDDEN_WIDTH,
        depth=HIDDEN_DEPTH,
        rff_dim=RFF_DIM,
        rff_scale=RFF_SCALE,
    ).to(device)
    rff_result = train_solution_model(
        rff_model,
        name="RFF-PINN",
        epochs=EPOCHS,
        lr=LR,
        lbfgs_max_iter=LBFGS_MAX_ITER,
    )

    std, rff = plot_results(std_result, rff_result)

    print("\n================ Final Errors ================")
    print(f"Standard PINN | Rel L2 u = {std['rel_u']:.6e} | Rel L2 f = {std['rel_f']:.6e}")
    print(f"RFF-PINN      | Rel L2 u = {rff['rel_u']:.6e} | Rel L2 f = {rff['rel_f']:.6e}")
    print(f"Standard PINN | Max |u error| = {std['max_u']:.6e} | Max |f error| = {std['max_f']:.6e}")
    print(f"RFF-PINN      | Max |u error| = {rff['max_u']:.6e} | Max |f error| = {rff['max_f']:.6e}")


if __name__ == "__main__":
    main()
