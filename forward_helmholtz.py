import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================================================
# 0. Settings
# =========================================================
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

SEED = 1234
K_TRUE = 4.0

# Training setup
FORWARD_EPOCHS = 5000
EVAL_EVERY = 100

LR_NET = 1e-3
BATCH_RES = 256

# Point sets
N_RES_SIDE = 36        # residual pool: 36 x 36
N_TEST_SIDE = 70       # test grid: 70 x 70

# Network setup
WIDTH = 48
DEPTH = 2
RFF_DIM = 64           # number of random frequencies
RFF_SCALE = 2.0        # frequency scale sigma

# Loss weight. The PDE loss is not normalised.
W_PDE = 1.0


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =========================================================
# 1. Manufactured Helmholtz problem
# =========================================================
MODES = [
    (1.00, 1, 1),
    (0.45, 3, 2),
    (0.30, 5, 4),
]


def u_exact(x, y):
    out = torch.zeros_like(x)
    for amp, m, n in MODES:
        out = out + amp * torch.sin(m * math.pi * x) * torch.sin(n * math.pi * y)
    return out


def f_known(x, y, k_true=K_TRUE):

    out = torch.zeros_like(x)
    for amp, m, n in MODES:
        lap_factor = (math.pi ** 2) * (m ** 2 + n ** 2)
        out = out + amp * (lap_factor - k_true ** 2) * (
            torch.sin(m * math.pi * x) * torch.sin(n * math.pi * y)
        )
    return out


def relative_l2(pred, true):
    return (
        torch.sqrt(torch.mean((pred - true) ** 2))
        / (torch.sqrt(torch.mean(true ** 2)) + 1e-12)
    ).item()


# =========================================================
# 2. Point generation
# =========================================================
def make_grid(n, lo=0.0, hi=1.0):
    xs = torch.linspace(lo, hi, n, device=device, dtype=dtype)
    ys = torch.linspace(lo, hi, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    return X.reshape(-1, 1), Y.reshape(-1, 1)


def build_points():
    # Interior residual points. No exact u is used here.
    x_res, y_res = make_grid(N_RES_SIDE, 0.01, 0.99)
    f_res = f_known(x_res, y_res).detach()

    # Test points for plotting and error measurement.
    x_test, y_test = make_grid(N_TEST_SIDE, 0.0, 1.0)
    u_test = u_exact(x_test, y_test).detach()
    f_test = f_known(x_test, y_test).detach()

    return {
        "x_res": x_res,
        "y_res": y_res,
        "f_res": f_res,
        "x_test": x_test,
        "y_test": y_test,
        "u_test": u_test,
        "f_test": f_test,
    }


# =========================================================
# 3. Networks
# =========================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, mapping_size=64, scale=1.0):
        super().__init__()
        B = scale * torch.randn(mapping_size, in_dim)
        self.register_buffer("B", B)

    def forward(self, xy):
        proj = 2.0 * math.pi * xy @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, width=48, depth=2):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, width))
            layers.append(nn.Tanh())
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class HardHelmholtzForwardPINN(nn.Module):
    def __init__(
        self,
        use_rff=False,
        width=48,
        depth=2,
        rff_dim=64,
        rff_scale=2.0,
    ):
        super().__init__()
        self.use_rff = use_rff

        if use_rff:
            self.rff = FourierFeatures(in_dim=2, mapping_size=rff_dim, scale=rff_scale)
            self.net = MLP(in_dim=2 * rff_dim, width=width, depth=depth)
        else:
            self.net = MLP(in_dim=2, width=width, depth=depth)

    def raw_output(self, x, y):
        xy = torch.cat([x, y], dim=1)
        if self.use_rff:
            xy = self.rff(xy)
        return self.net(xy)

    def forward(self, x, y):
        # Hard constraint for u=0 on the boundary of [0,1]^2.
        boundary_factor = torch.sin(math.pi * x) * torch.sin(math.pi * y)
        return boundary_factor * self.raw_output(x, y)


# =========================================================
# 4. PINN residual
# =========================================================
def laplacian_model(model, x, y):
    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)

    u = model(x_req, y_req)

    u_x = torch.autograd.grad(
        u, x_req, torch.ones_like(u), create_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u, y_req, torch.ones_like(u), create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x_req, torch.ones_like(u_x), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y_req, torch.ones_like(u_y), create_graph=True
    )[0]

    return u, u_xx + u_yy


def pde_residual_loss(model, x, y, f_val):
    u, lap_u = laplacian_model(model, x, y)
    res = -lap_u - K_TRUE ** 2 * u - f_val
    return torch.mean(res ** 2)


# =========================================================
# 5. Training
# =========================================================
def sample_batch(x, y, z, batch_size):
    n = x.shape[0]
    idx = torch.randint(0, n, (batch_size,), device=device)
    return x[idx], y[idx], z[idx]


def evaluate_model(model, pts):
    model.eval()

    with torch.no_grad():
        u_pred = model(pts["x_test"], pts["y_test"])
        rel_u = relative_l2(u_pred, pts["u_test"])

    model.train()
    return rel_u


def train_model(model, name, pts):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_NET)

    history = {
        "epoch": [],
        "rel_u": [],
        "loss": [],
        "loss_pde": [],
    }

    for epoch in range(1, FORWARD_EPOCHS + 1):
        xr, yr, fr = sample_batch(
            pts["x_res"], pts["y_res"], pts["f_res"], BATCH_RES
        )

        optimizer.zero_grad()

        loss_pde = pde_residual_loss(model, xr, yr, fr)
        loss = W_PDE * loss_pde

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            rel_u = evaluate_model(model, pts)

            history["epoch"].append(epoch)
            history["rel_u"].append(rel_u)
            history["loss"].append(loss.item())
            history["loss_pde"].append(loss_pde.item())

            print(
                f"[{name:13s}] epoch {epoch:5d} | "
                f"RelL2(u)={rel_u:.3e} | "
                f"loss_pde={loss_pde.item():.3e}"
            )

    return {"name": name, "model": model, "history": history}


# =========================================================
# 6. Plotting
# =========================================================
def grid_to_image(values):
    return values.detach().cpu().numpy().reshape(N_TEST_SIDE, N_TEST_SIDE)


def plot_results(std_result, rff_result, pts):

    std_model = std_result["model"]
    rff_model = rff_result["model"]

    with torch.no_grad():
        u_true = pts["u_test"]
        u_std = std_model(pts["x_test"], pts["y_test"])
        u_rff = rff_model(pts["x_test"], pts["y_test"])

    # Absolute errors
    e_std = torch.abs(u_std - u_true)
    e_rff = torch.abs(u_rff - u_true)

    # -----------------------------------------------------
    # Figure 1: two error heatmaps, each with its own colour scale
    # -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    im0 = axes[0].imshow(
        grid_to_image(e_std),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto"
    )
    axes[0].set_title("Standard |u error|")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        grid_to_image(e_rff),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto"
    )
    axes[1].set_title("RFF-PINN |u error|")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Figure 2: relative L2 error history of u
    # -----------------------------------------------------
    h0 = std_result["history"]
    h1 = rff_result["history"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))

    ax.semilogy(h0["epoch"], h0["rel_u"], lw=2, label="Standard PINN")
    ax.semilogy(h1["epoch"], h1["rel_u"], lw=2, label="RFF-PINN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Relative $L^2$ error of $u$")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    print("\n================ Final Results ================")
    for result in [std_result, rff_result]:
        h = result["history"]
        print(
            f"{result['name']:13s} | "
            f"RelL2(u)={h['rel_u'][-1]:.6e}"
        )


# =========================================================
# 7. Main
# =========================================================
if __name__ == "__main__":
    print("Using device:", device)
    print(f"Known parameter k = {K_TRUE}")

    set_seed(SEED)
    pts = build_points()

    set_seed(SEED)
    standard = train_model(
        HardHelmholtzForwardPINN(
            use_rff=False,
            width=WIDTH,
            depth=DEPTH,
            rff_dim=RFF_DIM,
            rff_scale=RFF_SCALE,
        ),
        "Standard PINN",
        pts,
    )

    set_seed(SEED)
    rff = train_model(
        HardHelmholtzForwardPINN(
            use_rff=True,
            width=WIDTH,
            depth=DEPTH,
            rff_dim=RFF_DIM,
            rff_scale=RFF_SCALE,
        ),
        "RFF-PINN",
        pts,
    )

    plot_results(standard, rff, pts)
