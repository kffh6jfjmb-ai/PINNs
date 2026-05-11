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
# =========================================================
def exact_solution(t, n):
   
    return np.sin(n * np.pi * t) / (n * np.pi * np.cos(n * np.pi))


def compute_l2_errors(x_pred, x_ref):
    abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
    return abs_l2, rel_l2


# =========================================================
# 2. Random Fourier Feature layer
# =========================================================
class RFFLayer(nn.Module):
    def __init__(self, out_features=64, sigma=1.0):
        super().__init__()

        if out_features % 2 != 0:
            raise ValueError("out_features must be even, because cos and sin features are concatenated.")

        self.B = nn.Parameter(
            torch.randn(1, out_features // 2) * sigma,
            requires_grad=False
        )

    def forward(self, t):
        proj = 2.0 * np.pi * torch.matmul(t, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# =========================================================
# 3. Hard RFF-PINN base network
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
# 4. Hard constraint output
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


# =========================================================
# 5. ODE residual loss for the hard RFF-PINN
# =========================================================
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
# 6. Training
# =========================================================
def train_hard_rff_pinn(
    model,
    n,
    t_min,
    t_max,
    t_ref,
    x_ref,
    epochs=30000,
    n_col=100,
    lr=2e-3,
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
            # hard_output computes x_hat'(1), so autograd must remain enabled here.
            x_pred = hard_output(model, t_eval_tensor).detach().cpu().numpy().flatten()
            abs_l2, rel_l2 = compute_l2_errors(x_pred, x_ref)

            l2_epochs.append(epoch + 1)
            abs_l2_history.append(abs_l2)
            rel_l2_history.append(rel_l2)

        if (epoch + 1) % 1000 == 0:
            print(f"[Hard RFF-PINN] Epoch {epoch+1:5d} | Loss = {loss.item():.3e}")

    return {
        "model": model,
        "loss": loss_history,
        "l2_epochs": l2_epochs,
        "abs_l2": abs_l2_history,
        "rel_l2": rel_l2_history
    }


# =========================================================
# 7. Check hard constraints
# =========================================================
def check_hard_constraints(model):
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
        "Hard BC check: "
        f"x(0)={x0.item():.8f}, target=0.00000000 | "
        f"x'(1)={x1_t.item():.8f}, target=1.00000000"
    )


# =========================================================
# 8. Main script
# =========================================================
if __name__ == "__main__":

    n = 6
    t_min, t_max = 0.0, 1.0

    # -----------------------------------------------------
    # Training setup
    # -----------------------------------------------------
    epochs = 30000
    n_col = 100
    lr = 2e-3
    eval_every = 50

    # Fourier feature setup
    sigma_rff = 1.0
    rff_features = 64
    width = 64

    # -----------------------------------------------------
    # Reference solution
    # -----------------------------------------------------
    t_ref = np.linspace(t_min, t_max, 1000)
    x_ref = exact_solution(t_ref, n)
    t_eval_tensor = torch.tensor(t_ref, dtype=torch.float32).reshape(-1, 1)

    # -----------------------------------------------------
    # Train Hard RFF-PINN
    # -----------------------------------------------------
    set_seed(0)
    model = HardRFFPINN(
        rff_features=rff_features,
        sigma=sigma_rff,
        width=width
    )

    result = train_hard_rff_pinn(
        model=model,
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
    # Final prediction and errors
    # -----------------------------------------------------
    x_pred = hard_output(result["model"], t_eval_tensor).detach().cpu().numpy().flatten()
    abs_l2, rel_l2 = compute_l2_errors(x_pred, x_ref)

    print("\n================ Final L2 Errors ================")
    print(f"n = {n}")
    print(f"Hard RFF-PINN | Abs L2 = {abs_l2:.6e} | Rel L2 = {rel_l2:.6e}")

    check_hard_constraints(result["model"])

    # -----------------------------------------------------
    # Solution comparison
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
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
        x_pred,
        color="purple",
        lw=2.2,
        linestyle="--",
        label="Hard RFF-PINN"
    )
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend(frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Absolute L2 error
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.semilogy(
        result["l2_epochs"],
        result["abs_l2"],
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

    # -----------------------------------------------------
    # Relative L2 error
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.semilogy(
        result["l2_epochs"],
        result["rel_l2"],
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
