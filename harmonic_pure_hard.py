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
# 3. Hard constraint output
# =========================================================
def hard_output(model, t):
    """
    Hard ansatz for:
        x(0)=0, x'(1)=1

        x(t) = x_hat(t) - x_hat(0) + t * (1 - x_hat'(1))
    """
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


def hard_predict(model, t):
    with torch.enable_grad():
        t_req = t.clone().detach().requires_grad_(True)
        x = hard_output(model, t_req)
    return x.detach()


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
# 4. Train hard model
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
            x_pred = hard_predict(model, t_eval_tensor).cpu().numpy().flatten()

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
# 5. Diagnostics
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
# 6. Main
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
    # Hard PINN
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

    # =====================================================
    # 7. Final prediction
    # =====================================================
    x_hard = hard_predict(hard_result["model"], t_eval_tensor).cpu().numpy().flatten()

    hard_abs_l2, hard_rel_l2 = compute_l2_errors(x_hard, x_ref)

    print("\n================ Final L2 Errors ================")
    print(f"n = {n}")
    print(f"Hard PINN | Abs L2 = {hard_abs_l2:.6e} | Rel L2 = {hard_rel_l2:.6e}")

    check_hard_constraints(hard_result["model"], "Hard PINN")

    # =====================================================
    # 8. Solution comparison
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
        x_hard,
        color="blue",
        lw=2.2,
        linestyle="--",
        label="Hard PINN"
    )
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend(frameon=True, fontsize=9)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("harmonic_pure_hard_solution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =====================================================
    # 9. Absolute L2 error
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(
        hard_result["l2_epochs"],
        hard_result["abs_l2"],
        color="blue",
        lw=2,
        label="Hard PINN"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Absolute L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("harmonic_pure_hard_abs_l2.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =====================================================
    # 10. Relative L2 error
    # =====================================================
    plt.figure(figsize=(11, 5))
    plt.semilogy(
        hard_result["l2_epochs"],
        hard_result["rel_l2"],
        color="blue",
        lw=2,
        label="Hard PINN"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("harmonic_pure_hard_rel_l2.png", dpi=300, bbox_inches="tight")
    plt.show()
