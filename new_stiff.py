import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.set_default_dtype(torch.float32)


# =====================================================
# Exact solution
# eps u'' + u' = 0, u(0)=0, u(1)=1
# =====================================================
def exact_solution(x, eps):
    return np.expm1(-x / eps) / np.expm1(-1.0 / eps)


# =====================================================
# Hard PINN with feature map
# =====================================================
class FeatureHardPINN(nn.Module):
    def __init__(self, eps, width=32):
        super().__init__()
        self.eps = float(eps)

        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )

    def feature_map(self, x):
        layer_feature = - x / (self.eps)
        return torch.cat([x, layer_feature], dim=1)

    def forward(self, x):
        z = self.feature_map(x)
        raw = self.net(z)

        return x + x * (1.0 - x) * raw


# =====================================================
# Residual loss
# =====================================================
def residual_loss(model, x_col, eps):
    x = x_col.clone().detach().requires_grad_(True)

    u = model(x)

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    res = eps * u_xx + u_x
    return torch.mean(res ** 2)


# =====================================================
# Training
# =====================================================
def train(eps, epochs=20000, n_col=200, lr=5e-3):
    torch.manual_seed(1234)
    np.random.seed(1234)

    model = FeatureHardPINN(eps=eps, width=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points
    x_col = torch.linspace(0.0, 1.0, n_col + 2).reshape(-1, 1)[1:-1]

    # Test grid
    x_test_np = np.linspace(0.0, 1.0, 2000)
    u_test_np = exact_solution(x_test_np, eps)
    x_test = torch.tensor(x_test_np, dtype=torch.float32).reshape(-1, 1)

    rel_errors = []
    epoch_list = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss = residual_loss(model, x_col, eps)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            with torch.no_grad():
                u_pred = model(x_test).numpy().flatten()

            rel_l2 = np.sqrt(np.mean((u_pred - u_test_np) ** 2)) / (
                np.sqrt(np.mean(u_test_np ** 2)) + 1e-12
            )

            rel_errors.append(rel_l2)
            epoch_list.append(epoch)

        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Loss = {loss.item():.3e} | "
                f"Rel L2 = {rel_errors[-1]:.3e}"
            )

    return model, x_test_np, u_test_np, epoch_list, rel_errors


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    eps = 1e-2

    model, x_test, u_exact, epochs, rel_errors = train(
        eps=eps,
        epochs=20000,
        n_col=300,
        lr=5e-3
    )

    x_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_tensor).numpy().flatten()

    # Solution plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_test, u_exact, color="gray", lw=3, label="Exact")
    plt.plot(x_test, u_pred, "--", lw=2, label="Feature Hard PINN")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Error history
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(epochs, rel_errors, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel(r"Relative $L^2$ error")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    print("Final relative L2 error:", rel_errors[-1])