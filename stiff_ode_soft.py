import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -----------------------------
# 0. Settings
# -----------------------------
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

# -----------------------------
# 1. Feature Layer (2D features)
# -----------------------------
class FeatureLayer(nn.Module):
    """
    Map x -> [x, x/eps]
    """
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps_t = torch.tensor(self.eps, dtype=x.dtype)
        return torch.cat([x, x / eps_t], dim=1)


# -----------------------------
# 2. Neural Network with Features
# -----------------------------
class PINN(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.feat = FeatureLayer(eps)

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feat(x)
        return self.net(features)


# -----------------------------
# 3. Loss Functions
# -----------------------------
def residual_loss(model: nn.Module, x: torch.Tensor, eps) -> torch.Tensor:

    x = x.clone().detach().requires_grad_(True)

    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    res = eps * u_xx + u_x
    return torch.mean(res**2)


def boundary_loss(model: nn.Module) -> torch.Tensor:
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])

    u0 = model(x0)
    u1 = model(x1)

    return (u0 - 0.0) ** 2 + (u1 - 1.0) ** 2


# -----------------------------
# 4. Training loop (random points)
# -----------------------------
def train_PINN(epochs: int, model: nn.Module, eps, n_col: int = 1000, lr = 1e-3, w_bc = 100.0,
    power = 0.30
):
    """
    Random collocation points each epoch:
        x = U^power, power<1 -> denser near x=0 (boundary layer at left end)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # random collocation points in [0,1], biased toward 0
        x_raw = torch.rand(n_col, 1)
        x_domain = x_raw ** power

        loss_pde = residual_loss(model, x_domain, eps)
        loss_bc = boundary_loss(model)
        loss = loss_pde + w_bc * loss_bc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch+1:5d} | Loss: {loss.item():.3e} "
                f"(PDE: {loss_pde.item():.2e}, BC: {loss_bc.item():.2e})"
            )

    return model, losses


# -----------------------------
# 5. Execution
# -----------------------------
if __name__ == "__main__":
    eps = 1e-4

    model = PINN(eps=eps)
    trained_model, loss_history = train_PINN(epochs=5000, model=model, eps=eps, n_col=1000, lr=1e-3, w_bc=100.0, power=0.30)

    # Reference
    x_plot = np.linspace(0, 1, 800)[:, None]
    u_exact = (1.0 - np.exp(-x_plot / eps)) / (1.0 - np.exp(-1.0 / eps))

    with torch.no_grad():
        x_plot_torch = torch.linspace(0, 1, 800).reshape(-1, 1)
        u_pinn = trained_model(x_plot_torch).numpy()

    # Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, u_exact, "k-", alpha=0.35, lw=3, label="Exact")
    plt.plot(x_plot, u_pinn, "r--", lw=2, label="PINN")
    plt.axvline(x=eps, color="gray", linestyle=":", label="x = eps scale")
    plt.title(f"eps*u'' + u' = 0 (eps={eps})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.semilogy(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.show()