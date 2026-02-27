import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

# -----------------------------
# 0. Settings
# -----------------------------
torch.set_default_dtype(torch.float32)

# -----------------------------
# 1. Random Fourier Feature Layer
# -----------------------------
class RFFLayer(nn.Module):
    def __init__(self, out_features, sigma=2.0):
        super().__init__()
        # Pre-set random frequencies. sigma determines the frequency range.
        # Frequencies distributed within [0, 10] are sufficient for mu=1 oscillations.
        self.B = nn.Parameter(torch.randn(1, out_features // 2) * sigma, requires_grad=False)

    def forward(self, x):
        # Map t to cos(2pi * B * t) and sin(2pi * B * t)
        proj = 2.0 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

# -----------------------------
# 2. Neural Network with RFF
# -----------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rff = RFFLayer(64, sigma=2.0) # RFF Pre-processing
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.rff(x)
        return self.net(features)

# -----------------------------
# 3. Loss Functions
# -----------------------------
def residual_loss(model, tau, mu, T):
    tau = tau.clone().detach().requires_grad_(True)
    x = model(tau)
    x_tau = torch.autograd.grad(x, tau, torch.ones_like(x), create_graph=True)[0]
    x_tautau = torch.autograd.grad(x_tau, tau, torch.ones_like(x_tau), create_graph=True)[0]
    
    # Physics Equation Residual (Normalized to tau coordinate system)
    # Equation: (1/T^2)x'' - mu(1-x^2)(1/T)x' + x = 0
    res = (1.0 / T**2) * x_tautau - mu * (1 - x**2) * (1.0 / T) * x_tau + x
    return torch.mean(res**2)

def boundary_loss(model, x0, v0, T):
    tau0 = torch.tensor([[0.0]]).requires_grad_(True)
    x_pred = model(tau0)
    x_tau_pred = torch.autograd.grad(x_pred, tau0, torch.ones_like(x_pred), create_graph=True)[0]
    
    loss_x = (x_pred - x0)**2
    loss_v = (x_tau_pred - (T * v0))**2 # Velocity scaling under tau coordinate
    return loss_x + loss_v

# -----------------------------
# 4. Training loop
# -----------------------------
def train_PINN(epochs, model, mu, T, x0, v0, n_col=1000, lr=1e-3, w_ic=1000.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        tau_domain = torch.linspace(0, 1, n_col).reshape(-1, 1)

        loss_ode = residual_loss(model, tau_domain, mu, T)
        loss_ic  = boundary_loss(model, x0, v0, T)

        loss = loss_ode + w_ic * loss_ic
        
        loss.backward()
        # Gradient clipping to prevent oscillations
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.3e} (ODE: {loss_ode.item():.2e}, IC: {loss_ic.item():.2e})")

    return model, losses

# -----------------------------
# 5. Execution
# -----------------------------
mu = 1.0
t_min, t_max = 0.0, 10.0
T = t_max - t_min
x0, v0 = 1.0, 0.0

model = PINN()
trained_model, loss_history = train_PINN(epochs=10000, model=model, mu=mu, T=T, x0=x0, v0=v0, w_ic=1000.0)

# Reference
t_ref = np.linspace(0, 10, 1000)
sol = solve_ivp(lambda t, z: [z[1], mu*(1-z[0]**2)*z[1]-z[0]], [0, 10], [1.0, 0.0], t_eval=t_ref)

with torch.no_grad():
    tau_plot = torch.linspace(0, 1, 1000).reshape(-1, 1)
    x_pinn = trained_model(tau_plot).numpy()

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], 'k-', alpha=0.3, lw=3, label="Reference")
plt.plot(sol.t, x_pinn, 'r--', lw=2, label="RFF-PINN")
plt.title(f"Van der Pol Oscillator (mu={mu})") 
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.semilogy(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.grid(True)
plt.tight_layout()
plt.show()