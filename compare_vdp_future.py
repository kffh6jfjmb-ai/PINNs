import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp


# =========================================================
# 0. Settings
# =========================================================
torch.set_default_dtype(torch.float32)


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =========================================================
# 1. Reference solution and errors
# =========================================================
def get_reference_solution(mu, t_min, t_max, x0, v0, n_eval=2000):
    t_ref = np.linspace(t_min, t_max, n_eval)

    sol = solve_ivp(
        lambda t, z: [
            z[1],
            mu * (1.0 - z[0] ** 2) * z[1] - z[0]
        ],
        [t_min, t_max],
        [x0, v0],
        t_eval=t_ref,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    return t_ref, sol.y[0]


def interpolate_reference(t_query, t_ref, x_ref):
    return np.interp(t_query, t_ref, x_ref)


def compute_l2_errors(x_pred, x_ref):
    abs_l2 = np.sqrt(np.mean((x_pred - x_ref) ** 2))
    rel_l2 = abs_l2 / (np.sqrt(np.mean(x_ref ** 2)) + 1e-12)
    return abs_l2, rel_l2


# =========================================================
# 2. Networks
# =========================================================
def make_mlp(in_dim, width=64):
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, 1),
    )


class RFFLayer(nn.Module):
    def __init__(self, out_features=64, sigma=1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be even.")

        self.B = nn.Parameter(
            torch.randn(1, out_features // 2) * sigma,
            requires_grad=False,
        )

    def forward(self, t):
        proj = 2.0 * np.pi * torch.matmul(t, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SoftPINN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = make_mlp(1, width)

    def forward(self, t):
        return self.net(t)


class SoftRFFPINN(nn.Module):
    def __init__(self, rff_features=64, sigma=1.0, width=64):
        super().__init__()
        self.rff = RFFLayer(out_features=rff_features, sigma=sigma)
        self.net = make_mlp(rff_features, width)

    def forward(self, t):
        return self.net(self.rff(t))


class HardPINN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = make_mlp(1, width)

    def forward(self, t):
        return self.net(t)


class HardRFFPINN(nn.Module):
    def __init__(self, rff_features=64, sigma=1.0, width=64):
        super().__init__()
        self.rff = RFFLayer(out_features=rff_features, sigma=sigma)
        self.net = make_mlp(rff_features, width)

    def forward(self, t):
        return self.net(self.rff(t))


# =========================================================
# 3. Hard constraint and losses
# =========================================================
def hard_output(model, t, x0, v0, t0=0.0):

    x_hat_t = model(t)

    t0_tensor = torch.tensor(
        [[t0]], dtype=t.dtype, device=t.device, requires_grad=True
    )
    x_hat_t0 = model(t0_tensor)
    dx_hat_t0 = torch.autograd.grad(
        x_hat_t0,
        t0_tensor,
        torch.ones_like(x_hat_t0),
        create_graph=True,
    )[0]

    return x_hat_t + x0 - x_hat_t0 + (t - t0_tensor) * (v0 - dx_hat_t0)


def residual_loss_soft(model, t, mu):
    t = t.clone().detach().requires_grad_(True)
    x = model(t)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


def residual_loss_hard(model, t, mu, x0, v0):
    t = t.clone().detach().requires_grad_(True)
    x = hard_output(model, t, x0, v0)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
    res = x_tt - mu * (1.0 - x ** 2) * x_t + x
    return torch.mean(res ** 2)


def initial_loss_soft(model, x0, v0):
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    x_pred = model(t0)
    x_t_pred = torch.autograd.grad(
        x_pred, t0, torch.ones_like(x_pred), create_graph=True
    )[0]
    return torch.mean((x_pred - x0) ** 2 + (x_t_pred - v0) ** 2)


def data_loss_soft(model, t_data, x_data):
    return torch.mean((model(t_data) - x_data) ** 2)


def data_loss_hard(model, t_data, x_data, x0, v0):
    x_pred = hard_output(model, t_data, x0, v0)
    return torch.mean((x_pred - x_data) ** 2)


# =========================================================
# 4. Prediction and evaluation
# =========================================================
def predict_soft(model, t_np):
    t = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        return model(t).cpu().numpy().flatten()


def predict_hard(model, t_np, x0, v0):
    t = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
    with torch.enable_grad():
        t_req = t.clone().detach().requires_grad_(True)
        x = hard_output(model, t_req, x0, v0)
    return x.detach().cpu().numpy().flatten()


def evaluate_model(model, is_hard, t_train, x_train_ref, t_future, x_future_ref, x0, v0):
    if is_hard:
        x_train_pred = predict_hard(model, t_train, x0, v0)
        x_future_pred = predict_hard(model, t_future, x0, v0)
    else:
        x_train_pred = predict_soft(model, t_train)
        x_future_pred = predict_soft(model, t_future)

    _, train_rel = compute_l2_errors(x_train_pred, x_train_ref)
    _, future_rel = compute_l2_errors(x_future_pred, x_future_ref)
    return train_rel, future_rel


# =========================================================
# 5. Training
# =========================================================
def train_model(
    model,
    model_name,
    is_hard,
    mu,
    x0,
    v0,
    t_col,
    t_data,
    x_data,
    t_train_eval,
    x_train_ref,
    t_future_eval,
    x_future_ref,
    epochs=20000,
    lr=3e-3,
    w_res=6e-2,
    w_ic=1.0,
    w_data=1.0,
    eval_every=100,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    l2_epochs = []
    train_rel_history = []
    future_rel_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        if is_hard:
            loss_res = residual_loss_hard(model, t_col, mu, x0, v0)
            loss_data = data_loss_hard(model, t_data, x_data, x0, v0)
            loss = w_res * loss_res + w_data * loss_data
        else:
            loss_res = residual_loss_soft(model, t_col, mu)
            loss_ic = initial_loss_soft(model, x0, v0)
            loss_data = data_loss_soft(model, t_data, x_data)
            loss = w_res * loss_res + w_ic * loss_ic + w_data * loss_data

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            train_rel, future_rel = evaluate_model(
                model, is_hard,
                t_train_eval, x_train_ref,
                t_future_eval, x_future_ref,
                x0, v0,
            )
            l2_epochs.append(epoch + 1)
            train_rel_history.append(train_rel)
            future_rel_history.append(future_rel)

        if (epoch + 1) % 1000 == 0:
            print(
                f"[{model_name}] Epoch {epoch+1:5d} | "
                f"Loss = {loss.item():.3e} | "
                f"Res = {loss_res.item():.3e} | Data = {loss_data.item():.3e}"
            )

    return {
        "model": model,
        "loss": loss_history,
        "l2_epochs": l2_epochs,
        "train_rel": train_rel_history,
        "future_rel": future_rel_history,
    }


# =========================================================
# 6. Plotting
# =========================================================
def plot_solution_grid(t_ref, x_ref, t_split, methods, x0, v0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, item in zip(axes.flat, methods):
        if item["is_hard"]:
            x_pred = predict_hard(item["result"]["model"], t_ref, x0, v0)
        else:
            x_pred = predict_soft(item["result"]["model"], t_ref)

        ax.plot(t_ref, x_ref, color="gray", lw=3, alpha=0.8, label="Reference")
        ax.plot(t_ref, x_pred, color=item["color"], lw=2, ls="--", label="Prediction")
        ax.axvline(t_split, color="black", lw=1, ls=":")
        ax.axvspan(t_split, t_ref[-1], color="gray", alpha=0.08)
        ax.text(
            0.03, 0.92,
            f"{item['name']}\ntrain Rel-$L^2$={item['train_rel']:.2e}\nfuture Rel-$L^2$={item['future_rel']:.2e}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax.legend(frameon=True, fontsize=8, loc="lower left")
        ax.grid(False)

    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("x(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    plt.tight_layout()
    plt.show()


def plot_future_grid(t_future, x_future_ref, methods, x0, v0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, item in zip(axes.flat, methods):
        if item["is_hard"]:
            x_pred = predict_hard(item["result"]["model"], t_future, x0, v0)
        else:
            x_pred = predict_soft(item["result"]["model"], t_future)

        ax.plot(t_future, x_future_ref, color="gray", lw=3, alpha=0.8, label="Reference")
        ax.plot(t_future, x_pred, color=item["color"], lw=2, ls="--", label="Prediction")
        ax.text(
            0.03, 0.92,
            f"{item['name']}\nfuture Rel-$L^2$={item['future_rel']:.2e}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax.legend(frameon=True, fontsize=8, loc="lower left")
        ax.grid(False)

    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("x(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    plt.tight_layout()
    plt.show()


def plot_error_history(methods):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    for item in methods:
        result = item["result"]
        axes[0].semilogy(result["l2_epochs"], result["train_rel"],
                         color=item["color"], lw=2, label=item["name"])
        axes[1].semilogy(result["l2_epochs"], result["future_rel"],
                         color=item["color"], lw=2, label=item["name"])

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Train relative $L^2$ error on $[0,10]$")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"Future relative $L^2$ error on $[10,20]$")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(False)

    plt.tight_layout()
    plt.show()


# =========================================================
# 7. Main
# =========================================================
if __name__ == "__main__":

    # Problem setup
    mu = 1.0
    x0, v0 = 1.0, 0.0
    t_train_min, t_train_max = 0.0, 10.0
    t_future_min, t_future_max = 10.0, 20.0
    t_col_min, t_col_max = 0.0, 20.0

    # Training setup
    epochs = 20000
    n_col = 500
    n_data = 80
    lr = 3e-3
    w_res = 6e-2
    w_ic = 1.0
    w_data = 1.0
    eval_every = 100

    # Network setup
    width = 64
    rff_features = 128
    rff_sigma = 1.0

    # Reference solution and training data
    t_ref_full, x_ref_full = get_reference_solution(
        mu=mu,
        t_min=t_col_min,
        t_max=t_col_max,
        x0=x0,
        v0=v0,
        n_eval=2000,
    )

    t_train_eval = np.linspace(t_train_min, t_train_max, 1000)
    x_train_ref = interpolate_reference(t_train_eval, t_ref_full, x_ref_full)
    t_future_eval = np.linspace(t_future_min, t_future_max, 1000)
    x_future_ref = interpolate_reference(t_future_eval, t_ref_full, x_ref_full)

    t_data_np = np.linspace(t_train_min, t_train_max, n_data)
    x_data_np = interpolate_reference(t_data_np, t_ref_full, x_ref_full)

    t_col = torch.linspace(t_col_min, t_col_max, n_col).reshape(-1, 1)
    t_data = torch.tensor(t_data_np, dtype=torch.float32).reshape(-1, 1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32).reshape(-1, 1)

    # Train models
    model_specs = [
        ("Soft PINN", SoftPINN(width), False, "red"),
        ("Hard PINN", HardPINN(width), True, "blue"),
        ("Soft RFF-PINN", SoftRFFPINN(rff_features, rff_sigma, width), False, "green"),
        ("Hard RFF-PINN", HardRFFPINN(rff_features, rff_sigma, width), True, "purple"),
    ]

    methods = []
    for name, model, is_hard, color in model_specs:
        set_seed(0)
        result = train_model(
            model=model,
            model_name=name,
            is_hard=is_hard,
            mu=mu,
            x0=x0,
            v0=v0,
            t_col=t_col,
            t_data=t_data,
            x_data=x_data,
            t_train_eval=t_train_eval,
            x_train_ref=x_train_ref,
            t_future_eval=t_future_eval,
            x_future_ref=x_future_ref,
            epochs=epochs,
            lr=lr,
            w_res=w_res,
            w_ic=w_ic,
            w_data=w_data,
            eval_every=eval_every,
        )
        train_rel, future_rel = evaluate_model(
            result["model"], is_hard,
            t_train_eval, x_train_ref,
            t_future_eval, x_future_ref,
            x0, v0,
        )
        methods.append({
            "name": name,
            "result": result,
            "color": color,
            "is_hard": is_hard,
            "train_rel": train_rel,
            "future_rel": future_rel,
        })

    print("\n================ Final Relative L2 Errors ================")
    for item in methods:
        print(
            f"{item['name']:<15s} | "
            f"train Rel L2 = {item['train_rel']:.6e} | "
            f"future Rel L2 = {item['future_rel']:.6e}"
        )

    # Plots only. No figures or CSV files are saved.
    plot_solution_grid(t_ref_full, x_ref_full, t_train_max, methods, x0, v0)
    plot_future_grid(t_future_eval, x_future_ref, methods, x0, v0)
    plot_error_history(methods)
