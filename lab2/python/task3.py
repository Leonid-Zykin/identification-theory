from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from utils import ensure_dir, load_variant_mat


def continuous_system(
    y: float, t: float, a: float, b: float, u_func
) -> float:
    """Continuous system: dy/dt = -a*y + b*u(t)."""
    u = u_func(t)
    return -a * y + b * u


def gradient_algorithm_2_continuous(
    y: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    gamma: float,
    theta0: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Gradient algorithm (2) in continuous time: dtheta/dt = gamma*Phi*e."""
    N = len(t)
    theta_hat = np.zeros((N, 2))
    theta_hat[0] = theta0

    # Compute derivative dy/dt numerically
    dy_dt = np.gradient(y, dt)

    # Numerical integration using Euler method
    for i in range(N - 1):
        phi = np.array([-y[i], u[i]])
        # Error: e(t) = dy/dt - Phi^T(t)*theta_hat(t)
        e = dy_dt[i] - phi @ theta_hat[i]
        dtheta_dt = gamma * e * phi
        theta_hat[i + 1] = theta_hat[i] + dtheta_dt * dt

    return theta_hat


def plot_identification_continuous(
    t: np.ndarray,
    theta_hat: np.ndarray,
    theta_true: np.ndarray,
    out_path: Path,
    title: str,
    gamma: float,
) -> None:
    """Plot parameter estimates vs time for continuous system."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(t, theta_hat[:, 0], label=f"$\hat{{a}}$", linewidth=2)
    axes[0].axhline(y=theta_true[0], color="r", linestyle="--", label="$a^*$")
    axes[0].set_ylabel("Параметр $a$")
    axes[0].set_title(f"$\gamma = {gamma}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, theta_hat[:, 1], label=f"$\hat{{b}}$", linewidth=2)
    axes[1].axhline(y=theta_true[1], color="r", linestyle="--", label="$b^*$")
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("Параметр $b$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    data = load_variant_mat(25)["zad3"]
    a_true = float(getattr(data, "a"))
    b_true = float(getattr(data, "b"))
    w = float(getattr(data, "w"))
    theta_true = np.array([a_true, b_true])

    dt = 0.01  # Simulation step <= 0.01
    t_sim = 30.0
    t = np.arange(0, t_sim, dt)

    # Input signal u(t) = sin(wt)
    u = np.sin(w * t)

    # Simulate continuous system
    def u_func(t_val):
        return np.sin(w * t_val)

    y0 = 0.0
    y = odeint(continuous_system, y0, t, args=(a_true, b_true, u_func))[:, 0]

    images_dir = Path(__file__).resolve().parents[1] / "images" / "task3"
    ensure_dir(images_dir)

    theta0 = np.array([0.5, 1.0])

    # Gradient algorithm (2) with different gamma
    for gamma in [1, 3, 10]:
        theta_hat = gradient_algorithm_2_continuous(y, u, t, gamma, theta0, dt)
        plot_identification_continuous(
            t,
            theta_hat,
            theta_true,
            images_dir / f"gamma{gamma}.png",
            f"Градиентный алгоритм (2), $\gamma = {gamma}$",
            gamma,
        )

    print("Task 3 completed. Plots saved to", images_dir)


if __name__ == "__main__":
    main()

