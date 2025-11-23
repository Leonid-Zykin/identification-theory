from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_variant_mat


def simulate_system(a: float, b: float, u: np.ndarray, y0: float = 0.0) -> np.ndarray:
    """Simulate discrete system y(k) = -a*y(k-1) + b*u(k-1)."""
    y = np.zeros_like(u)
    y[0] = y0
    for k in range(1, len(u)):
        y[k] = -a * y[k - 1] + b * u[k - 1]
    return y


def gradient_algorithm_3(
    y: np.ndarray, u: np.ndarray, gamma: float, theta0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient algorithm (3): theta(k) = theta(k-1) + gamma*Phi(k)*e0(k)/(1+gamma*Phi^T(k)*Phi(k))."""
    N = len(y)
    theta_hat = np.zeros((N, 2))
    theta_hat[0] = theta0
    e0 = np.zeros(N - 1)

    for k in range(1, N):
        phi = np.array([-y[k - 1], u[k - 1]])
        e0[k - 1] = y[k] - phi @ theta_hat[k - 1]
        denom = 1 + gamma * (phi @ phi)
        theta_hat[k] = theta_hat[k - 1] + (gamma * e0[k - 1] / denom) * phi

    return theta_hat, e0


def simple_gradient_algorithm_4(
    y: np.ndarray, u: np.ndarray, gamma: float, theta0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Simplified gradient algorithm (4): theta(k) = theta(k-1) + gamma*Phi(k)*e0(k)."""
    N = len(y)
    theta_hat = np.zeros((N, 2))
    theta_hat[0] = theta0
    e0 = np.zeros(N - 1)

    for k in range(1, N):
        phi = np.array([-y[k - 1], u[k - 1]])
        e0[k - 1] = y[k] - phi @ theta_hat[k - 1]
        theta_hat[k] = theta_hat[k - 1] + gamma * e0[k - 1] * phi

    return theta_hat, e0


def plot_identification(
    t: np.ndarray,
    theta_hat: np.ndarray,
    theta_true: np.ndarray,
    out_path: Path,
    title: str,
    gamma: float,
) -> None:
    """Plot parameter estimates vs time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)

    # Check for instability (very large values)
    # Use a reasonable threshold based on true values
    threshold = max(10 * abs(theta_true[0]), 10 * abs(theta_true[1]), 50)
    
    # Find where instability starts (values become unreasonably large)
    unstable_idx_a = np.where(np.abs(theta_hat[:, 0]) > threshold)[0]
    unstable_idx_b = np.where(np.abs(theta_hat[:, 1]) > threshold)[0]
    
    if len(unstable_idx_a) > 0 or len(unstable_idx_b) > 0:
        # Find first index where any parameter becomes unstable
        if len(unstable_idx_a) > 0 and len(unstable_idx_b) > 0:
            unstable_start = min(unstable_idx_a[0], unstable_idx_b[0])
        elif len(unstable_idx_a) > 0:
            unstable_start = unstable_idx_a[0]
        else:
            unstable_start = unstable_idx_b[0]
        
        # Plot only up to the point just before instability
        # Show at least 5 points, but stop before instability
        plot_end = max(5, min(unstable_start, 20))  # Show max 20 points or until instability
        t_plot = t[:plot_end]
        theta_plot = theta_hat[:plot_end, :]
        unstable_time = t[unstable_start] if unstable_start < len(t) else None
    else:
        t_plot = t
        theta_plot = theta_hat
        plot_end = len(t)
        unstable_time = None

    axes[0].plot(t_plot, theta_plot[:, 0], label=f"$\hat{{a}}$", linewidth=2)
    axes[0].axhline(y=theta_true[0], color="r", linestyle="--", label=f"$a^* = {theta_true[0]:.2f}$")
    axes[0].set_ylabel("Параметр $a$")
    axes[0].set_title(f"$\gamma = {gamma}$")
    if unstable_time is not None:
        axes[0].text(0.02, 0.98, f"Неустойчивость с t ≈ {unstable_time:.1f} с\n(график обрезан)", 
                    transform=axes[0].transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_plot, theta_plot[:, 1], label=f"$\hat{{b}}$", linewidth=2)
    axes[1].axhline(y=theta_true[1], color="r", linestyle="--", label=f"$b^* = {theta_true[1]:.2f}$")
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("Параметр $b$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    data = load_variant_mat(25)["zad1"]
    a_true = float(getattr(data, "a"))
    b_true = float(getattr(data, "b"))
    w = float(getattr(data, "w"))
    theta_true = np.array([a_true, b_true])

    Td = 0.1
    t_sim = 30.0
    t = np.arange(0, t_sim, Td)
    u = np.sin(w * t)

    y = simulate_system(a_true, b_true, u)

    images_dir = Path(__file__).resolve().parents[1] / "images" / "task1"
    ensure_dir(images_dir)

    theta0 = np.array([0.5, 1.0])

    # Gradient algorithm (3) with different gamma
    for gamma in [1, 3, 10]:
        theta_hat, _ = gradient_algorithm_3(y, u, gamma, theta0)
        plot_identification(
            t,
            theta_hat,
            theta_true,
            images_dir / f"gradient3_gamma{gamma}.png",
            f"Градиентный алгоритм (3), $\gamma = {gamma}$",
            gamma,
        )

    # Simple gradient algorithm (4)
    for gamma in [0.5, 10]:
        theta_hat, _ = simple_gradient_algorithm_4(y, u, gamma, theta0)
        plot_identification(
            t,
            theta_hat,
            theta_true,
            images_dir / f"simple4_gamma{gamma}.png",
            f"Упрощенный алгоритм (4), $\gamma = {gamma}$",
            gamma,
        )

    print("Task 1 completed. Plots saved to", images_dir)


if __name__ == "__main__":
    main()

