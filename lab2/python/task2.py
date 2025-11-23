from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_variant_mat


def simulate_system_2nd(
    a1: float, a2: float, b: float, u: np.ndarray, y0: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Simulate discrete 2nd order system y(k) = -a1*y(k-1) - a2*y(k-2) + b*u(k-1)."""
    y = np.zeros_like(u)
    y[0] = y0[0]
    if len(y) > 1:
        y[1] = y0[1]
    for k in range(2, len(u)):
        y[k] = -a1 * y[k - 1] - a2 * y[k - 2] + b * u[k - 1]
    return y


def gradient_algorithm_3_2nd(
    y: np.ndarray, u: np.ndarray, gamma: float, theta0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient algorithm (3) for 2nd order system."""
    N = len(y)
    theta_hat = np.zeros((N, 3))
    theta_hat[0] = theta0
    theta_hat[1] = theta0  # Initialize second element
    e0 = np.zeros(N - 2)

    for k in range(2, N):
        phi = np.array([-y[k - 1], -y[k - 2], u[k - 1]])
        e0[k - 2] = y[k] - phi @ theta_hat[k - 1]
        denom = 1 + gamma * (phi @ phi)
        theta_hat[k] = theta_hat[k - 1] + (gamma * e0[k - 2] / denom) * phi

    return theta_hat, e0


def plot_identification_2nd(
    t: np.ndarray,
    theta_hat: np.ndarray,
    theta_true: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot parameter estimates vs time for 2nd order system."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)

    labels = ["$a_1$", "$a_2$", "$b$"]
    for i, (label, ax) in enumerate(zip(labels, axes)):
        ax.plot(t, theta_hat[:, i], label=f"$\hat{{{label[1:-1]}}}$", linewidth=2)
        ax.axhline(y=theta_true[i], color="r", linestyle="--", label=f"${label[1:-1]}^*$")
        ax.set_ylabel(f"Параметр {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Время, с")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    data = load_variant_mat(25)["zad2"]
    a1_true = float(getattr(data, "a1"))
    a2_true = float(getattr(data, "a2"))
    b_true = float(getattr(data, "b"))
    w = float(getattr(data, "w"))
    theta_true = np.array([a1_true, a2_true, b_true])

    Td = 0.1
    t_sim = 60.0
    t = np.arange(0, t_sim, Td)

    images_dir = Path(__file__).resolve().parents[1] / "images" / "task2"
    ensure_dir(images_dir)

    gamma = 1.0
    theta0 = np.array([-1.0, 0.5, 1.0])

    # Input 1: u(t) = sin(wt)
    u1 = np.sin(w * t)
    y1 = simulate_system_2nd(a1_true, a2_true, b_true, u1)
    theta_hat1, _ = gradient_algorithm_3_2nd(y1, u1, gamma, theta0)
    plot_identification_2nd(
        t,
        theta_hat1,
        theta_true,
        images_dir / "sin_input.png",
        "Идентификация с входным сигналом $u(t) = \\sin(\\omega t)$",
    )

    # Input 2: u(t) = sin(wt) + 0.2*sin(0.5*wt)
    u2 = np.sin(w * t) + 0.2 * np.sin(0.5 * w * t)
    y2 = simulate_system_2nd(a1_true, a2_true, b_true, u2)
    theta_hat2, _ = gradient_algorithm_3_2nd(y2, u2, gamma, theta0)
    plot_identification_2nd(
        t,
        theta_hat2,
        theta_true,
        images_dir / "mixed_input.png",
        "Идентификация с входным сигналом $u(t) = \\sin(\\omega t) + 0.2\\sin(0.5\\omega t)$",
    )

    print("Task 2 completed. Plots saved to", images_dir)


if __name__ == "__main__":
    main()

