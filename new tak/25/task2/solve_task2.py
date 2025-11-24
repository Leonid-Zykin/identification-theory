"""Utilities for Task 2: identifying regressor types and running online LMS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit


@dataclass
class ExpComponent:
    """Represents a component of the form a * exp(b t + d) + c."""

    a: float
    b: float
    c: float
    d: float

    def __call__(self, t: ArrayLike) -> np.ndarray:
        t_arr = np.asarray(t)
        return self.a * np.exp(self.b * t_arr + self.d) + self.c


def fit_exponential_component(
    t: np.ndarray,
    x: np.ndarray,
    *,
    initial_guess: Tuple[float, float, float, float],
) -> ExpComponent:
    """Fit a single component with template a * exp(b t + d) + c."""

    def model(time: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return a * np.exp(b * time + d) + c

    params, _ = curve_fit(model, t, x, p0=initial_guess, maxfev=20000)
    return ExpComponent(*params)


def online_gradient_descent(
    t: np.ndarray,
    phi: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float,
    t_stop: float,
    theta0: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate theta_dot = gamma * phi * (y - phi^T theta) until t_stop."""
    theta = np.zeros(phi.shape[1]) if theta0 is None else theta0.astype(float)

    for i in range(len(t) - 1):
        if t[i] >= t_stop:
            break

        dt = min(t_stop, t[i + 1]) - t[i]
        if dt <= 0:
            continue

        err = y[i] - phi[i] @ theta
        theta += gamma * dt * phi[i] * err

        if t[i + 1] >= t_stop:
            break

    return theta


def analyze_persistent_excitation(
    t: np.ndarray, phi: np.ndarray, window: float
) -> Dict[Tuple[float, float], np.ndarray]:
    """Compute eigenvalues of integral(phi phi^T) over sliding windows."""
    results: Dict[Tuple[float, float], np.ndarray] = {}
    start = t[0]
    while start + window <= t[-1]:
        end = start + window
        mask = (t >= start) & (t <= end)
        if mask.sum() < 2:
            start += window
            continue

        integral = np.zeros((phi.shape[1], phi.shape[1]))
        t_masked = t[mask]
        phi_masked = phi[mask]
        for k in range(len(t_masked) - 1):
            dt = t_masked[k + 1] - t_masked[k]
            integral += dt * np.outer(phi_masked[k], phi_masked[k])
        eigvals = np.linalg.eigvalsh(integral)
        results[(start, end)] = eigvals
        start += window
    return results


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv(base_dir / "data.csv")
    t = df["t"].to_numpy()
    phi = df[["x_1", "x_2", "x_3"]].to_numpy()
    y = df["y"].to_numpy()

    print("=== Task 2: regressor identification ===")
    guesses = {
        "x_1": (10.0, -0.05, -5.0, 0.0),
        "x_2": (2.0, -0.4, -0.1, 0.0),
        "x_3": (30.0, -0.08, -2.0, -1.0),
    }
    components: Dict[str, ExpComponent] = {}
    for name, guess in guesses.items():
        comp = fit_exponential_component(t, df[name].to_numpy(), initial_guess=guess)
        rmse = np.sqrt(np.mean((comp(t) - df[name].to_numpy()) ** 2))
        components[name] = comp
        print(
            f"{name}(t) ≈ {comp.a:.6f} * exp({comp.b:.6f} * t + {comp.d:.6f}) + "
            f"{comp.c:.6f} (RMSE {rmse:.3e})"
        )

    print("\n=== Online gradient descent ===")
    t1 = float((base_dir / "t1.txt").read_text().strip())
    theta_t1 = online_gradient_descent(
        t,
        phi,
        y,
        gamma=1.0,
        t_stop=t1,
        theta0=np.zeros(3),
    )
    for i, val in enumerate(theta_t1, start=1):
        print(f"theta_{i}(t1={t1:.6f}) = {val:.9f}")

    print("\n=== Persistent excitation check ===")
    eig_windows = analyze_persistent_excitation(t, phi, window=5.0)
    for (start, end), eigvals in eig_windows.items():
        eig_list = ", ".join(f"{eig:.3e}" for eig in eigvals)
        print(f"Window [{start:.1f}, {end:.1f}] -> eigenvalues: {eig_list}")


if __name__ == "__main__":
    main()

