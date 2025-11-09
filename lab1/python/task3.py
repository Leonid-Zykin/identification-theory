from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_variant_mat


def fit_linear_basis(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return theta


def process_zad31(images_dir: Path) -> None:
    d = load_variant_mat(25)["zad31"]
    x = np.asarray(getattr(d, "x"), dtype=float)
    y = np.asarray(getattr(d, "y"), dtype=float)
    # func: y(x) = (7.0^p1) * (x^p2) = exp(p1*ln 7) * exp(p2*ln x)
    # Линейная по параметрам представление в логарифмах:
    # ln y = p1*ln 7 + p2*ln x
    ln_y = np.log(y)
    phi1 = np.log(7.0) * np.ones_like(x)
    phi2 = np.log(x)
    Phi = np.column_stack([phi1, phi2])
    theta = fit_linear_basis(Phi, ln_y)
    p1, p2 = theta  # непосредственно параметры p1, p2

    y_hat = (7.0 ** p1) * (x ** p2)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=30, label="эксперимент")
    plt.plot(x, y_hat, label=f"модель, p1={p1:.3f}, p2={p2:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("zad31: y(x) = (7^p1)*(x^p2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / "zad31_fit.png", dpi=150)
    plt.close()

    print("zad31 params:", p1, p2)


def process_zad32(images_dir: Path) -> None:
    d = load_variant_mat(25)["zad32"]
    x = np.asarray(getattr(d, "x"), dtype=float)
    y = np.asarray(getattr(d, "y"), dtype=float)
    # func: y(x) = p1 * exp(p2*x) -> y = p1 * exp(p2*x)
    # Представим как линейную регрессию по p1,p2 в логарифмах: ln y = ln p1 + p2*x
    ln_y = np.log(y)
    Phi = np.column_stack([np.ones_like(x), x])
    a0, p2 = fit_linear_basis(Phi, ln_y)
    p1 = np.exp(a0)

    y_hat = p1 * np.exp(p2 * x)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=30, label="эксперимент")
    plt.plot(x, y_hat, label=f"модель, p1={p1:.3f}, p2={p2:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("zad32: y(x) = p1*exp(p2*x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / "zad32_fit.png", dpi=150)
    plt.close()

    print("zad32 params:", p1, p2)


def main() -> None:
    images_dir = Path(__file__).resolve().parents[1] / "images" / "task3"
    ensure_dir(images_dir)
    process_zad31(images_dir)
    process_zad32(images_dir)


if __name__ == "__main__":
    main()





