from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_variant_mat


def poly_design_matrix(T: np.ndarray, degree: int) -> np.ndarray:
    # degree 1 -> [T, 1]; degree 2 -> [T^2, T, 1]
    cols = []
    for p in range(degree, 0, -1):
        cols.append(T ** p)
    cols.append(np.ones_like(T))
    return np.column_stack(cols)


def fit_model(X: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta, *_ = np.linalg.lstsq(X, V, rcond=None)
    V_hat = X @ theta
    e = V - V_hat
    sse = float(np.sum(e ** 2))
    return theta, V_hat, e, sse


def process(block_name: str, images_dir: Path) -> None:
    d = load_variant_mat(25)[block_name]
    T = np.asarray(getattr(d, "T")).astype(float)
    V = np.asarray(getattr(d, "V")).astype(float)

    X1 = poly_design_matrix(T, degree=1)
    t1, V1, e1, sse1 = fit_model(X1, V)

    X2 = poly_design_matrix(T, degree=2)
    t2, V2, e2, sse2 = fit_model(X2, V)

    # Plot models on data
    order = np.argsort(T)
    plt.figure(figsize=(8, 5))
    plt.scatter(T, V, s=30, label="эксперимент")
    plt.plot(T[order], V1[order], label="H1: линейная")
    plt.plot(T[order], V2[order], label="H2: квадратичная")
    plt.xlabel("T, °C")
    plt.ylabel("V, см^3/с")
    plt.title(f"{block_name}: аппроксимации")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / f"{block_name}_models.png", dpi=150)
    plt.close()

    # Residuals
    plt.figure(figsize=(9, 4))
    width = 0.35
    idx = np.arange(T.size)
    plt.bar(idx - width / 2, e1, width, label=f"H1, SSE={sse1:.2f}")
    plt.bar(idx + width / 2, e2, width, label=f"H2, SSE={sse2:.2f}")
    plt.xlabel("Номер точки")
    plt.ylabel("Ошибка")
    plt.title(f"{block_name}: ошибки аппроксимаций")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / f"{block_name}_residuals.png", dpi=150)
    plt.close()

    print(block_name, "SSE H1=", sse1, "SSE H2=", sse2)


def main() -> None:
    images_dir = Path(__file__).resolve().parents[1] / "images" / "task2"
    ensure_dir(images_dir)
    process("zad21", images_dir)
    process("zad22", images_dir)


if __name__ == "__main__":
    main()





