from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_variant_mat


def estimate_theta(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ theta
    residuals = y - y_hat
    return theta, y_hat, residuals


def plot_fit(x_axis: np.ndarray, y: np.ndarray, y_hat: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, y, label="y(k)")
    plt.plot(x_axis, y_hat, label="ŷ(k)")
    plt.xlabel("k")
    plt.ylabel("Амплитуда")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(x_axis: np.ndarray, e: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, e, label="e(k) = y-ŷ")
    plt.xlabel("k")
    plt.ylabel("Ошибка")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_block(block_name: str, images_dir: Path) -> np.ndarray:
    data = load_variant_mat(25)[block_name]
    x1 = np.asarray(getattr(data, "x1"))
    x2 = np.asarray(getattr(data, "x2"))
    x3 = np.asarray(getattr(data, "x3"))
    y = np.asarray(getattr(data, "y"))

    X = np.column_stack([x1, x2, x3])
    theta, y_hat, e = estimate_theta(X, y)

    k = np.arange(y.size)
    plot_fit(k, y, y_hat, images_dir / f"{block_name}_fit.png", f"{block_name}: данные и оценка")
    plot_residuals(k, e, images_dir / f"{block_name}_residuals.png", f"{block_name}: ошибка")

    return theta


def main() -> None:
    images_dir = Path(__file__).resolve().parents[1] / "images" / "task1"
    ensure_dir(images_dir)

    theta11 = process_block("zad11", images_dir)
    theta12 = process_block("zad12", images_dir)

    print("theta_zad11:", theta11)
    print("theta_zad12:", theta12)


if __name__ == "__main__":
    main()



