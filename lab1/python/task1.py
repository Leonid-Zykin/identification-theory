from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

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


def compute_diagnostics(e: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    mean_e = float(np.mean(e))
    var_e = float(np.var(e, ddof=1))
    # корреляции остатков с регрессорами
    corrs = []
    for i in range(X.shape[1]):
        xi = X[:, i] - np.mean(X[:, i])
        ei = e - np.mean(e)
        denom = np.sqrt(np.sum(xi**2) * np.sum(ei**2))
        corrs.append(float(np.sum(xi * ei) / denom) if denom > 0 else 0.0)
    # автокорреляция лаг-1
    e0 = e[:-1] - np.mean(e[:-1])
    e1 = e[1:] - np.mean(e[1:])
    denom_ac = np.sqrt(np.sum(e0**2) * np.sum(e1**2))
    ac1 = float(np.sum(e0 * e1) / denom_ac) if denom_ac > 0 else 0.0
    return {
        "mean_e": mean_e,
        "var_e": var_e,
        "corr_e_x1": corrs[0],
        "corr_e_x2": corrs[1],
        "corr_e_x3": corrs[2],
        "acf1_e": ac1,
    }


def process_block(block_name: str, images_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
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

    diags = compute_diagnostics(e, X)
    return theta, diags


def main() -> None:
    images_dir = Path(__file__).resolve().parents[1] / "images" / "task1"
    ensure_dir(images_dir)

    theta11, d11 = process_block("zad11", images_dir)
    theta12, d12 = process_block("zad12", images_dir)

    print("theta_zad11:", theta11)
    print("theta_zad12:", theta12)
    print("diag_zad11:", d11)
    print("diag_zad12:", d12)


if __name__ == "__main__":
    main()



