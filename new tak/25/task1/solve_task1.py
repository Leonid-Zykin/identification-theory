"""Solve task 1: estimate linear regression parameters with an intercept.

The script reads ``data.csv`` located in the same directory, augments the
regressor matrix with a column of ones (intercept), and reports the least
square estimates.  It also computes the residual sum of squares for the
intercept and no-intercept variants so that the user can justify the model
choice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data.csv"
    df = pd.read_csv(data_path)

    # Prepare matrices.
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    X_design = np.column_stack([np.ones(len(X)), X])
    theta_with_intercept, residuals_int, *_ = np.linalg.lstsq(
        X_design, y, rcond=None
    )

    theta_no_intercept, residuals_no_int, *_ = np.linalg.lstsq(X, y, rcond=None)

    print("=== Task 1: parameter estimates ===")
    for i, val in enumerate(theta_with_intercept, start=1):
        tag = "intercept" if i == 1 else f"x_{i-1}"
        print(f"{i}. {tag:>10s} = {val:.11f}")

    print("\nResidual sum of squares comparison:")
    print(f"  with intercept : {float(residuals_int):.6f}")
    print(f"  without intercept: {float(residuals_no_int):.6f}")


if __name__ == "__main__":
    main()

