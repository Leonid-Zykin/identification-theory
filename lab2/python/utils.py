from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat


def get_project_root() -> Path:
    """Return repository root path based on this file location."""
    return Path(__file__).resolve().parents[2]


def get_lab2_mat_path(variant: int = 25) -> Path:
    root = get_project_root()
    return root / "lab2" / "ident_lab2_vXX" / f"ident_lab2_v{variant:02d}.mat"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_variant_mat(variant: int = 25) -> Dict[str, Any]:
    mat_path = get_lab2_mat_path(variant)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return data


def to_numpy(x: Any) -> np.ndarray:
    """Convert MATLAB arrays or scalars to 1D numpy arrays when sensible."""
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    return arr

