from __future__ import annotations

from pprint import pprint
from typing import Any

import numpy as np

from utils import load_variant_mat


def summarize(name: str, obj: Any) -> Any:
    if hasattr(obj, "_fieldnames"):
        return {f: summarize(f, getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, (str, bytes)):
        s = obj if isinstance(obj, str) else obj.decode("utf-8", "ignore")
        return {"type": type(obj).__name__, "preview": s[:120]}
    arr = np.asarray(obj)
    if arr.dtype.kind in {"U", "S", "O"}:
        return {"type": str(arr.dtype), "shape": arr.shape}
    return {
        "type": type(obj).__name__,
        "shape": arr.shape,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "value": float(arr) if arr.size == 1 else None,
    }


def main() -> None:
    data = load_variant_mat(25)
    keys = sorted(k for k in data.keys() if not k.startswith("__"))
    print("Keys:", keys)
    for k in keys:
        print(f"\n== {k} ==")
        pprint(summarize(k, data[k]))


if __name__ == "__main__":
    main()

