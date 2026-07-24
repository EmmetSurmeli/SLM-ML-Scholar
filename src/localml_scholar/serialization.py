"""Small dependency-free persistence helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_savez(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Atomically replace an NPZ file after fully writing a sibling temporary."""
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("NPZ destination must end with '.npz'.")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("arrays must be a non-empty dictionary.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=f".{destination.stem}.",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination
