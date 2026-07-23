"""Reusable immutable boolean masks for attention."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]


def causal_attention_mask(sequence_length: int) -> BoolArray:
    """Return a read-only ``(1, T, T)`` mask where ``True`` means allowed.

    The leading singleton dimension broadcasts across an attention batch.
    Query row ``i`` allows key columns ``j <= i`` and blocks future columns.
    """
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    mask = np.tri(
        sequence_length,
        sequence_length,
        k=0,
        dtype=np.bool_,
    )[None, :, :]
    mask.setflags(write=False)
    return mask
