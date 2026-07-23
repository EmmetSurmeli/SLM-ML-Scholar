"""Seeded parameter initialization utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def validate_float_dtype(dtype: np.dtype | type[np.floating]) -> np.dtype:
    """Normalize and validate the supported project floating-point dtypes."""
    normalized = np.dtype(dtype)
    if normalized not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"dtype must be float32 or float64, received {normalized}.")
    return normalized


def _validate_matrix_shape(shape: Sequence[int]) -> tuple[int, int]:
    normalized = tuple(shape)
    if (
        len(normalized) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in normalized
        )
        or any(value <= 0 for value in normalized)
    ):
        raise ValueError(
            f"shape must contain two positive integer dimensions, got {normalized}."
        )
    return normalized[0], normalized[1]


def xavier_uniform(
    shape: Sequence[int],
    rng: np.random.Generator,
    *,
    dtype: np.dtype | type[np.floating] = np.float64,
) -> FloatArray:
    """Sample Xavier/Glorot uniform values for ``(fan_in, fan_out)``."""
    fan_in, fan_out = _validate_matrix_shape(shape)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    normalized_dtype = validate_float_dtype(dtype)
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(
        normalized_dtype, copy=False
    )


def he_normal(
    shape: Sequence[int],
    rng: np.random.Generator,
    *,
    dtype: np.dtype | type[np.floating] = np.float64,
) -> FloatArray:
    """Sample He-normal values for ``(fan_in, fan_out)``."""
    fan_in, fan_out = _validate_matrix_shape(shape)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    normalized_dtype = validate_float_dtype(dtype)
    standard_deviation = math.sqrt(2.0 / fan_in)
    return rng.normal(0.0, standard_deviation, size=(fan_in, fan_out)).astype(
        normalized_dtype, copy=False
    )
