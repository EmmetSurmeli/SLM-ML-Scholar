"""Numerically stable classification losses implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_logits(logits: NDArray[np.floating]) -> FloatArray:
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"logits must be 1D or 2D, received shape {array.shape}.")
    if array.size == 0 or array.shape[-1] == 0:
        raise ValueError("logits must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("logits must contain only finite values.")
    return array


def _validate_batch(
    values: NDArray[np.floating],
    targets: NDArray[np.integer],
    value_name: str,
) -> tuple[FloatArray, IntArray]:
    array = np.asarray(values, dtype=np.float64)
    target_array = np.asarray(targets)
    if array.ndim != 2:
        raise ValueError(
            f"{value_name} must have shape (batch, classes), got {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{value_name} must have non-zero batch and class axes.")
    if target_array.ndim != 1 or target_array.shape[0] != array.shape[0]:
        raise ValueError(
            f"targets must have shape ({array.shape[0]},), got {target_array.shape}."
        )
    if not np.issubdtype(target_array.dtype, np.integer):
        raise TypeError("targets must contain integer class indices.")
    normalized_targets = target_array.astype(np.int64, copy=False)
    if np.any(normalized_targets < 0) or np.any(normalized_targets >= array.shape[1]):
        raise ValueError(
            f"targets must be in [0, {array.shape[1]}), received "
            f"range [{normalized_targets.min()}, {normalized_targets.max()}]."
        )
    return array, normalized_targets


def stable_softmax(logits: NDArray[np.floating]) -> FloatArray:
    """Compute softmax after subtracting the row maximum.

    One-dimensional input produces one probability vector. Two-dimensional
    input is normalized independently across the final (class) axis.
    """
    array = _validate_logits(logits)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / denominator
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("Softmax produced non-finite probabilities.")
    return probabilities


def multiclass_cross_entropy(
    probabilities: NDArray[np.floating],
    targets: NDArray[np.integer],
) -> float:
    """Return mean indexed cross-entropy without constructing one-hot targets."""
    values, target_array = _validate_batch(probabilities, targets, "probabilities")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("Every probability row must sum to one.")

    selected = values[np.arange(values.shape[0]), target_array]
    with np.errstate(divide="ignore"):
        losses = -np.log(selected)
    return float(np.mean(losses))


def softmax_cross_entropy_forward(
    logits: NDArray[np.floating],
    targets: NDArray[np.integer],
) -> tuple[float, FloatArray]:
    """Return stable mean cross-entropy and softmax probabilities."""
    values, target_array = _validate_batch(logits, targets, "logits")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must contain only finite values.")

    shifted = values - np.max(values, axis=1, keepdims=True)
    log_normalizers = np.log(np.sum(np.exp(shifted), axis=1))
    selected_shifted_logits = shifted[np.arange(values.shape[0]), target_array]
    loss = np.mean(log_normalizers - selected_shifted_logits)
    probabilities = np.exp(shifted - log_normalizers[:, None])
    return float(loss), probabilities


def softmax_cross_entropy_backward(
    probabilities: NDArray[np.floating],
    targets: NDArray[np.integer],
) -> FloatArray:
    """Return ``(probabilities - one_hot(targets)) / batch_size``."""
    values, target_array = _validate_batch(probabilities, targets, "probabilities")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("Every probability row must sum to one.")

    gradient = values.copy()
    gradient[np.arange(values.shape[0]), target_array] -= 1.0
    gradient /= values.shape[0]
    return gradient
