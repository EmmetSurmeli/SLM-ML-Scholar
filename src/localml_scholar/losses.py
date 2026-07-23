"""Numerically stable indexed classification losses implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _validate_floating_array(
    values: ArrayLike,
    name: str,
    *,
    minimum_dimensions: int,
) -> FloatArray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating-point dtype, got {array.dtype}.")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must use float32 or float64, got {array.dtype}.")
    if array.ndim < minimum_dimensions:
        raise ValueError(
            f"{name} must have at least {minimum_dimensions} dimensions, "
            f"got shape {array.shape}."
        )
    if array.size == 0 or array.shape[-1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_classification(
    values: ArrayLike,
    targets: ArrayLike,
    value_name: str,
) -> tuple[FloatArray, IntArray]:
    array = _validate_floating_array(values, value_name, minimum_dimensions=2)
    target_array = np.asarray(targets)
    expected_target_shape = array.shape[:-1]
    if target_array.shape != expected_target_shape:
        raise ValueError(
            f"targets must have shape {expected_target_shape}, "
            f"got {target_array.shape}."
        )
    if not np.issubdtype(target_array.dtype, np.integer):
        raise TypeError("targets must contain integer class indices.")
    if target_array.size == 0:
        raise ValueError("targets must be non-empty.")
    if np.any(target_array < 0) or np.any(target_array >= array.shape[-1]):
        raise ValueError(
            f"targets must be in [0, {array.shape[-1]}), received "
            f"range [{int(target_array.min())}, {int(target_array.max())}]."
        )
    return array, target_array


def stable_softmax(logits: ArrayLike) -> FloatArray:
    """Compute softmax over the final dimension without changing dtype."""
    array = _validate_floating_array(logits, "logits", minimum_dimensions=1)
    # For opposite-sign values near float64 limits, the exact difference is
    # below -max_float and represents as -inf. exp(-inf)=0 is the correct
    # limiting probability, so suppress only that expected overflow warning.
    with np.errstate(over="ignore", invalid="raise"):
        shifted = array - np.max(array, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / denominator
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("Softmax produced non-finite probabilities.")
    return probabilities


def multiclass_cross_entropy(
    probabilities: ArrayLike,
    targets: ArrayLike,
) -> float:
    """Return mean indexed cross-entropy for 2D or higher probabilities."""
    values, target_array = _validate_classification(
        probabilities, targets, "probabilities"
    )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("Every probability vector must sum to one.")

    flat_values = values.reshape(-1, values.shape[-1])
    flat_targets = target_array.reshape(-1)
    selected = flat_values[np.arange(flat_targets.size), flat_targets]
    with np.errstate(divide="ignore"):
        losses = -np.log(selected)
    return float(np.mean(losses))


def softmax_cross_entropy_forward(
    logits: ArrayLike,
    targets: ArrayLike,
) -> tuple[float, FloatArray]:
    """Return stable mean cross-entropy and same-shaped probabilities.

    The final logits dimension is the class axis. Targets must match every
    leading logits dimension.
    """
    values, target_array = _validate_classification(logits, targets, "logits")
    class_count = values.shape[-1]
    flat_values = values.reshape(-1, class_count)
    flat_targets = target_array.reshape(-1)
    with np.errstate(over="ignore", invalid="raise"):
        shifted = flat_values - np.max(flat_values, axis=1, keepdims=True)
    log_normalizers = np.log(np.sum(np.exp(shifted), axis=1))
    selected_shifted_logits = shifted[np.arange(flat_targets.size), flat_targets]
    loss = np.mean(log_normalizers - selected_shifted_logits)
    flat_probabilities = np.exp(shifted - log_normalizers[:, None])
    return float(loss), flat_probabilities.reshape(values.shape)


def softmax_cross_entropy_backward(
    probabilities: ArrayLike,
    targets: ArrayLike,
) -> FloatArray:
    """Return ``(probabilities - one_hot(targets)) / example_count``."""
    values, target_array = _validate_classification(
        probabilities, targets, "probabilities"
    )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("Every probability vector must sum to one.")

    flat_gradient = values.reshape(-1, values.shape[-1]).copy()
    flat_targets = target_array.reshape(-1)
    flat_gradient[np.arange(flat_targets.size), flat_targets] -= 1.0
    flat_gradient /= np.asarray(flat_targets.size, dtype=values.dtype)
    return flat_gradient.reshape(values.shape)


def softmax_cross_entropy_loss_and_gradient(
    logits: ArrayLike,
    targets: ArrayLike,
) -> tuple[float, FloatArray]:
    """Return mean loss and its analytical gradient with respect to logits."""
    loss, probabilities = softmax_cross_entropy_forward(logits, targets)
    gradient = softmax_cross_entropy_backward(probabilities, targets)
    return loss, gradient
