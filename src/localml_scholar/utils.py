"""Reproducibility, numerical checks, and small shared utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from localml_scholar.models.bigram import BigramLanguageModel


@dataclass(frozen=True)
class GradientCheckResult:
    """Summary of a finite-difference gradient check."""

    checked_parameters: int
    maximum_relative_error: float
    worst_index: tuple[int, int]
    analytical_at_worst: float
    numerical_at_worst: float
    passed: bool


def seed_everything(seed: int) -> None:
    """Seed Python and legacy NumPy global RNGs for reproducible surrounding code."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    random.seed(seed)
    np.random.seed(seed)


def safe_perplexity(loss: float) -> float:
    """Compute ``exp(loss)`` without overflowing floating-point range."""
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise TypeError("loss must be a real number.")
    normalized = float(loss)
    if math.isnan(normalized):
        raise ValueError("loss must not be NaN.")
    if normalized == math.inf:
        return float(np.finfo(np.float64).max)
    if normalized == -math.inf:
        return 0.0
    maximum_log = math.log(float(np.finfo(np.float64).max))
    return math.exp(min(normalized, maximum_log))


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def check_bigram_gradients(
    model: BigramLanguageModel,
    input_ids: NDArray[np.integer],
    target_ids: NDArray[np.integer],
    *,
    epsilon: float = 1e-5,
    tolerance: float = 1e-5,
    denominator_delta: float = 1e-12,
    max_checks: int | None = None,
    seed: int = 0,
    raise_on_failure: bool = True,
) -> GradientCheckResult:
    """Compare analytical weight gradients with centered finite differences."""
    epsilon = _positive_finite(epsilon, "epsilon")
    tolerance = _positive_finite(tolerance, "tolerance")
    denominator_delta = _positive_finite(denominator_delta, "denominator_delta")
    if max_checks is not None and (
        isinstance(max_checks, bool)
        or not isinstance(max_checks, int)
        or max_checks <= 0
    ):
        raise ValueError("max_checks must be None or a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not isinstance(raise_on_failure, bool):
        raise TypeError("raise_on_failure must be a boolean.")

    model.grad_weights.fill(0.0)
    model.loss_and_backward(input_ids, target_ids)
    analytical = model.grad_weights.copy()

    total_parameters = model.weights.size
    check_count = (
        total_parameters if max_checks is None else min(max_checks, total_parameters)
    )
    if check_count == total_parameters:
        flat_indices = np.arange(total_parameters)
    else:
        rng = np.random.default_rng(seed)
        flat_indices = np.sort(
            rng.choice(total_parameters, size=check_count, replace=False)
        )

    maximum_relative_error = -1.0
    worst_index = (0, 0)
    analytical_at_worst = 0.0
    numerical_at_worst = 0.0

    for flat_index_value in flat_indices:
        index = np.unravel_index(int(flat_index_value), model.weights.shape)
        original = float(model.weights[index])
        try:
            model.weights[index] = original + epsilon
            loss_plus = model.loss(input_ids, target_ids)
            model.weights[index] = original - epsilon
            loss_minus = model.loss(input_ids, target_ids)
        finally:
            model.weights[index] = original

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical_value = float(analytical[index])
        relative_error = abs(analytical_value - numerical) / (
            abs(analytical_value) + abs(numerical) + denominator_delta
        )
        if relative_error > maximum_relative_error:
            maximum_relative_error = relative_error
            worst_index = (int(index[0]), int(index[1]))
            analytical_at_worst = analytical_value
            numerical_at_worst = numerical

    passed = maximum_relative_error <= tolerance
    result = GradientCheckResult(
        checked_parameters=check_count,
        maximum_relative_error=maximum_relative_error,
        worst_index=worst_index,
        analytical_at_worst=analytical_at_worst,
        numerical_at_worst=numerical_at_worst,
        passed=passed,
    )
    if not passed and raise_on_failure:
        raise AssertionError(
            "Bigram gradient check failed: "
            f"index={worst_index}, analytical={analytical_at_worst:.12e}, "
            f"numerical={numerical_at_worst:.12e}, "
            f"relative_error={maximum_relative_error:.12e}, "
            f"tolerance={tolerance:.12e}."
        )
    return result


def require_config_fields(config: dict[str, Any], fields: set[str]) -> None:
    """Raise an informative error when configuration fields are missing."""
    missing = sorted(fields - set(config))
    if missing:
        raise ValueError(f"Configuration is missing required fields: {missing}.")
