"""Global gradient norm measurement and uniform clipping."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from localml_scholar.nn.parameter import Parameter, validate_parameter_sequence


def global_gradient_norm(parameters: Iterable[Parameter]) -> float:
    """Return a scaled Euclidean norm without avoidable squaring overflow."""
    normalized = validate_parameter_sequence(tuple(parameters))
    scale = 0.0
    scaled_sum_squares = 0.0

    for index, parameter in enumerate(normalized):
        parameter._validate_gradient_buffer()
        if not np.all(np.isfinite(parameter.grad)):
            raise ValueError(f"Parameter {index} gradient contains non-finite values.")
        local_max = float(np.max(np.abs(parameter.grad)))
        if local_max == 0.0:
            continue
        local_scaled_sum = float(
            np.sum(
                np.square(parameter.grad / local_max),
                dtype=np.float64,
            )
        )
        if scale == 0.0:
            scale = local_max
            scaled_sum_squares = local_scaled_sum
        elif scale < local_max:
            ratio = scale / local_max
            scaled_sum_squares = local_scaled_sum + scaled_sum_squares * ratio * ratio
            scale = local_max
        else:
            ratio = local_max / scale
            scaled_sum_squares += local_scaled_sum * ratio * ratio

    if scale == 0.0:
        return 0.0
    return scale * math.sqrt(scaled_sum_squares)


def clip_grad_norm(
    parameters: Iterable[Parameter],
    max_norm: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Clip all gradients uniformly and return their pre-clipping norm."""
    normalized = validate_parameter_sequence(tuple(parameters))
    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
        raise TypeError("max_norm must be a real number.")
    if not np.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive.")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a real number.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    norm = global_gradient_norm(normalized)
    if norm > float(max_norm):
        scale = float(max_norm) / (norm + float(epsilon))
        for parameter in normalized:
            parameter.grad *= scale
    return norm
