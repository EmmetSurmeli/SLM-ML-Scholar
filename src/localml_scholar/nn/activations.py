"""Elementwise activations with explicit analytical derivatives."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from localml_scholar.nn.module import FloatArray, Module

_SQRT_TWO = math.sqrt(2.0)
_INVERSE_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(values: FloatArray) -> FloatArray:
    """Evaluate the standard-normal CDF with standard-library ``erf``."""
    flat_result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / _SQRT_TWO)) for value in values.flat),
        dtype=np.float64,
        count=values.size,
    )
    return flat_result.reshape(values.shape).astype(values.dtype, copy=False)


class ReLU(Module):
    """Rectified linear unit with derivative zero at the origin."""

    def forward(self, inputs: ArrayLike) -> FloatArray:
        values = self._validate_float_array(inputs, "inputs")
        self._store_forward_cache((values > 0.0, values.dtype))
        return np.maximum(values, np.asarray(0.0, dtype=values.dtype))

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        positive_mask, input_dtype = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_output, "grad_output", dtype=input_dtype
        )
        if gradient.shape != positive_mask.shape:
            raise ValueError(
                f"grad_output shape must be {positive_mask.shape}, "
                f"got {gradient.shape}."
            )
        result = gradient * positive_mask
        self._consume_forward_cache()
        return result


class GELU(Module):
    """Exact Gaussian Error Linear Unit, ``x * Phi(x)``."""

    def forward(self, inputs: ArrayLike) -> FloatArray:
        values = self._validate_float_array(inputs, "inputs")
        self._store_forward_cache(values)
        return values * _standard_normal_cdf(values)

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        values = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_output, "grad_output", dtype=values.dtype
        )
        if gradient.shape != values.shape:
            raise ValueError(
                f"grad_output shape must be {values.shape}, got {gradient.shape}."
            )
        cdf = _standard_normal_cdf(values)
        with np.errstate(over="ignore"):
            density = np.exp(-0.5 * values * values) * _INVERSE_SQRT_TWO_PI
        derivative = cdf + values * density
        result = gradient * derivative
        self._consume_forward_cache()
        return result
