"""Feature-wise normalization with a manual backward pass."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from localml_scholar.nn.initialization import validate_float_dtype
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.module import FloatArray, Module
from localml_scholar.nn.parameter import Parameter


class LayerNorm(Module):
    """Normalize the final dimension using population variance.

    The variance denominator is exactly the final feature count ``D``. A
    training forward caches ``x_hat`` and reciprocal standard deviation.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.feature_dim = Linear._validate_dimension(feature_dim, "feature_dim")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a real number.")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if not isinstance(affine, bool):
            raise TypeError("affine must be a boolean.")
        self.epsilon = float(epsilon)
        self.affine = affine
        self.dtype = validate_float_dtype(dtype)

        self.gamma: Parameter | None
        self.beta: Parameter | None
        if affine:
            self.gamma = Parameter(
                np.ones(self.feature_dim, dtype=self.dtype), name="gamma"
            )
            self.beta = Parameter(
                np.zeros(self.feature_dim, dtype=self.dtype), name="beta"
            )
            self.register_parameter("gamma", self.gamma)
            self.register_parameter("beta", self.beta)
        else:
            self.gamma = None
            self.beta = None

    def forward(self, inputs: ArrayLike) -> FloatArray:
        values = self._validate_float_array(
            inputs, "inputs", dtype=self.dtype, minimum_dimensions=1
        )
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"inputs final dimension must be {self.feature_dim}, "
                f"got shape {values.shape}."
            )

        mean = np.mean(values, axis=-1, keepdims=True, dtype=self.dtype)
        centered = values - mean
        variance = np.mean(
            centered * centered, axis=-1, keepdims=True, dtype=self.dtype
        )
        epsilon = np.asarray(self.epsilon, dtype=self.dtype)
        reciprocal_std = 1.0 / np.sqrt(variance + epsilon)
        normalized = centered * reciprocal_std
        self._store_forward_cache((normalized, reciprocal_std))

        if self.gamma is None or self.beta is None:
            return normalized
        return normalized * self.gamma.data + self.beta.data

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        normalized, reciprocal_std = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_output, "grad_output", dtype=self.dtype, minimum_dimensions=1
        )
        if gradient.shape != normalized.shape:
            raise ValueError(
                f"grad_output shape must be {normalized.shape}, got {gradient.shape}."
            )

        if self.gamma is None or self.beta is None:
            grad_normalized = gradient
        else:
            self.gamma._validate_gradient_buffer()
            self.beta._validate_gradient_buffer()
            reduction_axes = tuple(range(gradient.ndim - 1))
            self.gamma.grad += np.sum(gradient * normalized, axis=reduction_axes)
            self.beta.grad += np.sum(gradient, axis=reduction_axes)
            grad_normalized = gradient * self.gamma.data

        summed_gradient = np.sum(grad_normalized, axis=-1, keepdims=True)
        summed_projected_gradient = np.sum(
            grad_normalized * normalized, axis=-1, keepdims=True
        )
        grad_input = (
            reciprocal_std
            / self.feature_dim
            * (
                self.feature_dim * grad_normalized
                - summed_gradient
                - normalized * summed_projected_gradient
            )
        )
        self._consume_forward_cache()
        return grad_input
