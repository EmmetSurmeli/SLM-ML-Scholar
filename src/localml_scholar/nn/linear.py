"""A manually differentiated affine layer."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from localml_scholar.nn.initialization import (
    he_normal,
    validate_float_dtype,
    xavier_uniform,
)
from localml_scholar.nn.module import FloatArray, Module
from localml_scholar.nn.parameter import Parameter


class Linear(Module):
    """Apply ``Y = XW + b`` over the final input dimension.

    A training forward caches one reference to the input array until backward.
    Mutating that input before backward is unsupported.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,
        initialization: str = "xavier_uniform",
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.input_dim = self._validate_dimension(input_dim, "input_dim")
        self.output_dim = self._validate_dimension(output_dim, "output_dim")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean.")
        self.use_bias = bias
        self.dtype = validate_float_dtype(dtype)
        generator = self._resolve_rng(rng, seed)

        shape = (self.input_dim, self.output_dim)
        if initialization == "xavier_uniform":
            weight_values = xavier_uniform(shape, generator, dtype=self.dtype)
        elif initialization == "he_normal":
            weight_values = he_normal(shape, generator, dtype=self.dtype)
        else:
            raise ValueError("initialization must be 'xavier_uniform' or 'he_normal'.")

        self.weight = Parameter(weight_values, name="weight")
        self.register_parameter("weight", self.weight)
        self.bias: Parameter | None
        if bias:
            self.bias = Parameter(
                np.zeros(self.output_dim, dtype=self.dtype), name="bias"
            )
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

    @staticmethod
    def _validate_dimension(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return value

    @staticmethod
    def _resolve_rng(
        rng: np.random.Generator | None, seed: int | None
    ) -> np.random.Generator:
        if rng is not None and seed is not None:
            raise ValueError("Provide at most one of rng and seed.")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator.")
            return rng
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be None or an integer.")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative.")
        return np.random.default_rng(seed)

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Compute the affine transformation over the final dimension."""
        values = self._validate_float_array(
            inputs, "inputs", dtype=self.dtype, minimum_dimensions=1
        )
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs final dimension must be {self.input_dim}, "
                f"got shape {values.shape}."
            )
        self._store_forward_cache(values)
        output = values @ self.weight.data
        if self.bias is not None:
            output = output + self.bias.data
        return output

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        """Accumulate parameter gradients and return the input gradient."""
        inputs = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_output,
            "grad_output",
            dtype=self.dtype,
            minimum_dimensions=1,
        )
        expected_shape = inputs.shape[:-1] + (self.output_dim,)
        if gradient.shape != expected_shape:
            raise ValueError(
                f"grad_output shape must be {expected_shape}, got {gradient.shape}."
            )

        flat_inputs = inputs.reshape(-1, self.input_dim)
        flat_gradient = gradient.reshape(-1, self.output_dim)
        self.weight._validate_gradient_buffer()
        self.weight.grad += flat_inputs.T @ flat_gradient
        if self.bias is not None:
            self.bias._validate_gradient_buffer()
            self.bias.grad += np.sum(flat_gradient, axis=0)
        grad_input = (flat_gradient @ self.weight.data.T).reshape(inputs.shape)
        self._consume_forward_cache()
        return grad_input
