"""Manually differentiated integer embedding lookup."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from localml_scholar.nn.initialization import validate_float_dtype
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.module import FloatArray, Module
from localml_scholar.nn.parameter import Parameter

IntArray = NDArray[np.integer]


class Embedding(Module):
    """Map integer IDs to rows of a trainable embedding table."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.vocabulary_size = Linear._validate_dimension(
            vocabulary_size, "vocabulary_size"
        )
        self.embedding_dim = Linear._validate_dimension(embedding_dim, "embedding_dim")
        self.dtype = validate_float_dtype(dtype)
        generator = Linear._resolve_rng(rng, seed)
        scale = 1.0 / math.sqrt(self.embedding_dim)
        values = generator.normal(
            0.0,
            scale,
            size=(self.vocabulary_size, self.embedding_dim),
        ).astype(self.dtype, copy=False)
        self.weight = Parameter(values, name="weight")
        self.register_parameter("weight", self.weight)

    def _validate_indices(self, inputs: ArrayLike) -> IntArray:
        indices = np.asarray(inputs)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(
                f"Embedding indices must have an integer dtype, got {indices.dtype}."
            )
        if indices.size == 0:
            raise ValueError("Embedding indices must be non-empty.")
        if np.any(indices < 0) or np.any(indices >= self.vocabulary_size):
            minimum = int(np.min(indices))
            maximum = int(np.max(indices))
            raise ValueError(
                f"Embedding indices must lie in [0, {self.vocabulary_size}), "
                f"received range [{minimum}, {maximum}]."
            )
        return indices

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Look up rows with output shape ``input_shape + (embedding_dim,)``."""
        indices = self._validate_indices(inputs)
        self._store_forward_cache(indices.copy())
        return self.weight.data[indices]

    def backward(self, grad_output: ArrayLike) -> None:
        """Accumulate table gradients; integer indices have no gradient."""
        indices = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_output,
            "grad_output",
            dtype=self.dtype,
            minimum_dimensions=1,
        )
        expected_shape = indices.shape + (self.embedding_dim,)
        if gradient.shape != expected_shape:
            raise ValueError(
                f"grad_output shape must be {expected_shape}, got {gradient.shape}."
            )
        self.weight._validate_gradient_buffer()
        np.add.at(self.weight.grad, indices, gradient)
        self._consume_forward_cache()
        return None
