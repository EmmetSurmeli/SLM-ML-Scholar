"""Stable masked softmax primitives and single-head causal self-attention."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from localml_scholar.nn.initialization import validate_float_dtype
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.masks import causal_attention_mask
from localml_scholar.nn.module import FloatArray, Module


def _validate_attention_values(values: ArrayLike, name: str) -> NDArray[np.floating]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating-point dtype, got {array.dtype}.")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must use float32 or float64, got {array.dtype}.")
    if array.ndim < 1 or array.size == 0 or array.shape[-1] == 0:
        raise ValueError(f"{name} must be a non-empty array with a final axis.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _broadcast_allowed_mask(
    allowed_mask: ArrayLike,
    target_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    mask = np.asarray(allowed_mask)
    if mask.dtype != np.bool_:
        raise TypeError(f"allowed_mask must be boolean, got {mask.dtype}.")
    if mask.size == 0:
        raise ValueError("allowed_mask must be non-empty.")
    try:
        broadcast = np.broadcast_to(mask, target_shape)
    except ValueError as error:
        raise ValueError(
            f"allowed_mask shape {mask.shape} cannot broadcast to "
            f"score shape {target_shape}."
        ) from error
    if not np.all(np.any(broadcast, axis=-1)):
        raise ValueError(
            "Every masked-softmax row must contain at least one allowed entry."
        )
    return broadcast


def masked_softmax(
    logits: ArrayLike,
    allowed_mask: ArrayLike,
) -> NDArray[np.floating]:
    """Compute stable final-axis softmax with exact zeros where mask is false.

    ``True`` mask entries participate in the row maximum and normalization.
    ``False`` entries are blocked and receive exactly zero probability.
    """
    values = _validate_attention_values(logits, "logits")
    mask = _broadcast_allowed_mask(allowed_mask, values.shape)
    negative_infinity = np.asarray(-np.inf, dtype=values.dtype)
    valid_values = np.where(mask, values, negative_infinity)
    row_maximum = np.max(valid_values, axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="raise"):
        shifted = np.where(mask, values - row_maximum, negative_infinity)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / denominator
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("Masked softmax produced non-finite probabilities.")
    probabilities = np.where(
        mask,
        probabilities,
        np.asarray(0.0, dtype=values.dtype),
    )
    return probabilities


def masked_softmax_backward(
    probabilities: ArrayLike,
    grad_output: ArrayLike,
    allowed_mask: ArrayLike,
) -> NDArray[np.floating]:
    """Return the explicit masked-softmax gradient over the final axis."""
    values = _validate_attention_values(probabilities, "probabilities")
    gradient = _validate_attention_values(grad_output, "grad_output")
    if gradient.shape != values.shape:
        raise ValueError(
            f"grad_output shape {gradient.shape} must equal probability "
            f"shape {values.shape}."
        )
    if gradient.dtype != values.dtype:
        raise TypeError(
            f"grad_output dtype {gradient.dtype} does not match probability "
            f"dtype {values.dtype}."
        )
    mask = _broadcast_allowed_mask(allowed_mask, values.shape)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if np.any(values[~mask] != 0.0):
        raise ValueError("Blocked probabilities must be exactly zero.")
    if not np.allclose(
        np.sum(values, axis=-1),
        1.0,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Every probability row must sum to one.")

    projected = np.sum(gradient * values, axis=-1, keepdims=True)
    grad_logits = values * (gradient - projected)
    return np.where(
        mask,
        grad_logits,
        np.asarray(0.0, dtype=values.dtype),
    )


def _readonly_copy(values: NDArray) -> NDArray:
    copied = np.array(values, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class AttentionDetails:
    """Read-only tensors from one attention forward for inspection."""

    query: FloatArray
    key: FloatArray
    value: FloatArray
    scaled_scores: FloatArray
    allowed_mask: NDArray[np.bool_]
    probabilities: FloatArray


@dataclass(frozen=True)
class _AttentionCache:
    query: FloatArray
    key: FloatArray
    value: FloatArray
    probabilities: FloatArray
    allowed_mask: NDArray[np.bool_]
    output_shape: tuple[int, ...]


class CausalSelfAttentionHead(Module):
    """One scaled dot-product causal self-attention head.

    The query, key, and value projections are independent registered
    ``Linear`` modules. Training supports one unmatched forward. Returned
    ``AttentionDetails`` are inspection-only and are not separate gradient
    outputs.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        key_dim: int,
        *,
        value_dim: int | None = None,
        bias: bool = True,
        output_projection: bool = False,
        seed: int = 0,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.input_dim = Linear._validate_dimension(input_dim, "input_dim")
        self.key_dim = Linear._validate_dimension(key_dim, "key_dim")
        resolved_value_dim = key_dim if value_dim is None else value_dim
        self.value_dim = Linear._validate_dimension(resolved_value_dim, "value_dim")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean.")
        if not isinstance(output_projection, bool):
            raise TypeError("output_projection must be a boolean.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        self.use_bias = bias
        self.use_output_projection = output_projection
        self.seed = seed
        self.dtype = validate_float_dtype(dtype)
        self.scale = np.asarray(1.0 / math.sqrt(self.key_dim), dtype=self.dtype)

        rng = np.random.default_rng(seed)
        self.query_projection = Linear(
            self.input_dim,
            self.key_dim,
            bias=bias,
            rng=rng,
            dtype=self.dtype,
        )
        self.key_projection = Linear(
            self.input_dim,
            self.key_dim,
            bias=bias,
            rng=rng,
            dtype=self.dtype,
        )
        self.value_projection = Linear(
            self.input_dim,
            self.value_dim,
            bias=bias,
            rng=rng,
            dtype=self.dtype,
        )
        self.register_module("query_projection", self.query_projection)
        self.register_module("key_projection", self.key_projection)
        self.register_module("value_projection", self.value_projection)

        self.output_projection: Linear | None
        if output_projection:
            self.output_projection = Linear(
                self.value_dim,
                self.input_dim,
                bias=bias,
                rng=rng,
                dtype=self.dtype,
            )
            self.register_module("output_projection", self.output_projection)
        else:
            self.output_projection = None
        self._last_score_gradient: FloatArray | None = None

    @property
    def output_dim(self) -> int:
        """Return the final output feature dimension."""
        return self.input_dim if self.use_output_projection else self.value_dim

    @property
    def parameter_count(self) -> int:
        """Return the total number of scalar trainable parameters."""
        return sum(parameter.size for parameter in self.parameters())

    @property
    def configuration(self) -> dict[str, Any]:
        """Return the complete reconstruction configuration."""
        return {
            "input_dim": self.input_dim,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "bias": self.use_bias,
            "output_projection": self.use_output_projection,
            "seed": self.seed,
            "dtype": self.dtype.name,
        }

    @property
    def last_score_gradient(self) -> FloatArray | None:
        """Return a read-only copy of the most recent score gradient."""
        if self._last_score_gradient is None:
            return None
        return _readonly_copy(self._last_score_gradient)

    @overload
    def forward(
        self, inputs: ArrayLike, *, return_attention: Literal[False] = False
    ) -> FloatArray: ...

    @overload
    def forward(
        self, inputs: ArrayLike, *, return_attention: Literal[True]
    ) -> tuple[FloatArray, AttentionDetails]: ...

    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_attention: bool = False,
    ) -> FloatArray | tuple[FloatArray, AttentionDetails]:
        """Compute one batched causal attention head.

        Inputs must have shape ``(batch, sequence, input_dim)``. When
        ``return_attention`` is true, the second return value contains
        read-only copies for educational inspection.
        """
        if not isinstance(return_attention, bool):
            raise TypeError("return_attention must be a boolean.")
        values = self._validate_float_array(
            inputs,
            "inputs",
            dtype=self.dtype,
            minimum_dimensions=3,
        )
        if values.ndim != 3:
            raise ValueError(
                f"inputs must have exactly three dimensions (B, T, D), "
                f"got shape {values.shape}."
            )
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs final dimension must be {self.input_dim}, "
                f"got shape {values.shape}."
            )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("inputs batch and sequence dimensions must be positive.")
        if self.training and self.has_pending_cache():
            raise RuntimeError(
                "CausalSelfAttentionHead.forward cannot run twice in training "
                "mode before backward consumes the first cache."
            )

        try:
            query = self.query_projection.forward(values)
            key = self.key_projection.forward(values)
            value = self.value_projection.forward(values)
            if not (
                np.all(np.isfinite(query))
                and np.all(np.isfinite(key))
                and np.all(np.isfinite(value))
            ):
                raise FloatingPointError(
                    "Attention projections produced non-finite values."
                )

            scaled_scores = (query @ np.swapaxes(key, -1, -2)) * self.scale
            if not np.all(np.isfinite(scaled_scores)):
                raise FloatingPointError(
                    "Attention score calculation produced non-finite values."
                )
            allowed_mask = causal_attention_mask(values.shape[1])
            probabilities = masked_softmax(scaled_scores, allowed_mask)
            context = probabilities @ value
            if not np.all(np.isfinite(context)):
                raise FloatingPointError(
                    "Attention value aggregation produced non-finite values."
                )
            if self.output_projection is None:
                output = context
            else:
                output = self.output_projection.forward(context)
                if not np.all(np.isfinite(output)):
                    raise FloatingPointError(
                        "Attention output projection produced non-finite values."
                    )

            self._store_forward_cache(
                _AttentionCache(
                    query=query,
                    key=key,
                    value=value,
                    probabilities=probabilities,
                    allowed_mask=allowed_mask,
                    output_shape=output.shape,
                )
            )
            self._last_score_gradient = None
        except Exception:
            self.clear_cache()
            raise

        if not return_attention:
            return output
        details = AttentionDetails(
            query=_readonly_copy(query),
            key=_readonly_copy(key),
            value=_readonly_copy(value),
            scaled_scores=_readonly_copy(scaled_scores),
            allowed_mask=_readonly_copy(allowed_mask),
            probabilities=_readonly_copy(probabilities),
        )
        return output, details

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        """Backpropagate through aggregation, softmax, scores, and projections."""
        cache = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_output,
            "grad_output",
            dtype=self.dtype,
            minimum_dimensions=3,
        )
        if gradient.shape != cache.output_shape:
            raise ValueError(
                f"grad_output shape must be {cache.output_shape}, got {gradient.shape}."
            )

        if self.output_projection is None:
            grad_context = gradient
        else:
            grad_context = self.output_projection.backward(gradient)

        grad_attention = grad_context @ np.swapaxes(cache.value, -1, -2)
        grad_value = np.swapaxes(cache.probabilities, -1, -2) @ grad_context
        grad_scores = masked_softmax_backward(
            cache.probabilities,
            grad_attention,
            cache.allowed_mask,
        )
        grad_query = (grad_scores @ cache.key) * self.scale
        grad_key = (np.swapaxes(grad_scores, -1, -2) @ cache.query) * self.scale
        if not (
            np.all(np.isfinite(grad_query))
            and np.all(np.isfinite(grad_key))
            and np.all(np.isfinite(grad_value))
        ):
            raise FloatingPointError(
                "Attention backward produced non-finite projection gradients."
            )

        grad_input_query = self.query_projection.backward(grad_query)
        grad_input_key = self.key_projection.backward(grad_key)
        grad_input_value = self.value_projection.backward(grad_value)
        grad_input = grad_input_query + grad_input_key + grad_input_value
        if not np.all(np.isfinite(grad_input)):
            raise FloatingPointError(
                "Attention backward produced a non-finite input gradient."
            )
        self._last_score_gradient = np.array(grad_scores, copy=True)
        self._consume_forward_cache()
        return grad_input

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist versioned attention configuration and named parameters."""
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Checkpoint path must end with '.npz'.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "checkpoint_version": np.asarray(self.CHECKPOINT_VERSION, dtype=np.int64),
            "model_config_json": np.asarray(
                json.dumps(self.configuration, sort_keys=True)
            ),
        }
        for name, parameter in self.named_parameters():
            arrays[f"parameter::{name}"] = parameter.data
        np.savez(destination, **arrays)
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> CausalSelfAttentionHead:
        """Reconstruct an attention head and restore parameters exactly."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                if "checkpoint_version" not in checkpoint.files:
                    raise ValueError("Checkpoint is missing checkpoint_version.")
                version = int(checkpoint["checkpoint_version"])
                if version != cls.CHECKPOINT_VERSION:
                    raise ValueError(
                        f"Unsupported attention checkpoint version: {version}."
                    )
                if "model_config_json" not in checkpoint.files:
                    raise ValueError("Checkpoint is missing model_config_json.")
                try:
                    configuration = json.loads(str(checkpoint["model_config_json"]))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Checkpoint model configuration is not valid JSON."
                    ) from error
                if not isinstance(configuration, dict):
                    raise ValueError(
                        "Checkpoint model configuration must be an object."
                    )
                model = cls(**configuration)
                expected_keys = {
                    "checkpoint_version",
                    "model_config_json",
                    *(f"parameter::{name}" for name, _ in model.named_parameters()),
                }
                actual_keys = set(checkpoint.files)
                if actual_keys != expected_keys:
                    missing = sorted(expected_keys - actual_keys)
                    unexpected = sorted(actual_keys - expected_keys)
                    raise ValueError(
                        "Checkpoint parameter keys do not match attention; "
                        f"missing={missing}, unexpected={unexpected}."
                    )
                for name, parameter in model.named_parameters():
                    parameter.load_data(checkpoint[f"parameter::{name}"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint does not exist: {source}") from None
        return model
