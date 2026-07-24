"""Position-wise feed-forward and one pre-normalized decoder block."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from localml_scholar.nn.activations import GELU, ReLU
from localml_scholar.nn.attention import (
    AttentionDetails,
    CausalSelfAttentionHead,
)
from localml_scholar.nn.initialization import validate_float_dtype
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.module import FloatArray, Module
from localml_scholar.nn.normalization import LayerNorm

_CHECKPOINT_VERSION = 1
_MODEL_VERSION = "0.4.0"


def _positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _readonly_copy(values: NDArray) -> NDArray:
    copied = np.array(values, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_three_dimensional_input(
    module: Module,
    inputs: ArrayLike,
    *,
    model_dim: int,
    dtype: np.dtype[np.floating],
) -> FloatArray:
    values = module._validate_float_array(
        inputs,
        "inputs",
        dtype=dtype,
        minimum_dimensions=3,
    )
    if values.ndim != 3:
        raise ValueError(
            "inputs must have exactly three dimensions (B, T, D), "
            f"got shape {values.shape}."
        )
    if values.shape[-1] != model_dim:
        raise ValueError(
            f"inputs final dimension must be {model_dim}, got shape {values.shape}."
        )
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("inputs batch and sequence dimensions must be positive.")
    return values


def residual_add(
    identity: ArrayLike,
    transformed: ArrayLike,
    *,
    name: str = "residual",
) -> FloatArray:
    """Add equal-shaped floating tensors without permitting broadcasting."""
    if not isinstance(name, str) or not name:
        raise ValueError("Residual name must be a non-empty string.")
    identity_values = Module._validate_float_array(identity, f"{name} identity")
    transformed_values = Module._validate_float_array(
        transformed,
        f"{name} transformed",
    )
    if identity_values.shape != transformed_values.shape:
        raise ValueError(
            f"{name} residual shapes must match exactly; received "
            f"{identity_values.shape} and {transformed_values.shape}."
        )
    if identity_values.dtype != transformed_values.dtype:
        raise TypeError(
            f"{name} residual dtypes must match exactly; received "
            f"{identity_values.dtype} and {transformed_values.dtype}."
        )
    output = identity_values + transformed_values
    if not np.all(np.isfinite(output)):
        raise FloatingPointError(
            f"{name} residual addition produced non-finite values."
        )
    return output


def residual_add_backward(
    grad_output: ArrayLike,
    *,
    expected_shape: tuple[int, ...],
    dtype: np.dtype[np.floating],
    name: str = "residual",
) -> tuple[FloatArray, FloatArray]:
    """Return independent copies of the upstream gradient for both branches."""
    gradient = Module._validate_float_array(
        grad_output,
        f"{name} grad_output",
        dtype=dtype,
    )
    if gradient.shape != expected_shape:
        raise ValueError(
            f"{name} grad_output shape must be {expected_shape}, got {gradient.shape}."
        )
    return np.array(gradient, copy=True), np.array(gradient, copy=True)


def _save_checkpoint(
    module: Module,
    path: str | Path,
    *,
    model_type: str,
    configuration: dict[str, Any],
) -> Path:
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("Checkpoint path must end with '.npz'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "model_type": model_type,
        "configuration": configuration,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for name, parameter in module.named_parameters():
        arrays[f"parameter::{name}"] = parameter.data
    np.savez(destination, **arrays)
    return destination


def _load_checkpoint(
    path: str | Path,
    *,
    expected_model_type: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as checkpoint:
            if "metadata_json" not in checkpoint.files:
                raise ValueError("Checkpoint is missing metadata_json.")
            try:
                metadata = json.loads(str(checkpoint["metadata_json"]))
            except json.JSONDecodeError as error:
                raise ValueError("Checkpoint metadata is not valid JSON.") from error
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint metadata must be an object.")
            if metadata.get("checkpoint_version") != _CHECKPOINT_VERSION:
                raise ValueError(
                    "Unsupported checkpoint version: "
                    f"{metadata.get('checkpoint_version')!r}."
                )
            if metadata.get("model_type") != expected_model_type:
                raise ValueError(
                    f"Checkpoint model type {metadata.get('model_type')!r} "
                    f"does not match {expected_model_type!r}."
                )
            configuration = metadata.get("configuration")
            if not isinstance(configuration, dict):
                raise ValueError("Checkpoint configuration must be an object.")
            arrays = {
                key: np.array(checkpoint[key], copy=True)
                for key in checkpoint.files
                if key != "metadata_json"
            }
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint does not exist: {source}") from None
    return configuration, arrays


def _restore_checkpoint_parameters(
    module: Module,
    arrays: dict[str, np.ndarray],
) -> None:
    expected_keys = {f"parameter::{name}" for name, _ in module.named_parameters()}
    actual_keys = set(arrays)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Checkpoint parameter keys do not match the module; "
            f"missing={missing}, unexpected={unexpected}."
        )
    for name, parameter in module.named_parameters():
        parameter.load_data(arrays[f"parameter::{name}"])


def _constructor_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(configuration)
    model_version = normalized.pop("model_version", None)
    if model_version != _MODEL_VERSION:
        raise ValueError(
            f"Checkpoint model version {model_version!r} "
            f"does not match {_MODEL_VERSION!r}."
        )
    return normalized


@dataclass(frozen=True)
class FeedForwardDetails:
    """Read-only tensors from one position-wise feed-forward calculation."""

    pre_activation: FloatArray
    activation: FloatArray


@dataclass(frozen=True)
class _FeedForwardCache:
    output_shape: tuple[int, ...]


class TransformerFeedForward(Module):
    """Position-wise ``Linear -> activation -> Linear`` transformation."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        activation: str = "gelu",
        seed: int = 0,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.model_dim = Linear._validate_dimension(model_dim, "model_dim")
        self.hidden_dim = Linear._validate_dimension(hidden_dim, "hidden_dim")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean.")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        self.use_bias = bias
        self.activation_name = activation
        self.seed = seed
        self.dtype = validate_float_dtype(dtype)

        rng = np.random.default_rng(seed)
        self.linear1 = Linear(
            self.model_dim,
            self.hidden_dim,
            bias=bias,
            rng=rng,
            dtype=self.dtype,
        )
        self.activation: Module = GELU() if activation == "gelu" else ReLU()
        self.linear2 = Linear(
            self.hidden_dim,
            self.model_dim,
            bias=bias,
            rng=rng,
            dtype=self.dtype,
        )
        self.register_module("linear1", self.linear1)
        self.register_module("activation", self.activation)
        self.register_module("linear2", self.linear2)

    @property
    def configuration(self) -> dict[str, Any]:
        """Return the complete reconstruction configuration."""
        return {
            "model_version": _MODEL_VERSION,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "bias": self.use_bias,
            "activation": self.activation_name,
            "seed": self.seed,
            "dtype": self.dtype.name,
        }

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.size for parameter in self.parameters())

    @overload
    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_details: Literal[False] = False,
    ) -> FloatArray: ...

    @overload
    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_details: Literal[True],
    ) -> tuple[FloatArray, FeedForwardDetails]: ...

    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_details: bool = False,
    ) -> FloatArray | tuple[FloatArray, FeedForwardDetails]:
        """Transform every sequence position independently."""
        if not isinstance(return_details, bool):
            raise TypeError("return_details must be a boolean.")
        values = _validate_three_dimensional_input(
            self,
            inputs,
            model_dim=self.model_dim,
            dtype=self.dtype,
        )
        if self.training and self.has_pending_cache():
            raise RuntimeError(
                "TransformerFeedForward.forward cannot run twice in training "
                "mode before backward consumes the first cache."
            )

        try:
            pre_activation = self.linear1.forward(values)
            activation = self.activation.forward(pre_activation)
            output = self.linear2.forward(activation)
            if output.shape != values.shape:
                raise RuntimeError(
                    "Transformer feed-forward output did not preserve input shape."
                )
            if not np.all(np.isfinite(output)):
                raise FloatingPointError(
                    "Transformer feed-forward produced non-finite values."
                )
            self._store_forward_cache(_FeedForwardCache(output_shape=output.shape))
        except Exception:
            self.clear_cache()
            raise

        if not return_details:
            return output
        return output, FeedForwardDetails(
            pre_activation=_readonly_copy(pre_activation),
            activation=_readonly_copy(activation),
        )

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        """Backpropagate through the second affine, activation, and first affine."""
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

        grad_activation = self.linear2.backward(gradient)
        grad_pre_activation = self.activation.backward(grad_activation)
        grad_input = self.linear1.backward(grad_pre_activation)
        if not np.all(np.isfinite(grad_input)):
            raise FloatingPointError(
                "Transformer feed-forward backward produced non-finite values."
            )
        self._consume_forward_cache()
        return grad_input

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist configuration and named feed-forward parameters."""
        return _save_checkpoint(
            self,
            path,
            model_type=type(self).__name__,
            configuration=self.configuration,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> TransformerFeedForward:
        """Reconstruct a feed-forward module and restore parameters exactly."""
        configuration, arrays = _load_checkpoint(
            path,
            expected_model_type=cls.__name__,
        )
        model = cls(**_constructor_configuration(configuration))
        _restore_checkpoint_parameters(model, arrays)
        return model


@dataclass(frozen=True)
class DecoderBlockDetails:
    """Read-only tensors from one pre-normalized decoder-block forward."""

    normalized_attention_input: FloatArray
    attention: AttentionDetails
    attention_output: FloatArray
    first_residual: FloatArray
    normalized_feed_forward_input: FloatArray
    feed_forward: FeedForwardDetails
    feed_forward_output: FloatArray
    output: FloatArray


@dataclass(frozen=True)
class _DecoderBlockCache:
    output_shape: tuple[int, ...]


class PreNormDecoderBlock(Module):
    """One pre-normalized single-head causal decoder block."""

    def __init__(
        self,
        model_dim: int,
        key_dim: int,
        ff_hidden_dim: int,
        *,
        value_dim: int | None = None,
        attention_bias: bool = True,
        feed_forward_bias: bool = True,
        attention_output_projection: bool = True,
        layer_norm_epsilon: float = 1e-5,
        activation: str = "gelu",
        seed: int = 0,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.model_dim = Linear._validate_dimension(model_dim, "model_dim")
        self.key_dim = Linear._validate_dimension(key_dim, "key_dim")
        self.ff_hidden_dim = Linear._validate_dimension(
            ff_hidden_dim,
            "ff_hidden_dim",
        )
        resolved_value_dim = key_dim if value_dim is None else value_dim
        self.value_dim = Linear._validate_dimension(resolved_value_dim, "value_dim")
        if not isinstance(attention_bias, bool):
            raise TypeError("attention_bias must be a boolean.")
        if not isinstance(feed_forward_bias, bool):
            raise TypeError("feed_forward_bias must be a boolean.")
        if not isinstance(attention_output_projection, bool):
            raise TypeError("attention_output_projection must be a boolean.")
        if not attention_output_projection and self.value_dim != self.model_dim:
            raise ValueError(
                "value_dim must equal model_dim when attention output "
                "projection is disabled."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'.")
        self.attention_bias = attention_bias
        self.feed_forward_bias = feed_forward_bias
        self.attention_output_projection = attention_output_projection
        self.layer_norm_epsilon = _positive_finite_float(
            layer_norm_epsilon,
            "layer_norm_epsilon",
        )
        self.activation_name = activation
        self.seed = seed
        self.dtype = validate_float_dtype(dtype)

        seed_rng = np.random.default_rng(seed)
        maximum_seed = np.iinfo(np.int64).max
        attention_seed = int(seed_rng.integers(0, maximum_seed))
        feed_forward_seed = int(seed_rng.integers(0, maximum_seed))

        self.norm1 = LayerNorm(
            self.model_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
        )
        self.attention = CausalSelfAttentionHead(
            self.model_dim,
            self.key_dim,
            value_dim=self.value_dim,
            bias=attention_bias,
            output_projection=attention_output_projection,
            seed=attention_seed,
            dtype=self.dtype,
        )
        self.norm2 = LayerNorm(
            self.model_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
        )
        self.feed_forward = TransformerFeedForward(
            self.model_dim,
            self.ff_hidden_dim,
            bias=feed_forward_bias,
            activation=activation,
            seed=feed_forward_seed,
            dtype=self.dtype,
        )
        self.register_module("norm1", self.norm1)
        self.register_module("attention", self.attention)
        self.register_module("norm2", self.norm2)
        self.register_module("feed_forward", self.feed_forward)

    @property
    def configuration(self) -> dict[str, Any]:
        """Return the complete reconstruction configuration."""
        return {
            "model_version": _MODEL_VERSION,
            "model_dim": self.model_dim,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "ff_hidden_dim": self.ff_hidden_dim,
            "attention_bias": self.attention_bias,
            "feed_forward_bias": self.feed_forward_bias,
            "attention_output_projection": self.attention_output_projection,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "activation": self.activation_name,
            "seed": self.seed,
            "dtype": self.dtype.name,
        }

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.size for parameter in self.parameters())

    @overload
    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_details: Literal[False] = False,
    ) -> FloatArray: ...

    @overload
    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_details: Literal[True],
    ) -> tuple[FloatArray, DecoderBlockDetails]: ...

    def forward(
        self,
        inputs: ArrayLike,
        *,
        return_details: bool = False,
    ) -> FloatArray | tuple[FloatArray, DecoderBlockDetails]:
        """Apply pre-norm attention and feed-forward residual sublayers."""
        if not isinstance(return_details, bool):
            raise TypeError("return_details must be a boolean.")
        values = _validate_three_dimensional_input(
            self,
            inputs,
            model_dim=self.model_dim,
            dtype=self.dtype,
        )
        if self.training and self.has_pending_cache():
            raise RuntimeError(
                "PreNormDecoderBlock.forward cannot run twice in training "
                "mode before backward consumes the first cache."
            )

        try:
            normalized_attention_input = self.norm1.forward(values)
            if return_details:
                attention_output, attention_details = self.attention.forward(
                    normalized_attention_input,
                    return_attention=True,
                )
            else:
                attention_output = self.attention.forward(normalized_attention_input)
                attention_details = None
            first_residual = residual_add(
                values,
                attention_output,
                name="first",
            )
            normalized_feed_forward_input = self.norm2.forward(first_residual)
            if return_details:
                feed_forward_output, feed_forward_details = self.feed_forward.forward(
                    normalized_feed_forward_input,
                    return_details=True,
                )
            else:
                feed_forward_output = self.feed_forward.forward(
                    normalized_feed_forward_input
                )
                feed_forward_details = None
            output = residual_add(
                first_residual,
                feed_forward_output,
                name="second",
            )
            self._store_forward_cache(_DecoderBlockCache(output_shape=output.shape))
        except Exception:
            self.clear_cache()
            raise

        if not return_details:
            return output
        if attention_details is None or feed_forward_details is None:
            raise RuntimeError("Decoder-block inspection details were not produced.")
        return output, DecoderBlockDetails(
            normalized_attention_input=_readonly_copy(normalized_attention_input),
            attention=attention_details,
            attention_output=_readonly_copy(attention_output),
            first_residual=_readonly_copy(first_residual),
            normalized_feed_forward_input=_readonly_copy(normalized_feed_forward_input),
            feed_forward=feed_forward_details,
            feed_forward_output=_readonly_copy(feed_forward_output),
            output=_readonly_copy(output),
        )

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        """Backpropagate both residual branches with explicit accumulation."""
        cache = self._require_forward_cache()
        grad_first_residual_identity, grad_feed_forward = residual_add_backward(
            grad_output,
            expected_shape=cache.output_shape,
            dtype=self.dtype,
            name="second",
        )

        grad_normalized_feed_forward = self.feed_forward.backward(grad_feed_forward)
        grad_first_residual_branch = self.norm2.backward(grad_normalized_feed_forward)
        grad_first_residual = grad_first_residual_identity + grad_first_residual_branch
        if not np.all(np.isfinite(grad_first_residual)):
            raise FloatingPointError(
                "First residual gradient accumulation produced non-finite values."
            )

        grad_input_identity, grad_attention = residual_add_backward(
            grad_first_residual,
            expected_shape=cache.output_shape,
            dtype=self.dtype,
            name="first",
        )
        grad_normalized_attention = self.attention.backward(grad_attention)
        grad_input_branch = self.norm1.backward(grad_normalized_attention)
        grad_input = grad_input_identity + grad_input_branch
        if grad_input.shape != cache.output_shape:
            raise RuntimeError(
                "Decoder-block backward did not preserve the input shape."
            )
        if not np.all(np.isfinite(grad_input)):
            raise FloatingPointError(
                "Decoder-block backward produced non-finite input gradients."
            )
        self._consume_forward_cache()
        return grad_input

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist decoder configuration and every named parameter."""
        return _save_checkpoint(
            self,
            path,
            model_type=type(self).__name__,
            configuration=self.configuration,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> PreNormDecoderBlock:
        """Reconstruct a decoder block and restore parameters exactly."""
        configuration, arrays = _load_checkpoint(
            path,
            expected_model_type=cls.__name__,
        )
        model = cls(**_constructor_configuration(configuration))
        _restore_checkpoint_parameters(model, arrays)
        return model
