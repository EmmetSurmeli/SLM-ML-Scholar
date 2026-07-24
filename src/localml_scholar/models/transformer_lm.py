"""A manually differentiated single-head decoder-only language model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from localml_scholar.nn.containers import Sequential
from localml_scholar.nn.embedding import Embedding
from localml_scholar.nn.initialization import validate_float_dtype
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.module import FloatArray, Module
from localml_scholar.nn.normalization import LayerNorm
from localml_scholar.nn.transformer import (
    PreNormDecoderBlock,
    residual_add,
    residual_add_backward,
)
from localml_scholar.serialization import atomic_savez

_CHECKPOINT_VERSION = 1
_MODEL_VERSION = "0.5.0"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


@dataclass(frozen=True)
class TransformerConfig:
    """Validated architecture and initialization settings."""

    vocabulary_size: int
    maximum_context_length: int
    model_dimension: int
    number_of_layers: int
    key_dimension: int
    value_dimension: int
    feed_forward_dimension: int
    layer_norm_epsilon: float = 1e-5
    attention_bias: bool = True
    feed_forward_bias: bool = True
    vocabulary_bias: bool = True
    dtype: np.dtype | type[np.floating] | str = np.float64
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "vocabulary_size",
            "maximum_context_length",
            "model_dimension",
            "number_of_layers",
            "key_dimension",
            "value_dimension",
            "feed_forward_dimension",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "layer_norm_epsilon",
            _positive_finite_float(
                self.layer_norm_epsilon,
                "layer_norm_epsilon",
            ),
        )
        for name in (
            "attention_bias",
            "feed_forward_bias",
            "vocabulary_bias",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        object.__setattr__(self, "dtype", validate_float_dtype(self.dtype))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "vocabulary_size": self.vocabulary_size,
            "maximum_context_length": self.maximum_context_length,
            "model_dimension": self.model_dimension,
            "number_of_layers": self.number_of_layers,
            "key_dimension": self.key_dimension,
            "value_dimension": self.value_dimension,
            "feed_forward_dimension": self.feed_forward_dimension,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "attention_bias": self.attention_bias,
            "feed_forward_bias": self.feed_forward_bias,
            "vocabulary_bias": self.vocabulary_bias,
            "dtype": self.dtype.name,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TransformerConfig:
        """Construct a configuration from an exact serialized mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("Transformer configuration must be a mapping.")
        expected_keys = {
            "vocabulary_size",
            "maximum_context_length",
            "model_dimension",
            "number_of_layers",
            "key_dimension",
            "value_dimension",
            "feed_forward_dimension",
            "layer_norm_epsilon",
            "attention_bias",
            "feed_forward_bias",
            "vocabulary_bias",
            "dtype",
            "seed",
        }
        actual_keys = set(values)
        if actual_keys != expected_keys:
            missing = sorted(str(key) for key in expected_keys - actual_keys)
            unexpected = sorted(str(key) for key in actual_keys - expected_keys)
            raise ValueError(
                "Transformer configuration keys do not match; "
                f"missing={missing}, unexpected={unexpected}."
            )
        return cls(**dict(values))


@dataclass(frozen=True)
class _TransformerCache:
    token_shape: tuple[int, int]
    hidden_shape: tuple[int, int, int]
    logits_shape: tuple[int, int, int]


class TransformerLanguageModel(Module):
    """One decoder-only language model assembled from validated manual layers.

    The architecture uses learned token and position embeddings, a positive
    stack of independent single-head pre-norm decoder blocks, a final
    LayerNorm, and an untied vocabulary projection. Token IDs are discrete, so
    ``backward`` accumulates parameter gradients and returns ``None``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if not isinstance(config, TransformerConfig):
            raise TypeError("config must be a TransformerConfig.")
        self.config = config
        self.dtype = config.dtype

        child_seeds = np.random.SeedSequence(config.seed).spawn(
            config.number_of_layers + 3
        )

        def child_seed(index: int) -> int:
            return int(child_seeds[index].generate_state(1, dtype=np.uint64)[0])

        self.token_embedding = Embedding(
            config.vocabulary_size,
            config.model_dimension,
            seed=child_seed(0),
            dtype=self.dtype,
        )
        self.position_embedding = Embedding(
            config.maximum_context_length,
            config.model_dimension,
            seed=child_seed(1),
            dtype=self.dtype,
        )
        blocks = tuple(
            PreNormDecoderBlock(
                model_dim=config.model_dimension,
                key_dim=config.key_dimension,
                value_dim=config.value_dimension,
                ff_hidden_dim=config.feed_forward_dimension,
                attention_bias=config.attention_bias,
                feed_forward_bias=config.feed_forward_bias,
                attention_output_projection=True,
                layer_norm_epsilon=config.layer_norm_epsilon,
                activation="gelu",
                seed=child_seed(layer_index + 2),
                dtype=self.dtype,
            )
            for layer_index in range(config.number_of_layers)
        )
        self.decoder_blocks = Sequential(blocks)
        self.final_layer_norm = LayerNorm(
            config.model_dimension,
            epsilon=config.layer_norm_epsilon,
            dtype=self.dtype,
        )
        self.language_model_head = Linear(
            config.model_dimension,
            config.vocabulary_size,
            bias=config.vocabulary_bias,
            seed=child_seed(config.number_of_layers + 2),
            dtype=self.dtype,
        )

        self.register_module("token_embedding", self.token_embedding)
        self.register_module("position_embedding", self.position_embedding)
        self.register_module("decoder_blocks", self.decoder_blocks)
        self.register_module("final_layer_norm", self.final_layer_norm)
        self.register_module("language_model_head", self.language_model_head)

    @property
    def parameter_count(self) -> int:
        """Return the total number of trainable scalar parameters."""
        return sum(parameter.size for parameter in self.parameters())

    def _validate_token_ids(self, token_ids: ArrayLike) -> NDArray[np.integer]:
        values = np.asarray(token_ids)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(
                f"token_ids must have an integer dtype, got {values.dtype}."
            )
        if values.ndim != 2:
            raise ValueError(
                "token_ids must have exactly two dimensions (B, T), "
                f"got shape {values.shape}."
            )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(
                "token_ids batch and sequence dimensions must be positive."
            )
        if values.shape[1] > self.config.maximum_context_length:
            raise ValueError(
                f"token_ids sequence length {values.shape[1]} exceeds maximum "
                f"context length {self.config.maximum_context_length}."
            )
        if np.any(values < 0) or np.any(values >= self.config.vocabulary_size):
            minimum = int(np.min(values))
            maximum = int(np.max(values))
            raise ValueError(
                f"token_ids must lie in [0, {self.config.vocabulary_size}), "
                f"received range [{minimum}, {maximum}]."
            )
        return values

    def forward(self, token_ids: ArrayLike) -> FloatArray:
        """Return unnormalized vocabulary logits with shape ``(B, T, V)``."""
        values = self._validate_token_ids(token_ids)
        if self.training and self.has_pending_cache():
            raise RuntimeError(
                "TransformerLanguageModel.forward cannot run twice in training "
                "mode before backward consumes the first cache."
            )

        batch_size, sequence_length = values.shape
        try:
            token_embeddings = self.token_embedding.forward(values)
            positions = np.broadcast_to(
                np.arange(sequence_length, dtype=np.int64),
                values.shape,
            )
            position_embeddings = self.position_embedding.forward(positions)
            hidden = residual_add(
                token_embeddings,
                position_embeddings,
                name="embedding",
            )
            hidden = self.decoder_blocks.forward(hidden)
            hidden = self.final_layer_norm.forward(hidden)
            logits = self.language_model_head.forward(hidden)
            expected_logits_shape = (
                batch_size,
                sequence_length,
                self.config.vocabulary_size,
            )
            if logits.shape != expected_logits_shape:
                raise RuntimeError(
                    f"Language-model logits must have shape "
                    f"{expected_logits_shape}, got {logits.shape}."
                )
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError(
                    "Transformer language model produced non-finite logits."
                )
            self._store_forward_cache(
                _TransformerCache(
                    token_shape=(batch_size, sequence_length),
                    hidden_shape=(
                        batch_size,
                        sequence_length,
                        self.config.model_dimension,
                    ),
                    logits_shape=expected_logits_shape,
                )
            )
        except Exception:
            self.clear_cache()
            raise
        return logits

    def backward(self, grad_logits: ArrayLike) -> None:
        """Accumulate all parameter gradients from an explicit logit gradient."""
        cache = self._require_forward_cache()
        gradient = self._validate_float_array(
            grad_logits,
            "grad_logits",
            dtype=self.dtype,
            minimum_dimensions=3,
        )
        if gradient.shape != cache.logits_shape:
            raise ValueError(
                f"grad_logits shape must be {cache.logits_shape}, got {gradient.shape}."
            )

        grad_hidden = self.language_model_head.backward(gradient)
        grad_hidden = self.final_layer_norm.backward(grad_hidden)
        grad_hidden = self.decoder_blocks.backward(grad_hidden)
        grad_token_embedding, grad_position_embedding = residual_add_backward(
            grad_hidden,
            expected_shape=cache.hidden_shape,
            dtype=self.dtype,
            name="embedding",
        )
        self.position_embedding.backward(grad_position_embedding)
        self.token_embedding.backward(grad_token_embedding)
        self._consume_forward_cache()
        return None

    def state_dict(self) -> dict[str, FloatArray]:
        """Return deterministic copies of all named parameter arrays."""
        return {
            name: np.array(parameter.data, copy=True)
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state: Mapping[str, ArrayLike]) -> None:
        """Load an exact complete named state without partial mutation."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping of parameter names to arrays.")
        if self.has_pending_cache():
            raise RuntimeError(
                "Cannot load transformer state while a forward cache is pending."
            )
        named_parameters = self.named_parameters()
        expected_keys = {name for name, _ in named_parameters}
        actual_keys = set(state)
        if actual_keys != expected_keys:
            missing = sorted(str(key) for key in expected_keys - actual_keys)
            unexpected = sorted(str(key) for key in actual_keys - expected_keys)
            raise ValueError(
                "State parameter keys do not match the model; "
                f"missing={missing}, unexpected={unexpected}."
            )

        validated: dict[str, NDArray] = {}
        for name, parameter in named_parameters:
            values = np.asarray(state[name])
            if values.shape != parameter.shape:
                raise ValueError(
                    f"State parameter {name!r} shape {values.shape} "
                    f"does not match {parameter.shape}."
                )
            if values.dtype != parameter.dtype:
                raise TypeError(
                    f"State parameter {name!r} dtype {values.dtype} "
                    f"does not match {parameter.dtype}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"State parameter {name!r} must contain only finite values."
                )
            validated[name] = np.array(values, copy=True)

        for name, parameter in named_parameters:
            parameter.load_data(validated[name])

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist model version, configuration, and complete named state."""
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Checkpoint path must end with '.npz'.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "model_version": _MODEL_VERSION,
            "model_type": type(self).__name__,
            "configuration": self.config.to_dict(),
        }
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))
        }
        for name, values in self.state_dict().items():
            arrays[f"parameter::{name}"] = values
        return atomic_savez(destination, arrays)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> TransformerLanguageModel:
        """Reconstruct a model and reject every incompatible checkpoint."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                if "metadata_json" not in checkpoint.files:
                    raise ValueError("Checkpoint is missing metadata_json.")
                try:
                    metadata = json.loads(str(checkpoint["metadata_json"]))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Checkpoint metadata is not valid JSON."
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError("Checkpoint metadata must be an object.")
                if metadata.get("checkpoint_version") != _CHECKPOINT_VERSION:
                    raise ValueError(
                        "Unsupported checkpoint version: "
                        f"{metadata.get('checkpoint_version')!r}."
                    )
                if metadata.get("model_version") != _MODEL_VERSION:
                    raise ValueError(
                        f"Checkpoint model version "
                        f"{metadata.get('model_version')!r} "
                        f"does not match {_MODEL_VERSION!r}."
                    )
                if metadata.get("model_type") != cls.__name__:
                    raise ValueError(
                        f"Checkpoint model type {metadata.get('model_type')!r} "
                        f"does not match {cls.__name__!r}."
                    )
                configuration = metadata.get("configuration")
                if not isinstance(configuration, dict):
                    raise ValueError("Checkpoint configuration must be an object.")
                state = {
                    key.removeprefix("parameter::"): np.array(
                        checkpoint[key],
                        copy=True,
                    )
                    for key in checkpoint.files
                    if key != "metadata_json" and key.startswith("parameter::")
                }
                unexpected_arrays = [
                    key
                    for key in checkpoint.files
                    if key != "metadata_json" and not key.startswith("parameter::")
                ]
                if unexpected_arrays:
                    raise ValueError(
                        "Checkpoint contains unexpected arrays: "
                        f"{sorted(unexpected_arrays)}."
                    )
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint does not exist: {source}") from None

        model = cls(TransformerConfig.from_dict(configuration))
        model.load_state_dict(state)
        return model
