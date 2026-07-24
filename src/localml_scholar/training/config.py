"""Validated configuration for transformer training infrastructure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1).")
    return normalized


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Immutable training-only settings for a transformer language model."""

    batch_size: int = 4
    sequence_length: int = 16
    maximum_steps: int = 100
    evaluation_interval: int = 20
    evaluation_batches: int = 4
    checkpoint_interval: int = 50
    logging_interval: int = 10
    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    momentum_beta: float = 0.9
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 0.0
    maximum_gradient_norm: float | None = 1.0
    seed: int = 0
    output_directory: str = "outputs/transformer"

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "sequence_length",
            "maximum_steps",
            "evaluation_interval",
            "evaluation_batches",
            "checkpoint_interval",
            "logging_interval",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name),
            )
        if not isinstance(self.optimizer_name, str):
            raise TypeError("optimizer_name must be a string.")
        normalized_optimizer = self.optimizer_name.lower()
        if normalized_optimizer not in {"sgd", "momentum", "adam"}:
            raise ValueError("optimizer_name must be 'sgd', 'momentum', or 'adam'.")
        object.__setattr__(self, "optimizer_name", normalized_optimizer)
        object.__setattr__(
            self,
            "learning_rate",
            _positive_finite(self.learning_rate, "learning_rate"),
        )
        object.__setattr__(
            self,
            "adam_beta1",
            _probability(self.adam_beta1, "adam_beta1"),
        )
        object.__setattr__(
            self,
            "adam_beta2",
            _probability(self.adam_beta2, "adam_beta2"),
        )
        object.__setattr__(
            self,
            "momentum_beta",
            _probability(self.momentum_beta, "momentum_beta"),
        )
        object.__setattr__(
            self,
            "optimizer_epsilon",
            _positive_finite(self.optimizer_epsilon, "optimizer_epsilon"),
        )
        if isinstance(self.weight_decay, bool) or not isinstance(
            self.weight_decay,
            (int, float),
        ):
            raise TypeError("weight_decay must be a real number.")
        normalized_decay = float(self.weight_decay)
        if not np.isfinite(normalized_decay) or normalized_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")
        object.__setattr__(self, "weight_decay", normalized_decay)
        if self.maximum_gradient_norm is not None:
            object.__setattr__(
                self,
                "maximum_gradient_norm",
                _positive_finite(
                    self.maximum_gradient_norm,
                    "maximum_gradient_norm",
                ),
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not isinstance(self.output_directory, str) or not self.output_directory:
            raise ValueError("output_directory must be a non-empty string.")

    def validate_for_context(self, maximum_context_length: int) -> None:
        """Reject a training sequence longer than the model context."""
        context = _positive_integer(
            maximum_context_length,
            "maximum_context_length",
        )
        if self.sequence_length > context:
            raise ValueError(
                f"sequence_length {self.sequence_length} exceeds model context "
                f"length {context}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return an exact JSON-serializable representation."""
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "maximum_steps": self.maximum_steps,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_batches": self.evaluation_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "logging_interval": self.logging_interval,
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "momentum_beta": self.momentum_beta,
            "optimizer_epsilon": self.optimizer_epsilon,
            "weight_decay": self.weight_decay,
            "maximum_gradient_norm": self.maximum_gradient_norm,
            "seed": self.seed,
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TransformerTrainingConfig:
        """Construct from an exact serialized mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("Training configuration must be a mapping.")
        expected_keys = set(cls().to_dict())
        actual_keys = set(values)
        if actual_keys != expected_keys:
            missing = sorted(str(key) for key in expected_keys - actual_keys)
            unexpected = sorted(str(key) for key in actual_keys - expected_keys)
            raise ValueError(
                "Training configuration keys do not match; "
                f"missing={missing}, unexpected={unexpected}."
            )
        return cls(**dict(values))
