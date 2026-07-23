"""Shared validation and serialization for explicit optimizers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from localml_scholar.nn.parameter import Parameter, validate_parameter_sequence


class Optimizer(ABC):
    """Base class for optimizers over identity-keyed ``Parameter`` objects."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        learning_rate: float,
    ) -> None:
        self.parameters = validate_parameter_sequence(tuple(parameters))
        self.learning_rate = self._validate_positive_finite(
            learning_rate, "learning_rate"
        )

    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        normalized = float(value)
        if not np.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return normalized

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        normalized = float(value)
        if not np.isfinite(normalized) or not 0.0 <= normalized < 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1).")
        return normalized

    def _validate_parameters_and_gradients(self) -> None:
        for index, parameter in enumerate(self.parameters):
            parameter._validate_gradient_buffer()
            if not np.all(np.isfinite(parameter.data)):
                raise ValueError(f"Parameter {index} data contains non-finite values.")
            if not np.all(np.isfinite(parameter.grad)):
                raise ValueError(
                    f"Parameter {index} gradient contains non-finite values."
                )

    def zero_grad(self) -> None:
        """Zero every managed parameter gradient in place."""
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """Apply one optimizer update."""

    @abstractmethod
    def _hyperparameters(self) -> dict[str, float]:
        """Return JSON-serializable optimizer hyperparameters."""

    def _state_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def _load_state_arrays(
        self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
    ) -> None:
        if arrays:
            raise ValueError(
                f"{type(self).__name__} checkpoint has unexpected state arrays: "
                f"{sorted(arrays)}."
            )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save optimizer configuration and state arrays to NPZ."""
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Optimizer checkpoint path must end with '.npz'.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "optimizer_type": type(self).__name__,
            "hyperparameters": self._hyperparameters(),
            "parameter_shapes": [
                list(parameter.shape) for parameter in self.parameters
            ],
            "parameter_dtypes": [parameter.dtype.name for parameter in self.parameters],
        }
        arrays = {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            **self._state_arrays(),
        }
        np.savez(destination, **arrays)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore state into an optimizer with matching parameters and config."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                if "metadata_json" not in checkpoint.files:
                    raise ValueError("Optimizer checkpoint is missing metadata_json.")
                try:
                    metadata = json.loads(str(checkpoint["metadata_json"]))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Optimizer checkpoint metadata is not valid JSON."
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError("Optimizer checkpoint metadata must be an object.")
                arrays = {
                    key: np.array(checkpoint[key], copy=True)
                    for key in checkpoint.files
                    if key != "metadata_json"
                }
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Optimizer checkpoint does not exist: {source}"
            ) from None

        if metadata.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported optimizer checkpoint version: "
                f"{metadata.get('checkpoint_version')!r}."
            )
        if metadata.get("optimizer_type") != type(self).__name__:
            raise ValueError(
                f"Checkpoint optimizer type {metadata.get('optimizer_type')!r} "
                f"does not match {type(self).__name__!r}."
            )
        if metadata.get("hyperparameters") != self._hyperparameters():
            raise ValueError(
                "Optimizer checkpoint hyperparameters do not match this optimizer."
            )
        expected_shapes = [list(parameter.shape) for parameter in self.parameters]
        expected_dtypes = [parameter.dtype.name for parameter in self.parameters]
        if metadata.get("parameter_shapes") != expected_shapes:
            raise ValueError("Optimizer checkpoint parameter shapes do not match.")
        if metadata.get("parameter_dtypes") != expected_dtypes:
            raise ValueError("Optimizer checkpoint parameter dtypes do not match.")
        self._load_state_arrays(arrays, metadata)
