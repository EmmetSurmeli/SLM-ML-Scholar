"""A small manually differentiated multilayer perceptron."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from localml_scholar.nn.activations import GELU, ReLU
from localml_scholar.nn.containers import Sequential
from localml_scholar.nn.initialization import validate_float_dtype
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.module import FloatArray, Module


class MLP(Module):
    """Two affine layers separated by GELU or ReLU."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        activation: str = "gelu",
        seed: int = 0,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> None:
        super().__init__()
        self.input_dim = Linear._validate_dimension(input_dim, "input_dim")
        self.hidden_dim = Linear._validate_dimension(hidden_dim, "hidden_dim")
        self.output_dim = Linear._validate_dimension(output_dim, "output_dim")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        self.activation_name = activation
        self.seed = seed
        self.dtype = validate_float_dtype(dtype)

        rng = np.random.default_rng(seed)
        activation_module: Module = GELU() if activation == "gelu" else ReLU()
        self.network = Sequential(
            (
                Linear(
                    self.input_dim,
                    self.hidden_dim,
                    initialization="xavier_uniform",
                    rng=rng,
                    dtype=self.dtype,
                ),
                activation_module,
                Linear(
                    self.hidden_dim,
                    self.output_dim,
                    initialization="xavier_uniform",
                    rng=rng,
                    dtype=self.dtype,
                ),
            )
        )
        self.register_module("network", self.network)

    @property
    def configuration(self) -> dict[str, Any]:
        """Return the architecture configuration required for reconstruction."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "seed": self.seed,
            "dtype": self.dtype.name,
        }

    @property
    def parameter_count(self) -> int:
        """Return the total number of trainable scalar parameters."""
        return sum(parameter.size for parameter in self.parameters())

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Run the two-layer network."""
        return self.network.forward(inputs)

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        """Backpropagate through the two-layer network."""
        return self.network.backward(grad_output)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist versioned configuration and every named parameter."""
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
    def load_checkpoint(cls, path: str | Path) -> MLP:
        """Reconstruct an MLP and restore all parameters exactly."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                if "checkpoint_version" not in checkpoint.files:
                    raise ValueError("Checkpoint is missing checkpoint_version.")
                version = int(checkpoint["checkpoint_version"])
                if version != cls.CHECKPOINT_VERSION:
                    raise ValueError(f"Unsupported MLP checkpoint version: {version}.")
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
                        "Checkpoint parameter keys do not match the model; "
                        f"missing={missing}, unexpected={unexpected}."
                    )
                for name, parameter in model.named_parameters():
                    parameter.load_data(checkpoint[f"parameter::{name}"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint does not exist: {source}") from None
        return model
