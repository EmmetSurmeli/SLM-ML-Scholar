"""A character bigram language model with an explicit manual backward pass."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from localml_scholar.losses import (
    softmax_cross_entropy_backward,
    softmax_cross_entropy_forward,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BigramLanguageModel:
    """Learn one row of next-token logits for every current token.

    ``weights[i, j]`` is the unnormalized score for token ``j`` following
    token ``i``. Repeated input IDs accumulate into the same gradient row.
    """

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        vocabulary_size: int,
        seed: int = 0,
        initialization_scale: float = 0.01,
    ) -> None:
        if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
            raise TypeError("vocabulary_size must be an integer.")
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if (
            isinstance(initialization_scale, bool)
            or not isinstance(initialization_scale, (int, float))
            or not np.isfinite(initialization_scale)
            or initialization_scale < 0.0
        ):
            raise ValueError("initialization_scale must be finite and non-negative.")

        rng = np.random.default_rng(seed)
        self.weights: FloatArray = rng.normal(
            loc=0.0,
            scale=float(initialization_scale),
            size=(vocabulary_size, vocabulary_size),
        )
        self.grad_weights: FloatArray = np.zeros_like(self.weights)
        self._training = True

    @property
    def vocabulary_size(self) -> int:
        """Return the number of input and output token classes."""
        return int(self.weights.shape[0])

    @property
    def training(self) -> bool:
        """Report whether the model is in training mode."""
        return self._training

    @property
    def parameter_count(self) -> int:
        """Return the total number of scalar trainable parameters."""
        return int(self.weights.size)

    @property
    def configuration(self) -> dict[str, Any]:
        """Return the model configuration needed for reconstruction."""
        return {
            "vocabulary_size": self.vocabulary_size,
            "dtype": self.weights.dtype.name,
        }

    def train(self) -> BigramLanguageModel:
        """Switch to training mode and return ``self``."""
        self._training = True
        return self

    def eval(self) -> BigramLanguageModel:
        """Switch to evaluation mode and return ``self``."""
        self._training = False
        return self

    def parameters(self) -> Mapping[str, FloatArray]:
        """Return named trainable arrays for an optimizer."""
        return {"weights": self.weights}

    def gradients(self) -> Mapping[str, FloatArray]:
        """Return named gradient arrays corresponding to ``parameters``."""
        return {"weights": self.grad_weights}

    def _validate_token_ids(
        self, token_ids: NDArray[np.integer], name: str
    ) -> IntArray:
        array = np.asarray(token_ids)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
        if array.size == 0:
            raise ValueError(f"{name} must be non-empty.")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must contain integer token IDs.")
        normalized = array.astype(np.int64, copy=False)
        if np.any(normalized < 0) or np.any(normalized >= self.vocabulary_size):
            raise ValueError(
                f"{name} token IDs must lie in [0, {self.vocabulary_size})."
            )
        return normalized

    def forward(self, input_ids: NDArray[np.integer]) -> FloatArray:
        """Select the next-token logit row for each input token ID."""
        inputs = self._validate_token_ids(input_ids, "input_ids")
        return self.weights[inputs]

    def loss(
        self,
        input_ids: NDArray[np.integer],
        target_ids: NDArray[np.integer],
    ) -> float:
        """Compute mean softmax cross-entropy without changing gradients."""
        inputs = self._validate_token_ids(input_ids, "input_ids")
        targets = self._validate_token_ids(target_ids, "target_ids")
        if inputs.shape != targets.shape:
            raise ValueError(
                f"input_ids and target_ids must have equal shapes, got "
                f"{inputs.shape} and {targets.shape}."
            )
        loss, _ = softmax_cross_entropy_forward(self.weights[inputs], targets)
        return loss

    def backward(
        self,
        input_ids: NDArray[np.integer],
        grad_logits: NDArray[np.floating],
    ) -> None:
        """Accumulate logit gradients into the selected weight rows."""
        inputs = self._validate_token_ids(input_ids, "input_ids")
        gradient = np.asarray(grad_logits)
        expected_shape = (inputs.size, self.vocabulary_size)
        if gradient.shape != expected_shape:
            raise ValueError(
                f"grad_logits must have shape {expected_shape}, got {gradient.shape}."
            )
        if not np.issubdtype(gradient.dtype, np.floating):
            raise TypeError("grad_logits must have a floating-point dtype.")
        if gradient.dtype != self.weights.dtype:
            raise TypeError(
                f"grad_logits dtype {gradient.dtype} does not match model "
                f"dtype {self.weights.dtype}."
            )
        if not np.all(np.isfinite(gradient)):
            raise ValueError("grad_logits must contain only finite values.")

        # np.add.at is required because advanced-index `+=` does not correctly
        # accumulate when an input token appears more than once in the batch.
        np.add.at(self.grad_weights, inputs, gradient)

    def loss_and_backward(
        self,
        input_ids: NDArray[np.integer],
        target_ids: NDArray[np.integer],
    ) -> float:
        """Compute the mean loss and accumulate its analytical weight gradient."""
        inputs = self._validate_token_ids(input_ids, "input_ids")
        targets = self._validate_token_ids(target_ids, "target_ids")
        if inputs.shape != targets.shape:
            raise ValueError(
                f"input_ids and target_ids must have equal shapes, got "
                f"{inputs.shape} and {targets.shape}."
            )
        loss, probabilities = softmax_cross_entropy_forward(
            self.weights[inputs], targets
        )
        grad_logits = softmax_cross_entropy_backward(probabilities, targets)
        self.backward(inputs, grad_logits)
        return loss

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save versioned model configuration and weights in NPZ."""
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Checkpoint path must end with '.npz'.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            checkpoint_version=np.array(self.CHECKPOINT_VERSION, dtype=np.int64),
            model_config_json=np.asarray(
                json.dumps(self.configuration, sort_keys=True)
            ),
            weights=self.weights,
        )
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> BigramLanguageModel:
        """Load a validated bigram checkpoint."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                if "checkpoint_version" not in checkpoint.files:
                    raise ValueError("Checkpoint is missing checkpoint_version.")
                version = int(checkpoint["checkpoint_version"])
                if version == 1:
                    expected_files = {"checkpoint_version", "weights"}
                    configuration = None
                elif version == cls.CHECKPOINT_VERSION:
                    expected_files = {
                        "checkpoint_version",
                        "model_config_json",
                        "weights",
                    }
                    try:
                        configuration = json.loads(str(checkpoint["model_config_json"]))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            "Checkpoint model configuration is not valid JSON."
                        ) from error
                else:
                    raise ValueError(f"Unsupported checkpoint version: {version}.")
                if set(checkpoint.files) != expected_files:
                    raise ValueError(
                        f"Checkpoint version {version} has unexpected fields."
                    )
                weights = np.asarray(checkpoint["weights"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint does not exist: {source}") from None

        if (
            weights.ndim != 2
            or weights.shape[0] == 0
            or weights.shape[0] != weights.shape[1]
        ):
            raise ValueError(
                f"Checkpoint weights must be a non-empty square matrix, got "
                f"{weights.shape}."
            )
        if weights.dtype != np.float64:
            raise TypeError(
                f"Checkpoint weights must use float64, got {weights.dtype}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Checkpoint weights contain non-finite values.")
        expected_configuration = {
            "vocabulary_size": int(weights.shape[0]),
            "dtype": weights.dtype.name,
        }
        if configuration is not None and configuration != expected_configuration:
            raise ValueError(
                "Checkpoint model configuration does not match its weights."
            )

        model = cls(weights.shape[0], initialization_scale=0.0)
        model.weights[...] = weights
        return model
