"""Classical momentum with zero-initialized velocity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from localml_scholar.nn.parameter import Parameter
from localml_scholar.optim.base import Optimizer


class Momentum(Optimizer):
    """Use ``v = beta * v + grad`` then ``theta -= lr * v``."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        learning_rate: float,
        beta: float = 0.9,
    ) -> None:
        super().__init__(parameters, learning_rate=learning_rate)
        self.beta = self._validate_probability(beta, "beta")
        self._velocity = {
            parameter: np.zeros_like(parameter.data) for parameter in self.parameters
        }

    def step(self) -> None:
        """Apply one classical-momentum update."""
        self._validate_parameters_and_gradients()
        for parameter in self.parameters:
            velocity = self._velocity[parameter]
            velocity *= self.beta
            velocity += parameter.grad
            parameter.data -= self.learning_rate * velocity

    def _hyperparameters(self) -> dict[str, float]:
        return {"learning_rate": self.learning_rate, "beta": self.beta}

    def _state_arrays(self) -> dict[str, np.ndarray]:
        return {
            f"velocity::{index}": velocity
            for index, parameter in enumerate(self.parameters)
            for velocity in (self._velocity[parameter],)
        }

    def _load_state_arrays(
        self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
    ) -> None:
        expected = {f"velocity::{index}" for index in range(len(self.parameters))}
        if set(arrays) != expected:
            raise ValueError("Momentum checkpoint state keys do not match parameters.")
        validated: list[np.ndarray] = []
        for index, parameter in enumerate(self.parameters):
            values = arrays[f"velocity::{index}"]
            if (
                values.shape != parameter.shape
                or values.dtype != parameter.dtype
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"Momentum velocity {index} has invalid shape, dtype, or values."
                )
            validated.append(values)
        for parameter, values in zip(self.parameters, validated, strict=True):
            self._velocity[parameter][...] = values
