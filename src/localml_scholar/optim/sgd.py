"""Stochastic gradient descent for explicit Parameters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from localml_scholar.nn.parameter import Parameter
from localml_scholar.optim.base import Optimizer


class SGD(Optimizer):
    """SGD with optional coupled L2 weight decay."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, learning_rate=learning_rate)
        if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
            raise TypeError("weight_decay must be a real number.")
        self.weight_decay = float(weight_decay)
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")

    def step(self) -> None:
        """Apply ``theta -= lr * (grad + weight_decay * theta)``."""
        self._validate_parameters_and_gradients()
        for parameter in self.parameters:
            update = parameter.grad
            if self.weight_decay:
                update = update + self.weight_decay * parameter.data
            parameter.data -= self.learning_rate * update

    def _hyperparameters(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
