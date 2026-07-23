"""Optimizers that update explicit NumPy parameter arrays."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SGD:
    """Stochastic gradient descent with optional L2 weight decay."""

    def __init__(
        self,
        parameters: Mapping[str, FloatArray],
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        if not parameters:
            raise ValueError("parameters must contain at least one named array.")
        self.parameters = dict(parameters)
        for name, parameter in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be non-empty strings.")
            if not isinstance(parameter, np.ndarray):
                raise TypeError(f"Parameter {name!r} must be a NumPy array.")
            if not np.issubdtype(parameter.dtype, np.floating):
                raise TypeError(f"Parameter {name!r} must have a floating-point dtype.")
            if parameter.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError(
                    f"Parameter {name!r} must use float32 or float64, "
                    f"got {parameter.dtype}."
                )
            if parameter.size == 0:
                raise ValueError(f"Parameter {name!r} must be non-empty.")

        self.learning_rate = self._validate_non_negative_finite(
            learning_rate, "learning_rate", strictly_positive=True
        )
        self.weight_decay = self._validate_non_negative_finite(
            weight_decay, "weight_decay", strictly_positive=False
        )

    @staticmethod
    def _validate_non_negative_finite(
        value: float,
        name: str,
        *,
        strictly_positive: bool,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        normalized = float(value)
        lower_bound_invalid = (
            normalized <= 0.0 if strictly_positive else normalized < 0.0
        )
        if not np.isfinite(normalized) or lower_bound_invalid:
            comparison = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must be finite and {comparison}.")
        return normalized

    def _validate_gradients(
        self, gradients: Mapping[str, FloatArray]
    ) -> dict[str, FloatArray]:
        if set(gradients) != set(self.parameters):
            missing = sorted(set(self.parameters) - set(gradients))
            unexpected = sorted(set(gradients) - set(self.parameters))
            raise ValueError(
                f"Gradient names do not match parameters; missing={missing}, "
                f"unexpected={unexpected}."
            )

        normalized: dict[str, FloatArray] = {}
        for name, parameter in self.parameters.items():
            gradient = np.asarray(gradients[name])
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"Gradient {name!r} has shape {gradient.shape}; expected "
                    f"{parameter.shape}."
                )
            if not np.issubdtype(gradient.dtype, np.floating):
                raise TypeError(f"Gradient {name!r} must have a floating-point dtype.")
            if gradient.dtype != parameter.dtype:
                raise TypeError(
                    f"Gradient {name!r} dtype {gradient.dtype} does not "
                    f"match parameter dtype {parameter.dtype}."
                )
            if not np.all(np.isfinite(gradient)):
                raise ValueError(f"Gradient {name!r} contains non-finite values.")
            normalized[name] = gradient
        return normalized

    def zero_grad(self, gradients: Mapping[str, FloatArray]) -> None:
        """Set every validated gradient array to zero in place."""
        validated = self._validate_gradients(gradients)
        for gradient in validated.values():
            gradient.fill(0.0)

    def step(self, gradients: Mapping[str, FloatArray]) -> None:
        """Apply one in-place SGD update to all parameters."""
        validated = self._validate_gradients(gradients)
        for name, parameter in self.parameters.items():
            update = validated[name]
            if self.weight_decay:
                update = update + self.weight_decay * parameter
            parameter -= self.learning_rate * update
