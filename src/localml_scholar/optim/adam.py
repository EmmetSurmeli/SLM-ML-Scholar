"""Adam with bias correction and identity-keyed state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from localml_scholar.nn.parameter import Parameter
from localml_scholar.optim.base import Optimizer


class Adam(Optimizer):
    """Apply the original Adam moment and bias-correction convention."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate=learning_rate)
        self.beta1 = self._validate_probability(beta1, "beta1")
        self.beta2 = self._validate_probability(beta2, "beta2")
        self.epsilon = self._validate_positive_finite(epsilon, "epsilon")
        self._first_moment = {
            parameter: np.zeros_like(parameter.data) for parameter in self.parameters
        }
        self._second_moment = {
            parameter: np.zeros_like(parameter.data) for parameter in self.parameters
        }
        self.step_count = 0

    def step(self) -> None:
        """Apply one bias-corrected Adam update."""
        self._validate_parameters_and_gradients()
        self.step_count += 1
        first_correction = 1.0 - self.beta1**self.step_count
        second_correction = 1.0 - self.beta2**self.step_count

        for parameter in self.parameters:
            first = self._first_moment[parameter]
            second = self._second_moment[parameter]
            first *= self.beta1
            first += (1.0 - self.beta1) * parameter.grad
            second *= self.beta2
            second += (1.0 - self.beta2) * parameter.grad * parameter.grad
            corrected_first = first / first_correction
            corrected_second = second / second_correction
            parameter.data -= (
                self.learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + self.epsilon)
            )

    def _hyperparameters(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }

    def _state_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "step_count": np.asarray(self.step_count, dtype=np.int64)
        }
        for index, parameter in enumerate(self.parameters):
            arrays[f"first_moment::{index}"] = self._first_moment[parameter]
            arrays[f"second_moment::{index}"] = self._second_moment[parameter]
        return arrays

    def _load_state_arrays(
        self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
    ) -> None:
        expected = {"step_count"}
        for index in range(len(self.parameters)):
            expected.add(f"first_moment::{index}")
            expected.add(f"second_moment::{index}")
        if set(arrays) != expected:
            raise ValueError("Adam checkpoint state keys do not match parameters.")

        step_array = arrays["step_count"]
        if step_array.shape != () or not np.issubdtype(step_array.dtype, np.integer):
            raise ValueError("Adam checkpoint step_count must be an integer scalar.")
        step_count = int(step_array)
        if step_count < 0:
            raise ValueError("Adam checkpoint step_count cannot be negative.")

        validated: list[tuple[np.ndarray, np.ndarray]] = []
        for index, parameter in enumerate(self.parameters):
            first = arrays[f"first_moment::{index}"]
            second = arrays[f"second_moment::{index}"]
            for state_name, values in (
                ("first moment", first),
                ("second moment", second),
            ):
                if (
                    values.shape != parameter.shape
                    or values.dtype != parameter.dtype
                    or not np.all(np.isfinite(values))
                ):
                    raise ValueError(
                        f"Adam {state_name} {index} has invalid shape, "
                        "dtype, or values."
                    )
            validated.append((first, second))

        for parameter, (first, second) in zip(
            self.parameters,
            validated,
            strict=True,
        ):
            self._first_moment[parameter][...] = first
            self._second_moment[parameter][...] = second
        self.step_count = step_count
