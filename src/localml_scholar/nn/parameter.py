"""Explicit trainable parameters without an automatic-differentiation graph."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


class Parameter:
    """Own a floating-point value and a same-shaped gradient buffer.

    Parameters are identity objects. Optimizers therefore key state to the
    ``Parameter`` instance rather than to a potentially ambiguous string name.
    """

    def __init__(
        self,
        data: ArrayLike,
        *,
        name: str | None = None,
        trainable: bool = True,
    ) -> None:
        array = np.asarray(data)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                f"Parameter data must have a floating-point dtype, got {array.dtype}."
            )
        if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(
                f"Parameter data must use float32 or float64, got {array.dtype}."
            )
        if array.size == 0:
            raise ValueError("Parameter data must be non-empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Parameter data must contain only finite values.")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("Parameter name must be None or a non-empty string.")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")

        self.data: FloatArray = np.array(array, copy=True)
        self.grad: FloatArray = np.zeros_like(self.data)
        self.name = name
        self.trainable = trainable

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the parameter shape."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype[np.floating]:
        """Return the parameter dtype."""
        return self.data.dtype

    @property
    def size(self) -> int:
        """Return the number of scalar values."""
        return int(self.data.size)

    def zero_grad(self) -> None:
        """Set the persistent gradient buffer to zero in place."""
        self._validate_gradient_buffer()
        self.grad.fill(0.0)

    def _validate_gradient_buffer(self) -> None:
        if not isinstance(self.grad, np.ndarray):
            raise TypeError("Parameter gradient buffer must be a NumPy array.")
        if self.grad.shape != self.data.shape:
            raise ValueError(
                f"Parameter gradient shape {self.grad.shape} does not match "
                f"data shape {self.data.shape}."
            )
        if self.grad.dtype != self.data.dtype:
            raise TypeError(
                f"Parameter gradient dtype {self.grad.dtype} does not match "
                f"data dtype {self.data.dtype}."
            )

    def load_data(self, values: ArrayLike) -> None:
        """Copy validated values into the existing parameter storage."""
        array = np.asarray(values)
        if array.shape != self.data.shape:
            raise ValueError(
                f"Loaded parameter shape {array.shape} does not match "
                f"{self.data.shape}."
            )
        if array.dtype != self.data.dtype:
            raise TypeError(
                f"Loaded parameter dtype {array.dtype} does not match "
                f"{self.data.dtype}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Loaded parameter values must be finite.")
        self.data[...] = array


def validate_parameter_sequence(
    parameters: Sequence[Parameter],
    *,
    require_trainable: bool = True,
) -> tuple[Parameter, ...]:
    """Validate parameter types and reject duplicate identity references."""
    normalized = tuple(parameters)
    if not normalized:
        raise ValueError("At least one parameter is required.")

    seen: set[int] = set()
    for index, parameter in enumerate(normalized):
        if not isinstance(parameter, Parameter):
            raise TypeError(
                f"parameters[{index}] must be Parameter, "
                f"got {type(parameter).__name__}."
            )
        identity = id(parameter)
        if identity in seen:
            raise ValueError(
                f"Parameter at index {index} is registered more than once."
            )
        seen.add(identity)
        if require_trainable and not parameter.trainable:
            raise ValueError(f"Parameter at index {index} is not marked as trainable.")
    return normalized
