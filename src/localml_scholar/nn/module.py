"""Minimal explicit module registration and cache-lifecycle support."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from localml_scholar.nn.parameter import Parameter

FloatArray = NDArray[np.floating]


class Module:
    """Base class for manually differentiated components.

    Subclasses explicitly register parameters and child modules. In training
    mode, a layer may hold one forward cache until its matching backward call.
    A second cached forward is rejected rather than silently overwriting data.
    Evaluation forwards do not cache and therefore cannot be followed by
    backward.
    """

    def __init__(self) -> None:
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self._training = True
        self._forward_cache: Any | None = None

    @property
    def training(self) -> bool:
        """Return whether this module is in training mode."""
        return self._training

    def register_parameter(self, name: str, parameter: Parameter) -> None:
        """Register one direct parameter in deterministic insertion order."""
        self._validate_registration_name(name)
        if name in self._parameters or name in self._modules:
            raise ValueError(f"Module already contains a registration named {name!r}.")
        if not isinstance(parameter, Parameter):
            raise TypeError("parameter must be a Parameter instance.")
        if any(existing is parameter for existing in self._parameters.values()):
            raise ValueError("The same Parameter cannot be registered twice.")
        self._parameters[name] = parameter

    def register_module(self, name: str, module: Module) -> None:
        """Register one direct child module in deterministic insertion order."""
        self._validate_registration_name(name)
        if name in self._parameters or name in self._modules:
            raise ValueError(f"Module already contains a registration named {name!r}.")
        if not isinstance(module, Module):
            raise TypeError("module must be a Module instance.")
        if module is self:
            raise ValueError("A module cannot register itself as a child.")
        if any(existing is module for existing in self._modules.values()):
            raise ValueError("The same child Module cannot be registered twice.")
        self._modules[name] = module

    @staticmethod
    def _validate_registration_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Registration names must be non-empty strings.")
        if "." in name:
            raise ValueError("Registration names cannot contain '.'.")

    def named_parameters(
        self,
        prefix: str = "",
        *,
        include_non_trainable: bool = False,
    ) -> tuple[tuple[str, Parameter], ...]:
        """Return recursively named parameters in registration order."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string.")

        results: list[tuple[str, Parameter]] = []
        seen: set[int] = set()

        def visit(module: Module, current_prefix: str) -> None:
            for name, parameter in module._parameters.items():
                if not include_non_trainable and not parameter.trainable:
                    continue
                identity = id(parameter)
                if identity in seen:
                    raise RuntimeError(
                        "A Parameter is shared across the module tree; shared "
                        "parameters are not supported by this manual graph."
                    )
                seen.add(identity)
                full_name = f"{current_prefix}.{name}" if current_prefix else name
                results.append((full_name, parameter))
            for name, child in module._modules.items():
                child_prefix = f"{current_prefix}.{name}" if current_prefix else name
                visit(child, child_prefix)

        visit(self, prefix)
        return tuple(results)

    def parameters(
        self, *, include_non_trainable: bool = False
    ) -> tuple[Parameter, ...]:
        """Return recursively registered parameters in deterministic order."""
        return tuple(
            parameter
            for _, parameter in self.named_parameters(
                include_non_trainable=include_non_trainable
            )
        )

    def modules(self) -> tuple[Module, ...]:
        """Return direct child modules in registration order."""
        return tuple(self._modules.values())

    def zero_grad(self) -> None:
        """Zero every trainable parameter gradient recursively."""
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self) -> Module:
        """Enable training mode recursively."""
        if not self.training and self.has_pending_cache():
            raise RuntimeError(
                "Cannot change module mode while a forward cache is pending."
            )
        self._set_training_mode(True)
        return self

    def eval(self) -> Module:
        """Enable evaluation mode recursively."""
        if self.training and self.has_pending_cache():
            raise RuntimeError(
                "Cannot change module mode while a forward cache is pending."
            )
        self._set_training_mode(False)
        return self

    def _set_training_mode(self, training: bool) -> None:
        self._training = training
        for child in self._modules.values():
            child._set_training_mode(training)

    def has_pending_cache(self) -> bool:
        """Return whether this module tree contains an unmatched forward cache."""
        return self._forward_cache is not None or any(
            child.has_pending_cache() for child in self._modules.values()
        )

    def clear_cache(self) -> None:
        """Discard cached forward data recursively.

        This is intended for explicit recovery after an abandoned computation,
        not as a substitute for a matching backward call.
        """
        self._forward_cache = None
        for child in self._modules.values():
            child.clear_cache()

    def _store_forward_cache(self, cache: Any) -> None:
        if not self.training:
            return
        if self._forward_cache is not None:
            raise RuntimeError(
                f"{type(self).__name__}.forward cannot run twice in training "
                "mode before backward consumes the first cache. Reusing one "
                "module instance multiple times in a graph is unsupported."
            )
        self._forward_cache = cache

    def _require_forward_cache(self) -> Any:
        if self._forward_cache is None:
            raise RuntimeError(
                f"{type(self).__name__}.backward requires one unmatched "
                "training-mode forward call."
            )
        return self._forward_cache

    def _consume_forward_cache(self) -> None:
        self._forward_cache = None

    @staticmethod
    def _validate_float_array(
        values: ArrayLike,
        name: str,
        *,
        dtype: np.dtype[np.floating] | None = None,
        minimum_dimensions: int = 1,
    ) -> FloatArray:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                f"{name} must have a floating-point dtype, got {array.dtype}."
            )
        if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"{name} must use float32 or float64, got {array.dtype}.")
        if dtype is not None and array.dtype != dtype:
            raise TypeError(
                f"{name} dtype {array.dtype} does not match expected {dtype}."
            )
        if array.ndim < minimum_dimensions:
            raise ValueError(
                f"{name} must have at least {minimum_dimensions} dimensions, "
                f"got shape {array.shape}."
            )
        if array.size == 0:
            raise ValueError(f"{name} must be non-empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Compute module outputs. Subclasses must implement this."""
        raise NotImplementedError

    def backward(self, grad_output: ArrayLike) -> FloatArray | None:
        """Backpropagate one explicit output gradient."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
