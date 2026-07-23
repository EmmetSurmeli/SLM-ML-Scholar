"""Small containers for composing manually differentiated modules."""

from __future__ import annotations

from collections.abc import Sequence

from numpy.typing import ArrayLike

from localml_scholar.nn.module import FloatArray, Module


class Sequential(Module):
    """Apply child modules in order and backpropagate in reverse order."""

    def __init__(self, modules: Sequence[Module]) -> None:
        super().__init__()
        normalized = tuple(modules)
        if not normalized:
            raise ValueError("Sequential requires at least one module.")

        seen: set[int] = set()
        for index, module in enumerate(normalized):
            if not isinstance(module, Module):
                raise TypeError(
                    f"modules[{index}] must be Module, got {type(module).__name__}."
                )
            identity = id(module)
            if identity in seen:
                raise ValueError(
                    "Sequential cannot reuse one Module instance multiple times; "
                    "each occurrence needs an independent forward cache."
                )
            seen.add(identity)
            self.register_module(str(index), module)

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Run every child module in registration order."""
        output = inputs
        for module in self:
            output = module.forward(output)
        return output

    def backward(self, grad_output: ArrayLike) -> FloatArray:
        """Run child backward passes in reverse registration order."""
        gradient = grad_output
        for module in reversed(self.modules()):
            result = module.backward(gradient)
            if result is None:
                raise RuntimeError(
                    f"{type(module).__name__}.backward returned no input "
                    "gradient inside Sequential."
                )
            gradient = result
        return gradient

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Sequential index must be an integer.")
        return self.modules()[index]
