"""Centered finite-difference checks for manually differentiated modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from localml_scholar.nn.module import Module

FloatArray = NDArray[np.floating]
Objective = Callable[[FloatArray], tuple[float, FloatArray]]


@dataclass(frozen=True)
class TensorGradientCheck:
    """Diagnostics for one checked input or parameter tensor."""

    name: str
    checked_coordinates: int
    maximum_absolute_error: float
    maximum_relative_error: float
    worst_index: tuple[int, ...]
    analytical_at_worst: float
    numerical_at_worst: float
    passed: bool


@dataclass(frozen=True)
class ModuleGradientCheck:
    """Aggregate result for an input and all selected parameters."""

    loss: float
    tensors: tuple[TensorGradientCheck, ...]
    passed: bool

    @property
    def checked_coordinates(self) -> int:
        """Return the total number of checked scalar coordinates."""
        return sum(tensor.checked_coordinates for tensor in self.tensors)


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _validate_objective_result(
    output: FloatArray,
    result: tuple[float, FloatArray],
    *,
    require_gradient: bool,
) -> tuple[float, FloatArray]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("objective must return (scalar_loss, grad_output).")
    loss, grad_output = result
    if isinstance(loss, bool) or not isinstance(loss, (int, float, np.floating)):
        raise TypeError("objective loss must be a real scalar.")
    normalized_loss = float(loss)
    if not np.isfinite(normalized_loss):
        raise ValueError("objective loss must be finite.")
    gradient = np.asarray(grad_output)
    if require_gradient:
        if not np.issubdtype(gradient.dtype, np.floating):
            raise TypeError("objective grad_output must be floating point.")
        if gradient.dtype != output.dtype:
            raise TypeError(
                f"objective grad_output dtype {gradient.dtype} does not "
                f"match module output dtype {output.dtype}."
            )
        if gradient.shape != output.shape:
            raise ValueError(
                f"objective grad_output shape {gradient.shape} does not "
                f"match module output shape {output.shape}."
            )
        if not np.all(np.isfinite(gradient)):
            raise ValueError("objective grad_output must contain finite values.")
    return normalized_loss, gradient


def _coordinate_indices(
    size: int,
    max_checks: int | None,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    check_count = size if max_checks is None else min(size, max_checks)
    if check_count == size:
        return np.arange(size, dtype=np.int64)
    return np.sort(rng.choice(size, size=check_count, replace=False).astype(np.int64))


def _check_tensor(
    *,
    name: str,
    values: NDArray,
    analytical: FloatArray,
    evaluate_loss: Callable[[], float],
    epsilon: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    denominator_delta: float,
    max_checks: int | None,
    rng: np.random.Generator,
) -> TensorGradientCheck:
    if values.shape != analytical.shape:
        raise ValueError(
            f"Analytical gradient for {name!r} has shape {analytical.shape}; "
            f"expected {values.shape}."
        )
    flat_indices = _coordinate_indices(values.size, max_checks, rng)
    maximum_absolute_error = -1.0
    maximum_relative_error = -1.0
    worst_index = tuple(0 for _ in values.shape)
    analytical_at_worst = 0.0
    numerical_at_worst = 0.0
    passed = True

    for flat_index in flat_indices:
        index = tuple(
            int(value) for value in np.unravel_index(int(flat_index), values.shape)
        )
        original = values[index].copy()
        try:
            values[index] = original + epsilon
            loss_plus = evaluate_loss()
            values[index] = original - epsilon
            loss_minus = evaluate_loss()
        finally:
            values[index] = original

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical_value = float(analytical[index])
        absolute_error = abs(analytical_value - numerical)
        relative_error = absolute_error / (
            abs(analytical_value) + abs(numerical) + denominator_delta
        )
        coordinate_passed = absolute_error <= (
            absolute_tolerance
            + relative_tolerance * max(abs(analytical_value), abs(numerical))
        )
        passed = passed and coordinate_passed
        if absolute_error > maximum_absolute_error:
            maximum_absolute_error = absolute_error
            worst_index = index
            analytical_at_worst = analytical_value
            numerical_at_worst = numerical
        maximum_relative_error = max(maximum_relative_error, relative_error)

    return TensorGradientCheck(
        name=name,
        checked_coordinates=int(flat_indices.size),
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        worst_index=worst_index,
        analytical_at_worst=analytical_at_worst,
        numerical_at_worst=numerical_at_worst,
        passed=passed,
    )


def check_module_gradients(
    module: Module,
    inputs: ArrayLike,
    objective: Objective,
    *,
    check_input: bool = True,
    check_parameters: bool = True,
    epsilon: float = 1e-6,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-5,
    denominator_delta: float = 1e-12,
    max_checks_per_tensor: int | None = None,
    seed: int = 0,
    require_float64: bool = True,
    raise_on_failure: bool = True,
) -> ModuleGradientCheck:
    """Compare a module's analytical gradients with centered differences.

    The objective supplies both a scalar function of the module output and its
    analytical output gradient. All perturbed inputs and parameters are
    restored from exact copies in a ``finally`` block.
    """
    if not isinstance(module, Module):
        raise TypeError("module must be a Module.")
    if not callable(objective):
        raise TypeError("objective must be callable.")
    epsilon = _positive_finite(epsilon, "epsilon")
    absolute_tolerance = _positive_finite(absolute_tolerance, "absolute_tolerance")
    relative_tolerance = _positive_finite(relative_tolerance, "relative_tolerance")
    denominator_delta = _positive_finite(denominator_delta, "denominator_delta")
    if max_checks_per_tensor is not None and (
        isinstance(max_checks_per_tensor, bool)
        or not isinstance(max_checks_per_tensor, int)
        or max_checks_per_tensor <= 0
    ):
        raise ValueError("max_checks_per_tensor must be None or a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not isinstance(check_input, bool) or not isinstance(check_parameters, bool):
        raise TypeError("check_input and check_parameters must be booleans.")
    if not check_input and not check_parameters:
        raise ValueError("At least one gradient category must be checked.")
    if not isinstance(require_float64, bool) or not isinstance(raise_on_failure, bool):
        raise TypeError("require_float64 and raise_on_failure must be booleans.")
    if module.has_pending_cache():
        raise RuntimeError(
            "Gradient checking requires a module with no pending forward cache."
        )

    input_values = np.array(np.asarray(inputs), copy=True)
    original_inputs = input_values.copy()
    named_parameters = module.named_parameters()
    parameter_snapshots = {
        name: parameter.data.copy() for name, parameter in named_parameters
    }
    if check_input and not np.issubdtype(input_values.dtype, np.floating):
        raise TypeError("Input gradient checking requires floating-point inputs.")
    if require_float64:
        if check_input and input_values.dtype != np.float64:
            raise TypeError("Input gradient checking uses float64 by default.")
        for name, parameter in named_parameters:
            if check_parameters and parameter.dtype != np.float64:
                raise TypeError(
                    f"Parameter {name!r} must use float64 for this gradient check."
                )

    original_training = module.training
    analytical_input: FloatArray | None = None
    analytical_parameters: dict[str, FloatArray] = {}
    rng = np.random.default_rng(seed)
    tensor_results: list[TensorGradientCheck] = []

    try:
        module.clear_cache()
        module.train()
        module.zero_grad()
        output = module.forward(input_values)
        loss, grad_output = _validate_objective_result(
            output, objective(output), require_gradient=True
        )
        analytical_input = module.backward(grad_output)
        if module.has_pending_cache():
            raise RuntimeError(
                "Module backward did not consume every training forward cache."
            )
        analytical_parameters = {
            name: parameter.grad.copy() for name, parameter in named_parameters
        }

        module.eval()

        def evaluate_loss() -> float:
            numerical_output = module.forward(input_values)
            numerical_loss, _ = _validate_objective_result(
                numerical_output,
                objective(numerical_output),
                require_gradient=False,
            )
            return numerical_loss

        if check_input:
            if analytical_input is None:
                raise RuntimeError("Module returned no analytical input gradient.")
            tensor_results.append(
                _check_tensor(
                    name="input",
                    values=input_values,
                    analytical=analytical_input,
                    evaluate_loss=evaluate_loss,
                    epsilon=epsilon,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    denominator_delta=denominator_delta,
                    max_checks=max_checks_per_tensor,
                    rng=rng,
                )
            )

        if check_parameters:
            for name, parameter in named_parameters:
                tensor_results.append(
                    _check_tensor(
                        name=name,
                        values=parameter.data,
                        analytical=analytical_parameters[name],
                        evaluate_loss=evaluate_loss,
                        epsilon=epsilon,
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                        denominator_delta=denominator_delta,
                        max_checks=max_checks_per_tensor,
                        rng=rng,
                    )
                )
    finally:
        input_values[...] = original_inputs
        for name, parameter in named_parameters:
            parameter.data[...] = parameter_snapshots[name]
        module.clear_cache()
        if original_training:
            module.train()
        else:
            module.eval()

    passed = all(result.passed for result in tensor_results)
    report = ModuleGradientCheck(
        loss=loss,
        tensors=tuple(tensor_results),
        passed=passed,
    )
    if not passed and raise_on_failure:
        failures = [
            (
                f"{result.name} index={result.worst_index} "
                f"analytical={result.analytical_at_worst:.12e} "
                f"numerical={result.numerical_at_worst:.12e} "
                f"absolute_error={result.maximum_absolute_error:.12e} "
                f"maximum_relative_error={result.maximum_relative_error:.12e}"
            )
            for result in report.tensors
            if not result.passed
        ]
        raise AssertionError("Gradient check failed: " + "; ".join(failures))
    return report
