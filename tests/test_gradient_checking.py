import numpy as np
import pytest

from localml_scholar.nn.linear import Linear
from localml_scholar.training.gradient_check import check_module_gradients


def _weighted_objective(
    upstream: np.ndarray,
):
    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    return objective


def test_checker_restores_every_parameter_and_reports_per_tensor() -> None:
    layer = Linear(3, 2, seed=8)
    inputs = np.array([[0.2, -0.7, 1.1], [0.4, 0.9, -0.3]], dtype=np.float64)
    upstream = np.array([[0.5, -0.2], [1.3, 0.7]], dtype=np.float64)
    before = {
        name: parameter.data.copy() for name, parameter in layer.named_parameters()
    }

    report = check_module_gradients(
        layer,
        inputs,
        _weighted_objective(upstream),
    )

    assert report.passed
    assert report.checked_coordinates == inputs.size + sum(
        parameter.size for parameter in layer.parameters()
    )
    for name, parameter in layer.named_parameters():
        assert np.array_equal(parameter.data, before[name])


def test_sampled_coordinates_are_deterministic() -> None:
    layer = Linear(5, 4, seed=3)
    inputs = np.linspace(-1.0, 1.0, 15).reshape(3, 5)
    upstream = np.linspace(0.7, -0.4, 12).reshape(3, 4)

    first = check_module_gradients(
        layer,
        inputs,
        _weighted_objective(upstream),
        max_checks_per_tensor=3,
        seed=22,
    )
    second = check_module_gradients(
        layer,
        inputs,
        _weighted_objective(upstream),
        max_checks_per_tensor=3,
        seed=22,
    )

    assert first == second
    assert all(result.checked_coordinates == 3 for result in first.tensors)


def test_float64_is_required_by_default_but_float32_can_be_explicit() -> None:
    layer = Linear(2, 2, seed=1, dtype=np.float32)
    inputs = np.array([[0.2, -0.4]], dtype=np.float32)
    upstream = np.array([[0.7, 0.1]], dtype=np.float32)

    with pytest.raises(TypeError, match="float64 by default"):
        check_module_gradients(layer, inputs, _weighted_objective(upstream))

    report = check_module_gradients(
        layer,
        inputs,
        _weighted_objective(upstream),
        require_float64=False,
        epsilon=1e-3,
        absolute_tolerance=2e-4,
        relative_tolerance=2e-3,
    )
    assert report.passed


def test_checker_failure_includes_tensor_index_and_values() -> None:
    class IncorrectInputGradientLinear(Linear):
        def backward(self, grad_output):
            gradient = super().backward(grad_output)
            return gradient + 0.25

    layer = IncorrectInputGradientLinear(2, 2, seed=1)
    inputs = np.array([[0.2, -0.4]], dtype=np.float64)
    upstream = np.array([[0.7, 0.1]], dtype=np.float64)

    with pytest.raises(
        AssertionError,
        match=r"input index=.*analytical=.*numerical=",
    ):
        check_module_gradients(layer, inputs, _weighted_objective(upstream))
