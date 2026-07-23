import math

import numpy as np

from localml_scholar.nn.activations import GELU, ReLU
from localml_scholar.training.gradient_check import check_module_gradients


def test_relu_forward_and_derivative_include_zero_convention() -> None:
    layer = ReLU()
    inputs = np.array([-2.0, -0.0, 0.0, 3.0], dtype=np.float64)

    output = layer.forward(inputs)
    gradient = layer.backward(np.ones_like(inputs))

    assert np.array_equal(output, np.array([0.0, 0.0, 0.0, 3.0]))
    assert np.array_equal(gradient, np.array([0.0, 0.0, 0.0, 1.0]))


def test_relu_matches_finite_differences_away_from_kink() -> None:
    layer = ReLU()
    inputs = np.array([-2.0, -0.1, 0.2, 3.0], dtype=np.float64)
    upstream = np.array([0.2, -0.4, 1.3, -0.7], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    report = check_module_gradients(
        layer,
        inputs,
        objective,
        check_parameters=False,
    )

    assert report.passed


def test_exact_gelu_forward_and_derivative_at_difficult_values() -> None:
    inputs = np.array([-10.0, -1e-8, 0.0, 1e-8, 10.0], dtype=np.float64)
    layer = GELU()
    output = layer.forward(inputs)
    derivative = layer.backward(np.ones_like(inputs))
    cdf = np.array(
        [0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in inputs]
    )
    density = np.exp(-0.5 * inputs**2) / math.sqrt(2.0 * math.pi)

    assert np.allclose(output, inputs * cdf, rtol=0.0, atol=1e-15)
    assert np.allclose(derivative, cdf + inputs * density, atol=1e-15)
    assert derivative[2] == 0.5
    assert output[0] == 0.0
    assert output[-1] == 10.0


def test_gelu_matches_finite_differences_near_zero_and_in_tails() -> None:
    layer = GELU()
    inputs = np.array([-6.0, -0.01, 0.0, 0.01, 6.0], dtype=np.float64)
    upstream = np.array([0.2, -0.4, 1.3, -0.7, 0.8], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    report = check_module_gradients(
        layer,
        inputs,
        objective,
        check_parameters=False,
        absolute_tolerance=2e-7,
    )

    assert report.passed
