import numpy as np

from localml_scholar.nn.normalization import LayerNorm
from localml_scholar.training.gradient_check import check_module_gradients


def test_layer_norm_mean_and_population_variance() -> None:
    epsilon = 1e-5
    layer = LayerNorm(4, epsilon=epsilon, affine=False)
    inputs = np.array(
        [[1.0, 2.0, 4.0, 8.0], [-3.0, -1.0, 2.0, 5.0]],
        dtype=np.float64,
    )

    output = layer.forward(inputs)
    input_variance = np.var(inputs, axis=-1)
    expected_output_variance = input_variance / (input_variance + epsilon)

    assert np.allclose(np.mean(output, axis=-1), 0.0, atol=1e-15)
    assert np.allclose(np.var(output, axis=-1), expected_output_variance)


def test_layer_norm_affine_forward_and_parameter_gradients() -> None:
    layer = LayerNorm(3, epsilon=1e-5)
    assert layer.gamma is not None
    assert layer.beta is not None
    layer.gamma.load_data(np.array([2.0, -1.0, 0.5]))
    layer.beta.load_data(np.array([0.3, -0.2, 1.0]))
    inputs = np.array([[1.0, 2.0, 5.0], [2.0, -1.0, 4.0]])
    upstream = np.array([[0.2, 1.0, -0.5], [2.0, -1.0, 0.3]])
    centered = inputs - inputs.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt(
        np.mean(centered**2, axis=-1, keepdims=True) + layer.epsilon
    )

    output = layer.forward(inputs)
    layer.backward(upstream)

    assert np.allclose(output, normalized * layer.gamma.data + layer.beta.data)
    assert np.allclose(layer.gamma.grad, np.sum(upstream * normalized, axis=0))
    assert np.allclose(layer.beta.grad, np.sum(upstream, axis=0))


def test_layer_norm_supports_3d_inputs() -> None:
    layer = LayerNorm(4)
    inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    upstream = np.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)

    output = layer.forward(inputs)
    grad_input = layer.backward(upstream)

    assert output.shape == inputs.shape
    assert grad_input.shape == inputs.shape
    assert layer.gamma is not None
    assert layer.beta is not None
    assert layer.gamma.grad.shape == (4,)
    assert layer.beta.grad.shape == (4,)


def test_layer_norm_nearly_constant_input_remains_finite() -> None:
    layer = LayerNorm(4, epsilon=1e-5)
    inputs = np.array([[1.0, 1.0 + 1e-12, 1.0 - 1e-12, 1.0]], dtype=np.float64)

    output = layer.forward(inputs)
    grad_input = layer.backward(np.ones_like(output))

    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(grad_input))
    assert np.allclose(output.mean(axis=-1), 0.0, atol=1e-14)


def test_layer_norm_input_gamma_and_beta_pass_finite_differences() -> None:
    layer = LayerNorm(3, epsilon=1e-5)
    inputs = np.array([[0.2, -1.3, 2.1], [1.7, 0.4, -0.8]], dtype=np.float64)
    upstream = np.array([[0.7, -0.1, 0.5], [-0.4, 1.2, 0.3]], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    report = check_module_gradients(
        layer,
        inputs,
        objective,
        absolute_tolerance=3e-7,
        relative_tolerance=2e-5,
    )

    assert report.passed
    assert [result.name for result in report.tensors] == [
        "input",
        "gamma",
        "beta",
    ]
