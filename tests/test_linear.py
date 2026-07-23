import numpy as np

from localml_scholar.nn.linear import Linear
from localml_scholar.training.gradient_check import check_module_gradients


def test_linear_forward_and_all_gradients_match_hand_calculation() -> None:
    layer = Linear(2, 3, seed=0)
    layer.weight.load_data(
        np.array([[1.0, -2.0, 0.5], [3.0, 1.0, -1.0]], dtype=np.float64)
    )
    assert layer.bias is not None
    layer.bias.load_data(np.array([0.2, -0.3, 0.4], dtype=np.float64))
    inputs = np.array([[2.0, -1.0], [0.5, 4.0]], dtype=np.float64)
    upstream = np.array([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]], dtype=np.float64)

    output = layer.forward(inputs)
    grad_input = layer.backward(upstream)

    expected_output = inputs @ layer.weight.data + layer.bias.data
    expected_input_gradient = upstream @ layer.weight.data.T
    expected_weight_gradient = inputs.T @ upstream
    expected_bias_gradient = np.sum(upstream, axis=0)
    assert np.allclose(output, expected_output)
    assert np.allclose(grad_input, expected_input_gradient)
    assert np.allclose(layer.weight.grad, expected_weight_gradient)
    assert np.allclose(layer.bias.grad, expected_bias_gradient)


def test_linear_supports_3d_leading_dimensions() -> None:
    layer = Linear(2, 4, seed=2)
    inputs = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    upstream = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10.0

    output = layer.forward(inputs)
    grad_input = layer.backward(upstream)

    assert output.shape == (2, 3, 4)
    assert grad_input.shape == inputs.shape
    assert np.allclose(
        layer.weight.grad,
        inputs.reshape(-1, 2).T @ upstream.reshape(-1, 4),
    )
    assert layer.bias is not None
    assert np.allclose(layer.bias.grad, upstream.reshape(-1, 4).sum(axis=0))


def test_linear_without_bias_has_only_weight_parameter() -> None:
    layer = Linear(3, 2, bias=False, seed=7)
    inputs = np.ones((4, 3), dtype=np.float64)

    output = layer.forward(inputs)
    layer.backward(np.ones_like(output))

    assert layer.bias is None
    assert [name for name, _ in layer.named_parameters()] == ["weight"]


def test_linear_passes_input_weight_and_bias_gradient_check() -> None:
    layer = Linear(2, 3, seed=11)
    inputs = np.array([[0.3, -0.7], [1.2, 0.4]], dtype=np.float64)
    upstream = np.array([[0.2, -1.0, 0.5], [1.1, 0.4, -0.2]], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    report = check_module_gradients(layer, inputs, objective)

    assert report.passed
    assert [result.name for result in report.tensors] == [
        "input",
        "weight",
        "bias",
    ]


def test_linear_float32_is_explicit_and_preserved() -> None:
    layer = Linear(2, 2, seed=1, dtype=np.float32)
    inputs = np.ones((2, 2), dtype=np.float32)

    output = layer.forward(inputs)
    grad_input = layer.backward(np.ones_like(output))

    assert output.dtype == np.float32
    assert grad_input.dtype == np.float32
    assert layer.weight.grad.dtype == np.float32
