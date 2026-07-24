import numpy as np
import pytest

from localml_scholar.nn.transformer import TransformerFeedForward
from localml_scholar.optim.adam import Adam
from localml_scholar.training.clipping import clip_grad_norm
from localml_scholar.training.gradient_check import check_module_gradients


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "model_dim", "hidden_dim"),
    [(1, 1, 4, 2), (1, 5, 3, 7), (3, 2, 4, 4)],
)
def test_feed_forward_preserves_shape_for_arbitrary_leading_dimensions(
    batch_size: int,
    sequence_length: int,
    model_dim: int,
    hidden_dim: int,
) -> None:
    module = TransformerFeedForward(model_dim, hidden_dim, seed=3).eval()
    inputs = np.arange(
        batch_size * sequence_length * model_dim,
        dtype=np.float64,
    ).reshape(batch_size, sequence_length, model_dim)

    output, details = module.forward(inputs, return_details=True)

    assert output.shape == inputs.shape
    assert details.pre_activation.shape == (
        batch_size,
        sequence_length,
        hidden_dim,
    )
    assert details.activation.shape == details.pre_activation.shape
    assert not details.pre_activation.flags.writeable
    assert not details.activation.flags.writeable


def test_feed_forward_matches_hand_calculation() -> None:
    module = TransformerFeedForward(2, 2, bias=True, activation="relu", seed=5)
    module.linear1.weight.load_data(
        np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    )
    assert module.linear1.bias is not None
    module.linear1.bias.load_data(np.array([0.1, -0.2], dtype=np.float64))
    module.linear2.weight.load_data(
        np.array([[2.0, -1.0], [0.25, 0.5]], dtype=np.float64)
    )
    assert module.linear2.bias is not None
    module.linear2.bias.load_data(np.array([-0.3, 0.7], dtype=np.float64))
    inputs = np.array([[[1.0, 2.0], [-1.0, 0.5]]], dtype=np.float64)
    expected_pre_activation = np.array(
        [[[2.1, 3.8], [-0.65, 3.3]]],
        dtype=np.float64,
    )
    expected_activation = np.maximum(expected_pre_activation, 0.0)
    expected_output = (
        expected_activation @ module.linear2.weight.data + module.linear2.bias.data
    )

    output, details = module.forward(inputs, return_details=True)

    assert np.array_equal(details.pre_activation, expected_pre_activation)
    assert np.array_equal(details.activation, expected_activation)
    assert np.array_equal(output, expected_output)


def test_feed_forward_initialization_is_deterministic() -> None:
    first = TransformerFeedForward(4, 7, seed=101)
    second = TransformerFeedForward(4, 7, seed=101)
    third = TransformerFeedForward(4, 7, seed=102)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert np.array_equal(first_parameter.data, second_parameter.data)
    assert any(
        not np.array_equal(first_parameter.data, third_parameter.data)
        for first_parameter, third_parameter in zip(
            first.parameters(), third.parameters(), strict=True
        )
    )


def test_feed_forward_bias_modes_and_parameter_order() -> None:
    with_bias = TransformerFeedForward(3, 5, bias=True)
    without_bias = TransformerFeedForward(3, 5, bias=False)

    assert [name for name, _ in with_bias.named_parameters()] == [
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
    ]
    assert [name for name, _ in without_bias.named_parameters()] == [
        "linear1.weight",
        "linear2.weight",
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_feed_forward_preserves_supported_dtype(dtype: type[np.floating]) -> None:
    module = TransformerFeedForward(3, 4, seed=7, dtype=dtype)
    inputs = np.arange(18, dtype=dtype).reshape(2, 3, 3) / dtype(10.0)

    output = module.forward(inputs)
    grad_input = module.backward(np.ones_like(output))

    assert output.dtype == np.dtype(dtype)
    assert grad_input.dtype == np.dtype(dtype)
    assert all(parameter.dtype == np.dtype(dtype) for parameter in module.parameters())


@pytest.mark.parametrize(
    "bad_inputs",
    [
        np.ones((2, 3), dtype=np.float64),
        np.ones((1, 2, 3, 4), dtype=np.float64),
        np.ones((1, 2, 4), dtype=np.float64),
        np.ones((1, 2, 3), dtype=np.int64),
        np.empty((0, 2, 3), dtype=np.float64),
        np.empty((1, 0, 3), dtype=np.float64),
        np.array([[[np.inf, 0.0, 1.0]]]),
    ],
)
def test_feed_forward_rejects_malformed_inputs(bad_inputs: np.ndarray) -> None:
    module = TransformerFeedForward(3, 4, seed=11)

    with pytest.raises((TypeError, ValueError), match="inputs"):
        module.forward(bad_inputs)


def test_feed_forward_cache_and_upstream_validation() -> None:
    module = TransformerFeedForward(3, 4, seed=13)
    inputs = np.ones((1, 2, 3), dtype=np.float64)
    output = module.forward(inputs)

    with pytest.raises(RuntimeError, match="cannot run twice"):
        module.forward(inputs)
    with pytest.raises(ValueError, match="grad_output shape"):
        module.backward(np.ones((1, 1, 3), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        module.backward(np.ones(output.shape, dtype=np.float32))
    malformed = np.ones_like(output)
    malformed[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        module.backward(malformed)
    module.backward(np.ones_like(output))
    assert not module.has_pending_cache()
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        module.backward(np.ones_like(output))


def test_feed_forward_zero_grad_and_mode_propagation() -> None:
    module = TransformerFeedForward(3, 4, seed=17)
    for parameter in module.parameters():
        parameter.grad.fill(2.0)

    module.eval()

    assert not module.training
    assert all(not child.training for child in module.modules())
    module.zero_grad()
    assert all(
        np.count_nonzero(parameter.grad) == 0 for parameter in module.parameters()
    )
    module.train()
    assert module.training
    assert all(child.training for child in module.modules())


def test_feed_forward_checkpoint_preserves_configuration_and_float32_output(
    tmp_path,
) -> None:
    module = TransformerFeedForward(
        3,
        5,
        bias=False,
        activation="relu",
        seed=19,
        dtype=np.float32,
    ).eval()
    inputs = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    expected = module.forward(inputs)
    checkpoint = tmp_path / "feed_forward.npz"

    module.save_checkpoint(checkpoint)
    loaded = TransformerFeedForward.load_checkpoint(checkpoint)

    assert loaded.training
    assert loaded.configuration == module.configuration
    assert [name for name, _ in loaded.named_parameters()] == [
        name for name, _ in module.named_parameters()
    ]
    assert np.array_equal(loaded.eval().forward(inputs), expected)


def test_feed_forward_optimizer_and_clipping_compatibility() -> None:
    module = TransformerFeedForward(3, 5, seed=23, dtype=np.float32)
    optimizer = Adam(module.parameters(), learning_rate=0.01)
    inputs = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / np.float32(20.0)
    output = module.forward(inputs)
    module.backward(np.ones_like(output))
    before = [parameter.data.copy() for parameter in module.parameters()]

    norm = clip_grad_norm(module.parameters(), max_norm=0.5)
    optimizer.step()

    assert np.isfinite(norm) and norm > 0.0
    assert any(
        not np.array_equal(previous, parameter.data)
        for previous, parameter in zip(before, module.parameters(), strict=True)
    )


def test_feed_forward_all_gradients_match_finite_differences() -> None:
    module = TransformerFeedForward(
        2,
        3,
        bias=True,
        activation="gelu",
        seed=29,
        dtype=np.float64,
    )
    inputs = np.array([[[0.2, -0.3], [0.7, 0.1]]], dtype=np.float64)
    weights = np.array([[[0.4, -0.2], [0.3, 0.8]]], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * weights)), weights.copy()

    result = check_module_gradients(
        module,
        inputs,
        objective,
        absolute_tolerance=2e-7,
        relative_tolerance=2e-5,
    )

    assert result.passed
    assert result.checked_coordinates == 21
    assert [tensor.name for tensor in result.tensors] == [
        "input",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
    ]


def test_feed_forward_configuration_validation() -> None:
    with pytest.raises(ValueError, match="model_dim"):
        TransformerFeedForward(0, 3)
    with pytest.raises(ValueError, match="hidden_dim"):
        TransformerFeedForward(3, 0)
    with pytest.raises(TypeError, match="bias"):
        TransformerFeedForward(3, 4, bias=1)
    with pytest.raises(ValueError, match="activation"):
        TransformerFeedForward(3, 4, activation="swiglu")
    with pytest.raises(ValueError, match="non-negative"):
        TransformerFeedForward(3, 4, seed=-1)
    with pytest.raises(TypeError, match="dtype"):
        TransformerFeedForward(3, 4, dtype=np.float16)
