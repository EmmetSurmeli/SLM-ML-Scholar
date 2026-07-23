import math

import numpy as np
import pytest

from experiments.inspect_single_head_attention import inspect_attention
from localml_scholar.nn.attention import CausalSelfAttentionHead
from localml_scholar.optim.adam import Adam
from localml_scholar.training.clipping import clip_grad_norm
from localml_scholar.training.gradient_check import check_module_gradients


def _weighted_sum_objective(
    weights: np.ndarray,
):
    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * weights)), weights.copy()

    return objective


def test_attention_forward_shapes_probabilities_and_unequal_dimensions() -> None:
    model = CausalSelfAttentionHead(
        input_dim=4,
        key_dim=2,
        value_dim=3,
        bias=True,
        seed=11,
    ).eval()
    inputs = np.arange(40, dtype=np.float64).reshape(2, 5, 4) / 20.0

    output, details = model.forward(inputs, return_attention=True)

    assert output.shape == (2, 5, 3)
    assert details.query.shape == (2, 5, 2)
    assert details.key.shape == (2, 5, 2)
    assert details.value.shape == (2, 5, 3)
    assert details.scaled_scores.shape == (2, 5, 5)
    assert details.allowed_mask.shape == (1, 5, 5)
    assert details.probabilities.shape == (2, 5, 5)
    assert np.allclose(np.sum(details.probabilities, axis=-1), 1.0)
    blocked = np.broadcast_to(~details.allowed_mask, details.probabilities.shape)
    assert np.all(details.probabilities[blocked] == 0.0)
    assert not details.probabilities.flags.writeable


@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 1), (1, 4), (3, 2)])
def test_attention_supports_small_batches_and_sequences(
    batch_size: int,
    sequence_length: int,
) -> None:
    model = CausalSelfAttentionHead(3, 2, value_dim=4, seed=3).eval()
    inputs = np.ones((batch_size, sequence_length, 3), dtype=np.float64)

    output, details = model.forward(inputs, return_attention=True)

    assert output.shape == (batch_size, sequence_length, 4)
    assert details.probabilities.shape == (
        batch_size,
        sequence_length,
        sequence_length,
    )
    if sequence_length == 1:
        assert np.array_equal(details.probabilities, np.ones((batch_size, 1, 1)))


def test_attention_hand_calculated_two_token_example() -> None:
    model = CausalSelfAttentionHead(2, 2, bias=False, seed=0)
    identity = np.eye(2, dtype=np.float64)
    for projection in (
        model.query_projection,
        model.key_projection,
        model.value_projection,
    ):
        projection.weight.load_data(identity)
    inputs = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)

    output, details = model.forward(inputs, return_attention=True)

    scaled = 1.0 / math.sqrt(2.0)
    second_self = math.exp(scaled) / (1.0 + math.exp(scaled))
    expected_scores = np.array(
        [[[scaled, 0.0], [0.0, scaled]]],
        dtype=np.float64,
    )
    expected_probabilities = np.array(
        [[[1.0, 0.0], [1.0 - second_self, second_self]]],
        dtype=np.float64,
    )
    expected_output = np.array(
        [[[1.0, 0.0], [1.0 - second_self, second_self]]],
        dtype=np.float64,
    )

    assert np.allclose(details.scaled_scores, expected_scores, atol=1e-15)
    assert np.allclose(details.probabilities, expected_probabilities, atol=1e-15)
    assert np.allclose(output, expected_output, atol=1e-15)


def test_attention_is_causal_under_future_input_changes() -> None:
    model = CausalSelfAttentionHead(3, 2, value_dim=3, seed=19).eval()
    original = np.array(
        [[[0.2, -0.1, 0.4], [0.7, 0.3, -0.2], [-0.5, 0.9, 0.1]]],
        dtype=np.float64,
    )
    modified = original.copy()
    modified[:, 2, :] = np.array([8.0, -7.0, 5.0])

    original_output = model.forward(original)
    modified_output = model.forward(modified)

    assert np.array_equal(original_output[:, :2, :], modified_output[:, :2, :])
    assert not np.allclose(original_output[:, 2, :], modified_output[:, 2, :])


def test_earlier_output_loss_has_no_gradient_to_future_tokens() -> None:
    model = CausalSelfAttentionHead(3, 2, value_dim=3, seed=29)
    inputs = np.array(
        [[[0.3, -0.2, 0.6], [0.1, 0.9, -0.4], [-0.7, 0.2, 0.5]]],
        dtype=np.float64,
    )
    output = model.forward(inputs)
    grad_output = np.zeros_like(output)
    grad_output[:, 0, :] = np.array([0.5, -1.0, 0.7])

    grad_input = model.backward(grad_output)

    assert np.count_nonzero(grad_input[:, 0, :]) > 0
    assert np.array_equal(grad_input[:, 1:, :], np.zeros((1, 2, 3)))


def test_backward_shapes_and_masked_score_gradients() -> None:
    model = CausalSelfAttentionHead(4, 3, value_dim=2, seed=13)
    inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10.0
    output, details = model.forward(inputs, return_attention=True)

    grad_input = model.backward(np.ones_like(output))
    grad_scores = model.last_score_gradient

    assert grad_input.shape == inputs.shape
    assert grad_scores is not None
    assert grad_scores.shape == details.scaled_scores.shape
    blocked = np.broadcast_to(~details.allowed_mask, grad_scores.shape)
    assert np.array_equal(grad_scores[blocked], np.zeros(np.count_nonzero(blocked)))
    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.data.shape


def test_attention_output_projection_restores_input_dimension() -> None:
    model = CausalSelfAttentionHead(
        4,
        2,
        value_dim=3,
        output_projection=True,
        seed=5,
    )
    inputs = np.ones((2, 3, 4), dtype=np.float64)

    output = model.forward(inputs)
    grad_input = model.backward(np.ones_like(output))

    assert output.shape == inputs.shape
    assert grad_input.shape == inputs.shape
    assert model.output_projection is not None
    assert model.output_projection.weight.grad.shape == (3, 4)


def test_attention_without_bias_has_only_projection_weights() -> None:
    model = CausalSelfAttentionHead(
        3,
        2,
        value_dim=4,
        bias=False,
        output_projection=True,
        seed=2,
    )

    assert [name for name, _ in model.named_parameters()] == [
        "query_projection.weight",
        "key_projection.weight",
        "value_projection.weight",
        "output_projection.weight",
    ]


def test_attention_projection_biases_are_applied() -> None:
    model = CausalSelfAttentionHead(2, 2, value_dim=3, bias=True, seed=17).eval()
    for projection in (
        model.query_projection,
        model.key_projection,
        model.value_projection,
    ):
        projection.weight.data.fill(0.0)
    assert model.query_projection.bias is not None
    assert model.key_projection.bias is not None
    assert model.value_projection.bias is not None
    model.query_projection.bias.data[...] = np.array([0.2, -0.4])
    model.key_projection.bias.data[...] = np.array([0.7, 0.1])
    model.value_projection.bias.data[...] = np.array([1.0, -2.0, 0.5])
    inputs = np.array([[[8.0, -3.0], [0.5, 4.0]]], dtype=np.float64)

    output, details = model.forward(inputs, return_attention=True)

    assert np.allclose(details.query, np.array([[[0.2, -0.4], [0.2, -0.4]]]))
    assert np.allclose(details.key, np.array([[[0.7, 0.1], [0.7, 0.1]]]))
    assert np.allclose(details.value, np.array([[[1.0, -2.0, 0.5]] * 2]))
    assert np.allclose(output, details.value)


def test_attention_seeded_initialization_is_deterministic() -> None:
    first = CausalSelfAttentionHead(4, 2, value_dim=3, seed=101)
    second = CausalSelfAttentionHead(4, 2, value_dim=3, seed=101)
    third = CausalSelfAttentionHead(4, 2, value_dim=3, seed=102)

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


def test_attention_input_and_projection_gradients_match_finite_differences() -> None:
    model = CausalSelfAttentionHead(
        2,
        2,
        value_dim=2,
        bias=True,
        seed=7,
        dtype=np.float64,
    )
    inputs = np.array([[[0.2, -0.3], [0.7, 0.1]]], dtype=np.float64)
    weights = np.array([[[0.4, -0.2], [0.3, 0.8]]], dtype=np.float64)

    result = check_module_gradients(
        model,
        inputs,
        _weighted_sum_objective(weights),
        absolute_tolerance=2e-7,
        relative_tolerance=2e-5,
    )

    assert result.passed
    assert result.checked_coordinates == 22
    assert [tensor.name for tensor in result.tensors] == [
        "input",
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "key_projection.bias",
        "value_projection.weight",
        "value_projection.bias",
    ]


def test_attention_output_projection_gradients_match_finite_differences() -> None:
    model = CausalSelfAttentionHead(
        2,
        1,
        value_dim=2,
        bias=True,
        output_projection=True,
        seed=8,
        dtype=np.float64,
    )
    inputs = np.array([[[0.2, -0.3], [0.7, 0.1]]], dtype=np.float64)
    weights = np.array([[[0.4, -0.2], [0.3, 0.8]]], dtype=np.float64)

    result = check_module_gradients(
        model,
        inputs,
        _weighted_sum_objective(weights),
        absolute_tolerance=2e-7,
        relative_tolerance=2e-5,
    )

    assert result.passed
    assert {tensor.name for tensor in result.tensors} >= {
        "output_projection.weight",
        "output_projection.bias",
    }


def test_attention_checkpoint_preserves_configuration_parameters_and_output(
    tmp_path,
) -> None:
    model = CausalSelfAttentionHead(
        3,
        2,
        value_dim=4,
        bias=True,
        output_projection=True,
        seed=23,
        dtype=np.float32,
    ).eval()
    inputs = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / np.float32(10.0)
    expected = model.forward(inputs)
    checkpoint = tmp_path / "attention.npz"

    model.save_checkpoint(checkpoint)
    loaded = CausalSelfAttentionHead.load_checkpoint(checkpoint).eval()

    assert loaded.configuration == model.configuration
    assert loaded.parameter_count == model.parameter_count
    assert [name for name, _ in loaded.named_parameters()] == [
        name for name, _ in model.named_parameters()
    ]
    assert np.array_equal(loaded.forward(inputs), expected)


def test_attention_inspection_experiment_writes_verified_summary(tmp_path) -> None:
    summary = inspect_attention(seed=31, output_directory=tmp_path)

    assert summary["future_probabilities_exactly_zero"]
    assert summary["shapes"]["input"] == [1, 4, 4]
    assert summary["shapes"]["query"] == [1, 4, 2]
    assert summary["shapes"]["value"] == [1, 4, 3]
    assert summary["shapes"]["output"] == [1, 4, 3]
    assert summary["synthetic_loss"] > 0.0
    assert all(
        np.isfinite(norm) and norm >= 0.0 for norm in summary["gradient_norms"].values()
    )
    assert (tmp_path / "run_summary.json").is_file()


def test_attention_integrates_with_optimizer_clipping_and_zero_grad() -> None:
    model = CausalSelfAttentionHead(3, 2, seed=31, dtype=np.float32)
    optimizer = Adam(model.parameters(), learning_rate=0.01)
    inputs = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / np.float32(20.0)
    output = model.forward(inputs)
    model.backward(np.ones_like(output))
    parameters_before = [parameter.data.copy() for parameter in model.parameters()]

    norm = clip_grad_norm(model.parameters(), max_norm=0.5)
    optimizer.step()

    assert np.isfinite(norm)
    assert norm > 0.0
    assert any(
        not np.array_equal(before, parameter.data)
        for before, parameter in zip(parameters_before, model.parameters(), strict=True)
    )
    optimizer.zero_grad()
    assert all(
        np.array_equal(parameter.grad, np.zeros_like(parameter.grad))
        for parameter in model.parameters()
    )


def test_attention_named_parameter_order_and_modes_are_recursive() -> None:
    model = CausalSelfAttentionHead(
        3,
        2,
        value_dim=4,
        output_projection=True,
        seed=37,
    )

    assert [name for name, _ in model.named_parameters()] == [
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "key_projection.bias",
        "value_projection.weight",
        "value_projection.bias",
        "output_projection.weight",
        "output_projection.bias",
    ]
    model.eval()
    assert not model.training
    assert all(not child.training for child in model.modules())
    model.train()
    assert model.training
    assert all(child.training for child in model.modules())


def test_attention_cache_and_gradient_validation_errors_are_informative() -> None:
    model = CausalSelfAttentionHead(2, 2, seed=41)
    inputs = np.ones((1, 2, 2), dtype=np.float64)
    output = model.forward(inputs)

    with pytest.raises(RuntimeError, match="cannot run twice"):
        model.forward(inputs)
    with pytest.raises(ValueError, match="grad_output shape"):
        model.backward(np.ones((1, 1, 2), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        model.backward(np.ones(output.shape, dtype=np.float32))

    model.backward(np.ones_like(output))
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        model.backward(np.ones_like(output))


@pytest.mark.parametrize(
    "bad_inputs",
    [
        np.ones((2, 3), dtype=np.float64),
        np.ones((1, 2, 3, 2), dtype=np.float64),
        np.ones((1, 2, 3), dtype=np.float64),
        np.ones((1, 2, 2), dtype=np.int64),
        np.empty((0, 2, 2), dtype=np.float64),
        np.empty((1, 0, 2), dtype=np.float64),
        np.array([[[np.nan, 0.0], [1.0, 2.0]]]),
    ],
)
def test_attention_rejects_invalid_inputs(bad_inputs: np.ndarray) -> None:
    model = CausalSelfAttentionHead(2, 2, seed=43)

    with pytest.raises((TypeError, ValueError), match="inputs"):
        model.forward(bad_inputs)


def test_attention_configuration_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        CausalSelfAttentionHead(0, 2)
    with pytest.raises(ValueError, match="key_dim"):
        CausalSelfAttentionHead(2, 0)
    with pytest.raises(ValueError, match="value_dim"):
        CausalSelfAttentionHead(2, 2, value_dim=0)
    with pytest.raises(TypeError, match="bias"):
        CausalSelfAttentionHead(2, 2, bias=1)
    with pytest.raises(TypeError, match="output_projection"):
        CausalSelfAttentionHead(2, 2, output_projection=1)
    with pytest.raises(ValueError, match="non-negative"):
        CausalSelfAttentionHead(2, 2, seed=-1)
    with pytest.raises(TypeError, match="dtype"):
        CausalSelfAttentionHead(2, 2, dtype=np.float16)
