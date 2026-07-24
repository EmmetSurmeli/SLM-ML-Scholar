import json

import numpy as np
import pytest

from experiments.inspect_pre_norm_decoder_block import inspect_decoder_block
from localml_scholar.nn.transformer import (
    PreNormDecoderBlock,
    residual_add,
    residual_add_backward,
)
from localml_scholar.optim.adam import Adam
from localml_scholar.optim.momentum import Momentum
from localml_scholar.optim.sgd import SGD
from localml_scholar.training.clipping import clip_grad_norm
from localml_scholar.training.gradient_check import check_module_gradients


def _zero_attention_output(block: PreNormDecoderBlock) -> None:
    if block.attention.output_projection is None:
        block.attention.value_projection.weight.data.fill(0.0)
        if block.attention.value_projection.bias is not None:
            block.attention.value_projection.bias.data.fill(0.0)
    else:
        block.attention.output_projection.weight.data.fill(0.0)
        if block.attention.output_projection.bias is not None:
            block.attention.output_projection.bias.data.fill(0.0)


def _zero_feed_forward_output(block: PreNormDecoderBlock) -> None:
    block.feed_forward.linear2.weight.data.fill(0.0)
    if block.feed_forward.linear2.bias is not None:
        block.feed_forward.linear2.bias.data.fill(0.0)


def test_residual_add_preserves_shape_and_rejects_broadcasting() -> None:
    identity = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    transformed = np.ones_like(identity)

    output = residual_add(identity, transformed, name="test")

    assert output.shape == identity.shape
    assert np.array_equal(output, identity + transformed)
    with pytest.raises(ValueError, match="match exactly"):
        residual_add(identity, np.ones((1, 1, 4)), name="test")
    with pytest.raises(TypeError, match="dtypes"):
        residual_add(identity, transformed.astype(np.float32), name="test")
    with pytest.raises(TypeError, match="floating"):
        residual_add(identity.astype(np.int64), identity.astype(np.int64))
    with pytest.raises(ValueError, match="finite"):
        residual_add(identity, np.full_like(identity, np.nan))


def test_residual_backward_returns_two_independent_identity_gradients() -> None:
    gradient = np.arange(6, dtype=np.float64).reshape(1, 3, 2)

    identity_gradient, transformed_gradient = residual_add_backward(
        gradient,
        expected_shape=gradient.shape,
        dtype=np.dtype(np.float64),
        name="test",
    )

    assert np.array_equal(identity_gradient, gradient)
    assert np.array_equal(transformed_gradient, gradient)
    assert identity_gradient is not transformed_gradient
    identity_gradient.fill(-1.0)
    assert np.array_equal(transformed_gradient, gradient)
    with pytest.raises(ValueError, match="shape"):
        residual_add_backward(
            gradient,
            expected_shape=(1, 2, 3),
            dtype=np.dtype(np.float64),
        )


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "key_dim", "value_dim"),
    [(1, 1, 1, 1), (1, 4, 2, 3), (3, 2, 5, 2)],
)
def test_decoder_block_preserves_shape_across_valid_dimensions(
    batch_size: int,
    sequence_length: int,
    key_dim: int,
    value_dim: int,
) -> None:
    block = PreNormDecoderBlock(
        model_dim=4,
        key_dim=key_dim,
        value_dim=value_dim,
        ff_hidden_dim=7,
        attention_output_projection=True,
        seed=3,
    ).eval()
    inputs = np.arange(
        batch_size * sequence_length * 4,
        dtype=np.float64,
    ).reshape(batch_size, sequence_length, 4)

    output, details = block.forward(inputs, return_details=True)

    assert output.shape == inputs.shape
    assert details.normalized_attention_input.shape == inputs.shape
    assert details.attention.query.shape == (
        batch_size,
        sequence_length,
        key_dim,
    )
    assert details.attention.value.shape == (
        batch_size,
        sequence_length,
        value_dim,
    )
    assert details.attention.scaled_scores.shape == (
        batch_size,
        sequence_length,
        sequence_length,
    )
    assert details.attention_output.shape == inputs.shape
    assert details.first_residual.shape == inputs.shape
    assert details.normalized_feed_forward_input.shape == inputs.shape
    assert details.feed_forward.activation.shape == (
        batch_size,
        sequence_length,
        7,
    )
    assert details.feed_forward_output.shape == inputs.shape
    assert details.output.shape == inputs.shape
    blocked = np.broadcast_to(
        ~details.attention.allowed_mask,
        details.attention.probabilities.shape,
    )
    assert np.all(details.attention.probabilities[blocked] == 0.0)
    assert np.allclose(np.sum(details.attention.probabilities, axis=-1), 1.0)
    assert np.array_equal(
        details.first_residual,
        inputs + details.attention_output,
    )
    assert np.array_equal(
        output,
        details.first_residual + details.feed_forward_output,
    )
    assert all(
        not tensor.flags.writeable
        for tensor in (
            details.normalized_attention_input,
            details.attention_output,
            details.first_residual,
            details.normalized_feed_forward_input,
            details.feed_forward_output,
            details.output,
        )
    )


def test_decoder_block_without_output_projection_requires_compatible_value_dim() -> (
    None
):
    compatible = PreNormDecoderBlock(
        model_dim=3,
        key_dim=2,
        value_dim=3,
        ff_hidden_dim=5,
        attention_output_projection=False,
        seed=5,
    ).eval()
    inputs = np.ones((1, 2, 3), dtype=np.float64)

    assert compatible.forward(inputs).shape == inputs.shape
    with pytest.raises(ValueError, match="value_dim must equal model_dim"):
        PreNormDecoderBlock(
            model_dim=3,
            key_dim=2,
            value_dim=2,
            ff_hidden_dim=5,
            attention_output_projection=False,
        )


def test_identity_block_is_exact_in_forward_and_backward() -> None:
    block = PreNormDecoderBlock(3, 2, 5, seed=7)
    _zero_attention_output(block)
    _zero_feed_forward_output(block)
    inputs = np.array(
        [[[0.2, -0.1, 0.4], [0.7, 0.3, -0.2]]],
        dtype=np.float64,
    )
    upstream = np.array(
        [[[0.5, -1.0, 0.7], [0.2, 0.4, -0.3]]],
        dtype=np.float64,
    )

    output = block.forward(inputs)
    grad_input = block.backward(upstream)

    assert np.array_equal(output, inputs)
    assert np.array_equal(grad_input, upstream)


def test_attention_only_residual_matches_independent_calculation() -> None:
    epsilon = 1e-5
    block = PreNormDecoderBlock(
        model_dim=2,
        key_dim=1,
        value_dim=2,
        ff_hidden_dim=3,
        attention_bias=False,
        feed_forward_bias=False,
        attention_output_projection=False,
        layer_norm_epsilon=epsilon,
        seed=11,
    ).eval()
    block.attention.value_projection.weight.load_data(np.eye(2, dtype=np.float64))
    _zero_feed_forward_output(block)
    inputs = np.array([[[1.0, 3.0]]], dtype=np.float64)
    centered = np.array([[[-1.0, 1.0]]], dtype=np.float64)
    expected_normalized = centered / np.sqrt(1.0 + epsilon)
    expected = inputs + expected_normalized

    output = block.forward(inputs)

    assert np.allclose(output, expected, rtol=0.0, atol=1e-15)
    assert not np.allclose(output.mean(axis=-1), 0.0)


def test_decoder_block_initialization_is_deterministic() -> None:
    first = PreNormDecoderBlock(4, 2, 7, value_dim=3, seed=13)
    second = PreNormDecoderBlock(4, 2, 7, value_dim=3, seed=13)
    third = PreNormDecoderBlock(4, 2, 7, value_dim=3, seed=14)

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


def test_decoder_block_is_causal_under_future_input_changes() -> None:
    block = PreNormDecoderBlock(3, 2, 5, value_dim=2, seed=17).eval()
    original = np.array(
        [[[0.2, -0.1, 0.4], [0.7, 0.3, -0.2], [-0.5, 0.9, 0.1]]],
        dtype=np.float64,
    )
    modified = original.copy()
    modified[:, 2, :] = np.array([8.0, -7.0, 5.0])

    original_output = block.forward(original)
    modified_output = block.forward(modified)

    assert np.array_equal(original_output[:, :2, :], modified_output[:, :2, :])
    assert not np.allclose(original_output[:, 2, :], modified_output[:, 2, :])


def test_earlier_decoder_loss_has_zero_gradient_to_future_tokens() -> None:
    block = PreNormDecoderBlock(3, 2, 5, value_dim=2, seed=19)
    inputs = np.array(
        [[[0.3, -0.2, 0.6], [0.1, 0.9, -0.4], [-0.7, 0.2, 0.5]]],
        dtype=np.float64,
    )
    output = block.forward(inputs)
    grad_output = np.zeros_like(output)
    grad_output[:, 0, :] = np.array([0.5, -1.0, 0.7])

    grad_input = block.backward(grad_output)

    assert np.count_nonzero(grad_input[:, 0, :]) > 0
    assert np.array_equal(grad_input[:, 1:, :], np.zeros((1, 2, 3)))


def test_decoder_all_gradients_match_exhaustive_finite_differences() -> None:
    block = PreNormDecoderBlock(
        model_dim=2,
        key_dim=1,
        value_dim=1,
        ff_hidden_dim=2,
        attention_bias=True,
        feed_forward_bias=True,
        attention_output_projection=True,
        seed=23,
        dtype=np.float64,
    )
    inputs = np.array([[[0.2, -0.3], [0.7, 0.1]]], dtype=np.float64)
    weights = np.array([[[0.4, -0.2], [0.3, 0.8]]], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * weights)), weights.copy()

    result = check_module_gradients(
        block,
        inputs,
        objective,
        absolute_tolerance=1e-6,
        relative_tolerance=5e-5,
    )

    assert result.passed
    assert (
        result.checked_coordinates
        == sum(parameter.size for parameter in block.parameters()) + inputs.size
    )
    assert [tensor.name for tensor in result.tensors] == [
        "input",
        "norm1.gamma",
        "norm1.beta",
        "attention.query_projection.weight",
        "attention.query_projection.bias",
        "attention.key_projection.weight",
        "attention.key_projection.bias",
        "attention.value_projection.weight",
        "attention.value_projection.bias",
        "attention.output_projection.weight",
        "attention.output_projection.bias",
        "norm2.gamma",
        "norm2.beta",
        "feed_forward.linear1.weight",
        "feed_forward.linear1.bias",
        "feed_forward.linear2.weight",
        "feed_forward.linear2.bias",
    ]


def test_decoder_named_parameters_and_modes_are_recursive() -> None:
    block = PreNormDecoderBlock(3, 2, 5, value_dim=4, seed=29)
    expected_names = [
        "norm1.gamma",
        "norm1.beta",
        "attention.query_projection.weight",
        "attention.query_projection.bias",
        "attention.key_projection.weight",
        "attention.key_projection.bias",
        "attention.value_projection.weight",
        "attention.value_projection.bias",
        "attention.output_projection.weight",
        "attention.output_projection.bias",
        "norm2.gamma",
        "norm2.beta",
        "feed_forward.linear1.weight",
        "feed_forward.linear1.bias",
        "feed_forward.linear2.weight",
        "feed_forward.linear2.bias",
    ]

    assert [name for name, _ in block.named_parameters()] == expected_names
    assert len({id(parameter) for parameter in block.parameters()}) == len(
        block.parameters()
    )
    block.eval()
    assert not block.training
    assert all(not child.training for child in block.modules())
    assert all(
        not grandchild.training
        for child in block.modules()
        for grandchild in child.modules()
    )
    block.train()
    assert block.training
    assert all(child.training for child in block.modules())
    assert all(
        grandchild.training
        for child in block.modules()
        for grandchild in child.modules()
    )
    for parameter in block.parameters():
        parameter.grad.fill(3.0)
    block.zero_grad()
    assert all(
        np.array_equal(parameter.grad, np.zeros_like(parameter.grad))
        for parameter in block.parameters()
    )


@pytest.mark.parametrize("optimizer_name", ["sgd", "momentum", "adam"])
def test_decoder_optimizers_support_repeated_update_cycles(
    optimizer_name: str,
) -> None:
    block = PreNormDecoderBlock(3, 2, 5, seed=31, dtype=np.float32)
    if optimizer_name == "sgd":
        optimizer = SGD(block.parameters(), learning_rate=0.01)
    elif optimizer_name == "momentum":
        optimizer = Momentum(block.parameters(), learning_rate=0.01, beta=0.8)
    else:
        optimizer = Adam(block.parameters(), learning_rate=0.01)
    inputs = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / np.float32(20.0)
    before = [parameter.data.copy() for parameter in block.parameters()]

    for _ in range(2):
        optimizer.zero_grad()
        output = block.forward(inputs)
        block.backward(np.ones_like(output))
        clip_grad_norm(block.parameters(), max_norm=1.0)
        optimizer.step()

    assert any(
        not np.array_equal(previous, parameter.data)
        for previous, parameter in zip(before, block.parameters(), strict=True)
    )
    assert not block.has_pending_cache()


def test_decoder_checkpoint_preserves_configuration_names_and_float32_output(
    tmp_path,
) -> None:
    block = PreNormDecoderBlock(
        3,
        2,
        5,
        value_dim=4,
        attention_bias=False,
        feed_forward_bias=False,
        attention_output_projection=True,
        layer_norm_epsilon=2e-5,
        activation="relu",
        seed=37,
        dtype=np.float32,
    ).eval()
    inputs = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / np.float32(10.0)
    expected = block.forward(inputs)
    checkpoint = tmp_path / "decoder.npz"

    block.save_checkpoint(checkpoint)
    loaded = PreNormDecoderBlock.load_checkpoint(checkpoint)

    assert loaded.training
    assert loaded.configuration == block.configuration
    assert loaded.configuration["model_version"] == "0.4.0"
    assert [name for name, _ in loaded.named_parameters()] == [
        name for name, _ in block.named_parameters()
    ]
    assert np.array_equal(loaded.eval().forward(inputs), expected)


def test_decoder_checkpoint_rejects_incompatible_model_version(tmp_path) -> None:
    block = PreNormDecoderBlock(2, 1, 3, seed=41)
    original = tmp_path / "original.npz"
    malformed = tmp_path / "malformed.npz"
    block.save_checkpoint(original)
    with np.load(original, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        metadata["configuration"]["model_version"] = "9.9.9"
        arrays = {
            key: np.array(checkpoint[key], copy=True)
            for key in checkpoint.files
            if key != "metadata_json"
        }
    np.savez(
        malformed,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )

    with pytest.raises(ValueError, match="model version"):
        PreNormDecoderBlock.load_checkpoint(malformed)


def test_decoder_inspection_experiment_writes_verified_summary(tmp_path) -> None:
    summary = inspect_decoder_block(seed=41, output_directory=tmp_path)

    assert summary["future_probabilities_exactly_zero"]
    assert summary["output_shape_matches_input"]
    assert summary["earlier_outputs_unchanged_after_future_token_change"]
    assert summary["optimizer_changed_parameter"]
    assert summary["shapes"]["embeddings"] == [1, 4, 4]
    assert summary["shapes"]["query"] == [1, 4, 2]
    assert summary["shapes"]["value"] == [1, 4, 3]
    assert summary["shapes"]["feed_forward_hidden"] == [1, 4, 7]
    assert summary["shapes"]["output"] == [1, 4, 4]
    assert summary["synthetic_loss"] > 0.0
    assert all(
        np.isfinite(norm) and norm >= 0.0 for norm in summary["gradient_norms"].values()
    )
    assert (tmp_path / "run_summary.json").is_file()


def test_decoder_cache_and_malformed_gradient_errors_are_informative() -> None:
    block = PreNormDecoderBlock(3, 2, 5, seed=43)
    inputs = np.ones((1, 2, 3), dtype=np.float64)
    output = block.forward(inputs)

    with pytest.raises(RuntimeError, match="cannot run twice"):
        block.forward(inputs)
    with pytest.raises(ValueError, match="grad_output shape"):
        block.backward(np.ones((1, 1, 3), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        block.backward(np.ones(output.shape, dtype=np.float32))
    malformed = np.ones_like(output)
    malformed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        block.backward(malformed)
    block.backward(np.ones_like(output))
    assert not block.has_pending_cache()
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        block.backward(np.ones_like(output))


@pytest.mark.parametrize(
    "bad_inputs",
    [
        np.ones((2, 3), dtype=np.float64),
        np.ones((1, 2, 3, 4), dtype=np.float64),
        np.ones((1, 2, 4), dtype=np.float64),
        np.ones((1, 2, 3), dtype=np.int64),
        np.empty((0, 2, 3), dtype=np.float64),
        np.empty((1, 0, 3), dtype=np.float64),
        np.array([[[np.nan, 0.0, 1.0]]]),
    ],
)
def test_decoder_rejects_malformed_inputs(bad_inputs: np.ndarray) -> None:
    block = PreNormDecoderBlock(3, 2, 5, seed=47)

    with pytest.raises((TypeError, ValueError), match="inputs"):
        block.forward(bad_inputs)


def test_decoder_configuration_validation() -> None:
    with pytest.raises(ValueError, match="model_dim"):
        PreNormDecoderBlock(0, 2, 4)
    with pytest.raises(ValueError, match="key_dim"):
        PreNormDecoderBlock(3, 0, 4)
    with pytest.raises(ValueError, match="ff_hidden_dim"):
        PreNormDecoderBlock(3, 2, 0)
    with pytest.raises(ValueError, match="value_dim"):
        PreNormDecoderBlock(3, 2, 4, value_dim=0)
    with pytest.raises(TypeError, match="attention_bias"):
        PreNormDecoderBlock(3, 2, 4, attention_bias=1)
    with pytest.raises(TypeError, match="feed_forward_bias"):
        PreNormDecoderBlock(3, 2, 4, feed_forward_bias=1)
    with pytest.raises(TypeError, match="attention_output_projection"):
        PreNormDecoderBlock(3, 2, 4, attention_output_projection=1)
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        PreNormDecoderBlock(3, 2, 4, layer_norm_epsilon=0.0)
    with pytest.raises(ValueError, match="activation"):
        PreNormDecoderBlock(3, 2, 4, activation="swiglu")
    with pytest.raises(ValueError, match="non-negative"):
        PreNormDecoderBlock(3, 2, 4, seed=-1)
    with pytest.raises(TypeError, match="dtype"):
        PreNormDecoderBlock(3, 2, 4, dtype=np.float16)
