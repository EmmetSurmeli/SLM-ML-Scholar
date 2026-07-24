import json
import math
from typing import Any

import numpy as np
import pytest

from localml_scholar.nn.attention import (
    CausalSelfAttentionHead,
    MultiHeadCausalSelfAttention,
)
from localml_scholar.optim.adam import Adam
from localml_scholar.training.clipping import clip_grad_norm
from localml_scholar.training.gradient_check import check_module_gradients


def _module(**overrides: Any) -> MultiHeadCausalSelfAttention:
    values: dict[str, Any] = {
        "input_dim": 4,
        "number_of_heads": 2,
        "key_dim": 2,
        "value_dim": 3,
        "bias": True,
        "output_bias": True,
        "seed": 17,
        "dtype": np.float64,
    }
    values.update(overrides)
    return MultiHeadCausalSelfAttention(**values)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("input_dim", 0, ValueError),
        ("number_of_heads", 0, ValueError),
        ("number_of_heads", -1, ValueError),
        ("number_of_heads", 1.5, TypeError),
        ("key_dim", 0, ValueError),
        ("value_dim", 0, ValueError),
        ("bias", 1, TypeError),
        ("output_bias", 1, TypeError),
        ("seed", -1, ValueError),
        ("dtype", np.float16, TypeError),
    ],
)
def test_multi_head_configuration_rejects_invalid_values(
    field: str,
    value: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=field.replace("_dim", "")):
        _module(**{field: value})


def test_multi_head_configuration_is_complete_and_deterministic() -> None:
    module = _module(output_bias=False)

    assert module.configuration == {
        "input_dim": 4,
        "number_of_heads": 2,
        "key_dim": 2,
        "value_dim": 3,
        "bias": True,
        "output_bias": False,
        "seed": 17,
        "dtype": "float64",
    }
    assert json.loads(json.dumps(module.configuration)) == module.configuration
    assert [name for name, _ in module.named_parameters()] == [
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "key_projection.bias",
        "value_projection.weight",
        "value_projection.bias",
        "output_projection.weight",
    ]


def test_multi_head_rejects_malformed_inputs_and_upstream_gradients() -> None:
    module = _module()
    valid = np.ones((1, 2, 4), dtype=np.float64)

    with pytest.raises(TypeError, match="floating-point"):
        module.forward(np.ones((1, 2, 4), dtype=np.int64))
    with pytest.raises(TypeError, match="float32 or float64"):
        module.forward(np.ones((1, 2, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="at least 3"):
        module.forward(np.ones((2, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="exactly three"):
        module.forward(np.ones((1, 1, 2, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="final dimension"):
        module.forward(np.ones((1, 2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="non-empty"):
        module.forward(np.empty((0, 2, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        module.forward(np.full((1, 2, 4), np.nan, dtype=np.float64))

    output = module.forward(valid)
    with pytest.raises(ValueError, match="shape"):
        module.backward(np.ones((1, 1, 4), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        module.backward(np.ones_like(output, dtype=np.float32))
    module.backward(np.ones_like(output))


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "heads", "key_dim", "value_dim"),
    [
        (1, 1, 1, 1, 1),
        (1, 4, 2, 2, 3),
        (3, 2, 3, 1, 2),
    ],
)
def test_multi_head_forward_shapes_and_read_only_inspection(
    batch_size: int,
    sequence_length: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    module = _module(
        number_of_heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
    ).eval()
    inputs = np.arange(
        batch_size * sequence_length * 4,
        dtype=np.float64,
    ).reshape(batch_size, sequence_length, 4)

    output, details = module.forward(inputs, return_attention=True)

    assert details.query_flat.shape == (
        batch_size,
        sequence_length,
        heads * key_dim,
    )
    assert details.key_flat.shape == details.query_flat.shape
    assert details.value_flat.shape == (
        batch_size,
        sequence_length,
        heads * value_dim,
    )
    assert details.query.shape == (
        batch_size,
        heads,
        sequence_length,
        key_dim,
    )
    assert details.key.shape == details.query.shape
    assert details.value.shape == (
        batch_size,
        heads,
        sequence_length,
        value_dim,
    )
    assert details.scaled_scores.shape == (
        batch_size,
        heads,
        sequence_length,
        sequence_length,
    )
    assert details.allowed_mask.shape == (1, 1, sequence_length, sequence_length)
    assert details.probabilities.shape == details.scaled_scores.shape
    assert details.head_outputs.shape == details.value.shape
    assert details.concatenated.shape == (
        batch_size,
        sequence_length,
        heads * value_dim,
    )
    assert (
        details.output.shape
        == output.shape
        == (
            batch_size,
            sequence_length,
            4,
        )
    )
    assert all(
        not tensor.flags.writeable
        for tensor in (
            details.query_flat,
            details.key_flat,
            details.value_flat,
            details.query,
            details.key,
            details.value,
            details.scaled_scores,
            details.allowed_mask,
            details.probabilities,
            details.head_outputs,
            details.concatenated,
            details.output,
        )
    )


def test_multi_head_probabilities_are_independently_normalized_and_causal() -> None:
    module = _module(number_of_heads=3, key_dim=1, value_dim=2).eval()
    inputs = np.full((2, 5, 4), 1e3, dtype=np.float64)

    output, details = module.forward(inputs, return_attention=True)
    blocked = np.broadcast_to(
        ~details.allowed_mask,
        details.probabilities.shape,
    )

    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(details.probabilities))
    assert np.array_equal(
        details.probabilities[blocked],
        np.zeros(np.count_nonzero(blocked)),
    )
    assert np.allclose(
        np.sum(details.probabilities, axis=-1),
        np.ones((2, 3, 5)),
        rtol=0.0,
        atol=1e-15,
    )


def test_two_head_hand_computed_example_has_distinct_head_patterns() -> None:
    module = MultiHeadCausalSelfAttention(
        input_dim=2,
        number_of_heads=2,
        key_dim=1,
        value_dim=1,
        bias=False,
        output_bias=False,
        seed=1,
        dtype=np.float64,
    ).eval()
    identity = np.eye(2, dtype=np.float64)
    module.query_projection.weight.load_data(identity)
    module.key_projection.weight.load_data(identity)
    module.value_projection.weight.load_data(identity)
    module.output_projection.weight.load_data(identity)
    inputs = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    second_head_self = math.e / (1.0 + math.e)
    expected_probabilities = np.array(
        [
            [
                [[1.0, 0.0], [0.5, 0.5]],
                [[1.0, 0.0], [1.0 - second_head_self, second_head_self]],
            ]
        ],
        dtype=np.float64,
    )
    expected_head_outputs = np.array(
        [
            [
                [[1.0], [0.5]],
                [[0.0], [second_head_self]],
            ]
        ],
        dtype=np.float64,
    )
    expected_output = np.array(
        [[[1.0, 0.0], [0.5, second_head_self]]],
        dtype=np.float64,
    )

    output, details = module.forward(inputs, return_attention=True)

    assert np.allclose(
        details.probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.allclose(
        details.head_outputs,
        expected_head_outputs,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.allclose(details.concatenated, expected_output, atol=1e-15)
    assert np.allclose(output, expected_output, atol=1e-15)
    assert not np.array_equal(
        details.probabilities[0, 0, 1],
        details.probabilities[0, 1, 1],
    )


def _copy_one_head_parameters(
    legacy: CausalSelfAttentionHead,
    fused: MultiHeadCausalSelfAttention,
) -> None:
    for name, legacy_parameter in legacy.named_parameters():
        fused_parameter = dict(fused.named_parameters())[name]
        fused_parameter.load_data(legacy_parameter.data)


def test_one_head_matches_legacy_forward_backward_and_every_parameter() -> None:
    legacy = CausalSelfAttentionHead(
        input_dim=3,
        key_dim=2,
        value_dim=4,
        bias=True,
        output_projection=True,
        seed=3,
        dtype=np.float64,
    )
    fused = MultiHeadCausalSelfAttention(
        input_dim=3,
        number_of_heads=1,
        key_dim=2,
        value_dim=4,
        bias=True,
        output_bias=True,
        seed=9,
        dtype=np.float64,
    )
    _copy_one_head_parameters(legacy, fused)
    inputs = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 7.0
    upstream = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 11.0

    legacy_output, legacy_details = legacy.forward(inputs, return_attention=True)
    fused_output, fused_details = fused.forward(inputs, return_attention=True)
    legacy_grad_input = legacy.backward(upstream)
    fused_grad_input = fused.backward(upstream)

    assert np.array_equal(fused_details.query[:, 0], legacy_details.query)
    assert np.array_equal(fused_details.key[:, 0], legacy_details.key)
    assert np.array_equal(fused_details.value[:, 0], legacy_details.value)
    assert np.array_equal(
        fused_details.scaled_scores[:, 0],
        legacy_details.scaled_scores,
    )
    assert np.array_equal(
        fused_details.allowed_mask[:, 0],
        legacy_details.allowed_mask,
    )
    assert np.array_equal(
        fused_details.probabilities[:, 0],
        legacy_details.probabilities,
    )
    legacy_context = legacy_details.probabilities @ legacy_details.value
    assert np.array_equal(fused_details.head_outputs[:, 0], legacy_context)
    assert np.array_equal(fused_details.concatenated, legacy_context)
    assert np.array_equal(fused_output, legacy_output)
    assert np.array_equal(fused_grad_input, legacy_grad_input)
    assert np.array_equal(
        fused.last_score_gradient[:, 0],
        legacy.last_score_gradient,
    )
    for (legacy_name, legacy_parameter), (fused_name, fused_parameter) in zip(
        legacy.named_parameters(),
        fused.named_parameters(),
        strict=True,
    ):
        assert fused_name == legacy_name
        assert np.array_equal(fused_parameter.grad, legacy_parameter.grad)


def test_multi_head_matches_independent_per_head_numpy_reference() -> None:
    module = _module(
        input_dim=3,
        number_of_heads=2,
        key_dim=2,
        value_dim=1,
        seed=23,
    ).eval()
    inputs = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 5.0

    output, details = module.forward(inputs, return_attention=True)

    query_flat = (
        inputs @ module.query_projection.weight.data + module.query_projection.bias.data
    )
    key_flat = (
        inputs @ module.key_projection.weight.data + module.key_projection.bias.data
    )
    value_flat = (
        inputs @ module.value_projection.weight.data + module.value_projection.bias.data
    )
    reference_scores = np.empty((2, 2, 3, 3), dtype=np.float64)
    reference_probabilities = np.zeros_like(reference_scores)
    reference_heads = np.empty((2, 2, 3, 1), dtype=np.float64)
    for batch in range(2):
        for head in range(2):
            query = query_flat[batch, :, head * 2 : (head + 1) * 2]
            key = key_flat[batch, :, head * 2 : (head + 1) * 2]
            value = value_flat[batch, :, head : head + 1]
            scores = query @ key.T / math.sqrt(2.0)
            reference_scores[batch, head] = scores
            for row in range(3):
                valid = scores[row, : row + 1]
                shifted = valid - np.max(valid)
                probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
                reference_probabilities[batch, head, row, : row + 1] = probabilities
            reference_heads[batch, head] = reference_probabilities[batch, head] @ value
    reference_concatenated = reference_heads.transpose(0, 2, 1, 3).reshape(
        2,
        3,
        2,
    )
    reference_output = (
        reference_concatenated @ module.output_projection.weight.data
        + module.output_projection.bias.data
    )

    assert np.allclose(details.scaled_scores, reference_scores, atol=1e-15)
    assert np.allclose(
        details.probabilities,
        reference_probabilities,
        atol=1e-15,
    )
    assert np.allclose(details.head_outputs, reference_heads, atol=1e-15)
    assert np.allclose(
        details.concatenated,
        reference_concatenated,
        atol=1e-15,
    )
    assert np.allclose(output, reference_output, atol=1e-15)


@pytest.mark.parametrize(
    ("heads", "key_dim", "value_dim", "batch_size"),
    [(1, 2, 1, 1), (2, 1, 2, 2)],
)
def test_multi_head_input_and_parameter_gradients_match_finite_differences(
    heads: int,
    key_dim: int,
    value_dim: int,
    batch_size: int,
) -> None:
    module = MultiHeadCausalSelfAttention(
        input_dim=2,
        number_of_heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        bias=True,
        output_bias=True,
        seed=29,
        dtype=np.float64,
    )
    inputs = np.arange(batch_size * 4, dtype=np.float64).reshape(batch_size, 2, 2) / 9.0
    upstream = (
        np.arange(batch_size * 4, dtype=np.float64).reshape(batch_size, 2, 2) / 7.0
    )

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    result = check_module_gradients(
        module,
        inputs,
        objective,
        check_input=True,
        absolute_tolerance=2e-6,
        relative_tolerance=3e-5,
    )

    assert result.passed
    assert result.checked_coordinates == inputs.size + module.parameter_count


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_multi_head_forward_and_backward_are_causal(heads: int) -> None:
    module = _module(
        number_of_heads=heads,
        key_dim=1,
        value_dim=1,
        seed=31,
    )
    original = np.arange(16, dtype=np.float64).reshape(1, 4, 4) / 10.0
    modified = original.copy()
    modified[:, 3, :] += 7.0

    with module.inference_mode():
        original_output = module.forward(original)
        modified_output = module.forward(modified)
    assert np.array_equal(original_output[:, :3], modified_output[:, :3])

    output, details = module.forward(original, return_attention=True)
    upstream = np.zeros_like(output)
    upstream[:, 1, :] = 1.0
    grad_input = module.backward(upstream)
    blocked = np.broadcast_to(
        ~details.allowed_mask,
        details.probabilities.shape,
    )
    assert np.array_equal(grad_input[:, 2:, :], np.zeros_like(grad_input[:, 2:, :]))
    assert np.array_equal(
        module.last_score_gradient[blocked],
        np.zeros(np.count_nonzero(blocked)),
    )


def test_multi_head_lifecycle_modes_optimizer_clipping_and_checkpoint(tmp_path) -> None:
    module = _module(dtype=np.float32, seed=37)
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / np.float32(10)
    output = module.forward(inputs)
    with pytest.raises(RuntimeError, match="cannot run twice"):
        module.forward(inputs)
    module.backward(np.ones_like(output))
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        module.backward(np.ones_like(output))

    with module.inference_mode():
        first = module.forward(inputs)
        second = module.forward(inputs)
        assert np.array_equal(first, second)
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        module.backward(np.ones_like(first))

    optimizer = Adam(module.parameters(), learning_rate=0.01)
    before = [parameter.data.copy() for parameter in module.parameters()]
    optimizer.zero_grad()
    output = module.forward(inputs)
    module.backward(np.ones_like(output))
    norm = clip_grad_norm(module.parameters(), 0.1)
    optimizer.step()
    assert norm > 0.1
    assert any(
        not np.array_equal(old, parameter.data)
        for old, parameter in zip(before, module.parameters(), strict=True)
    )

    expected = module.eval().forward(inputs)
    path = module.save_checkpoint(tmp_path / "multi_head.npz")
    loaded = MultiHeadCausalSelfAttention.load_checkpoint(path).eval()
    assert loaded.configuration == module.configuration
    assert np.array_equal(loaded.forward(inputs), expected)
