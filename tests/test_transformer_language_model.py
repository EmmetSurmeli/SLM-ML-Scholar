import json
from typing import Any

import numpy as np
import pytest

from localml_scholar.losses import softmax_cross_entropy_loss_and_gradient
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.nn.transformer import PreNormDecoderBlock
from localml_scholar.optim.adam import Adam
from localml_scholar.optim.momentum import Momentum
from localml_scholar.optim.sgd import SGD
from localml_scholar.training.clipping import clip_grad_norm
from localml_scholar.training.gradient_check import check_module_gradients


def _config(**overrides: Any) -> TransformerConfig:
    values: dict[str, Any] = {
        "vocabulary_size": 7,
        "maximum_context_length": 5,
        "model_dimension": 4,
        "number_of_layers": 2,
        "key_dimension": 2,
        "value_dimension": 3,
        "feed_forward_dimension": 7,
        "layer_norm_epsilon": 1e-5,
        "attention_bias": True,
        "feed_forward_bias": True,
        "vocabulary_bias": True,
        "dtype": np.float64,
        "seed": 17,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def _make_decoder_stack_identity(model: TransformerLanguageModel) -> None:
    for module in model.decoder_blocks:
        assert isinstance(module, PreNormDecoderBlock)
        output_projection = module.attention.output_projection
        assert output_projection is not None
        output_projection.weight.data.fill(0.0)
        if output_projection.bias is not None:
            output_projection.bias.data.fill(0.0)
        module.feed_forward.linear2.weight.data.fill(0.0)
        if module.feed_forward.linear2.bias is not None:
            module.feed_forward.linear2.bias.data.fill(0.0)


def test_transformer_config_round_trip_and_dtype_normalization() -> None:
    config = _config(dtype="float32")

    restored = TransformerConfig.from_dict(config.to_dict())

    assert restored == config
    assert config.dtype == np.dtype(np.float32)
    assert config.to_dict()["dtype"] == "float32"


def test_transformer_config_preserves_legacy_positional_optional_arguments() -> None:
    config = TransformerConfig(
        7,
        5,
        4,
        2,
        2,
        3,
        7,
        2e-5,
        False,
        False,
        False,
        np.float32,
        19,
    )

    assert config.number_of_heads == 1
    assert config.layer_norm_epsilon == 2e-5
    assert config.seed == 19


@pytest.mark.parametrize(
    "field",
    [
        "vocabulary_size",
        "maximum_context_length",
        "model_dimension",
        "number_of_layers",
        "key_dimension",
        "value_dimension",
        "feed_forward_dimension",
        "number_of_heads",
    ],
)
def test_transformer_config_rejects_nonpositive_dimensions(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: 0})


def test_transformer_config_rejects_malformed_values() -> None:
    with pytest.raises(TypeError, match="vocabulary_size"):
        _config(vocabulary_size=3.5)
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        _config(layer_norm_epsilon=0.0)
    with pytest.raises(TypeError, match="attention_bias"):
        _config(attention_bias=1)
    with pytest.raises(TypeError, match="feed_forward_bias"):
        _config(feed_forward_bias=1)
    with pytest.raises(TypeError, match="vocabulary_bias"):
        _config(vocabulary_bias=1)
    with pytest.raises(TypeError, match="dtype"):
        _config(dtype=np.float16)
    with pytest.raises(ValueError, match="non-negative"):
        _config(seed=-1)
    with pytest.raises(ValueError, match="configuration keys"):
        TransformerConfig.from_dict({"vocabulary_size": 3})


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "number_of_layers"),
    [(1, 1, 1), (1, 5, 2), (3, 2, 3)],
)
def test_transformer_forward_returns_sequence_vocabulary_logits(
    batch_size: int,
    sequence_length: int,
    number_of_layers: int,
) -> None:
    config = _config(number_of_layers=number_of_layers)
    model = TransformerLanguageModel(config).eval()
    token_ids = (
        np.arange(batch_size * sequence_length, dtype=np.int64).reshape(
            batch_size, sequence_length
        )
        % config.vocabulary_size
    )

    logits = model.forward(token_ids)

    assert logits.shape == (
        batch_size,
        sequence_length,
        config.vocabulary_size,
    )
    assert logits.dtype == config.dtype
    assert np.all(np.isfinite(logits))


def test_transformer_matches_controlled_embedding_norm_and_projection() -> None:
    config = _config(
        vocabulary_size=3,
        maximum_context_length=2,
        model_dimension=2,
        number_of_layers=1,
        key_dimension=1,
        value_dimension=1,
        feed_forward_dimension=2,
        vocabulary_bias=True,
        seed=19,
    )
    model = TransformerLanguageModel(config).eval()
    _make_decoder_stack_identity(model)
    model.token_embedding.weight.data[...] = np.array(
        [[1.0, 3.0], [2.0, 0.0], [-1.0, 0.5]],
        dtype=np.float64,
    )
    model.position_embedding.weight.data[...] = np.array(
        [[0.0, 0.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    model.language_model_head.weight.data[...] = np.array(
        [[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]],
        dtype=np.float64,
    )
    assert model.language_model_head.bias is not None
    model.language_model_head.bias.data[...] = np.array(
        [0.1, -0.2, 0.3],
        dtype=np.float64,
    )
    token_ids = np.array([[0, 1]], dtype=np.int64)
    combined = np.array([[[1.0, 3.0], [3.0, 0.0]]], dtype=np.float64)
    mean = np.mean(combined, axis=-1, keepdims=True)
    centered = combined - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / np.sqrt(variance + config.layer_norm_epsilon)
    expected = (
        normalized @ model.language_model_head.weight.data
        + model.language_model_head.bias.data
    )

    logits = model.forward(token_ids)

    assert np.allclose(logits, expected, rtol=0.0, atol=1e-15)
    assert not np.allclose(np.sum(logits, axis=-1), 1.0)


def test_transformer_uses_independent_deterministic_child_initialization() -> None:
    first = TransformerLanguageModel(_config(seed=23, number_of_layers=3))
    second = TransformerLanguageModel(_config(seed=23, number_of_layers=3))
    third = TransformerLanguageModel(_config(seed=24, number_of_layers=3))

    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        assert np.array_equal(first_parameter.data, second_parameter.data)
    assert any(
        not np.array_equal(first_parameter.data, third_parameter.data)
        for first_parameter, third_parameter in zip(
            first.parameters(),
            third.parameters(),
            strict=True,
        )
    )

    blocks = tuple(first.decoder_blocks)
    assert len(blocks) == 3
    assert len({id(block) for block in blocks}) == 3
    first_weights = [
        block.attention.query_projection.weight.data
        for block in blocks
        if isinstance(block, PreNormDecoderBlock)
    ]
    assert len(first_weights) == 3
    assert all(
        not np.array_equal(first_weights[index], first_weights[index + 1])
        for index in range(len(first_weights) - 1)
    )
    assert len({id(parameter) for parameter in first.parameters()}) == len(
        first.parameters()
    )


def test_transformer_parameter_order_and_weight_untiedness() -> None:
    model = TransformerLanguageModel(_config(number_of_layers=2))
    names = [name for name, _ in model.named_parameters()]

    assert names[:4] == [
        "token_embedding.weight",
        "position_embedding.weight",
        "decoder_blocks.0.norm1.gamma",
        "decoder_blocks.0.norm1.beta",
    ]
    assert "decoder_blocks.1.norm1.gamma" in names
    assert names[-4:] == [
        "final_layer_norm.gamma",
        "final_layer_norm.beta",
        "language_model_head.weight",
        "language_model_head.bias",
    ]
    assert len(names) == len(set(names))
    assert model.token_embedding.weight is not model.language_model_head.weight
    assert not np.shares_memory(
        model.token_embedding.weight.data,
        model.language_model_head.weight.data,
    )


def test_position_and_repeated_token_gradients_accumulate_across_batch() -> None:
    config = _config(
        vocabulary_size=4,
        maximum_context_length=5,
        model_dimension=3,
        number_of_layers=1,
        key_dimension=2,
        value_dimension=2,
        feed_forward_dimension=4,
        seed=29,
    )
    single = TransformerLanguageModel(config)
    doubled = TransformerLanguageModel(config)
    single_ids = np.array([[0, 1, 0]], dtype=np.int64)
    doubled_ids = np.repeat(single_ids, 2, axis=0)
    single_output = single.forward(single_ids)
    doubled.forward(doubled_ids)
    single_upstream = (
        np.arange(
            single_output.size,
            dtype=np.float64,
        ).reshape(single_output.shape)
        / 10.0
    )
    doubled_upstream = np.repeat(single_upstream, 2, axis=0)

    single.backward(single_upstream)
    doubled.backward(doubled_upstream)

    assert np.allclose(
        doubled.position_embedding.weight.grad,
        2.0 * single.position_embedding.weight.grad,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        doubled.token_embedding.weight.grad,
        2.0 * single.token_embedding.weight.grad,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.array_equal(
        doubled.position_embedding.weight.grad[3:],
        np.zeros_like(doubled.position_embedding.weight.grad[3:]),
    )


def test_transformer_forward_is_causal_across_multiple_blocks() -> None:
    model = TransformerLanguageModel(_config(number_of_layers=3, seed=31)).eval()
    original = np.array([[0, 1, 2, 3]], dtype=np.int64)
    modified = original.copy()
    modified[:, 3] = 6

    original_logits = model.forward(original)
    modified_logits = model.forward(modified)

    assert np.array_equal(original_logits[:, :3, :], modified_logits[:, :3, :])
    assert not np.allclose(original_logits[:, 3, :], modified_logits[:, 3, :])


def test_earlier_logit_loss_has_zero_future_embedding_gradients() -> None:
    config = _config(vocabulary_size=7, number_of_layers=2, seed=37)
    model = TransformerLanguageModel(config)
    token_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    logits = model.forward(token_ids)
    grad_logits = np.zeros_like(logits)
    grad_logits[:, 0, :] = np.array(
        [0.5, -1.0, 0.7, 0.2, -0.4, 0.8, -0.3],
        dtype=np.float64,
    )

    result = model.backward(grad_logits)

    assert result is None
    assert np.count_nonzero(model.token_embedding.weight.grad[0]) > 0
    assert np.array_equal(
        model.token_embedding.weight.grad[1:4],
        np.zeros((3, config.model_dimension)),
    )
    assert np.count_nonzero(model.position_embedding.weight.grad[0]) > 0
    assert np.array_equal(
        model.position_embedding.weight.grad[1:],
        np.zeros_like(model.position_embedding.weight.grad[1:]),
    )


def test_logits_integrate_with_existing_cross_entropy_and_backward() -> None:
    model = TransformerLanguageModel(_config(dtype=np.float32, seed=41))
    token_ids = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    targets = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)

    logits = model.forward(token_ids)
    loss, grad_logits = softmax_cross_entropy_loss_and_gradient(logits, targets)
    result = model.backward(grad_logits)

    assert np.isfinite(loss)
    assert grad_logits.shape == logits.shape
    assert grad_logits.dtype == np.float32
    assert result is None
    assert not model.has_pending_cache()
    assert all(
        parameter.grad.shape == parameter.data.shape for parameter in model.parameters()
    )


def test_transformer_parameter_gradients_match_finite_differences() -> None:
    config = TransformerConfig(
        vocabulary_size=3,
        maximum_context_length=2,
        model_dimension=2,
        number_of_layers=1,
        key_dimension=1,
        value_dimension=1,
        feed_forward_dimension=2,
        dtype=np.float64,
        seed=43,
    )
    model = TransformerLanguageModel(config)
    token_ids = np.array([[0, 1]], dtype=np.int64)
    weights = np.arange(6, dtype=np.float64).reshape(1, 2, 3) / 10.0

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * weights)), weights.copy()

    result = check_module_gradients(
        model,
        token_ids,
        objective,
        check_input=False,
        absolute_tolerance=2e-6,
        relative_tolerance=8e-5,
    )

    assert result.passed
    assert result.checked_coordinates == model.parameter_count
    assert [tensor.name for tensor in result.tensors] == [
        name for name, _ in model.named_parameters()
    ]


def test_state_dict_is_deterministic_copied_and_exactly_loadable() -> None:
    original = TransformerLanguageModel(_config(seed=47))
    restored = TransformerLanguageModel(_config(seed=48))
    state = original.state_dict()

    assert list(state) == [name for name, _ in original.named_parameters()]
    first_name = next(iter(state))
    snapshot = original.state_dict()[first_name]
    state[first_name].fill(999.0)
    assert np.array_equal(original.state_dict()[first_name], snapshot)

    restored.load_state_dict(original.state_dict())

    for original_parameter, restored_parameter in zip(
        original.parameters(),
        restored.parameters(),
        strict=True,
    ):
        assert np.array_equal(original_parameter.data, restored_parameter.data)


def test_load_state_dict_rejects_bad_state_without_partial_mutation() -> None:
    model = TransformerLanguageModel(_config(seed=53))
    before = model.state_dict()
    malformed = model.state_dict()
    names = list(malformed)
    malformed[names[0]] = malformed[names[0]] + 1.0
    malformed[names[-1]] = malformed[names[-1]][:-1]

    with pytest.raises(ValueError, match="shape"):
        model.load_state_dict(malformed)

    after = model.state_dict()
    assert all(np.array_equal(before[name], after[name]) for name in before)
    missing = dict(before)
    missing.pop(names[0])
    with pytest.raises(ValueError, match="keys"):
        model.load_state_dict(missing)


def test_load_state_dict_rejects_pending_forward_cache() -> None:
    model = TransformerLanguageModel(_config(seed=57))
    state = model.state_dict()
    token_ids = np.array([[0, 1]], dtype=np.int64)
    logits = model.forward(token_ids)

    with pytest.raises(RuntimeError, match="cache is pending"):
        model.load_state_dict(state)

    model.backward(np.ones_like(logits))


def test_transformer_checkpoint_preserves_float32_logits_and_config(tmp_path) -> None:
    config = _config(dtype=np.float32, number_of_layers=2, seed=59)
    model = TransformerLanguageModel(config).eval()
    token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    expected = model.forward(token_ids)
    checkpoint = tmp_path / "transformer.npz"

    model.save_checkpoint(checkpoint)
    loaded = TransformerLanguageModel.load_checkpoint(checkpoint)

    assert loaded.training
    assert loaded.config == config
    assert loaded.config.to_dict() == config.to_dict()
    assert list(loaded.state_dict()) == list(model.state_dict())
    assert np.array_equal(loaded.eval().forward(token_ids), expected)


def test_transformer_checkpoint_rejects_incompatible_model_version(tmp_path) -> None:
    model = TransformerLanguageModel(_config())
    original = tmp_path / "original.npz"
    malformed = tmp_path / "malformed.npz"
    model.save_checkpoint(original)
    with np.load(original, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        metadata["model_version"] = "9.9.9"
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
        TransformerLanguageModel.load_checkpoint(malformed)


@pytest.mark.parametrize("optimizer_name", ["sgd", "momentum", "adam"])
def test_transformer_optimizers_and_clipping_support_repeated_cycles(
    optimizer_name: str,
) -> None:
    model = TransformerLanguageModel(
        _config(dtype=np.float32, number_of_layers=1, seed=61)
    )
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), learning_rate=0.01)
    elif optimizer_name == "momentum":
        optimizer = Momentum(model.parameters(), learning_rate=0.01, beta=0.8)
    else:
        optimizer = Adam(model.parameters(), learning_rate=0.01)
    token_ids = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    targets = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
    before = [parameter.data.copy() for parameter in model.parameters()]

    for _ in range(2):
        optimizer.zero_grad()
        logits = model.forward(token_ids)
        _, grad_logits = softmax_cross_entropy_loss_and_gradient(logits, targets)
        model.backward(grad_logits)
        norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()

    assert np.isfinite(norm) and norm > 0.0
    assert any(
        not np.array_equal(previous, parameter.data)
        for previous, parameter in zip(
            before,
            model.parameters(),
            strict=True,
        )
    )
    assert not model.has_pending_cache()


def test_transformer_modes_and_zero_grad_propagate_recursively() -> None:
    model = TransformerLanguageModel(_config(number_of_layers=2))
    for parameter in model.parameters():
        parameter.grad.fill(3.0)

    model.eval()

    assert not model.training
    assert all(not module.training for module in model.modules())
    assert all(not block.training for block in model.decoder_blocks)
    model.zero_grad()
    assert all(
        np.array_equal(parameter.grad, np.zeros_like(parameter.grad))
        for parameter in model.parameters()
    )
    model.train()
    assert model.training
    assert all(module.training for module in model.modules())
    assert all(block.training for block in model.decoder_blocks)


def test_transformer_cache_and_upstream_validation_are_explicit() -> None:
    model = TransformerLanguageModel(_config(seed=67))
    token_ids = np.array([[0, 1, 2]], dtype=np.int64)
    logits = model.forward(token_ids)

    with pytest.raises(RuntimeError, match="cannot run twice"):
        model.forward(token_ids)
    with pytest.raises(ValueError, match="grad_logits shape"):
        model.backward(np.ones((1, 2, 7), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        model.backward(np.ones(logits.shape, dtype=np.float32))
    malformed = np.ones_like(logits)
    malformed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.backward(malformed)
    model.backward(np.ones_like(logits))
    assert not model.has_pending_cache()
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        model.backward(np.ones_like(logits))


@pytest.mark.parametrize(
    "bad_token_ids",
    [
        np.array([0, 1], dtype=np.int64),
        np.zeros((1, 2, 3), dtype=np.int64),
        np.zeros((0, 2), dtype=np.int64),
        np.zeros((1, 0), dtype=np.int64),
        np.array([[0.0, 1.0]], dtype=np.float64),
        np.array([[-1, 0]], dtype=np.int64),
        np.array([[0, 7]], dtype=np.int64),
    ],
)
def test_transformer_rejects_malformed_token_ids(
    bad_token_ids: np.ndarray,
) -> None:
    model = TransformerLanguageModel(_config())

    with pytest.raises((TypeError, ValueError), match="token_ids"):
        model.forward(bad_token_ids)


def test_transformer_rejects_sequence_beyond_context_and_nonconfig() -> None:
    model = TransformerLanguageModel(_config(maximum_context_length=3))

    with pytest.raises(ValueError, match="exceeds maximum"):
        model.forward(np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(TypeError, match="TransformerConfig"):
        TransformerLanguageModel({})  # type: ignore[arg-type]
