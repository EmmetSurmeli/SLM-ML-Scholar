import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from localml_scholar.generation import generate_transformer_ids
from localml_scholar.losses import softmax_cross_entropy_loss_and_gradient
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.nn.attention import CausalSelfAttentionHead
from localml_scholar.nn.transformer import PreNormDecoderBlock
from localml_scholar.tokenizer import CharacterTokenizer
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.training.gradient_check import check_module_gradients
from localml_scholar.training.transformer import TransformerTrainer


def _model_config(**overrides: Any) -> TransformerConfig:
    values: dict[str, Any] = {
        "vocabulary_size": 5,
        "maximum_context_length": 4,
        "model_dimension": 4,
        "number_of_layers": 1,
        "key_dimension": 2,
        "value_dimension": 2,
        "feed_forward_dimension": 6,
        "number_of_heads": 2,
        "dtype": np.float64,
        "seed": 71,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def _rewrite_npz_metadata(
    source: Path,
    destination: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    with np.load(source, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        arrays = {
            name: np.array(checkpoint[name], copy=True)
            for name in checkpoint.files
            if name != "metadata_json"
        }
    mutate(metadata)
    np.savez(
        destination,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


def test_two_head_decoder_shapes_causality_and_checkpoint(tmp_path: Path) -> None:
    block = PreNormDecoderBlock(
        model_dim=4,
        number_of_heads=2,
        key_dim=2,
        value_dim=3,
        ff_hidden_dim=7,
        seed=73,
        dtype=np.float64,
    ).eval()
    inputs = np.arange(48, dtype=np.float64).reshape(3, 4, 4) / 17.0

    output, details = block.forward(inputs, return_details=True)

    assert output.shape == inputs.shape
    assert details.attention.query.shape == (3, 2, 4, 2)
    assert details.attention.value.shape == (3, 2, 4, 3)
    assert details.attention.probabilities.shape == (3, 2, 4, 4)
    changed = inputs.copy()
    changed[:, 3, :] += 100.0
    assert np.array_equal(block.forward(inputs)[:, :3], block.forward(changed)[:, :3])

    checkpoint = block.save_checkpoint(tmp_path / "decoder_multi_head.npz")
    loaded = PreNormDecoderBlock.load_checkpoint(checkpoint).eval()
    assert loaded.configuration == block.configuration
    assert np.array_equal(loaded.forward(inputs), output)


def test_two_head_decoder_gradients_match_finite_differences() -> None:
    block = PreNormDecoderBlock(
        model_dim=2,
        number_of_heads=2,
        key_dim=1,
        value_dim=1,
        ff_hidden_dim=2,
        seed=79,
        dtype=np.float64,
    )
    inputs = np.array([[[0.2, -0.3], [0.7, 0.1]]], dtype=np.float64)
    upstream = np.array([[[0.4, -0.2], [0.3, 0.8]]], dtype=np.float64)

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    result = check_module_gradients(
        block,
        inputs,
        objective,
        absolute_tolerance=2e-6,
        relative_tolerance=6e-5,
    )

    assert result.passed
    assert result.checked_coordinates == inputs.size + block.parameter_count


def test_one_head_decoder_matches_equivalent_legacy_attention_path() -> None:
    block = PreNormDecoderBlock(
        model_dim=3,
        number_of_heads=1,
        key_dim=2,
        value_dim=2,
        ff_hidden_dim=5,
        seed=83,
        dtype=np.float64,
    ).eval()
    legacy_attention = CausalSelfAttentionHead(
        input_dim=3,
        key_dim=2,
        value_dim=2,
        bias=True,
        output_projection=True,
        seed=0,
        dtype=np.float64,
    ).eval()
    fused_parameters = dict(block.attention.named_parameters())
    for name, parameter in legacy_attention.named_parameters():
        parameter.load_data(fused_parameters[name].data)
    inputs = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 13.0

    actual = block.forward(inputs)
    normalized_attention = block.norm1.forward(inputs)
    legacy_attention_output = legacy_attention.forward(normalized_attention)
    first_residual = inputs + legacy_attention_output
    normalized_feed_forward = block.norm2.forward(first_residual)
    expected = first_residual + block.feed_forward.forward(normalized_feed_forward)

    assert np.array_equal(actual, expected)


def test_multi_head_language_model_trains_and_generates_deterministically() -> None:
    model = TransformerLanguageModel(_model_config())
    token_ids = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    targets = np.array([[1, 2, 3], [3, 4, 0]], dtype=np.int64)

    logits = model.forward(token_ids)
    loss, grad_logits = softmax_cross_entropy_loss_and_gradient(logits, targets)
    model.backward(grad_logits)

    assert logits.shape == (2, 3, 5)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(parameter.grad)) for parameter in model.parameters())
    prompt = np.array([[0, 1]], dtype=np.int64)
    first = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=5,
        temperature=0.8,
        top_k=3,
        seed=89,
    )
    second = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=5,
        temperature=0.8,
        top_k=3,
        seed=89,
    )
    assert np.array_equal(first, second)


def test_legacy_model_checkpoint_migrates_to_one_head_exactly(
    tmp_path: Path,
) -> None:
    model = TransformerLanguageModel(_model_config(number_of_heads=1, seed=97)).eval()
    token_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    expected = model.forward(token_ids)
    current = model.save_checkpoint(tmp_path / "current.npz")
    legacy = tmp_path / "legacy_0_5_0.npz"

    def make_legacy(metadata: dict[str, Any]) -> None:
        metadata["checkpoint_version"] = 1
        metadata["model_version"] = "0.5.0"
        metadata["configuration"].pop("number_of_heads")

    _rewrite_npz_metadata(current, legacy, make_legacy)
    loaded = TransformerLanguageModel.load_checkpoint(legacy).eval()

    assert loaded.config.number_of_heads == 1
    assert np.array_equal(loaded.forward(token_ids), expected)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            loaded.state_dict().values(),
            model.state_dict().values(),
            strict=True,
        )
    )


def _training_components(
    tmp_path: Path,
    *,
    number_of_heads: int,
) -> tuple[
    CharacterTokenizer,
    np.ndarray,
    np.ndarray,
    TransformerConfig,
    TransformerTrainingConfig,
]:
    tokenizer = CharacterTokenizer("abcde")
    tokens = tokenizer.encode("abcde" * 8)
    model_config = _model_config(
        number_of_heads=number_of_heads,
        seed=101,
    )
    training_config = TransformerTrainingConfig(
        batch_size=2,
        sequence_length=3,
        maximum_steps=4,
        evaluation_interval=1,
        evaluation_batches=1,
        checkpoint_interval=2,
        logging_interval=1,
        optimizer_name="adam",
        learning_rate=0.01,
        maximum_gradient_norm=0.7,
        seed=103,
        output_directory=str(tmp_path),
    )
    return (
        tokenizer,
        tokens[:25],
        tokens[5:],
        model_config,
        training_config,
    )


def _trainer(tmp_path: Path, number_of_heads: int) -> TransformerTrainer:
    tokenizer, train, validation, model_config, training_config = _training_components(
        tmp_path, number_of_heads=number_of_heads
    )
    return TransformerTrainer(
        TransformerLanguageModel(model_config),
        tokenizer,
        train,
        validation,
        training_config,
    )


def _advance(trainer: TransformerTrainer, steps: int) -> list[float]:
    losses: list[float] = []
    for _ in range(steps):
        metrics = trainer.train_step()
        losses.append(metrics.loss)
        trainer.record_evaluation(metrics, trainer.evaluate())
    return losses


def test_two_head_training_resume_is_bitwise_exact(tmp_path: Path) -> None:
    uninterrupted = _trainer(tmp_path / "uninterrupted", 2)
    expected_losses = _advance(uninterrupted, 4)

    interrupted = _trainer(tmp_path / "interrupted", 2)
    actual_losses = _advance(interrupted, 2)
    checkpoint = interrupted.save_checkpoint(tmp_path / "multi_head_resume.npz")
    tokenizer, train, validation, model_config, _ = _training_components(
        tmp_path / "interrupted",
        number_of_heads=2,
    )
    resumed = TransformerTrainer.load_checkpoint(
        checkpoint,
        train_tokens=train,
        validation_tokens=validation,
        tokenizer=tokenizer,
        expected_model_config=model_config,
    )
    actual_losses.extend(_advance(resumed, 2))

    assert actual_losses == expected_losses
    assert resumed.history == uninterrupted.history
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            resumed.model.state_dict().values(),
            uninterrupted.model.state_dict().values(),
            strict=True,
        )
    )
    resumed_optimizer = resumed.optimizer.state_dict()
    uninterrupted_optimizer = uninterrupted.optimizer.state_dict()
    resumed_arrays = resumed_optimizer.pop("arrays")
    uninterrupted_arrays = uninterrupted_optimizer.pop("arrays")
    assert resumed_optimizer == uninterrupted_optimizer
    assert resumed_arrays.keys() == uninterrupted_arrays.keys()
    assert all(
        np.array_equal(resumed_arrays[name], uninterrupted_arrays[name])
        for name in resumed_arrays
    )


def test_legacy_training_checkpoint_migrates_and_rejects_head_mismatch(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path / "legacy", 1)
    _advance(trainer, 2)
    current = trainer.save_checkpoint(tmp_path / "current_training.npz")
    legacy = tmp_path / "legacy_0_5_1_training.npz"

    def make_legacy(metadata: dict[str, Any]) -> None:
        metadata["checkpoint_version"] = 1
        metadata["package_version"] = "0.5.1"
        metadata["model_configuration"].pop("number_of_heads")

    _rewrite_npz_metadata(current, legacy, make_legacy)
    tokenizer, train, validation, one_head_config, training_config = (
        _training_components(tmp_path / "legacy", number_of_heads=1)
    )
    restored = TransformerTrainer.load_checkpoint(
        legacy,
        train_tokens=train,
        validation_tokens=validation,
        tokenizer=tokenizer,
        expected_model_config=one_head_config,
        expected_training_config=training_config,
    )

    assert restored.model.config.number_of_heads == 1
    assert restored.completed_steps == trainer.completed_steps
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            restored.model.state_dict().values(),
            trainer.model.state_dict().values(),
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="configuration"):
        TransformerTrainer.load_checkpoint(
            legacy,
            train_tokens=train,
            validation_tokens=validation,
            tokenizer=tokenizer,
            expected_model_config=_model_config(number_of_heads=2, seed=101),
        )
