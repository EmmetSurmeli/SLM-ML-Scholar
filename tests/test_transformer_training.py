import copy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from experiments.overfit_tiny_transformer import run_experiment
from localml_scholar.data import SequenceBatchSampler
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.tokenizer import CharacterTokenizer
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.training.transformer import (
    TransformerTrainer,
    evaluate_language_model,
)


def _model_config(**overrides: Any) -> TransformerConfig:
    values: dict[str, Any] = {
        "vocabulary_size": 3,
        "maximum_context_length": 4,
        "model_dimension": 4,
        "number_of_layers": 1,
        "key_dimension": 2,
        "value_dimension": 2,
        "feed_forward_dimension": 7,
        "dtype": np.float64,
        "seed": 31,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def _training_config(tmp_path, **overrides: Any) -> TransformerTrainingConfig:
    values: dict[str, Any] = {
        "batch_size": 3,
        "sequence_length": 3,
        "maximum_steps": 8,
        "evaluation_interval": 2,
        "evaluation_batches": 2,
        "checkpoint_interval": 4,
        "logging_interval": 1,
        "optimizer_name": "adam",
        "learning_rate": 0.01,
        "maximum_gradient_norm": 0.5,
        "seed": 19,
        "output_directory": str(tmp_path),
    }
    values.update(overrides)
    return TransformerTrainingConfig(**values)


def _trainer(tmp_path, **training_overrides: Any) -> TransformerTrainer:
    tokenizer = CharacterTokenizer("abc")
    tokens = tokenizer.encode("abcabcabcabcabcabc")
    return TransformerTrainer(
        TransformerLanguageModel(_model_config()),
        tokenizer,
        tokens[:12],
        tokens[3:],
        _training_config(tmp_path, **training_overrides),
    )


def test_training_config_round_trip_and_context_validation(tmp_path) -> None:
    config = _training_config(tmp_path)

    restored = TransformerTrainingConfig.from_dict(config.to_dict())

    assert restored == config
    config.validate_for_context(3)
    with pytest.raises(ValueError, match="exceeds"):
        config.validate_for_context(2)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("batch_size", 0, ValueError),
        ("sequence_length", 0, ValueError),
        ("maximum_steps", 0, ValueError),
        ("evaluation_batches", 0, ValueError),
        ("optimizer_name", "rmsprop", ValueError),
        ("learning_rate", 0.0, ValueError),
        ("adam_beta1", 1.0, ValueError),
        ("optimizer_epsilon", 0.0, ValueError),
        ("weight_decay", -0.1, ValueError),
        ("maximum_gradient_norm", 0.0, ValueError),
        ("seed", -1, ValueError),
    ],
)
def test_training_config_rejects_invalid_values(
    tmp_path,
    field: str,
    value: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=field):
        _training_config(tmp_path, **{field: value})


@pytest.mark.parametrize("optimizer_name", ["sgd", "momentum", "adam"])
def test_repeated_training_steps_update_parameters_and_clear_caches(
    tmp_path,
    optimizer_name: str,
) -> None:
    trainer = _trainer(
        tmp_path,
        optimizer_name=optimizer_name,
        maximum_gradient_norm=0.05,
    )
    before = trainer.model.state_dict()

    first = trainer.train_step()
    second = trainer.train_step()

    assert np.isfinite(first.loss)
    assert second.step == 2
    assert first.pre_clipping_gradient_norm >= first.post_clipping_gradient_norm
    assert first.post_clipping_gradient_norm <= 0.05 + 1e-12
    assert any(
        not np.array_equal(before[name], values)
        for name, values in trainer.model.state_dict().items()
    )
    assert not trainer.model.has_pending_cache()
    assert all(
        np.all(np.isfinite(parameter.grad)) for parameter in trainer.model.parameters()
    )


def test_training_weight_decay_is_coupled_before_norm_clipping(tmp_path) -> None:
    without_decay = _trainer(
        tmp_path / "none",
        weight_decay=0.0,
        maximum_gradient_norm=None,
    )
    with_decay = _trainer(
        tmp_path / "decay",
        weight_decay=0.2,
        maximum_gradient_norm=None,
    )

    plain = without_decay.train_step()
    decayed = with_decay.train_step()

    assert decayed.pre_clipping_gradient_norm != plain.pre_clipping_gradient_norm
    assert not all(
        np.array_equal(left.data, right.data)
        for left, right in zip(
            without_decay.model.parameters(),
            with_decay.model.parameters(),
            strict=True,
        )
    )


def test_explicit_training_batch_rejects_malformed_targets_and_recovers(
    tmp_path,
) -> None:
    trainer = _trainer(tmp_path)
    inputs = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(TypeError, match="integer"):
        trainer.train_batch(inputs, np.array([[1.0, 2.0, 0.0]]))
    assert not trainer.model.has_pending_cache()
    with pytest.raises(ValueError, match="shape"):
        trainer.train_batch(inputs, np.array([[1, 2]], dtype=np.int64))
    assert not trainer.model.has_pending_cache()

    metrics = trainer.train_batch(
        inputs,
        np.array([[1, 2, 0]], dtype=np.int64),
    )
    assert metrics.step == 1


def test_evaluation_restores_mode_parameters_gradients_and_sampler(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    trainer.train_step()
    trainer.model.eval()
    parameters_before = trainer.model.state_dict()
    gradients_before = [
        parameter.grad.copy() for parameter in trainer.model.parameters()
    ]
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    sampler_state = trainer.train_sampler.state_dict()
    clone = SequenceBatchSampler(
        trainer.train_tokens,
        batch_size=trainer.config.batch_size,
        sequence_length=trainer.config.sequence_length,
        seed=trainer.config.seed + 1,
    )
    clone.load_state_dict(sampler_state)
    expected_next_batch = clone.next_batch()

    first = trainer.evaluate()
    second = trainer.evaluate()
    actual_next_batch = trainer.train_sampler.next_batch()

    assert first == second
    assert not trainer.model.training
    assert not trainer.model.has_pending_cache()
    assert all(
        np.array_equal(parameters_before[name], values)
        for name, values in trainer.model.state_dict().items()
    )
    assert all(
        np.array_equal(before, parameter.grad)
        for before, parameter in zip(
            gradients_before,
            trainer.model.parameters(),
            strict=True,
        )
    )
    assert optimizer_before.keys() == trainer.optimizer.state_dict().keys()
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            expected_next_batch,
            actual_next_batch,
            strict=True,
        )
    )


def test_evaluation_uses_token_weighted_mean_and_safe_perplexity() -> None:
    model = TransformerLanguageModel(_model_config()).eval()
    tokens = np.tile(np.array([0, 1, 2], dtype=np.int64), 8)

    metrics = evaluate_language_model(
        model,
        tokens,
        batch_size=2,
        sequence_length=3,
        batches=4,
        seed=2,
    )

    assert np.isfinite(metrics.loss)
    assert metrics.predicted_tokens == 2 * 3 * 4
    assert metrics.perplexity == pytest.approx(np.exp(metrics.loss))
    assert not model.has_pending_cache()


def test_inference_lifecycle_preserves_strict_training_guards() -> None:
    model = TransformerLanguageModel(_model_config())
    inputs = np.array([[0, 1, 2]], dtype=np.int64)

    with model.inference_mode():
        first = model.forward(inputs)
        second = model.forward(inputs)
        assert np.array_equal(first, second)
        assert not model.has_pending_cache()
        with pytest.raises(RuntimeError, match="requires one unmatched"):
            model.backward(np.ones_like(first))

    assert model.training
    training_logits = model.forward(inputs)
    with (
        pytest.raises(RuntimeError, match="inference mode.*cache"),
        model.inference_mode(),
    ):
        pass
    model.backward(np.ones_like(training_logits))
    assert not model.has_pending_cache()


def test_inference_context_restores_individual_nested_modes() -> None:
    model = TransformerLanguageModel(_model_config())
    model.final_layer_norm.eval()
    original_modes = [
        model.training,
        model.token_embedding.training,
        model.final_layer_norm.training,
    ]

    with model.inference_mode():
        assert not model.training
        assert all(not module.training for module in model.modules())

    assert [
        model.training,
        model.token_embedding.training,
        model.final_layer_norm.training,
    ] == original_modes


def test_trainer_rejects_wrong_target_stream_and_optimizer_config(tmp_path) -> None:
    tokenizer = CharacterTokenizer("abc")
    tokens = tokenizer.encode("abcabc")
    config = _training_config(tmp_path, sequence_length=6)

    with pytest.raises(ValueError, match="must exceed"):
        TransformerTrainer(
            TransformerLanguageModel(_model_config(maximum_context_length=6)),
            tokenizer,
            tokens,
            tokens,
            config,
        )
    with pytest.raises(ValueError, match="vocabulary"):
        TransformerTrainer(
            TransformerLanguageModel(_model_config(vocabulary_size=4)),
            tokenizer,
            tokens,
            tokens,
            replace(config, sequence_length=2),
        )


def test_tiny_transformer_overfits_repetitive_pattern_and_reloads(tmp_path) -> None:
    summary = run_experiment(
        seed=7,
        steps=40,
        output_directory=tmp_path / "tiny_overfit",
    )

    assert summary["final_validation_loss"] < 0.5 * summary["initial_validation_loss"]
    assert summary["final_pattern_agreement"] == 1.0
    assert summary["checkpoint_reload_logits_equal"]
    assert summary["checkpoint_reload_generation_equal"]
    assert summary["resumed_to_step"] == 40
