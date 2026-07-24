import copy
import json
from typing import Any

import numpy as np
import pytest

from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.tokenizer import CharacterTokenizer
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.training.transformer import TransformerTrainer


def _components(tmp_path):
    tokenizer = CharacterTokenizer("abc")
    tokens = tokenizer.encode("abcabcabcabcabcabcabcabc")
    model_config = TransformerConfig(
        vocabulary_size=3,
        maximum_context_length=4,
        model_dimension=4,
        number_of_layers=1,
        key_dimension=2,
        value_dimension=2,
        feed_forward_dimension=7,
        dtype=np.float64,
        seed=41,
    )
    training_config = TransformerTrainingConfig(
        batch_size=3,
        sequence_length=3,
        maximum_steps=6,
        evaluation_interval=1,
        evaluation_batches=2,
        checkpoint_interval=3,
        logging_interval=1,
        optimizer_name="adam",
        learning_rate=0.01,
        maximum_gradient_norm=0.5,
        seed=29,
        output_directory=str(tmp_path),
    )
    return tokenizer, tokens[:15], tokens[3:], model_config, training_config


def _new_trainer(tmp_path) -> TransformerTrainer:
    tokenizer, train, validation, model_config, training_config = _components(tmp_path)
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


def _assert_optimizer_states_equal(
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    left_arrays = left.pop("arrays")
    right_arrays = right.pop("arrays")
    assert left == right
    assert left_arrays.keys() == right_arrays.keys()
    assert all(
        np.array_equal(left_arrays[name], right_arrays[name]) for name in left_arrays
    )


def test_full_training_checkpoint_round_trip_preserves_every_state(tmp_path) -> None:
    trainer = _new_trainer(tmp_path)
    _advance(trainer, 2)
    prompt = np.array([[0, 1, 2]], dtype=np.int64)
    with trainer.model.inference_mode():
        expected_logits = trainer.model.forward(prompt)
    checkpoint = trainer.save_checkpoint(tmp_path / "training.npz")
    tokenizer, train, validation, model_config, training_config = _components(tmp_path)

    restored = TransformerTrainer.load_checkpoint(
        checkpoint,
        train_tokens=train,
        validation_tokens=validation,
        tokenizer=tokenizer,
        expected_model_config=model_config,
        expected_training_config=training_config,
    )

    with restored.model.inference_mode():
        actual_logits = restored.model.forward(prompt)
    assert np.array_equal(actual_logits, expected_logits)
    assert restored.completed_steps == trainer.completed_steps
    assert restored.best_validation_loss == trainer.best_validation_loss
    assert restored.best_validation_step == trainer.best_validation_step
    assert restored.history == trainer.history
    assert restored.train_sampler.state_dict() == trainer.train_sampler.state_dict()
    _assert_optimizer_states_equal(
        copy.deepcopy(restored.optimizer.state_dict()),
        copy.deepcopy(trainer.optimizer.state_dict()),
    )


def test_interrupted_training_resumes_exact_uninterrupted_trajectory(tmp_path) -> None:
    uninterrupted = _new_trainer(tmp_path / "continuous")
    uninterrupted_losses = _advance(uninterrupted, 6)
    expected_next_batch = uninterrupted.train_sampler.next_batch()

    interrupted = _new_trainer(tmp_path / "resumed")
    resumed_losses = _advance(interrupted, 2)
    checkpoint = interrupted.save_checkpoint(tmp_path / "resume.npz")
    tokenizer, train, validation, model_config, _ = _components(tmp_path / "resumed")
    resumed = TransformerTrainer.load_checkpoint(
        checkpoint,
        train_tokens=train,
        validation_tokens=validation,
        tokenizer=tokenizer,
        expected_model_config=model_config,
    )
    resumed_losses.extend(_advance(resumed, 4))
    actual_next_batch = resumed.train_sampler.next_batch()

    assert resumed_losses == uninterrupted_losses
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            expected_next_batch,
            actual_next_batch,
            strict=True,
        )
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            uninterrupted.model.state_dict().values(),
            resumed.model.state_dict().values(),
            strict=True,
        )
    )
    _assert_optimizer_states_equal(
        copy.deepcopy(uninterrupted.optimizer.state_dict()),
        copy.deepcopy(resumed.optimizer.state_dict()),
    )
    assert resumed.best_validation_loss == uninterrupted.best_validation_loss
    assert resumed.best_validation_step == uninterrupted.best_validation_step
    assert resumed.history == uninterrupted.history


def _rewrite_metadata(source, destination, mutate) -> None:
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


def _rewrite_arrays(source, destination, mutate) -> None:
    with np.load(source, allow_pickle=False) as checkpoint:
        metadata = np.array(checkpoint["metadata_json"], copy=True)
        arrays = {
            name: np.array(checkpoint[name], copy=True)
            for name in checkpoint.files
            if name != "metadata_json"
        }
    mutate(arrays)
    np.savez(destination, metadata_json=metadata, **arrays)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update(checkpoint_version=99), "version"),
        (lambda metadata: metadata.update(checkpoint_type="model_only"), "not a full"),
        (
            lambda metadata: metadata["model_configuration"].update(model_dimension=99),
            "configuration",
        ),
        (
            lambda metadata: metadata["training_configuration"].update(
                optimizer_name="sgd"
            ),
            "optimizer",
        ),
        (
            lambda metadata: metadata["tokenizer"].update(characters=["a", "b"]),
            "tokenizer",
        ),
    ],
)
def test_training_checkpoint_rejects_malformed_or_incompatible_metadata(
    tmp_path,
    mutation,
    message: str,
) -> None:
    trainer = _new_trainer(tmp_path)
    _advance(trainer, 1)
    original = trainer.save_checkpoint(tmp_path / "original.npz")
    malformed = tmp_path / "malformed.npz"
    _rewrite_metadata(original, malformed, mutation)
    tokenizer, train, validation, model_config, training_config = _components(tmp_path)

    with pytest.raises((ValueError, TypeError), match=message):
        TransformerTrainer.load_checkpoint(
            malformed,
            train_tokens=train,
            validation_tokens=validation,
            tokenizer=tokenizer,
            expected_model_config=model_config,
            expected_training_config=training_config,
        )


def test_model_only_checkpoint_is_not_accepted_as_training_state(tmp_path) -> None:
    trainer = _new_trainer(tmp_path)
    model_checkpoint = trainer.model.save_checkpoint(tmp_path / "model.npz")
    tokenizer, train, validation, _, _ = _components(tmp_path)

    with pytest.raises(ValueError, match="metadata keys"):
        TransformerTrainer.load_checkpoint(
            model_checkpoint,
            train_tokens=train,
            validation_tokens=validation,
            tokenizer=tokenizer,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda arrays: arrays.pop("optimizer::first_moment::0"),
            "state keys",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "model::token_embedding.weight",
                arrays["model::token_embedding.weight"][:-1],
            ),
            "shape",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "optimizer::second_moment::0",
                arrays["optimizer::second_moment::0"].astype(np.float32),
            ),
            "invalid shape, dtype, or values",
        ),
    ],
)
def test_training_checkpoint_rejects_missing_or_invalid_state_arrays(
    tmp_path,
    mutation,
    message: str,
) -> None:
    trainer = _new_trainer(tmp_path)
    _advance(trainer, 1)
    original = trainer.save_checkpoint(tmp_path / "original.npz")
    malformed = tmp_path / "malformed.npz"
    _rewrite_arrays(original, malformed, mutation)
    tokenizer, train, validation, _, _ = _components(tmp_path)

    with pytest.raises((ValueError, TypeError), match=message):
        TransformerTrainer.load_checkpoint(
            malformed,
            train_tokens=train,
            validation_tokens=validation,
            tokenizer=tokenizer,
        )


def test_atomic_training_checkpoint_leaves_no_temporary_file(tmp_path) -> None:
    trainer = _new_trainer(tmp_path)

    destination = trainer.save_checkpoint(tmp_path / "training.npz")

    assert destination.is_file()
    assert not list(tmp_path.glob(".training.*.npz"))


def test_training_checkpoint_rejects_corpus_and_tokenizer_mismatch(tmp_path) -> None:
    trainer = _new_trainer(tmp_path)
    _advance(trainer, 1)
    checkpoint = trainer.save_checkpoint(tmp_path / "training.npz")
    tokenizer, train, validation, _, _ = _components(tmp_path)
    changed_train = train.copy()
    changed_train[0] = (changed_train[0] + 1) % 3

    with pytest.raises(ValueError, match="corpus"):
        TransformerTrainer.load_checkpoint(
            checkpoint,
            train_tokens=changed_train,
            validation_tokens=validation,
            tokenizer=tokenizer,
        )
    with pytest.raises(ValueError, match="tokenizer"):
        TransformerTrainer.load_checkpoint(
            checkpoint,
            train_tokens=train,
            validation_tokens=validation,
            tokenizer=CharacterTokenizer("abcd"),
        )
