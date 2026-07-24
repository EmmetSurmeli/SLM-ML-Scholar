import json
from pathlib import Path

import numpy as np
import pytest

from localml_scholar.data import prepare_token_stream_dataset
from localml_scholar.generation import (
    generate_transformer_ids,
    generate_transformer_text,
)
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.tokenizer import BPETrainingConfig
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.training.transformer import TransformerTrainer


def _components(tmp_path: Path, tokenizer_type: str):
    text = ("abcde café ∑\n" * 10) + ("abcde café ∑\n" * 4)
    dataset = prepare_token_stream_dataset(
        text,
        0.7,
        tokenizer=tokenizer_type,
        bpe_config=(
            BPETrainingConfig(
                target_vocabulary_size=264,
                minimum_pair_frequency=2,
            )
            if tokenizer_type == "bpe"
            else None
        ),
        source_name="integration_fixture",
    )
    model_config = TransformerConfig(
        vocabulary_size=dataset.tokenizer.vocabulary_size,
        maximum_context_length=4,
        model_dimension=4,
        number_of_layers=1,
        key_dimension=1,
        value_dimension=1,
        feed_forward_dimension=6,
        number_of_heads=2,
        dtype=np.float64,
        seed=131,
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
        maximum_gradient_norm=0.8,
        seed=137,
        output_directory=str(tmp_path),
    )
    return dataset, model_config, training_config


def _trainer(tmp_path: Path, tokenizer_type: str) -> TransformerTrainer:
    dataset, model_config, training_config = _components(tmp_path, tokenizer_type)
    return TransformerTrainer(
        TransformerLanguageModel(model_config),
        dataset.tokenizer,
        dataset.train_tokens,
        dataset.validation_tokens,
        training_config,
        raw_corpus_metadata=dataset.metadata.to_dict(),
    )


def _advance(trainer: TransformerTrainer, steps: int) -> list[float]:
    losses: list[float] = []
    for _ in range(steps):
        metrics = trainer.train_step()
        losses.append(metrics.loss)
        trainer.record_evaluation(metrics, trainer.evaluate())
    return losses


def _rewrite_checkpoint_metadata(source: Path, destination: Path, mutate) -> None:
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


@pytest.mark.parametrize("tokenizer_type", ["character", "byte", "bpe"])
def test_all_tokenizers_support_full_language_model_path(
    tmp_path: Path,
    tokenizer_type: str,
) -> None:
    trainer = _trainer(tmp_path / tokenizer_type, tokenizer_type)
    initial = trainer.evaluate()["validation"]
    metrics = trainer.train_step()
    final = trainer.evaluate()["validation"]
    prompt = "abc"
    prompt_ids = trainer.tokenizer.encode(prompt)[None, :]

    greedy = generate_transformer_ids(
        trainer.model,
        prompt_ids,
        max_new_tokens=3,
        greedy=True,
    )
    sampled = generate_transformer_ids(
        trainer.model,
        prompt_ids,
        max_new_tokens=3,
        top_k=min(5, trainer.tokenizer.vocabulary_size),
        seed=139,
    )
    displayed = generate_transformer_text(
        trainer.model,
        trainer.tokenizer,
        prompt,
        max_new_tokens=3,
        greedy=True,
        decode_errors="replace",
    )

    assert np.isfinite(initial.loss)
    assert np.isfinite(metrics.loss)
    assert np.isfinite(final.loss)
    assert greedy.shape == sampled.shape
    assert np.all(greedy < trainer.tokenizer.vocabulary_size)
    assert displayed.startswith(prompt)

    model_path = trainer.model.save_checkpoint(
        tmp_path / f"{tokenizer_type}_model.npz",
        tokenizer=trainer.tokenizer,
    )
    loaded_model, loaded_tokenizer = (
        TransformerLanguageModel.load_checkpoint_with_tokenizer(model_path)
    )
    assert loaded_tokenizer.state_dict() == trainer.tokenizer.state_dict()
    with trainer.model.inference_mode():
        expected_logits = trainer.model.forward(prompt_ids)
    with loaded_model.inference_mode():
        actual_logits = loaded_model.forward(prompt_ids)
    assert np.array_equal(actual_logits, expected_logits)

    checkpoint = trainer.save_checkpoint(tmp_path / f"{tokenizer_type}_training.npz")
    dataset, model_config, training_config = _components(
        tmp_path / tokenizer_type,
        tokenizer_type,
    )
    restored = TransformerTrainer.load_checkpoint(
        checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=None,
        raw_corpus_metadata=dataset.metadata.to_dict(),
        expected_model_config=model_config,
        expected_training_config=training_config,
    )
    assert restored.tokenizer.state_dict() == trainer.tokenizer.state_dict()


@pytest.mark.parametrize("tokenizer_type", ["character", "byte", "bpe"])
def test_all_tokenizers_resume_exactly(
    tmp_path: Path,
    tokenizer_type: str,
) -> None:
    uninterrupted = _trainer(tmp_path / "continuous", tokenizer_type)
    expected_losses = _advance(uninterrupted, 4)

    interrupted = _trainer(tmp_path / "resumed", tokenizer_type)
    actual_losses = _advance(interrupted, 2)
    checkpoint = interrupted.save_checkpoint(tmp_path / f"{tokenizer_type}_resume.npz")
    dataset, model_config, _ = _components(
        tmp_path / "resumed",
        tokenizer_type,
    )
    resumed = TransformerTrainer.load_checkpoint(
        checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=None,
        raw_corpus_metadata=dataset.metadata.to_dict(),
        expected_model_config=model_config,
    )
    actual_losses.extend(_advance(resumed, 2))

    assert actual_losses == expected_losses
    assert resumed.history == uninterrupted.history
    assert (
        resumed.train_sampler.state_dict() == uninterrupted.train_sampler.state_dict()
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            resumed.model.state_dict().values(),
            uninterrupted.model.state_dict().values(),
            strict=True,
        )
    )


def test_realistic_version_060_training_checkpoint_migrates_in_memory(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path / "current", "character")
    trainer.train_step()
    current = trainer.save_checkpoint(tmp_path / "current.npz")
    legacy = tmp_path / "legacy_060.npz"

    def make_legacy(metadata: dict) -> None:
        characters = metadata["tokenizer"]["state"]["characters"]
        metadata["checkpoint_version"] = 2
        metadata["package_version"] = "0.6.0"
        metadata["tokenizer"] = {
            "format_version": 1,
            "type": "character",
            "characters": characters,
        }
        metadata.pop("tokenizer_state_sha256")
        metadata["corpus"].pop("raw_corpus")

    _rewrite_checkpoint_metadata(current, legacy, make_legacy)
    dataset, _, _ = _components(tmp_path / "current", "character")

    restored = TransformerTrainer.load_checkpoint(
        legacy,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
    )

    assert restored.tokenizer.state_dict() == trainer.tokenizer.state_dict()
    assert restored.completed_steps == trainer.completed_steps
    assert restored.model.config == trainer.model.config


def test_realistic_version_060_model_checkpoint_preserves_logits(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path / "model", "character")
    current = trainer.model.save_checkpoint(
        tmp_path / "current_model.npz",
        tokenizer=trainer.tokenizer,
    )
    legacy = tmp_path / "legacy_model_060.npz"

    def make_legacy(metadata: dict) -> None:
        metadata["checkpoint_version"] = 2
        metadata["model_version"] = "0.6.0"
        metadata.pop("tokenizer")
        metadata.pop("tokenizer_state_sha256")

    _rewrite_checkpoint_metadata(current, legacy, make_legacy)
    restored = TransformerLanguageModel.load_checkpoint(legacy)
    prompt = trainer.tokenizer.encode("abc")[None, :]
    with trainer.model.inference_mode():
        expected = trainer.model.forward(prompt)
    with restored.inference_mode():
        actual = restored.forward(prompt)

    assert np.array_equal(actual, expected)
