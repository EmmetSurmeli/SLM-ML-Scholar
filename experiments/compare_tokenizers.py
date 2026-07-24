#!/usr/bin/env python3
"""Compare character, byte, and BPE behavior on one controlled tiny corpus."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from localml_scholar._version import __version__  # noqa: E402
from localml_scholar.data import (  # noqa: E402
    SequenceBatchSampler,
    prepare_token_stream_dataset,
)
from localml_scholar.generation import generate_transformer_text  # noqa: E402
from localml_scholar.losses import (  # noqa: E402
    softmax_cross_entropy_loss_and_gradient,
)
from localml_scholar.models.transformer_lm import (  # noqa: E402
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.tokenizer import (  # noqa: E402
    BPETrainingConfig,
    BytePairTokenizer,
    ByteTokenizer,
    CharacterTokenizer,
    Tokenizer,
)
from localml_scholar.training.config import (  # noqa: E402
    TransformerTrainingConfig,
)
from localml_scholar.training.transformer import (  # noqa: E402
    TransformerTrainer,
)

COMPARISON_CORPUS = (
    "banana bandana and local models.\ncafé math uses ∑ and λ; emoji 🧠 stays UTF-8.\n"
) * 20


def _fit_tokenizer(
    tokenizer_type: str,
    training_text: str,
) -> Tokenizer:
    if tokenizer_type == "character":
        return CharacterTokenizer.from_text(training_text)
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "bpe":
        return BytePairTokenizer.train(
            training_text,
            BPETrainingConfig(
                target_vocabulary_size=272,
                minimum_pair_frequency=2,
            ),
        )
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type!r}.")


def _sampled_bits_per_byte(
    model: TransformerLanguageModel,
    tokenizer: Tokenizer,
    token_ids: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    sampler = SequenceBatchSampler(
        token_ids,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    total_nll = 0.0
    evaluated_tokens = 0
    evaluated_bytes = 0
    with model.inference_mode():
        for _ in range(batches):
            inputs, targets = sampler.next_batch()
            logits = model.forward(inputs)
            loss, _ = softmax_cross_entropy_loss_and_gradient(logits, targets)
            total_nll += loss * targets.size
            evaluated_tokens += int(targets.size)
            evaluated_bytes += sum(
                len(tokenizer.token_bytes(int(token_id))) for token_id in targets.flat
            )
    return {
        "bits_per_byte": total_nll / (evaluated_bytes * math.log(2.0)),
        "total_negative_log_likelihood": total_nll,
        "evaluated_tokens": evaluated_tokens,
        "evaluated_bytes": evaluated_bytes,
        "sampling_note": (
            "Fixed-seed sampled target tokens; each target NLL is normalized "
            "by the exact UTF-8 bytes represented by that target token."
        ),
    }


def _run_tokenizer(
    tokenizer_type: str,
    *,
    text: str,
    seed: int,
    steps: int,
    output_directory: Path,
) -> dict[str, Any]:
    split_index = int(len(text) * 0.8)
    training_text = text[:split_index]
    validation_text = text[split_index:]
    start = time.perf_counter()
    tokenizer = _fit_tokenizer(tokenizer_type, training_text)
    tokenizer_training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    train_tokens = tokenizer.encode(training_text)
    validation_tokens = tokenizer.encode(validation_text)
    encoding_seconds = time.perf_counter() - start
    start = time.perf_counter()
    round_trip = (
        tokenizer.decode(train_tokens) == training_text
        and tokenizer.decode(validation_tokens) == validation_text
    )
    decoding_seconds = time.perf_counter() - start

    dataset = prepare_token_stream_dataset(
        text,
        0.8,
        tokenizer=tokenizer,
        source_name="controlled_tokenizer_comparison",
    )
    model_config = TransformerConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        maximum_context_length=6,
        model_dimension=6,
        number_of_layers=1,
        key_dimension=2,
        value_dimension=2,
        feed_forward_dimension=12,
        number_of_heads=2,
        dtype=np.float32,
        seed=seed,
    )
    training_config = TransformerTrainingConfig(
        batch_size=4,
        sequence_length=6,
        maximum_steps=steps,
        evaluation_interval=max(1, steps // 3),
        evaluation_batches=2,
        checkpoint_interval=max(1, steps // 2),
        logging_interval=max(1, steps // 4),
        optimizer_name="adam",
        learning_rate=0.015,
        maximum_gradient_norm=1.0,
        seed=seed,
        output_directory=str(output_directory),
    )
    trainer = TransformerTrainer(
        TransformerLanguageModel(model_config),
        tokenizer,
        dataset.train_tokens,
        dataset.validation_tokens,
        training_config,
        raw_corpus_metadata=dataset.metadata.to_dict(),
    )
    initial = trainer.evaluate()
    training_start = time.perf_counter()
    interruption_step = max(1, steps // 2)
    trainer.run(until_step=interruption_step)
    interruption_checkpoint = trainer.save_checkpoint(
        output_directory / "interrupted_training.npz"
    )
    resumed = TransformerTrainer.load_checkpoint(
        interruption_checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=None,
        raw_corpus_metadata=dataset.metadata.to_dict(),
        expected_model_config=model_config,
        expected_training_config=training_config,
    )
    resumed.run(until_step=steps)
    training_seconds = time.perf_counter() - training_start
    final = resumed.evaluate()
    normalized = _sampled_bits_per_byte(
        resumed.model,
        resumed.tokenizer,
        dataset.validation_tokens,
        batch_size=training_config.batch_size,
        sequence_length=training_config.sequence_length,
        batches=training_config.evaluation_batches,
        seed=seed + 19,
    )
    generated = generate_transformer_text(
        resumed.model,
        resumed.tokenizer,
        "banana",
        max_new_tokens=18,
        temperature=0.8,
        top_k=min(10, tokenizer.vocabulary_size),
        seed=seed + 23,
        decode_errors="replace",
    )
    final_checkpoint = resumed.save_checkpoint(output_directory / "final_training.npz")
    reloaded = TransformerTrainer.load_checkpoint(
        final_checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=None,
        raw_corpus_metadata=dataset.metadata.to_dict(),
    )
    probe = tokenizer.encode("banana")[None, :]
    with resumed.model.inference_mode():
        expected_logits = resumed.model.forward(probe)
    with reloaded.model.inference_mode():
        actual_logits = reloaded.model.forward(probe)
    checkpoint_reload_equal = bool(np.array_equal(expected_logits, actual_logits))
    tokenizer_path = tokenizer.save(output_directory / "tokenizer.json")
    final_training_loss = next(
        record["training_step"]["loss"]
        for record in reversed(resumed.history)
        if "training_step" in record
    )
    training_bytes = len(training_text.encode("utf-8"))
    validation_bytes = len(validation_text.encode("utf-8"))
    selected_samples = [
        "banana",
        "café ∑",
        "emoji 🧠",
        "banana\ncafé",
    ]
    return {
        "tokenizer_type": tokenizer_type,
        "vocabulary_size": tokenizer.vocabulary_size,
        "learned_merges": (
            len(tokenizer.merge_rules)
            if isinstance(tokenizer, BytePairTokenizer)
            else 0
        ),
        "tokenizer_state_sha256": tokenizer.state_hash(),
        "raw_character_count": len(text),
        "raw_byte_count": len(text.encode("utf-8")),
        "training_byte_count": training_bytes,
        "validation_byte_count": validation_bytes,
        "training_token_count": int(train_tokens.size),
        "validation_token_count": int(validation_tokens.size),
        "training_token_to_byte_ratio": int(train_tokens.size) / training_bytes,
        "validation_token_to_byte_ratio": (
            int(validation_tokens.size) / validation_bytes
        ),
        "training_average_bytes_per_token": training_bytes / int(train_tokens.size),
        "validation_average_bytes_per_token": (
            validation_bytes / int(validation_tokens.size)
        ),
        "maximum_selected_sample_tokens": max(
            tokenizer.encode(sample).size for sample in selected_samples
        ),
        "round_trip_exact": round_trip,
        "tokenizer_training_seconds": tokenizer_training_seconds,
        "encoding_seconds": encoding_seconds,
        "decoding_seconds": decoding_seconds,
        "model_parameter_count": resumed.model.parameter_count,
        "initial_train_loss": initial["train"].loss,
        "initial_validation_loss": initial["validation"].loss,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final["validation"].loss,
        "final_validation_perplexity": final["validation"].perplexity,
        "byte_normalized_validation": normalized,
        "total_predicted_training_tokens": (
            steps * training_config.batch_size * training_config.sequence_length
        ),
        "training_seconds": training_seconds,
        "generated_sample": generated,
        "interruption_step": interruption_step,
        "resumed_to_step": resumed.completed_steps,
        "checkpoint_reload_logits_equal": checkpoint_reload_equal,
        "artifacts": {
            "tokenizer": str(tokenizer_path),
            "interruption_checkpoint": str(interruption_checkpoint),
            "final_checkpoint": str(final_checkpoint),
        },
    }


def compare_tokenizers(
    *,
    seed: int = 149,
    steps: int = 12,
    output_directory: str | Path = "outputs/tokenizer_comparison",
) -> dict[str, Any]:
    """Run a controlled integration study over three token units."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer of at least two.")
    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    runs = [
        _run_tokenizer(
            tokenizer_type,
            text=COMPARISON_CORPUS,
            seed=seed,
            steps=steps,
            output_directory=destination / tokenizer_type,
        )
        for tokenizer_type in ("character", "byte", "bpe")
    ]
    if not all(
        run["round_trip_exact"] and run["checkpoint_reload_logits_equal"]
        for run in runs
    ):
        raise RuntimeError("A tokenizer comparison integration invariant failed.")
    summary: dict[str, Any] = {
        "milestone": 7,
        "package_version": __version__,
        "purpose": "controlled tokenizer behavior and integration study",
        "control_policy": (
            "All runs use the same raw split, seed, update count, batch size, "
            "context length, model dimensions, and optimizer settings. "
            "Vocabulary-dependent parameter counts and token coverage differ."
        ),
        "perplexity_warning": (
            "Token-level perplexity is not directly comparable across tokenizers. "
            "Use the reported sampled bits-per-byte with its evaluated-byte count."
        ),
        "runs": runs,
    }
    summary_path = destination / "comparison_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare character, raw-byte, and byte-level BPE tokenization."
    )
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/tokenizer_comparison"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    compare_tokenizers(
        seed=args.seed,
        steps=args.steps,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
