#!/usr/bin/env python3
"""Deterministically overfit the manual transformer to a transparent pattern."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from localml_scholar._version import __version__  # noqa: E402
from localml_scholar.data import prepare_token_stream_dataset  # noqa: E402
from localml_scholar.generation import generate_transformer_ids  # noqa: E402
from localml_scholar.models.transformer_lm import (  # noqa: E402
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.training.config import (  # noqa: E402
    TransformerTrainingConfig,
)
from localml_scholar.training.transformer import (  # noqa: E402
    TransformerTrainer,
)


def _pattern_agreement(token_ids: np.ndarray, vocabulary_size: int) -> float:
    if token_ids.ndim != 1 or token_ids.size < 2:
        raise ValueError("token_ids must contain at least two generated tokens.")
    expected = (token_ids[:-1] + 1) % vocabulary_size
    return float(np.mean(token_ids[1:] == expected))


def run_experiment(
    *,
    seed: int,
    steps: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Run an interrupted-and-resumed tiny-pattern overfit demonstration."""
    corpus = "abc" * 80
    dataset = prepare_token_stream_dataset(corpus, train_fraction=0.8)
    model_config = TransformerConfig(
        vocabulary_size=dataset.tokenizer.vocabulary_size,
        maximum_context_length=6,
        model_dimension=8,
        number_of_layers=1,
        key_dimension=4,
        value_dimension=4,
        feed_forward_dimension=16,
        dtype=np.float32,
        seed=seed,
    )
    training_config = TransformerTrainingConfig(
        batch_size=8,
        sequence_length=6,
        maximum_steps=steps,
        evaluation_interval=max(1, steps // 8),
        evaluation_batches=4,
        checkpoint_interval=max(1, steps // 4),
        logging_interval=max(1, steps // 10),
        optimizer_name="adam",
        learning_rate=0.02,
        maximum_gradient_norm=1.0,
        seed=seed,
        output_directory=str(output_directory),
    )
    trainer = TransformerTrainer(
        TransformerLanguageModel(model_config),
        dataset.tokenizer,
        dataset.train_tokens,
        dataset.validation_tokens,
        training_config,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer_path = dataset.tokenizer.save(output_directory / "tokenizer.json")
    prompt = dataset.tokenizer.encode("a")[None, :]
    initial_metrics = trainer.evaluate()["validation"]
    initial_generation = generate_transformer_ids(
        trainer.model,
        prompt,
        max_new_tokens=30,
        greedy=True,
    )

    interruption_step = max(1, steps // 2)
    trainer.run(until_step=interruption_step)
    interruption_checkpoint = trainer.save_checkpoint(
        output_directory / "interrupted_training_checkpoint.npz"
    )
    resumed = TransformerTrainer.load_checkpoint(
        interruption_checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=dataset.tokenizer,
        expected_model_config=model_config,
        expected_training_config=training_config,
    )
    resumed.run(until_step=steps)
    final_metrics = resumed.evaluate()["validation"]
    final_generation = generate_transformer_ids(
        resumed.model,
        prompt,
        max_new_tokens=30,
        greedy=True,
    )
    final_checkpoint = resumed.save_checkpoint(
        output_directory / "final_training_checkpoint.npz"
    )
    reloaded = TransformerTrainer.load_checkpoint(
        final_checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=dataset.tokenizer,
        expected_model_config=model_config,
        expected_training_config=training_config,
    )
    with resumed.model.inference_mode():
        expected_logits = resumed.model.forward(prompt)
    with reloaded.model.inference_mode():
        reloaded_logits = reloaded.model.forward(prompt)
    reload_logits_equal = bool(np.array_equal(expected_logits, reloaded_logits))
    reload_generation_equal = bool(
        np.array_equal(
            final_generation,
            generate_transformer_ids(
                reloaded.model,
                prompt,
                max_new_tokens=30,
                greedy=True,
            ),
        )
    )
    gradient_norms = [
        record["training_step"]["pre_clipping_gradient_norm"]
        for record in resumed.history
        if "training_step" in record
    ]
    initial_ids = initial_generation[0]
    final_ids = final_generation[0]
    summary: dict[str, Any] = {
        "milestone": "5 Part 2",
        "package_version": __version__,
        "model_configuration": model_config.to_dict(),
        "training_configuration": training_config.to_dict(),
        "parameter_count": resumed.model.parameter_count,
        "corpus": corpus,
        "corpus_characters": len(corpus),
        "train_tokens": int(dataset.train_tokens.size),
        "validation_tokens": int(dataset.validation_tokens.size),
        "initial_validation_loss": initial_metrics.loss,
        "final_validation_loss": final_metrics.loss,
        "initial_validation_perplexity": initial_metrics.perplexity,
        "final_validation_perplexity": final_metrics.perplexity,
        "best_validation_loss": resumed.best_validation_loss,
        "best_validation_step": resumed.best_validation_step,
        "initial_generation": dataset.tokenizer.decode(initial_ids),
        "final_generation": dataset.tokenizer.decode(final_ids),
        "initial_pattern_agreement": _pattern_agreement(
            initial_ids,
            dataset.tokenizer.vocabulary_size,
        ),
        "final_pattern_agreement": _pattern_agreement(
            final_ids,
            dataset.tokenizer.vocabulary_size,
        ),
        "gradient_norm_minimum": min(gradient_norms),
        "gradient_norm_maximum": max(gradient_norms),
        "interruption_step": interruption_step,
        "resumed_to_step": resumed.completed_steps,
        "checkpoint_reload_logits_equal": reload_logits_equal,
        "checkpoint_reload_generation_equal": reload_generation_equal,
        "artifacts": {
            "tokenizer": str(tokenizer_path),
            "interruption_checkpoint": str(interruption_checkpoint),
            "final_checkpoint": str(final_checkpoint),
        },
    }
    if not final_metrics.loss < initial_metrics.loss * 0.5:
        raise RuntimeError(
            "Tiny transformer did not reduce validation loss by at least 50%."
        )
    if summary["final_pattern_agreement"] < 0.9:
        raise RuntimeError("Tiny transformer did not learn the repetitive pattern.")
    if not reload_logits_equal or not reload_generation_equal:
        raise RuntimeError("Checkpoint reload did not preserve transformer outputs.")
    summary_path = output_directory / "run_summary.json"
    summary["artifacts"]["summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit the manual transformer to the pattern 'abc'."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tiny_transformer_overfit",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")
    summary = run_experiment(
        seed=args.seed,
        steps=args.steps,
        output_directory=args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
