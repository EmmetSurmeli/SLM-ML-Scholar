#!/usr/bin/env python3
"""Train and generate with the manual multi-head transformer language model."""

from __future__ import annotations

import argparse
import json
import logging
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
from localml_scholar.data import (  # noqa: E402
    FALLBACK_CORPUS,
    load_utf8_text,
    prepare_token_stream_dataset,
)
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

LOGGER = logging.getLogger("train_transformer_lm")


def _decode_generation(
    trainer: TransformerTrainer,
    prompt: str,
    length: int,
    seed: int,
) -> str:
    prompt_ids = trainer.tokenizer.encode(prompt)[None, :]
    generated = generate_transformer_ids(
        trainer.model,
        prompt_ids,
        max_new_tokens=length,
        temperature=0.8,
        top_k=min(10, trainer.tokenizer.vocabulary_size),
        seed=seed,
    )
    return trainer.tokenizer.decode(generated[0])


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run a configured local-corpus training experiment and save artifacts."""
    if args.input is None:
        text = FALLBACK_CORPUS
        corpus_source = "built-in fallback corpus"
        LOGGER.warning("No --input supplied; using the built-in smoke corpus.")
    else:
        text = load_utf8_text(args.input)
        corpus_source = str(args.input)
    dataset = prepare_token_stream_dataset(text, args.train_fraction)
    output_directory = args.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model_config = TransformerConfig(
        vocabulary_size=dataset.tokenizer.vocabulary_size,
        maximum_context_length=args.context_length,
        model_dimension=args.model_dimension,
        number_of_layers=args.layers,
        number_of_heads=args.heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        feed_forward_dimension=args.feed_forward_dimension,
        dtype=np.float32,
        seed=args.seed,
    )
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        sequence_length=args.context_length,
        maximum_steps=args.steps,
        evaluation_interval=args.evaluation_interval,
        evaluation_batches=args.evaluation_batches,
        checkpoint_interval=args.checkpoint_interval,
        logging_interval=args.logging_interval,
        optimizer_name="adam",
        learning_rate=args.learning_rate,
        maximum_gradient_norm=args.maximum_gradient_norm,
        seed=args.seed,
        output_directory=str(output_directory),
    )
    if args.resume is None:
        trainer = TransformerTrainer(
            TransformerLanguageModel(model_config),
            dataset.tokenizer,
            dataset.train_tokens,
            dataset.validation_tokens,
            training_config,
        )
        resumed_from = None
    else:
        trainer = TransformerTrainer.load_checkpoint(
            args.resume,
            train_tokens=dataset.train_tokens,
            validation_tokens=dataset.validation_tokens,
            tokenizer=dataset.tokenizer,
            expected_model_config=model_config,
            expected_training_config=training_config,
        )
        resumed_from = str(args.resume)

    prompt = args.prompt if args.prompt is not None else text[0]
    initial_metrics = trainer.evaluate()
    initial_generation = _decode_generation(
        trainer,
        prompt,
        args.generation_length,
        args.seed + trainer.completed_steps,
    )
    trainer.run(until_step=args.until_step)
    final_metrics = trainer.evaluate()
    final_generation = _decode_generation(
        trainer,
        prompt,
        args.generation_length,
        args.seed + trainer.completed_steps,
    )
    final_training_checkpoint = trainer.save_checkpoint(
        output_directory / "final_training_checkpoint.npz"
    )
    final_model_checkpoint = trainer.model.save_checkpoint(
        output_directory / "final_model.npz"
    )
    tokenizer_path = dataset.tokenizer.save(output_directory / "tokenizer.json")
    history_path = output_directory / "history.json"
    history_path.write_text(
        json.dumps(trainer.history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best_checkpoint = output_directory / "best_training_checkpoint.npz"
    reloaded_best = TransformerTrainer.load_checkpoint(
        best_checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=dataset.tokenizer,
        expected_model_config=model_config,
        expected_training_config=training_config,
    )
    best_generation = _decode_generation(
        reloaded_best,
        prompt,
        args.generation_length,
        args.seed + int(reloaded_best.best_validation_step or 0),
    )
    summary: dict[str, Any] = {
        "milestone": 6,
        "package_version": __version__,
        "corpus_source": corpus_source,
        "corpus_characters": len(text),
        "chronological_split_index": dataset.split_index,
        "train_tokens": int(dataset.train_tokens.size),
        "validation_tokens": int(dataset.validation_tokens.size),
        "tokenizer_vocabulary_size": dataset.tokenizer.vocabulary_size,
        "model_configuration": model_config.to_dict(),
        "training_configuration": training_config.to_dict(),
        "parameter_count": trainer.model.parameter_count,
        "initial_metrics": {
            name: vars(metrics) for name, metrics in initial_metrics.items()
        },
        "final_metrics": {
            name: vars(metrics) for name, metrics in final_metrics.items()
        },
        "best_validation_loss": trainer.best_validation_loss,
        "best_validation_step": trainer.best_validation_step,
        "generation": {
            "prompt": prompt,
            "length": args.generation_length,
            "initial": initial_generation,
            "final": final_generation,
            "best_checkpoint": best_generation,
        },
        "resume": {
            "resumed_from": resumed_from,
            "completed_steps": trainer.completed_steps,
            "configured_maximum_steps": training_config.maximum_steps,
        },
        "artifacts": {
            "latest_training_checkpoint": str(
                output_directory / "latest_training_checkpoint.npz"
            ),
            "best_training_checkpoint": str(best_checkpoint),
            "final_training_checkpoint": str(final_training_checkpoint),
            "final_model_checkpoint": str(final_model_checkpoint),
            "tokenizer": str(tokenizer_path),
            "history": str(history_path),
        },
    }
    summary_path = output_directory / "run_summary.json"
    summary["artifacts"]["summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the from-scratch NumPy multi-head transformer."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "transformer_lm_smoke",
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-dimension", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--key-dimension", type=int, default=4)
    parser.add_argument("--value-dimension", type=int, default=4)
    parser.add_argument("--feed-forward-dimension", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--until-step",
        type=int,
        default=None,
        help="Stop this invocation at an absolute step before --steps.",
    )
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--evaluation-batches", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=15)
    parser.add_argument("--logging-interval", type=int, default=5)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--generation-length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(arguments)
    summary = run_training(args)
    LOGGER.info(
        "complete step=%d best_validation_loss=%.6f summary=%s",
        summary["resume"]["completed_steps"],
        summary["best_validation_loss"],
        summary["artifacts"]["summary"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
