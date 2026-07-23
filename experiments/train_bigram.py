#!/usr/bin/env python3
"""Train the NumPy bigram baseline from a JSON configuration."""

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

from localml_scholar.data import (  # noqa: E402
    FALLBACK_CORPUS,
    MiniBatchSampler,
    load_utf8_text,
    prepare_bigram_dataset,
)
from localml_scholar.generation import generate_text  # noqa: E402
from localml_scholar.models.bigram import BigramLanguageModel  # noqa: E402
from localml_scholar.optimizers import SGD  # noqa: E402
from localml_scholar.utils import (  # noqa: E402
    require_config_fields,
    safe_perplexity,
    seed_everything,
)

LOGGER = logging.getLogger("train_bigram")

REQUIRED_CONFIG_FIELDS = {
    "seed",
    "train_fraction",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "num_steps",
    "evaluation_interval",
    "evaluation_batches",
    "generation_length",
    "sampling_temperature",
    "checkpoint_directory",
}


def _validate_integer(config: dict[str, Any], name: str, *, minimum: int) -> int:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Configuration field {name!r} must be an integer.")
    if value < minimum:
        raise ValueError(
            f"Configuration field {name!r} must be at least {minimum}, got {value}."
        )
    return value


def _validate_float(
    config: dict[str, Any],
    name: str,
    *,
    minimum: float,
    strict: bool,
) -> float:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Configuration field {name!r} must be a real number.")
    normalized = float(value)
    boundary_invalid = normalized <= minimum if strict else normalized < minimum
    if not np.isfinite(normalized) or boundary_invalid:
        relation = "greater than" if strict else "at least"
        raise ValueError(
            f"Configuration field {name!r} must be finite and {relation} {minimum}."
        )
    return normalized


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the bigram training configuration."""
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file does not exist: {source}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Configuration is not valid JSON: {source}") from error
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object.")
    require_config_fields(config, REQUIRED_CONFIG_FIELDS)

    _validate_integer(config, "seed", minimum=0)
    _validate_integer(config, "batch_size", minimum=1)
    _validate_integer(config, "num_steps", minimum=1)
    _validate_integer(config, "evaluation_interval", minimum=1)
    _validate_integer(config, "evaluation_batches", minimum=1)
    _validate_integer(config, "generation_length", minimum=0)
    train_fraction = _validate_float(config, "train_fraction", minimum=0.0, strict=True)
    if train_fraction >= 1.0:
        raise ValueError("Configuration field 'train_fraction' must be below 1.0.")
    _validate_float(config, "learning_rate", minimum=0.0, strict=True)
    _validate_float(config, "weight_decay", minimum=0.0, strict=False)
    _validate_float(config, "sampling_temperature", minimum=0.0, strict=True)
    checkpoint_directory = config["checkpoint_directory"]
    if not isinstance(checkpoint_directory, str) or not checkpoint_directory.strip():
        raise ValueError(
            "Configuration field 'checkpoint_directory' must be a non-empty string."
        )
    return config


def evaluate(
    model: BigramLanguageModel,
    sampler: MiniBatchSampler,
    evaluation_batches: int,
) -> float:
    """Estimate mean loss across a fixed number of sampled minibatches."""
    was_training = model.training
    model.eval()
    try:
        losses = [model.loss(*sampler.next_batch()) for _ in range(evaluation_batches)]
    finally:
        if was_training:
            model.train()
    return float(np.mean(losses))


def _output_directory(configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train(config: dict[str, Any], text: str, corpus_source: str) -> dict[str, Any]:
    """Run reproducible minibatch SGD and persist all milestone artifacts."""
    seed = int(config["seed"])
    seed_everything(seed)
    dataset = prepare_bigram_dataset(text, float(config["train_fraction"]))

    train_sampler = MiniBatchSampler(
        dataset.train_inputs,
        dataset.train_targets,
        int(config["batch_size"]),
        seed + 1,
    )
    train_evaluation_sampler = MiniBatchSampler(
        dataset.train_inputs,
        dataset.train_targets,
        int(config["batch_size"]),
        seed + 2,
    )
    validation_sampler = MiniBatchSampler(
        dataset.validation_inputs,
        dataset.validation_targets,
        int(config["batch_size"]),
        seed + 3,
    )

    model = BigramLanguageModel(dataset.tokenizer.vocabulary_size, seed=seed)
    optimizer = SGD(
        model.parameters(),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    output_directory = _output_directory(str(config["checkpoint_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_directory / "best_model.npz"
    final_checkpoint_path = output_directory / "final_model.npz"
    tokenizer_path = output_directory / "tokenizer.json"
    history_path = output_directory / "history.json"
    summary_path = output_directory / "run_summary.json"
    dataset.tokenizer.save(tokenizer_path)

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_step = 0

    for step in range(1, int(config["num_steps"]) + 1):
        optimizer.zero_grad(model.gradients())
        batch_inputs, batch_targets = train_sampler.next_batch()
        model.loss_and_backward(batch_inputs, batch_targets)
        optimizer.step(model.gradients())

        should_evaluate = (
            step == 1
            or step % int(config["evaluation_interval"]) == 0
            or step == int(config["num_steps"])
        )
        if not should_evaluate:
            continue

        train_loss = evaluate(
            model, train_evaluation_sampler, int(config["evaluation_batches"])
        )
        validation_loss = evaluate(
            model, validation_sampler, int(config["evaluation_batches"])
        )
        sample = generate_text(
            model,
            dataset.tokenizer,
            max_new_tokens=int(config["generation_length"]),
            seed=seed + step,
            temperature=float(config["sampling_temperature"]),
            seed_text=text[0],
        )
        record = {
            "step": step,
            "train_loss": train_loss,
            "train_perplexity": safe_perplexity(train_loss),
            "validation_loss": validation_loss,
            "validation_perplexity": safe_perplexity(validation_loss),
            "sample": sample,
        }
        history.append(record)
        LOGGER.info(
            "step=%d train_loss=%.6f validation_loss=%.6f "
            "train_perplexity=%.3f validation_perplexity=%.3f",
            step,
            train_loss,
            validation_loss,
            record["train_perplexity"],
            record["validation_perplexity"],
        )
        LOGGER.info("sample=%r", sample)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_step = step
            model.save_checkpoint(checkpoint_path)

    model.save_checkpoint(final_checkpoint_path)
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    summary: dict[str, Any] = {
        "configuration": config,
        "corpus_source": corpus_source,
        "corpus_characters": len(text),
        "chronological_split_index": dataset.split_index,
        "training_examples": int(dataset.train_inputs.size),
        "validation_examples": int(dataset.validation_inputs.size),
        "vocabulary_size": dataset.tokenizer.vocabulary_size,
        "parameter_count": model.parameter_count,
        "seed": seed,
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": safe_perplexity(best_validation_loss),
        "output_paths": {
            "best_checkpoint": display_path(checkpoint_path),
            "final_checkpoint": display_path(final_checkpoint_path),
            "tokenizer": display_path(tokenizer_path),
            "history": display_path(history_path),
            "summary": display_path(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the from-scratch NumPy character bigram model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local UTF-8 corpus. Omit only for the built-in smoke corpus.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "bigram_small.json",
        help="Path to a JSON training configuration.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(arguments)
    config = load_config(args.config)
    if args.input is None:
        text = FALLBACK_CORPUS
        corpus_source = "built-in fallback corpus"
        LOGGER.warning(
            "No --input supplied; using the tiny built-in corpus for a smoke run."
        )
    else:
        text = load_utf8_text(args.input)
        corpus_source = str(args.input)

    summary = train(config, text, corpus_source)
    LOGGER.info(
        "complete best_step=%d best_validation_loss=%.6f summary=%s",
        summary["best_step"],
        summary["best_validation_loss"],
        summary["output_paths"]["summary"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
