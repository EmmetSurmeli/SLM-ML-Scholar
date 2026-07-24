#!/usr/bin/env python3
"""Compare controlled one-head and two-head tiny-transformer runs."""

from __future__ import annotations

import argparse
import json
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


def _run_configuration(
    *,
    number_of_heads: int,
    seed: int,
    steps: int,
    output_directory: Path,
) -> dict[str, Any]:
    corpus = "abcde" * 48
    dataset = prepare_token_stream_dataset(corpus, train_fraction=0.8)
    model_config = TransformerConfig(
        vocabulary_size=dataset.tokenizer.vocabulary_size,
        maximum_context_length=6,
        model_dimension=8,
        number_of_layers=1,
        number_of_heads=number_of_heads,
        key_dimension=2,
        value_dimension=2,
        feed_forward_dimension=16,
        dtype=np.float32,
        seed=seed,
    )
    training_config = TransformerTrainingConfig(
        batch_size=6,
        sequence_length=6,
        maximum_steps=steps,
        evaluation_interval=max(1, steps // 4),
        evaluation_batches=2,
        checkpoint_interval=max(1, steps // 2),
        logging_interval=max(1, steps // 5),
        optimizer_name="adam",
        learning_rate=0.015,
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
    initial = trainer.evaluate()
    start = time.perf_counter()
    trainer.run()
    elapsed_seconds = time.perf_counter() - start
    final = trainer.evaluate()
    prompt = dataset.tokenizer.encode("a")[None, :]
    generated = generate_transformer_ids(
        trainer.model,
        prompt,
        max_new_tokens=20,
        greedy=True,
    )
    checkpoint = trainer.save_checkpoint(output_directory / "final_training.npz")
    reloaded = TransformerTrainer.load_checkpoint(
        checkpoint,
        train_tokens=dataset.train_tokens,
        validation_tokens=dataset.validation_tokens,
        tokenizer=dataset.tokenizer,
        expected_model_config=model_config,
        expected_training_config=training_config,
    )
    with trainer.model.inference_mode():
        expected_logits = trainer.model.forward(prompt)
    with reloaded.model.inference_mode():
        actual_logits = reloaded.model.forward(prompt)
    gradient_norms = [
        record["training_step"]["pre_clipping_gradient_norm"]
        for record in trainer.history
        if "training_step" in record
    ]
    return {
        "number_of_heads": number_of_heads,
        "per_head_key_dimension": model_config.key_dimension,
        "per_head_value_dimension": model_config.value_dimension,
        "total_query_width": number_of_heads * model_config.key_dimension,
        "total_value_width": number_of_heads * model_config.value_dimension,
        "parameter_count": trainer.model.parameter_count,
        "model_configuration": model_config.to_dict(),
        "initial_train_loss": initial["train"].loss,
        "initial_validation_loss": initial["validation"].loss,
        "final_train_loss": final["train"].loss,
        "final_validation_loss": final["validation"].loss,
        "initial_validation_perplexity": initial["validation"].perplexity,
        "final_validation_perplexity": final["validation"].perplexity,
        "best_validation_loss": trainer.best_validation_loss,
        "gradient_norm_minimum": min(gradient_norms),
        "gradient_norm_maximum": max(gradient_norms),
        "elapsed_seconds": elapsed_seconds,
        "generated_text": dataset.tokenizer.decode(generated[0]),
        "checkpoint_reload_logits_equal": bool(
            np.array_equal(expected_logits, actual_logits)
        ),
        "checkpoint": str(checkpoint),
    }


def compare_single_and_multi_head(
    *,
    seed: int = 67,
    steps: int = 30,
    output_directory: str | Path = "outputs/single_vs_multi_head",
) -> dict[str, Any]:
    """Run a controlled comparison whose only architectural change is H."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)

    runs = [
        _run_configuration(
            number_of_heads=heads,
            seed=seed,
            steps=steps,
            output_directory=destination / f"heads_{heads}",
        )
        for heads in (1, 2)
    ]
    if not all(run["checkpoint_reload_logits_equal"] for run in runs):
        raise RuntimeError("A comparison checkpoint changed model logits.")
    summary: dict[str, Any] = {
        "milestone": 6,
        "package_version": __version__,
        "purpose": "controlled one-head versus two-head correctness comparison",
        "claim_boundary": (
            "This tiny run demonstrates integration only; it is not evidence "
            "that either head count is generally better."
        ),
        "controlled_settings": {
            "seed": seed,
            "steps": steps,
            "corpus": "abcde repeated 48 times",
            "per_head_dimensions_held_constant": True,
            "changed_architectural_field": "number_of_heads",
        },
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
        description="Compare deterministic one-head and two-head transformer runs."
    )
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/single_vs_multi_head"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    compare_single_and_multi_head(
        seed=args.seed,
        steps=args.steps,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
