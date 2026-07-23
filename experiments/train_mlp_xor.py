#!/usr/bin/env python3
"""Train the manually differentiated MLP on the four XOR examples."""

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

from localml_scholar.losses import (  # noqa: E402
    softmax_cross_entropy_loss_and_gradient,
    stable_softmax,
)
from localml_scholar.models.mlp import MLP  # noqa: E402
from localml_scholar.optim.adam import Adam  # noqa: E402
from localml_scholar.utils import seed_everything  # noqa: E402

LOGGER = logging.getLogger("train_mlp_xor")


def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic four-example XOR classification dataset."""
    inputs = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    targets = np.asarray([0, 1, 1, 0], dtype=np.int64)
    return inputs, targets


def train_xor(
    *,
    seed: int = 17,
    steps: int = 1_000,
    learning_rate: float = 0.05,
    hidden_dim: int = 8,
    output_directory: str | Path = "outputs/mlp_xor",
    report_interval: int = 200,
) -> dict[str, Any]:
    """Train and save a deterministic XOR correctness demonstration."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    for name, value in (
        ("steps", steps),
        ("hidden_dim", hidden_dim),
        ("report_interval", report_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be a real number.")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")

    seed_everything(seed)
    inputs, targets = xor_dataset()
    model = MLP(2, hidden_dim, 2, activation="gelu", seed=seed)
    optimizer = Adam(model.parameters(), learning_rate=float(learning_rate))

    model.eval()
    initial_logits = model.forward(inputs)
    initial_loss, _ = softmax_cross_entropy_loss_and_gradient(initial_logits, targets)
    model.train()

    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model.forward(inputs)
        loss, grad_logits = softmax_cross_entropy_loss_and_gradient(logits, targets)
        model.backward(grad_logits)
        optimizer.step()
        if step == 1 or step % report_interval == 0 or step == steps:
            history.append({"step": step, "loss_before_update": loss})
            LOGGER.info("step=%d loss_before_update=%.12f", step, loss)

    model.eval()
    final_logits = model.forward(inputs)
    final_loss, _ = softmax_cross_entropy_loss_and_gradient(final_logits, targets)
    probabilities = stable_softmax(final_logits)
    predictions = np.argmax(probabilities, axis=-1)

    output_path = Path(output_directory)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    model_checkpoint = output_path / "model.npz"
    optimizer_checkpoint = output_path / "optimizer.npz"
    summary_path = output_path / "run_summary.json"
    model.save_checkpoint(model_checkpoint)
    optimizer.save_checkpoint(optimizer_checkpoint)

    reloaded = MLP.load_checkpoint(model_checkpoint).eval()
    reloaded_logits = reloaded.forward(inputs)
    checkpoint_round_trip_exact = np.array_equal(reloaded_logits, final_logits)
    summary: dict[str, Any] = {
        "purpose": "manual-backpropagation XOR correctness demonstration",
        "configuration": {
            "seed": seed,
            "steps": steps,
            "learning_rate": float(learning_rate),
            "hidden_dim": hidden_dim,
            "activation": "gelu",
            "dtype": "float64",
            "report_interval": report_interval,
        },
        "parameter_count": model.parameter_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "correct_predictions": int(np.sum(predictions == targets)),
        "checkpoint_round_trip_exact": checkpoint_round_trip_exact,
        "history": history,
        "output_paths": {
            "model_checkpoint": str(model_checkpoint),
            "optimizer_checkpoint": str(optimizer_checkpoint),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "complete initial_loss=%.12f final_loss=%.12f predictions=%s summary=%s",
        initial_loss,
        final_loss,
        predictions.tolist(),
        summary_path,
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the from-scratch MLP on deterministic XOR."
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("outputs/mlp_xor")
    )
    parser.add_argument("--report-interval", type=int, default=200)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(arguments)
    train_xor(
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        output_directory=args.output_directory,
        report_interval=args.report_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
