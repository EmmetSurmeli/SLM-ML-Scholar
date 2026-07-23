#!/usr/bin/env python3
"""Inspect one deterministic embedding-to-causal-attention computation."""

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

from localml_scholar.nn.attention import (  # noqa: E402
    CausalSelfAttentionHead,
)
from localml_scholar.nn.embedding import Embedding  # noqa: E402
from localml_scholar.utils import seed_everything  # noqa: E402

LOGGER = logging.getLogger("inspect_single_head_attention")


def _gradient_norm(*gradients: np.ndarray) -> float:
    """Return a stable joint Euclidean norm for the supplied gradients."""
    maximum = max(
        (float(np.max(np.abs(gradient))) for gradient in gradients),
        default=0.0,
    )
    if maximum == 0.0:
        return 0.0
    scaled_sum = sum(float(np.sum((gradient / maximum) ** 2)) for gradient in gradients)
    return maximum * float(np.sqrt(scaled_sum))


def _projection_gradient_norm(model: CausalSelfAttentionHead, name: str) -> float:
    projection = getattr(model, name)
    gradients = [projection.weight.grad]
    if projection.bias is not None:
        gradients.append(projection.bias.grad)
    return _gradient_norm(*gradients)


def inspect_attention(
    *,
    seed: int = 31,
    output_directory: str | Path = "outputs/attention_inspection",
) -> dict[str, Any]:
    """Run forward/backward inspection and save a machine-readable summary."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    seed_everything(seed)

    token_ids = np.asarray([[0, 2, 1, 3]], dtype=np.int64)
    embedding = Embedding(
        vocabulary_size=5,
        embedding_dim=4,
        seed=seed,
        dtype=np.float64,
    )
    attention = CausalSelfAttentionHead(
        input_dim=4,
        key_dim=2,
        value_dim=3,
        bias=True,
        output_projection=False,
        seed=seed + 1,
        dtype=np.float64,
    )

    input_embeddings = embedding.forward(token_ids)
    output, details = attention.forward(input_embeddings, return_attention=True)
    blocked = np.broadcast_to(~details.allowed_mask, details.probabilities.shape)
    future_probabilities_exactly_zero = bool(
        np.all(details.probabilities[blocked] == 0.0)
    )
    if not future_probabilities_exactly_zero:
        raise AssertionError("A future-token attention probability was nonzero.")

    loss = 0.5 * float(np.sum(output * output))
    grad_input_embeddings = attention.backward(output)
    embedding.backward(grad_input_embeddings)

    gradient_norms = {
        "input_embeddings": _gradient_norm(grad_input_embeddings),
        "embedding_table": _gradient_norm(embedding.weight.grad),
        "query_projection": _projection_gradient_norm(attention, "query_projection"),
        "key_projection": _projection_gradient_norm(attention, "key_projection"),
        "value_projection": _projection_gradient_norm(attention, "value_projection"),
    }
    summary: dict[str, Any] = {
        "purpose": "single-head causal-attention forward/backward inspection",
        "seed": seed,
        "token_ids": token_ids.tolist(),
        "configuration": attention.configuration,
        "parameter_count": attention.parameter_count + embedding.weight.size,
        "shapes": {
            "input": list(input_embeddings.shape),
            "query": list(details.query.shape),
            "key": list(details.key.shape),
            "value": list(details.value.shape),
            "scores": list(details.scaled_scores.shape),
            "attention_probabilities": list(details.probabilities.shape),
            "output": list(output.shape),
        },
        "query": details.query.tolist(),
        "key": details.key.tolist(),
        "value": details.value.tolist(),
        "scaled_scores": details.scaled_scores.tolist(),
        "causal_allowed_mask": details.allowed_mask.tolist(),
        "attention_probabilities": details.probabilities.tolist(),
        "output": output.tolist(),
        "synthetic_loss": loss,
        "gradient_norms": gradient_norms,
        "future_probabilities_exactly_zero": future_probabilities_exactly_zero,
    }

    output_path = Path(output_directory)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "run_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    array_options = {"precision": 5, "suppress_small": True}
    print(f"input shape: {input_embeddings.shape}")
    print(
        f"query/key/value shapes: {details.query.shape}, "
        f"{details.key.shape}, {details.value.shape}"
    )
    print(
        "raw scaled scores:\n" + np.array2string(details.scaled_scores, **array_options)
    )
    print(f"causal allowed mask (True = allowed):\n{details.allowed_mask}")
    print(
        "attention probabilities:\n"
        + np.array2string(details.probabilities, **array_options)
    )
    print(f"output shape: {output.shape}")
    print(f"future probabilities exactly zero: {future_probabilities_exactly_zero}")
    print(f"synthetic loss: {loss:.12f}")
    for name, norm in gradient_norms.items():
        print(f"{name} gradient norm: {norm:.12f}")
    print(f"summary: {summary_path}")
    LOGGER.info("Attention inspection completed successfully.")
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect one deterministic manually differentiated attention head."
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/attention_inspection"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(arguments)
    inspect_attention(seed=args.seed, output_directory=args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
