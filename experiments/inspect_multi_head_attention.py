#!/usr/bin/env python3
"""Inspect a deterministic fused multi-head causal-attention calculation."""

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
from localml_scholar.nn.attention import (  # noqa: E402
    CausalSelfAttentionHead,
    MultiHeadCausalSelfAttention,
)
from localml_scholar.nn.embedding import Embedding  # noqa: E402


def _gradient_norm(*gradients: np.ndarray) -> float:
    maximum = max(
        (float(np.max(np.abs(gradient))) for gradient in gradients),
        default=0.0,
    )
    if maximum == 0.0:
        return 0.0
    scaled_sum = sum(float(np.sum((gradient / maximum) ** 2)) for gradient in gradients)
    return maximum * float(np.sqrt(scaled_sum))


def _projection_gradient_norm(
    attention: MultiHeadCausalSelfAttention,
    name: str,
) -> float:
    projection = getattr(attention, name)
    gradients = [projection.weight.grad]
    if projection.bias is not None:
        gradients.append(projection.bias.grad)
    return _gradient_norm(*gradients)


def inspect_multi_head_attention(
    *,
    seed: int = 61,
    output_directory: str | Path = "outputs/multi_head_attention_inspection",
) -> dict[str, Any]:
    """Run forward, backward, causality, and one-head equivalence checks."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    token_ids = np.asarray([[0, 2, 1, 3]], dtype=np.int64)
    embedding = Embedding(5, 4, seed=seed, dtype=np.float64)
    attention = MultiHeadCausalSelfAttention(
        input_dim=4,
        number_of_heads=2,
        key_dim=2,
        value_dim=2,
        bias=True,
        seed=seed + 1,
        dtype=np.float64,
    )
    embeddings = embedding.forward(token_ids)
    output, details = attention.forward(embeddings, return_attention=True)
    blocked = np.broadcast_to(~details.allowed_mask, details.probabilities.shape)
    future_probabilities_exactly_zero = bool(
        np.all(details.probabilities[blocked] == 0.0)
    )
    probability_row_sums = np.sum(details.probabilities, axis=-1)

    loss = 0.5 * float(np.sum(output * output))
    grad_embeddings = attention.backward(output)
    embedding.backward(grad_embeddings)
    gradient_norms = {
        "input_embeddings": _gradient_norm(grad_embeddings),
        "embedding_table": _gradient_norm(embedding.weight.grad),
        "query_projection": _projection_gradient_norm(
            attention,
            "query_projection",
        ),
        "key_projection": _projection_gradient_norm(attention, "key_projection"),
        "value_projection": _projection_gradient_norm(
            attention,
            "value_projection",
        ),
        "output_projection": _projection_gradient_norm(
            attention,
            "output_projection",
        ),
    }

    changed = np.array(embeddings, copy=True)
    changed[:, -1, :] += 20.0
    with attention.inference_mode():
        original_output = attention.forward(embeddings)
        changed_output = attention.forward(changed)
    earlier_outputs_unchanged = bool(
        np.array_equal(original_output[:, :-1], changed_output[:, :-1])
    )

    legacy = CausalSelfAttentionHead(
        input_dim=4,
        key_dim=2,
        value_dim=2,
        bias=True,
        output_projection=True,
        seed=seed + 3,
        dtype=np.float64,
    ).eval()
    fused_one_head = MultiHeadCausalSelfAttention(
        input_dim=4,
        number_of_heads=1,
        key_dim=2,
        value_dim=2,
        bias=True,
        seed=seed + 3,
        dtype=np.float64,
    ).eval()
    one_head_maximum_absolute_difference = float(
        np.max(np.abs(legacy.forward(embeddings) - fused_one_head.forward(embeddings)))
    )

    per_head_summary = [
        {
            "head": head,
            "minimum_probability": float(np.min(details.probabilities[:, head])),
            "maximum_probability": float(np.max(details.probabilities[:, head])),
            "maximum_row_sum_error": float(
                np.max(np.abs(probability_row_sums[:, head] - 1.0))
            ),
        }
        for head in range(attention.number_of_heads)
    ]
    summary: dict[str, Any] = {
        "milestone": 6,
        "package_version": __version__,
        "purpose": "fused multi-head causal-attention inspection",
        "seed": seed,
        "token_ids": token_ids.tolist(),
        "configuration": attention.configuration,
        "parameter_count": attention.parameter_count + embedding.weight.size,
        "shapes": {
            "embeddings": list(embeddings.shape),
            "query_flat": list(details.query_flat.shape),
            "query_per_head": list(details.query.shape),
            "scores": list(details.scaled_scores.shape),
            "probabilities": list(details.probabilities.shape),
            "head_outputs": list(details.head_outputs.shape),
            "concatenated": list(details.concatenated.shape),
            "output": list(output.shape),
        },
        "scaled_scores": details.scaled_scores.tolist(),
        "causal_allowed_mask": details.allowed_mask.tolist(),
        "attention_probabilities": details.probabilities.tolist(),
        "probability_row_sums": probability_row_sums.tolist(),
        "per_head_probability_summary": per_head_summary,
        "synthetic_loss": loss,
        "gradient_norms": gradient_norms,
        "future_probabilities_exactly_zero": future_probabilities_exactly_zero,
        "earlier_outputs_unchanged_after_future_change": earlier_outputs_unchanged,
        "one_head_maximum_absolute_difference": (one_head_maximum_absolute_difference),
    }
    if not future_probabilities_exactly_zero:
        raise RuntimeError("A blocked future probability was nonzero.")
    if not earlier_outputs_unchanged:
        raise RuntimeError("A future-token change altered an earlier output.")
    if one_head_maximum_absolute_difference != 0.0:
        raise RuntimeError("Fused one-head output differs from legacy attention.")

    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "run_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"input shape: {embeddings.shape}")
    print(f"flat query shape: {details.query_flat.shape}")
    print(f"per-head query shape: {details.query.shape}")
    print(f"score/probability shape: {details.probabilities.shape}")
    print(f"concatenated/output shapes: {details.concatenated.shape}, {output.shape}")
    print(f"causal allowed mask (True = allowed):\n{details.allowed_mask}")
    print(
        "attention probabilities:\n"
        + np.array2string(
            details.probabilities,
            precision=5,
            suppress_small=True,
        )
    )
    print(f"future probabilities exactly zero: {future_probabilities_exactly_zero}")
    print(
        f"one-head maximum absolute difference: {one_head_maximum_absolute_difference}"
    )
    for name, norm in gradient_norms.items():
        print(f"{name} gradient norm: {norm:.12f}")
    print(f"summary: {summary_path}")
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect fused manually differentiated multi-head attention."
    )
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/multi_head_attention_inspection"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    inspect_multi_head_attention(
        seed=args.seed,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
