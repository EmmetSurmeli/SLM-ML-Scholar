#!/usr/bin/env python3
"""Inspect one deterministic manually differentiated pre-norm decoder block."""

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

from localml_scholar.nn.embedding import Embedding  # noqa: E402
from localml_scholar.nn.module import Module  # noqa: E402
from localml_scholar.nn.transformer import (  # noqa: E402
    PreNormDecoderBlock,
)
from localml_scholar.optim.adam import Adam  # noqa: E402
from localml_scholar.utils import seed_everything  # noqa: E402

LOGGER = logging.getLogger("inspect_pre_norm_decoder_block")


def _gradient_norm(*gradients: np.ndarray) -> float:
    maximum = max(
        (float(np.max(np.abs(gradient))) for gradient in gradients),
        default=0.0,
    )
    if maximum == 0.0:
        return 0.0
    scaled_sum = sum(float(np.sum((gradient / maximum) ** 2)) for gradient in gradients)
    return maximum * float(np.sqrt(scaled_sum))


def _module_gradient_norm(module: Module) -> float:
    return _gradient_norm(*(parameter.grad for parameter in module.parameters()))


def inspect_decoder_block(
    *,
    seed: int = 41,
    output_directory: str | Path = "outputs/decoder_block_inspection",
) -> dict[str, Any]:
    """Run a deterministic decoder-block forward, backward, and optimizer step."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    seed_everything(seed)

    token_ids = np.asarray([[0, 2, 1, 3]], dtype=np.int64)
    modified_token_ids = token_ids.copy()
    modified_token_ids[:, -1] = 4
    embedding = Embedding(
        vocabulary_size=6,
        embedding_dim=4,
        seed=seed,
        dtype=np.float64,
    )
    block = PreNormDecoderBlock(
        model_dim=4,
        key_dim=2,
        value_dim=3,
        ff_hidden_dim=7,
        attention_bias=True,
        feed_forward_bias=True,
        attention_output_projection=True,
        layer_norm_epsilon=1e-5,
        activation="gelu",
        seed=seed + 1,
        dtype=np.float64,
    )

    embedding.eval()
    block.eval()
    input_embeddings = embedding.forward(token_ids)
    modified_embeddings = embedding.forward(modified_token_ids)
    output, inspection = block.forward(input_embeddings, return_details=True)
    modified_output = block.forward(modified_embeddings)

    blocked = np.broadcast_to(
        ~inspection.attention.allowed_mask,
        inspection.attention.probabilities.shape,
    )
    future_probabilities_exactly_zero = bool(
        np.all(inspection.attention.probabilities[blocked] == 0.0)
    )
    output_shape_matches_input = output.shape == input_embeddings.shape
    earlier_outputs_unchanged = bool(
        np.array_equal(output[:, :-1, :], modified_output[:, :-1, :])
    )
    if not future_probabilities_exactly_zero:
        raise AssertionError("A future-token attention probability was nonzero.")
    if not output_shape_matches_input:
        raise AssertionError("Decoder-block output shape did not match its input.")
    if not earlier_outputs_unchanged:
        raise AssertionError("Changing a future token changed an earlier output.")

    embedding.train()
    block.train()
    block.zero_grad()
    embedding.zero_grad()
    training_embeddings = embedding.forward(token_ids)
    training_output = block.forward(training_embeddings)
    loss = 0.5 * float(np.sum(training_output * training_output))
    grad_embeddings = block.backward(training_output)
    embedding.backward(grad_embeddings)

    attention_output_projection = block.attention.output_projection
    if attention_output_projection is None:
        raise RuntimeError("Inspection requires an attention output projection.")
    gradient_norms = {
        "embedding_parameters": _module_gradient_norm(embedding),
        "norm1": _module_gradient_norm(block.norm1),
        "query_projection": _module_gradient_norm(block.attention.query_projection),
        "key_projection": _module_gradient_norm(block.attention.key_projection),
        "value_projection": _module_gradient_norm(block.attention.value_projection),
        "attention_output_projection": _module_gradient_norm(
            attention_output_projection
        ),
        "norm2": _module_gradient_norm(block.norm2),
        "feed_forward_linear1": _module_gradient_norm(block.feed_forward.linear1),
        "feed_forward_linear2": _module_gradient_norm(block.feed_forward.linear2),
    }

    parameters_before = [parameter.data.copy() for parameter in block.parameters()]
    optimizer = Adam(block.parameters(), learning_rate=0.01)
    optimizer.step()
    optimizer_changed_parameter = any(
        not np.array_equal(before, parameter.data)
        for before, parameter in zip(
            parameters_before,
            block.parameters(),
            strict=True,
        )
    )
    if not optimizer_changed_parameter:
        raise AssertionError("Adam did not update any decoder-block parameter.")

    summary: dict[str, Any] = {
        "purpose": "pre-norm decoder-block correctness inspection",
        "seed": seed,
        "token_ids": token_ids.tolist(),
        "modified_token_ids": modified_token_ids.tolist(),
        "configuration": block.configuration,
        "embedding_parameter_count": embedding.weight.size,
        "decoder_parameter_count": block.parameter_count,
        "shapes": {
            "embeddings": list(input_embeddings.shape),
            "normalized_attention_input": list(
                inspection.normalized_attention_input.shape
            ),
            "query": list(inspection.attention.query.shape),
            "key": list(inspection.attention.key.shape),
            "value": list(inspection.attention.value.shape),
            "attention_scores": list(inspection.attention.scaled_scores.shape),
            "attention_probabilities": list(inspection.attention.probabilities.shape),
            "first_residual": list(inspection.first_residual.shape),
            "feed_forward_hidden": list(inspection.feed_forward.activation.shape),
            "feed_forward_output": list(inspection.feed_forward_output.shape),
            "output": list(output.shape),
        },
        "causal_allowed_mask": inspection.attention.allowed_mask.tolist(),
        "attention_probabilities": inspection.attention.probabilities.tolist(),
        "synthetic_loss": loss,
        "gradient_norms": gradient_norms,
        "future_probabilities_exactly_zero": (future_probabilities_exactly_zero),
        "output_shape_matches_input": output_shape_matches_input,
        "earlier_outputs_unchanged_after_future_token_change": (
            earlier_outputs_unchanged
        ),
        "optimizer_changed_parameter": optimizer_changed_parameter,
    }

    output_path = Path(output_directory)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "run_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    array_options = {"precision": 5, "suppress_small": True}
    print(f"token IDs: {token_ids.tolist()}")
    print(f"embedding shape: {input_embeddings.shape}")
    print(
        "normalized attention input shape: "
        f"{inspection.normalized_attention_input.shape}"
    )
    print(
        "query/key/value shapes: "
        f"{inspection.attention.query.shape}, "
        f"{inspection.attention.key.shape}, "
        f"{inspection.attention.value.shape}"
    )
    print(f"attention-score shape: {inspection.attention.scaled_scores.shape}")
    print(f"causal allowed mask (True = allowed):\n{inspection.attention.allowed_mask}")
    print(
        "attention probabilities:\n"
        + np.array2string(
            inspection.attention.probabilities,
            **array_options,
        )
    )
    print(f"first residual shape: {inspection.first_residual.shape}")
    print(f"feed-forward hidden shape: {inspection.feed_forward.activation.shape}")
    print(f"feed-forward output shape: {inspection.feed_forward_output.shape}")
    print(f"final output shape: {output.shape}")
    print(f"future probabilities exactly zero: {future_probabilities_exactly_zero}")
    print(f"earlier outputs unchanged: {earlier_outputs_unchanged}")
    print(f"synthetic loss: {loss:.12f}")
    for name, norm in gradient_norms.items():
        print(f"{name} gradient norm: {norm:.12f}")
    print(f"optimizer changed a parameter: {optimizer_changed_parameter}")
    print(f"summary: {summary_path}")
    LOGGER.info("Decoder-block inspection completed successfully.")
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect one manually differentiated pre-norm decoder block."
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/decoder_block_inspection"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(arguments)
    inspect_decoder_block(
        seed=args.seed,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
