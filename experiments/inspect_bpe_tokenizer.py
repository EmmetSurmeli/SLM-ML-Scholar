#!/usr/bin/env python3
"""Inspect deterministic byte-pair training on a transparent tiny corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from localml_scholar._version import __version__  # noqa: E402
from localml_scholar.tokenizer import (  # noqa: E402
    BPETrainingConfig,
    BytePairTokenizer,
)


def inspect_bpe(
    *,
    text: str = "banana bandana",
    output_directory: str | Path = "outputs/bpe_inspection",
) -> dict[str, Any]:
    """Train a tiny BPE vocabulary and save every inspectable stage."""
    config = BPETrainingConfig(
        target_vocabulary_size=259,
        minimum_pair_frequency=2,
    )
    tokenizer, trace = BytePairTokenizer.train_with_trace(text, config)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    raw_bytes = list(text.encode("utf-8"))
    expansions = {
        str(token_id): list(tokenizer.token_bytes(token_id))
        for token_id in range(256, tokenizer.vocabulary_size)
    }
    summary: dict[str, Any] = {
        "milestone": 7,
        "package_version": __version__,
        "purpose": "transparent deterministic byte-pair training inspection",
        "text": text,
        "utf8_bytes": raw_bytes,
        "initial_byte_tokens": raw_bytes,
        "training_configuration": config.to_dict(),
        "merge_trace": trace,
        "merge_rules": [rule.to_dict() for rule in tokenizer.merge_rules],
        "encoded_tokens": encoded.tolist(),
        "token_expansions": expansions,
        "decoded_text": decoded,
        "round_trip_exact": decoded == text,
        "raw_byte_count": len(raw_bytes),
        "encoded_token_count": int(encoded.size),
        "average_bytes_per_token": len(raw_bytes) / int(encoded.size),
        "token_to_byte_ratio": int(encoded.size) / len(raw_bytes),
        "tokenizer_state_sha256": tokenizer.state_hash(),
    }
    if not summary["round_trip_exact"]:
        raise RuntimeError("BPE inspection failed exact text round-trip.")

    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer.save(destination / "tokenizer.json")
    summary_path = destination / "run_summary.json"
    summary["artifacts"] = {
        "tokenizer": str(tokenizer_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"text: {text!r}")
    print(f"UTF-8 bytes: {raw_bytes}")
    for step in trace:
        print(
            f"rank {step['rank']}: selected={step['selected_pair']} "
            f"frequency={step['selected_frequency']} "
            f"new_token={step['new_token_id']}"
        )
        print(f"  pair counts: {step['pair_counts']}")
        print(f"  sequence after: {step['sequences_after'][0]}")
    print(f"encoded: {encoded.tolist()}")
    print(f"expansions: {expansions}")
    print(f"decoded: {decoded!r}")
    print(f"round trip exact: {summary['round_trip_exact']}")
    print(f"summary: {summary_path}")
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect independently implemented byte-level BPE training."
    )
    parser.add_argument("--text", default="banana bandana")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/bpe_inspection"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    inspect_bpe(text=args.text, output_directory=args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
