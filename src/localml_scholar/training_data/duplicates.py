"""Stable exact and near-duplicate clustering for instruction examples."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from localml_scholar.training_data.schemas import GroundedInstructionExample

_TOKEN = re.compile(r"[a-z0-9]+")


def normalized_example_text(example: GroundedInstructionExample) -> str:
    """Return punctuation-insensitive question/answer text for comparisons."""
    if not isinstance(example, GroundedInstructionExample):
        raise TypeError("example must be GroundedInstructionExample.")
    question = next(
        (turn.content for turn in reversed(example.turns) if turn.role == "user"), ""
    )
    return " ".join(_TOKEN.findall(f"{question} {example.final_answer}".casefold()))


def _jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def cluster_duplicates(
    examples: tuple[GroundedInstructionExample, ...],
    *,
    near_duplicate_threshold: float = 0.90,
) -> dict[str, Any]:
    """Cluster exact/near duplicates with deterministic union-find ordering."""
    if not isinstance(examples, tuple) or not all(
        isinstance(item, GroundedInstructionExample) for item in examples
    ):
        raise TypeError("examples must be a tuple of GroundedInstructionExample.")
    if not isinstance(near_duplicate_threshold, (int, float)) or isinstance(
        near_duplicate_threshold, bool
    ):
        raise TypeError("near_duplicate_threshold must be numeric.")
    threshold = float(near_duplicate_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in [0, 1].")
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    parent = list(range(len(ordered)))

    def root(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    texts = [normalized_example_text(item) for item in ordered]
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            exact = texts[left] == texts[right]
            near = _jaccard(texts[left], texts[right]) >= threshold
            evidence_repeat = (
                ordered[left].evidence == ordered[right].evidence
                and ordered[left].final_answer.casefold()
                == ordered[right].final_answer.casefold()
            )
            if exact or near or evidence_repeat:
                first, second = root(left), root(right)
                if first != second:
                    parent[second] = first
    groups: dict[int, list[str]] = {}
    for position, example in enumerate(ordered):
        groups.setdefault(root(position), []).append(example.example_id)
    mapping: dict[str, str] = {}
    clusters = []
    for members in sorted(groups.values(), key=lambda values: values[0]):
        digest = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]
        cluster_id = f"duplicate_cluster_{digest}"
        for member in members:
            mapping[member] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "example_ids": members,
                "size": len(members),
                "duplicate": len(members) > 1,
            }
        )
    return {
        "near_duplicate_threshold": threshold,
        "cluster_by_example_id": mapping,
        "clusters": clusters,
        "duplicate_cluster_count": sum(item["duplicate"] for item in clusters),
    }
