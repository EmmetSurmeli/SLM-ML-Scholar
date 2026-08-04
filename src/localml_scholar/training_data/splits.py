"""Leakage-resistant paper-level dataset split assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


def assign_paper_splits(
    paper_ids: Sequence[str],
    *,
    seed: int = 0,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    manual_assignments: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign each paper to exactly one split using a stable hash ordering."""
    if not isinstance(paper_ids, Sequence) or isinstance(paper_ids, (str, bytes)):
        raise TypeError("paper_ids must be a sequence of strings.")
    if not all(isinstance(item, str) and item.strip() for item in paper_ids):
        raise ValueError("paper_ids must contain non-empty strings.")
    unique = tuple(sorted(set(paper_ids)))
    if not unique:
        raise ValueError("paper_ids must not be empty.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below 1.")
    manual = {} if manual_assignments is None else dict(manual_assignments)
    unknown = set(manual) - set(unique)
    if unknown:
        raise ValueError(
            f"Manual assignments contain unknown papers: {sorted(unknown)}."
        )
    if any(split not in {"train", "validation", "test"} for split in manual.values()):
        raise ValueError("Manual split values must be train, validation, or test.")

    output = dict(manual)
    for paper_id in unique:
        if paper_id in output:
            continue
        digest = hashlib.sha256(f"{seed}:{paper_id}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
        if unit < test_fraction:
            split = "test"
        elif unit < test_fraction + validation_fraction:
            split = "validation"
        else:
            split = "train"
        output[paper_id] = split
    return dict(sorted(output.items()))


def validate_prompt_variation_splits(
    example_to_parent: Mapping[str, str | None],
    example_splits: Mapping[str, str],
) -> None:
    """Reject prompt variations assigned apart from their parent target."""
    for example_id, parent_id in example_to_parent.items():
        if parent_id is None:
            continue
        if parent_id not in example_splits:
            raise ValueError(f"Prompt variation {example_id} has an unknown parent.")
        if example_splits.get(example_id) != example_splits[parent_id]:
            raise ValueError(
                f"Prompt variation {example_id} is split apart from parent {parent_id}."
            )
