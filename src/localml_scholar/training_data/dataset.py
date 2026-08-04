"""Approved-only dataset construction, persistence, and reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from localml_scholar.serialization import atomic_write_text
from localml_scholar.training_data.diversity import (
    diversity_metrics,
    diversity_warnings,
)
from localml_scholar.training_data.schemas import (
    GroundedInstructionDataset,
    GroundedInstructionExample,
)
from localml_scholar.training_data.splits import assign_paper_splits
from localml_scholar.training_data.trust import TRUST_TIERS, select_trusted_examples


def _coalesce_cross_paper_splits(
    examples: tuple[GroundedInstructionExample, ...],
    splits: dict[str, str],
    manual_assignments: dict[str, str] | None,
) -> dict[str, str]:
    parent = {paper_id: paper_id for paper_id in splits}

    def root(paper_id: str) -> str:
        while parent[paper_id] != paper_id:
            parent[paper_id] = parent[parent[paper_id]]
            paper_id = parent[paper_id]
        return paper_id

    for example in examples:
        anchor = root(example.paper_ids[0])
        for paper_id in example.paper_ids[1:]:
            other = root(paper_id)
            if anchor != other:
                parent[other] = anchor
    groups: dict[str, list[str]] = {}
    for paper_id in splits:
        groups.setdefault(root(paper_id), []).append(paper_id)
    manual = {} if manual_assignments is None else manual_assignments
    output = dict(splits)
    for papers in groups.values():
        explicit = {manual[paper] for paper in papers if paper in manual}
        if len(explicit) > 1:
            raise ValueError(
                "Manual assignments place connected cross-paper examples in "
                "different splits."
            )
        chosen = next(iter(explicit)) if explicit else splits[min(papers)]
        for paper_id in papers:
            output[paper_id] = chosen
    return dict(sorted(output.items()))


def build_dataset(
    examples: tuple[GroundedInstructionExample, ...],
    *,
    dataset_version: str = "1.0",
    seed: int = 0,
    manual_paper_splits: dict[str, str] | None = None,
    approved_only: bool = True,
    trust_tier: str = "human-only",
    deduplicate: bool = True,
) -> GroundedInstructionDataset:
    """Create a trust-filtered dataset with paper-grouped, leakage-free splits."""
    if not isinstance(examples, tuple) or not all(
        isinstance(item, GroundedInstructionExample) for item in examples
    ):
        raise TypeError("examples must be a tuple of GroundedInstructionExample.")
    if trust_tier not in TRUST_TIERS:
        raise ValueError(f"trust_tier must be one of {sorted(TRUST_TIERS)}.")
    selected = (
        select_trusted_examples(
            examples, trust_tier=trust_tier, deduplicate=deduplicate
        )
        if approved_only
        else examples
    )
    if not approved_only and any(
        item.review_status != "human_approved" for item in selected
    ):
        raise ValueError(
            "Training dataset exports cannot include proposed or rejected examples."
        )
    leaking = sorted(
        {
            paper_id
            for item in selected
            for paper_id in item.paper_ids
            if item.metadata.get("test_only") is True
            or paper_id in set(item.metadata.get("test_only_paper_ids", []))
        }
    )
    if leaking:
        raise ValueError(
            "Training export would leak corrections from designated test-only "
            f"papers: {leaking}."
        )
    paper_ids = tuple(paper for item in selected for paper in item.paper_ids)
    if not paper_ids:
        raise ValueError(
            "No human-approved or otherwise trust-tier-eligible examples are "
            f"available for {trust_tier!r}."
        )
    splits = assign_paper_splits(
        paper_ids,
        seed=seed,
        manual_assignments=manual_paper_splits,
    )
    splits = _coalesce_cross_paper_splits(selected, splits, manual_paper_splits)
    assigned = tuple(
        replace(item, split=splits[item.paper_ids[0]]) for item in selected
    )
    metrics = diversity_metrics(assigned)
    return GroundedInstructionDataset(
        dataset_version=dataset_version,
        examples=tuple(sorted(assigned, key=lambda item: item.example_id)),
        paper_splits=splits,
        metadata={
            "approved_only": True,
            "trust_tier": trust_tier,
            "deduplicated": deduplicate,
            "trust_weights": {
                item.example_id: item.metadata["trust_weight"] for item in assigned
            },
            "split_seed": seed,
            "diversity": metrics,
            "warnings": list(diversity_warnings(metrics)),
        },
    )


def save_dataset(dataset: GroundedInstructionDataset, path: str | Path) -> Path:
    """Atomically save one finite, deterministic JSON artifact."""
    if not isinstance(dataset, GroundedInstructionDataset):
        raise TypeError("dataset must be GroundedInstructionDataset.")
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("Dataset output path must end with .json.")
    payload = json.dumps(
        dataset.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )
    return atomic_write_text(destination, payload + "\n")


def load_dataset(path: str | Path) -> GroundedInstructionDataset:
    """Load and verify one dataset artifact, including its content hash."""
    source = Path(path)
    try:
        state = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset does not exist: {source}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Dataset is not valid UTF-8 JSON: {source}") from error
    if not isinstance(state, dict):
        raise ValueError("Dataset JSON must contain one object.")
    claimed_hash = state.pop("dataset_sha256", None)
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    if claimed_hash != hashlib.sha256(encoded).hexdigest():
        raise ValueError("Dataset SHA-256 does not match its contents.")
    examples = tuple(
        GroundedInstructionExample.from_dict(item) for item in state["examples"]
    )
    return GroundedInstructionDataset(
        format_version=state["format_version"],
        dataset_version=state["dataset_version"],
        examples=examples,
        paper_splits=state["paper_splits"],
        metadata=state.get("metadata", {}),
    )


def dataset_report(dataset: GroundedInstructionDataset) -> dict[str, Any]:
    """Return split, diversity, and leakage diagnostics for one dataset."""
    if not isinstance(dataset, GroundedInstructionDataset):
        raise TypeError("dataset must be GroundedInstructionDataset.")
    metrics = diversity_metrics(dataset.examples)
    split_counts = {
        split: sum(item.split == split for item in dataset.examples)
        for split in ("train", "validation", "test")
    }
    paper_split_counts = {
        split: sum(value == split for value in dataset.paper_splits.values())
        for split in ("train", "validation", "test")
    }
    return {
        "dataset_version": dataset.dataset_version,
        "split_example_counts": split_counts,
        "split_paper_counts": paper_split_counts,
        "diversity": metrics,
        "warnings": list(diversity_warnings(metrics)),
        "paper_level_leakage": False,
    }
