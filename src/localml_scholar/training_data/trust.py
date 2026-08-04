"""Trust-tier selection and per-example training weights."""

from __future__ import annotations

from dataclasses import replace

from localml_scholar.training_data.duplicates import cluster_duplicates
from localml_scholar.training_data.schemas import GroundedInstructionExample

TRUST_TIERS = {
    "human-only",
    "human-and-audited",
    "include-codex-approved",
    # Compatibility alias used by early 1.2.1 development snapshots.
    "include-codex",
}
TRUST_WEIGHTS = {
    "human_approved": 1.0,
    "codex_approved_audited": 0.9,
    "codex_approved": 0.6,
}


def _eligible(example: GroundedInstructionExample, trust_tier: str) -> bool:
    if example.review_status == "human_approved":
        return True
    if example.review_status != "codex_approved":
        return False
    provenance = example.metadata.get(
        "approval_provenance", example.metadata.get("review_provenance", {})
    )
    if isinstance(provenance, dict) and provenance.get("circular_warnings"):
        return False
    if trust_tier == "human-only":
        return False
    audited = example.metadata.get("audit_status") in {
        "human_confirmed",
        "passed",
    }
    return trust_tier in {"include-codex", "include-codex-approved"} or audited


def _weight(example: GroundedInstructionExample) -> float:
    if example.review_status == "human_approved":
        return TRUST_WEIGHTS["human_approved"]
    if example.metadata.get("audit_status") in {"human_confirmed", "passed"}:
        return TRUST_WEIGHTS["codex_approved_audited"]
    return TRUST_WEIGHTS["codex_approved"]


def select_trusted_examples(
    examples: tuple[GroundedInstructionExample, ...],
    *,
    trust_tier: str = "human-only",
    deduplicate: bool = True,
) -> tuple[GroundedInstructionExample, ...]:
    """Select eligible examples and annotate stable trust/duplicate metadata."""
    if trust_tier not in TRUST_TIERS:
        raise ValueError(f"trust_tier must be one of {sorted(TRUST_TIERS)}.")
    if not isinstance(deduplicate, bool):
        raise TypeError("deduplicate must be boolean.")
    selected = tuple(item for item in examples if _eligible(item, trust_tier))
    if not selected:
        return ()
    clusters = cluster_duplicates(selected)
    cluster_by_id = clusters["cluster_by_example_id"]
    annotated = tuple(
        replace(
            item,
            metadata={
                **item.metadata,
                "trust_tier": trust_tier,
                "trust_weight": _weight(item),
                "duplicate_cluster_id": cluster_by_id[item.example_id],
            },
        )
        for item in selected
    )
    if not deduplicate:
        return tuple(sorted(annotated, key=lambda item: item.example_id))
    rank = {"human_approved": 0, "codex_approved": 1}
    representatives: dict[str, GroundedInstructionExample] = {}
    for item in sorted(
        annotated,
        key=lambda example: (
            rank[example.review_status],
            -float(example.metadata["trust_weight"]),
            example.example_id,
        ),
    ):
        cluster = item.metadata["duplicate_cluster_id"]
        representatives.setdefault(cluster, item)
    return tuple(sorted(representatives.values(), key=lambda item: item.example_id))
