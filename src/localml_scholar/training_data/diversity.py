"""Dataset composition metrics and actionable imbalance warnings."""

from __future__ import annotations

from collections import Counter
from typing import Any

from localml_scholar.training_data.schemas import GroundedInstructionExample


def diversity_metrics(
    examples: tuple[GroundedInstructionExample, ...],
) -> dict[str, Any]:
    """Summarize task, paper, profile, provenance, and review-label coverage."""
    if not isinstance(examples, tuple) or not all(
        isinstance(item, GroundedInstructionExample) for item in examples
    ):
        raise TypeError("examples must be a tuple of GroundedInstructionExample.")
    task_counts = Counter(item.task_type for item in examples)
    paper_counts = Counter(paper for item in examples for paper in item.paper_ids)
    audience_counts = Counter(
        item.instruction_profile.canonical_audience or "adaptive_only"
        for item in examples
    )
    review_counts = Counter(item.review_label for item in examples)
    provenance_counts = Counter(
        fact.provenance
        for item in examples
        for group in (
            item.target.facts,
            item.target.equations,
            item.target.derivation_steps,
            item.target.assumptions,
            item.target.qualifications,
            item.target.limitations,
        )
        for fact in group
    )
    return {
        "example_count": len(examples),
        "paper_count": len(paper_counts),
        "task_type_counts": dict(sorted(task_counts.items())),
        "paper_counts": dict(sorted(paper_counts.items())),
        "audience_metadata_counts": dict(sorted(audience_counts.items())),
        "review_label_counts": dict(sorted(review_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "multi_turn_count": sum(len(item.turns) > 1 for item in examples),
        "multi_paper_count": sum(len(item.paper_ids) > 1 for item in examples),
        "abstention_count": sum(
            item.review_label == "should_abstain" for item in examples
        ),
        "derivation_count": sum(
            bool(item.target.derivation_steps) for item in examples
        ),
    }


def diversity_warnings(metrics: dict[str, Any]) -> tuple[str, ...]:
    """Return deterministic warnings; empty datasets are reported, not hidden."""
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary.")
    count = metrics.get("example_count", 0)
    if not isinstance(count, int):
        raise ValueError("metrics example_count must be an integer.")
    if count == 0:
        return ("Dataset has no human-approved examples.",)
    warnings = []
    task_counts = metrics.get("task_type_counts", {})
    if len(task_counts) < 4:
        warnings.append("Fewer than four task types are represented.")
    if metrics.get("paper_count", 0) < 3:
        warnings.append(
            "Fewer than three papers are represented; paper-level "
            "generalization cannot be assessed well."
        )
    if metrics.get("abstention_count", 0) == 0:
        warnings.append("No approved abstention examples are present.")
    if metrics.get("multi_turn_count", 0) == 0:
        warnings.append("No multi-turn examples are present.")
    if metrics.get("multi_paper_count", 0) == 0:
        warnings.append("No cross-paper examples are present.")
    if metrics.get("derivation_count", 0) == 0:
        warnings.append("No reviewed mathematical derivation examples are present.")
    paper_counts = metrics.get("paper_counts", {})
    if paper_counts and max(paper_counts.values()) / count > 0.6:
        warnings.append("More than 60% of examples reference one paper.")
    return tuple(warnings)


def progress_status(approved_count: int) -> dict[str, Any]:
    """Report progress toward the 100/300/600 review workflow targets."""
    if (
        isinstance(approved_count, bool)
        or not isinstance(approved_count, int)
        or approved_count < 0
    ):
        raise ValueError("approved_count must be a non-negative integer.")
    targets = (100, 300, 600)
    next_target = next((target for target in targets if approved_count < target), None)
    return {
        "approved_examples": approved_count,
        "targets": [
            {
                "count": target,
                "reached": approved_count >= target,
                "progress": min(1.0, approved_count / target),
            }
            for target in targets
        ],
        "next_target": next_target,
    }


def review_priority(item: dict[str, Any]) -> tuple[int, str]:
    """Rank high-value/risky reviews ahead of trivial metadata."""
    if not isinstance(item, dict):
        raise TypeError("item must be a dictionary.")
    kind = item.get("question_type", item.get("task_type", "unknown"))
    priority = {
        "false_premise": 0,
        "insufficient_evidence": 0,
        "derivation": 1,
        "critical_reasoning": 1,
        "comparison": 2,
        "external_context": 2,
        "limitation": 3,
        "reproduction": 4,
        "equation": 4,
        "experiment": 5,
        "method": 6,
        "metadata": 9,
    }.get(str(kind), 7)
    identity = str(item.get("question_id", item.get("interaction_id", "")))
    return priority, identity
