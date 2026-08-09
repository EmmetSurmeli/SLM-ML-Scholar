"""Deterministic calibration sampling, metrics, and activation policy."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

MINIMUM_CALIBRATION_EXAMPLES = 50
CONFIDENCE_BUCKETS = (
    (0.50, 0.70, "0.50-0.69"),
    (0.70, 0.80, "0.70-0.79"),
    (0.80, 0.90, "0.80-0.89"),
    (0.90, 0.95, "0.90-0.94"),
    (0.95, 0.98, "0.95-0.97"),
    (0.98, 1.000000000001, "0.98-1.00"),
)


@dataclass(frozen=True)
class CalibrationPolicy:
    """Conservative, configurable requirements for bulk automatic approval.

    ``minimum_agreement`` and ``maximum_brier_score`` remain for artifact/API
    compatibility. Readiness is principally governed by false approvals and
    precision because an incorrect automatic approval is the costly failure.
    Confidence is a deterministic heuristic score, not a calibrated probability.
    """

    minimum_examples: int = MINIMUM_CALIBRATION_EXAMPLES
    minimum_auto_approval_candidates: int = 20
    minimum_agreement: float = 0.95
    minimum_auto_approval_precision: float = 0.95
    maximum_override_rate: float = 0.05
    maximum_false_approval_rate: float = 0.05
    maximum_brier_score: float = 0.08
    maximum_near_threshold_error_rate: float = 0.10

    def __post_init__(self) -> None:
        for name in ("minimum_examples", "minimum_auto_approval_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "minimum_agreement",
            "minimum_auto_approval_precision",
            "maximum_override_rate",
            "maximum_false_approval_rate",
            "maximum_brier_score",
            "maximum_near_threshold_error_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")


def _confidence(record: dict[str, Any], position: int) -> float:
    value = record.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"records[{position}].confidence must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"records[{position}].confidence must be in [0, 1].")
    return result


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        raw = row.get(field, "unknown")
        keys: Iterable[object] = raw if isinstance(raw, list) else (raw,)
        for key in keys:
            values.setdefault(str(key or "unknown"), []).append(row)
    return {
        key: {
            "count": len(group),
            "agreement": _rate(
                sum(
                    item["automated_approved"] == item["human_approved"]
                    for item in group
                ),
                len(group),
            ),
            "false_approval_rate": _rate(
                sum(
                    item["automated_approved"] and not item["human_approved"]
                    for item in group
                ),
                sum(item["automated_approved"] for item in group),
            ),
        }
        for key, group in sorted(values.items())
    }


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise TypeError("records must be a list of dictionaries.")
    normalized = []
    for position, item in enumerate(records):
        automatic = item.get("automated_approved")
        human = item.get("human_approved")
        if not isinstance(automatic, bool) or not isinstance(human, bool):
            raise TypeError(f"records[{position}] approval values must be booleans.")
        normalized.append(
            {
                **item,
                "confidence": _confidence(item, position),
                "automated_approved": automatic,
                "human_approved": human,
            }
        )
    return normalized


def calibration_report(
    records: list[dict[str, Any]],
    *,
    policy: CalibrationPolicy | None = None,
    explicit_enable: bool = False,
    integrity: dict[str, Any] | None = None,
    approval_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compare automatic recommendations with subsequent human decisions.

    The report is descriptive. It never enables approval itself; qualifying
    metrics only make the explicit enable operation available.
    """
    policy = CalibrationPolicy() if policy is None else policy
    if not isinstance(policy, CalibrationPolicy):
        raise TypeError("policy must be CalibrationPolicy or None.")
    if not isinstance(explicit_enable, bool):
        raise TypeError("explicit_enable must be boolean.")
    rows = _normalize_records(records)
    count = len(rows)
    tp = sum(row["automated_approved"] and row["human_approved"] for row in rows)
    fp = sum(row["automated_approved"] and not row["human_approved"] for row in rows)
    tn = sum(
        not row["automated_approved"] and not row["human_approved"] for row in rows
    )
    fn = sum(not row["automated_approved"] and row["human_approved"] for row in rows)
    auto_positive = tp + fp
    auto_negative = tn + fn
    human_positive = tp + fn
    agreement = _rate(tp + tn, count)
    override_rate = _rate(fp + fn, count)
    precision = _rate(tp, auto_positive)
    recall = _rate(tp, human_positive)
    false_approval_rate = _rate(fp, auto_positive)
    false_rejection_rate = _rate(fn, auto_negative)
    brier = _rate(
        sum((row["confidence"] - float(row["human_approved"])) ** 2 for row in rows),
        count,
    )
    bins = []
    for lower, upper, label in CONFIDENCE_BUCKETS:
        selected = [row for row in rows if lower <= row["confidence"] < upper]
        bins.append(
            {
                "label": label,
                "lower": lower,
                "upper": min(1.0, upper),
                "count": len(selected),
                "mean_confidence": (
                    sum(row["confidence"] for row in selected) / len(selected)
                    if selected
                    else None
                ),
                "human_approval_rate": (
                    sum(row["human_approved"] for row in selected) / len(selected)
                    if selected
                    else None
                ),
                "false_approval_rate": _rate(
                    sum(
                        row["automated_approved"] and not row["human_approved"]
                        for row in selected
                    ),
                    sum(row["automated_approved"] for row in selected),
                ),
            }
        )
    near = [
        row
        for row in rows
        if abs(row["confidence"] - float(approval_threshold)) <= 0.02
    ]
    near_error_rate = _rate(
        sum(row["automated_approved"] != row["human_approved"] for row in near),
        len(near),
    )
    mandatory = [row for row in rows if row.get("mandatory_human_categories")]
    mandatory_route_accuracy = (
        _rate(sum(not row["automated_approved"] for row in mandatory), len(mandatory))
        if mandatory
        else 1.0
    )
    integrity = {} if integrity is None else dict(integrity)
    integrity_defaults = {
        "source_hash_errors": 0,
        "test_leakage_errors": 0,
        "provenance_errors": 0,
        "duplicate_errors": 0,
    }
    integrity_defaults.update(integrity)
    checks = {
        "minimum_pairs": count >= policy.minimum_examples,
        "minimum_auto_approval_candidates": auto_positive
        >= policy.minimum_auto_approval_candidates,
        "agreement": agreement >= policy.minimum_agreement,
        "override_rate": override_rate <= policy.maximum_override_rate,
        "auto_approval_precision": precision >= policy.minimum_auto_approval_precision,
        "false_approval_rate": false_approval_rate
        <= policy.maximum_false_approval_rate,
        "near_threshold_error_rate": near_error_rate
        <= policy.maximum_near_threshold_error_rate,
        "mandatory_human_routing": mandatory_route_accuracy == 1.0,
        "source_hashes": integrity_defaults["source_hash_errors"] == 0,
        "test_leakage": integrity_defaults["test_leakage_errors"] == 0,
        "provenance": integrity_defaults["provenance_errors"] == 0,
        "duplicates": integrity_defaults["duplicate_errors"] == 0,
    }
    metrics_pass = all(checks.values())
    reasons = []
    descriptions = {
        "minimum_pairs": (
            f"Need at least {policy.minimum_examples} human-reviewed pairs; "
            f"have {count}."
        ),
        "minimum_auto_approval_candidates": (
            "Need at least "
            f"{policy.minimum_auto_approval_candidates} auto-approval candidates; "
            f"have {auto_positive}."
        ),
        "agreement": "Human/automatic agreement is below policy minimum.",
        "override_rate": "Human override rate exceeds policy maximum.",
        "auto_approval_precision": "Auto-approval precision is below policy minimum.",
        "false_approval_rate": "False-approval rate exceeds policy maximum.",
        "near_threshold_error_rate": "Near-threshold errors are excessive.",
        "mandatory_human_routing": (
            "A mandatory-human case was routed for automatic approval."
        ),
        "source_hashes": "Source-hash validation errors are present.",
        "test_leakage": "Test-only data leakage was detected.",
        "provenance": "Provenance validation errors are present.",
        "duplicates": "Duplicate validation errors are present.",
    }
    reasons.extend(descriptions[key] for key, passed in checks.items() if not passed)
    if count < policy.minimum_examples:
        state = "calibration_required"
    elif not metrics_pass:
        state = (
            "auto_approval_suspended"
            if count >= policy.minimum_examples
            else "calibration_active"
        )
    elif explicit_enable:
        state = "auto_approval_enabled"
    else:
        state = "calibration_active"
        reasons.append(
            "Metrics qualify, but a human must explicitly enable auto approval."
        )
    return {
        "state": state,
        "example_count": count,
        "minimum_examples": policy.minimum_examples,
        "auto_approval_candidate_count": auto_positive,
        "agreement": agreement,
        "approval_agreement": _rate(tp, auto_positive),
        "rejection_agreement": _rate(tn, auto_negative),
        "override_count": fp + fn,
        "override_rate": override_rate,
        "auto_approval_precision": precision,
        "auto_approval_recall": recall,
        "false_approval_count": fp,
        "false_approval_rate": false_approval_rate,
        "false_rejection_count": fn,
        "false_rejection_rate": false_rejection_rate,
        "brier_score": brier,
        "confidence_is_heuristic": True,
        "near_threshold_count": len(near),
        "near_threshold_error_rate": near_error_rate,
        "mandatory_human_route_accuracy": mandatory_route_accuracy,
        "metrics_pass": metrics_pass,
        "explicit_enable": explicit_enable,
        "reasons": reasons,
        "checks": checks,
        "bins": bins,
        "by_question_type": _group_metrics(rows, "question_type"),
        "by_failure_category": _group_metrics(rows, "failure_categories"),
        "by_paper": _group_metrics(rows, "paper_ids"),
        "by_reviewer_profile": _group_metrics(rows, "reviewer_profile"),
        "integrity": integrity_defaults,
        "policy": {
            "minimum_examples": policy.minimum_examples,
            "minimum_auto_approval_candidates": policy.minimum_auto_approval_candidates,
            "minimum_agreement": policy.minimum_agreement,
            "minimum_auto_approval_precision": policy.minimum_auto_approval_precision,
            "maximum_override_rate": policy.maximum_override_rate,
            "maximum_false_approval_rate": policy.maximum_false_approval_rate,
            "maximum_brier_score": policy.maximum_brier_score,
            "maximum_near_threshold_error_rate": (
                policy.maximum_near_threshold_error_rate
            ),
        },
    }


def _rank(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest(), "big")


def confidence_bucket(confidence: float) -> str:
    """Return the documented bucket for a finite confidence in ``[0, 1]``."""
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be finite and in [0, 1].")
    for lower, upper, label in CONFIDENCE_BUCKETS:
        if lower <= confidence < upper:
            return label
    return "below-0.50"


def select_calibration_sample(
    reviews: list[dict[str, Any]], *, target_count: int = 50, seed: int = 42
) -> dict[str, Any]:
    """Select a deterministic coverage-seeking sample without changing reviews."""
    if isinstance(target_count, bool) or not isinstance(target_count, int):
        raise TypeError("target_count must be an integer.")
    if target_count < 1 or target_count > 500:
        raise ValueError("target_count must be between 1 and 500.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not isinstance(reviews, list) or not all(
        isinstance(item, dict) for item in reviews
    ):
        raise TypeError("reviews must be a list of dictionaries.")
    candidates = []
    for position, review in enumerate(reviews):
        review_id = review.get("review_id")
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError(f"reviews[{position}] requires review_id.")
        confidence = review.get("second_pass", {}).get(
            "confidence", review.get("proposed_confidence", 0.0)
        )
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"reviews[{position}] confidence must be numeric.")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"reviews[{position}] confidence must be in [0, 1].")
        second = review.get("second_pass", {})
        failures = sorted(
            {
                gate
                for result in second.get("reviewer_results", [])
                for gate, passed in result.get("gates", {}).items()
                if not passed
            }
            | set(second.get("mandatory_human_categories", []))
        )
        candidates.append(
            {
                "review_id": review_id,
                "paper_ids": list(review.get("paper_ids", [])),
                "question_type": review.get("question_type", "unknown"),
                "automatic_label": review.get("proposed_label", "unknown"),
                "abstained": bool(review.get("answer", {}).get("abstained", False)),
                "confidence": confidence,
                "confidence_bucket": confidence_bucket(confidence),
                "failure_categories": failures,
                "near_threshold": abs(confidence - 0.95) <= 0.02,
            }
        )
    remaining = sorted(
        candidates, key=lambda item: (_rank(item["review_id"], seed), item["review_id"])
    )
    selected = []
    coverage: Counter[str] = Counter()
    selection_target = min(target_count, len(remaining))
    while remaining and len(selected) < selection_target:

        def score(item: dict[str, Any]) -> tuple[int, int, str]:
            strata = [
                *(f"paper:{value}" for value in item["paper_ids"]),
                f"type:{item['question_type']}",
                f"label:{item['automatic_label']}",
                f"abstain:{item['abstained']}",
                f"confidence:{item['confidence_bucket']}",
                *(f"failure:{value}" for value in item["failure_categories"]),
            ]
            novelty = sum(1 for value in strata if coverage[value] == 0)
            scarcity = -sum(coverage[value] for value in strata)
            return novelty, scarcity, item["review_id"]

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        for paper_id in chosen["paper_ids"]:
            coverage[f"paper:{paper_id}"] += 1
        coverage[f"type:{chosen['question_type']}"] += 1
        coverage[f"label:{chosen['automatic_label']}"] += 1
        coverage[f"abstain:{chosen['abstained']}"] += 1
        coverage[f"confidence:{chosen['confidence_bucket']}"] += 1
        for failure in chosen["failure_categories"]:
            coverage[f"failure:{failure}"] += 1
    population_strata = (
        {f"paper:{value}" for item in candidates for value in item["paper_ids"]}
        | {f"type:{item['question_type']}" for item in candidates}
        | {f"label:{item['automatic_label']}" for item in candidates}
        | {f"confidence:{item['confidence_bucket']}" for item in candidates}
        | {
            f"failure:{value}"
            for item in candidates
            for value in item["failure_categories"]
        }
    )
    gaps = sorted(value for value in population_strata if coverage[value] == 0)
    return {
        "target_count": target_count,
        "seed": seed,
        "population_count": len(candidates),
        "selected_count": len(selected),
        "review_ids": [item["review_id"] for item in selected],
        "items": selected,
        "coverage": dict(sorted(coverage.items())),
        "coverage_gaps": gaps,
        "warnings": (
            ["Population is smaller than the requested sample."]
            if len(candidates) < target_count
            else []
        )
        + (
            ["Some available strata are not represented: " + ", ".join(gaps)]
            if gaps
            else []
        ),
    }


def recommend_threshold(
    records: list[dict[str, Any]], *, maximum_override_rate: float = 0.05
) -> float | None:
    """Return the lowest observed threshold meeting the false-approval target."""
    if not 0 <= maximum_override_rate <= 1:
        raise ValueError("maximum_override_rate must be in [0, 1].")
    candidates = sorted(
        {
            float(item["confidence"])
            for item in records
            if isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
        }
    )
    for threshold in candidates:
        selected = [item for item in records if float(item["confidence"]) >= threshold]
        if (
            selected
            and _rate(
                sum(not bool(item["human_approved"]) for item in selected),
                len(selected),
            )
            <= maximum_override_rate
        ):
            return threshold
    return None
