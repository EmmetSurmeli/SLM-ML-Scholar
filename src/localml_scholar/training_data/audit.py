"""Deterministic risk-aware audit sampling for automated review decisions."""

from __future__ import annotations

import hashlib
import math
from typing import Any


def _rank(example_id: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{example_id}".encode()).digest(), "big"
    )


def select_audit_sample(
    reviews: list[dict[str, Any]],
    *,
    sample_fraction: float = 0.10,
    seed: int = 42,
    approval_threshold: float = 0.95,
    near_threshold_margin: float = 0.02,
) -> dict[str, Any]:
    """Select mandatory-risk cases plus a deterministic random-like sample."""
    if not isinstance(reviews, list) or not all(
        isinstance(item, dict) for item in reviews
    ):
        raise TypeError("reviews must be a list of dictionaries.")
    for name, value in {
        "sample_fraction": sample_fraction,
        "approval_threshold": approval_threshold,
        "near_threshold_margin": near_threshold_margin,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    identities = []
    for position, item in enumerate(reviews):
        identity = item.get("example_id", item.get("review_id"))
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"reviews[{position}] requires example_id or review_id.")
        confidence = item.get("confidence", item.get("proposed_confidence", 0.0))
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"reviews[{position}] confidence must be numeric.")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"reviews[{position}] confidence must be in [0, 1].")
        reasons = []
        if abs(confidence - approval_threshold) <= near_threshold_margin:
            reasons.append("near_threshold")
        if item.get("reviewer_disagreement") or "reviewer_disagreement" in item.get(
            "mandatory_human_categories", []
        ):
            reasons.append("reviewer_disagreement")
        if item.get("novel_failure"):
            reasons.append("novel_failure")
        identities.append((identity.strip(), reasons))
    mandatory = {identity for identity, reasons in identities if reasons}
    target = math.ceil(len(identities) * float(sample_fraction))
    optional = sorted(
        (identity for identity, _ in identities if identity not in mandatory),
        key=lambda identity: (_rank(identity, seed), identity),
    )
    selected = set(mandatory)
    selected.update(optional[: max(0, target - len(selected))])
    reason_lookup = dict(identities)
    items = []
    for identity in sorted(selected):
        reasons = reason_lookup[identity] or ["deterministic_random_sample"]
        items.append({"example_id": identity, "reasons": reasons, "status": "pending"})
    return {
        "seed": seed,
        "sample_fraction": float(sample_fraction),
        "population_count": len(identities),
        "selected_count": len(items),
        "mandatory_count": len(mandatory),
        "items": items,
    }
