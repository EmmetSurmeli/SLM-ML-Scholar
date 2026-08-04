"""Calibration reports and explicit enable/suspension policy for auto-review."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MINIMUM_CALIBRATION_EXAMPLES = 50


@dataclass(frozen=True)
class CalibrationPolicy:
    """Conservative thresholds for activating automated approval."""

    minimum_examples: int = MINIMUM_CALIBRATION_EXAMPLES
    minimum_agreement: float = 0.95
    maximum_override_rate: float = 0.05
    maximum_brier_score: float = 0.08

    def __post_init__(self) -> None:
        if isinstance(self.minimum_examples, bool) or self.minimum_examples < 1:
            raise ValueError("minimum_examples must be a positive integer.")
        for name in (
            "minimum_agreement",
            "maximum_override_rate",
            "maximum_brier_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")


def calibration_report(
    records: list[dict[str, Any]],
    *,
    policy: CalibrationPolicy | None = None,
    explicit_enable: bool = False,
) -> dict[str, Any]:
    """Compare automated recommendations with later human decisions.

    Each record needs ``confidence``, ``automated_approved`` and
    ``human_approved``. Reports with fewer than the policy minimum remain
    calibration-required regardless of observed agreement.
    """
    policy = CalibrationPolicy() if policy is None else policy
    if not isinstance(policy, CalibrationPolicy):
        raise TypeError("policy must be CalibrationPolicy or None.")
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise TypeError("records must be a list of dictionaries.")
    normalized: list[tuple[float, bool, bool]] = []
    for position, item in enumerate(records):
        confidence = item.get("confidence")
        automatic = item.get("automated_approved")
        human = item.get("human_approved")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"records[{position}].confidence must be numeric.")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"records[{position}].confidence must be in [0, 1].")
        if not isinstance(automatic, bool) or not isinstance(human, bool):
            raise TypeError(f"records[{position}] approval values must be booleans.")
        normalized.append((confidence, automatic, human))
    count = len(normalized)
    agreement = (
        sum(automatic == human for _, automatic, human in normalized) / count
        if count
        else 0.0
    )
    overrides = sum(automatic != human for _, automatic, human in normalized)
    override_rate = overrides / count if count else 0.0
    brier = (
        sum((confidence - float(human)) ** 2 for confidence, _, human in normalized)
        / count
        if count
        else 0.0
    )
    bins = []
    for lower in (0.0, 0.5, 0.8, 0.9, 0.95):
        upper = {0.0: 0.5, 0.5: 0.8, 0.8: 0.9, 0.9: 0.95, 0.95: 1.0000001}[lower]
        selected = [item for item in normalized if lower <= item[0] < upper]
        bins.append(
            {
                "lower": lower,
                "upper": min(1.0, upper),
                "count": len(selected),
                "mean_confidence": (
                    sum(item[0] for item in selected) / len(selected)
                    if selected
                    else None
                ),
                "human_approval_rate": (
                    sum(item[2] for item in selected) / len(selected)
                    if selected
                    else None
                ),
            }
        )
    enough = count >= policy.minimum_examples
    metrics_pass = (
        enough
        and agreement >= policy.minimum_agreement
        and override_rate <= policy.maximum_override_rate
        and brier <= policy.maximum_brier_score
    )
    if not enough:
        state = "calibration_required"
    elif not metrics_pass:
        state = "auto_approval_suspended"
    elif explicit_enable:
        state = "auto_approval_enabled"
    else:
        state = "calibration_active"
    reasons = []
    if not enough:
        reasons.append(
            "Need at least "
            f"{policy.minimum_examples} human-reviewed examples; have {count}."
        )
    if enough and agreement < policy.minimum_agreement:
        reasons.append("Human/automatic agreement is below policy minimum.")
    if enough and override_rate > policy.maximum_override_rate:
        reasons.append("Human override rate exceeds policy maximum.")
    if enough and brier > policy.maximum_brier_score:
        reasons.append("Confidence calibration error exceeds policy maximum.")
    if metrics_pass and not explicit_enable:
        reasons.append(
            "Metrics qualify, but a human must explicitly enable auto approval."
        )
    return {
        "state": state,
        "example_count": count,
        "minimum_examples": policy.minimum_examples,
        "agreement": agreement,
        "override_count": overrides,
        "override_rate": override_rate,
        "brier_score": brier,
        "metrics_pass": metrics_pass,
        "explicit_enable": explicit_enable,
        "reasons": reasons,
        "bins": bins,
        "policy": {
            "minimum_agreement": policy.minimum_agreement,
            "maximum_override_rate": policy.maximum_override_rate,
            "maximum_brier_score": policy.maximum_brier_score,
        },
    }


def recommend_threshold(
    records: list[dict[str, Any]],
    *,
    maximum_override_rate: float = 0.05,
) -> float | None:
    """Return the lowest observed threshold meeting the override target."""
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
        if not selected:
            continue
        override_rate = sum(
            not bool(item["human_approved"]) for item in selected
        ) / len(selected)
        if override_rate <= maximum_override_rate:
            return threshold
    return None
