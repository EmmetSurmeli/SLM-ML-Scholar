"""Deterministic adaptive instruction interpretation without an external model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from localml_scholar.training_data.schemas import ConversationTurn, InstructionProfile

_BACKGROUND_SIGNALS = (
    (("phd", "professor", "researcher", "expert"), "researcher", "researcher"),
    (("undergrad", "undergraduate", "college level"), "undergraduate", "undergraduate"),
    (
        ("high school", "beginner", "inexperienced", "new to", "eli5"),
        "high_school",
        "beginner",
    ),
)


def infer_instruction_profile(
    prompt: str,
    *,
    recent_turns: Sequence[ConversationTurn] = (),
    stored_preferences: Mapping[str, Any] | None = None,
    explicit_overrides: Mapping[str, Any] | None = None,
) -> InstructionProfile:
    """Infer presentation requirements with deterministic precedence.

    Current prompt signals override recent conversational signals, which override
    opt-in stored preferences. ``explicit_overrides`` always wins. The function
    does not inspect or alter retrieved evidence.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must contain non-whitespace text.")
    if not isinstance(recent_turns, Sequence) or not all(
        isinstance(turn, ConversationTurn) for turn in recent_turns
    ):
        raise TypeError("recent_turns must contain ConversationTurn instances.")
    if stored_preferences is not None and not isinstance(stored_preferences, Mapping):
        raise TypeError("stored_preferences must be a mapping.")
    if explicit_overrides is not None and not isinstance(explicit_overrides, Mapping):
        raise TypeError("explicit_overrides must be a mapping.")

    values: dict[str, Any] = {
        "desired_depth": "standard",
        "mathematical_depth": "moderate",
        "assumed_background": "undergraduate",
        "explanation_style": "direct",
        "output_format": "prose",
        "verbosity": "standard",
        "use_analogy": False,
        "include_derivation": False,
        "include_critique": False,
        "include_comparison": False,
        "simplify_previous": False,
        "constraints": (),
        "confidence": 0.5,
        "canonical_audience": None,
    }
    allowed_overrides = set(values) - {"confidence"}
    signals: list[str] = []

    if stored_preferences:
        for key in allowed_overrides:
            if key in stored_preferences:
                values[key] = stored_preferences[key]
                signals.append(f"stored:{key}")

    context_text = " ".join(
        turn.content.casefold() for turn in recent_turns[-4:] if turn.role == "user"
    )
    current = prompt.casefold()

    def has(needles: tuple[str, ...], *, include_context: bool = True) -> bool:
        return any(item in current for item in needles) or (
            include_context and any(item in context_text for item in needles)
        )

    for needles, background, audience in reversed(_BACKGROUND_SIGNALS):
        if any(item in context_text for item in needles):
            values["assumed_background"] = background
            values["canonical_audience"] = audience
            signals.append(f"context:background:{background}")
    for needles, background, audience in reversed(_BACKGROUND_SIGNALS):
        if any(item in current for item in needles):
            values["assumed_background"] = background
            values["canonical_audience"] = audience
            signals.append(f"prompt:background:{background}")

    if has(("simple", "plain language", "intuitive", "eli5")):
        values.update(
            desired_depth="brief" if "brief" in current else "standard",
            mathematical_depth="basic",
            explanation_style="intuitive",
        )
        signals.append("instruction:simplify")
    if has(("more detail", "deep dive", "in depth", "thorough", "rigorous")):
        values["desired_depth"] = "deep"
        values["verbosity"] = "detailed"
        signals.append("instruction:deep")
    if has(("concise", "briefly", "short answer", "one paragraph")):
        values["verbosity"] = "concise"
        values["desired_depth"] = "brief"
        signals.append("instruction:concise")
    if has(("derive", "derivation", "step by step", "show the math")):
        values["include_derivation"] = True
        values["mathematical_depth"] = "advanced"
        values["output_format"] = "derivation"
        signals.append("task:derivation")
    if has(("analogy", "metaphor")) and not has(("no analogy", "without analogy")):
        values["use_analogy"] = True
        signals.append("instruction:analogy")
    if has(("no analogy", "without analogy")):
        values["use_analogy"] = False
        signals.append("constraint:no_analogy")
    if has(("critique", "limitations", "weakness", "assumptions")):
        values["include_critique"] = True
        signals.append("task:critique")
    if has(("compare", "contrast", "versus", " vs ")):
        values["include_comparison"] = True
        signals.append("task:comparison")
    if has(
        (
            "simplify that",
            "explain that more simply",
            "i don't understand",
            "too advanced",
        )
    ):
        values["simplify_previous"] = True
        values["explanation_style"] = "intuitive"
        values["mathematical_depth"] = "basic"
        signals.append("conversation:simplify_previous")
    if has(("bullet", "list")):
        values["output_format"] = "bullets"
        signals.append("format:bullets")
    if has(("checklist", "implementation steps")):
        values["output_format"] = "checklist"
        signals.append("format:checklist")
    if has(("table", "matrix")):
        values["output_format"] = "table"
        signals.append("format:table")
    if has(("formal proof", "formally")):
        values["explanation_style"] = "formal"
        values["mathematical_depth"] = "advanced"
        signals.append("style:formal")

    constraint_signals = (
        "use only the paper",
        "no external knowledge",
        "cite every claim",
        "define every symbol",
        "do not speculate",
    )
    constraints = list(values.get("constraints", ()))
    for item in constraint_signals:
        if item in current:
            constraints.append(item)
            signals.append(f"constraint:{item.replace(' ', '_')}")
    values["constraints"] = tuple(dict.fromkeys(constraints))

    if explicit_overrides:
        unknown = set(explicit_overrides) - allowed_overrides
        if unknown:
            raise ValueError(f"Unknown instruction overrides: {sorted(unknown)}.")
        for key, value in explicit_overrides.items():
            values[key] = value
            signals.append(f"explicit:{key}")

    values["confidence"] = min(1.0, 0.5 + 0.08 * len(signals))
    values["signals"] = tuple(signals)
    return InstructionProfile(**values)
