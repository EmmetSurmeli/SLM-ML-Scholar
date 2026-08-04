"""Tests for deterministic, context-aware instruction interpretation."""

from __future__ import annotations

import pytest

from localml_scholar.training_data import ConversationTurn, infer_instruction_profile


def test_beginner_prompt_infers_intuitive_profile_without_static_selector():
    profile = infer_instruction_profile(
        "I'm a high school beginner. Explain this in plain language with an analogy."
    )
    assert profile.assumed_background == "high_school"
    assert profile.canonical_audience == "beginner"
    assert profile.explanation_style == "intuitive"
    assert profile.mathematical_depth == "basic"
    assert profile.use_analogy is True


def test_researcher_derivation_prompt_is_deep_and_mathematical():
    profile = infer_instruction_profile(
        "As a PhD researcher, derive this step by step and define every symbol."
    )
    assert profile.assumed_background == "researcher"
    assert profile.canonical_audience == "researcher"
    assert profile.include_derivation is True
    assert profile.mathematical_depth == "advanced"
    assert profile.output_format == "derivation"
    assert "define every symbol" in profile.constraints


def test_current_prompt_overrides_recent_context_and_explicit_override_wins():
    turns = (ConversationTurn("user", "Explain this to a beginner."),)
    current = infer_instruction_profile(
        "Now answer at a researcher level, but keep it concise.",
        recent_turns=turns,
        explicit_overrides={"output_format": "bullets"},
    )
    assert current.assumed_background == "researcher"
    assert current.verbosity == "concise"
    assert current.output_format == "bullets"


def test_simplify_previous_uses_conversation_semantics():
    profile = infer_instruction_profile(
        "I don't understand. Explain that more simply.",
        recent_turns=(ConversationTurn("assistant", "A formal derivation."),),
    )
    assert profile.simplify_previous is True
    assert profile.mathematical_depth == "basic"


def test_evidence_independence_is_visible_in_schema():
    profile = infer_instruction_profile(
        "Compare the methods and critique their assumptions."
    )
    assert profile.include_comparison is True
    assert profile.include_critique is True
    assert "evidence" not in profile.to_dict()


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt):
    with pytest.raises(ValueError, match="prompt"):
        infer_instruction_profile(prompt)


def test_unknown_explicit_override_rejected():
    with pytest.raises(ValueError, match="Unknown instruction overrides"):
        infer_instruction_profile("Explain it.", explicit_overrides={"magic": True})
