"""Tests for proposed-only diverse paper question generation."""

from __future__ import annotations

import pytest

from localml_scholar.training_data.questions import (
    ATTENTION_QUESTIONS,
    generate_paper_questions,
    generate_prompt_variations,
    question_type_counts,
)


def test_attention_catalog_contains_exact_required_80_candidates():
    assert len(ATTENTION_QUESTIONS) == 80
    assert len({question for question, _ in ATTENTION_QUESTIONS}) == 80
    assert ATTENTION_QUESTIONS[0][0] == "Who wrote Attention Is All You Need?"
    assert (
        ATTENTION_QUESTIONS[-1][0]
        == "What should a researcher be skeptical about in this paper?"
    )


def test_attention_generation_is_deterministic_and_never_auto_approves():
    first = generate_paper_questions(
        "paper-attention", "Attention Is All You Need", count=80
    )
    second = generate_paper_questions(
        "paper-attention", "Attention Is All You Need", count=80
    )
    assert first == second
    assert len(first) == 80
    assert all(item.review_status == "proposed" for item in first)
    assert all(item.metadata["trusted_gold"] is False for item in first)
    assert len(question_type_counts(first)) >= 10


def test_generic_generation_covers_requested_range_and_natural_prompts():
    candidates = generate_paper_questions("paper-1", "A Tiny Method", count=60)
    assert len(candidates) == 60
    assert any("I don't get" in item.question for item in candidates)
    assert any(item.question_type == "false_premise" for item in candidates)
    assert any(item.question_type == "insufficient_evidence" for item in candidates)


@pytest.mark.parametrize("count", [0, 39, 81, True])
def test_generation_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match=r"\[40, 80\]"):
        generate_paper_questions("paper", "Title", count=count)


def test_variations_preserve_target_link_and_require_approval():
    source = generate_paper_questions("paper", "Title", count=40)[0]
    variations = generate_prompt_variations(source)
    assert len(variations) == 4
    assert all(item.parent_question_id == source.question_id for item in variations)
    assert all(item.review_status == "proposed" for item in variations)
    assert all(
        item.metadata["variation_requires_human_approval"] for item in variations
    )
