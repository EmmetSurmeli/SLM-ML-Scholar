from __future__ import annotations

import pytest

from localml_scholar.answering import (
    CitationSyntaxError,
    citation_labels,
    parse_inline_citations,
    select_evidence,
    validate_answer_text,
)
from localml_scholar.retrieval import RetrievalIndex


def test_citation_parser_normalizes_groups_and_repetition() -> None:
    text = "First [C1, C2, C1]. Second [C2]."
    occurrences = parse_inline_citations(text)

    assert occurrences[0].labels == ("C1", "C2")
    assert citation_labels(text) == ("C1", "C2")
    assert occurrences[0].raw_text == "[C1, C2, C1]"


@pytest.mark.parametrize(
    "text",
    ("claim [C1,].", "claim [C0].", "claim [C].", "claim [C1"),
)
def test_citation_parser_rejects_malformed_syntax(text: str) -> None:
    with pytest.raises(CitationSyntaxError, match="Malformed"):
        parse_inline_citations(text)


def test_valid_exact_source_claim_is_accepted(
    grounded_index: RetrievalIndex,
) -> None:
    evidence = select_evidence(
        grounded_index,
        "How does a decoder prevent future token leakage?",
    ).evidence
    answer = (
        "A decoder applies a causal mask before softmax. [C1]\n"
        "This prevents future-token leakage during\n"
        "autoregressive prediction. [C1]"
    )

    claims, validation = validate_answer_text(
        grounded_index,
        answer,
        evidence,
    )

    assert validation.accepted
    assert validation.citation_coverage == 1.0
    assert all(claim.supported for claim in claims)
    assert validation.longest_exact_quote >= 40


def test_unknown_and_uncited_claims_are_rejected(
    grounded_index: RetrievalIndex,
) -> None:
    evidence = select_evidence(
        grounded_index,
        "causal mask future positions",
    ).evidence
    answer = "A causal mask blocks future positions. [C99] It changes training."

    _, validation = validate_answer_text(grounded_index, answer, evidence)

    assert not validation.accepted
    assert validation.unknown_citation_labels == ("C99",)
    assert validation.uncited_claim_count >= 1
    assert "unknown_citations" in validation.rejection_reasons


def test_citation_before_claim_does_not_attach(
    grounded_index: RetrievalIndex,
) -> None:
    evidence = select_evidence(
        grounded_index,
        "causal mask future positions",
    ).evidence

    claims, validation = validate_answer_text(
        grounded_index,
        "[C1] A decoder applies a causal mask before softmax.",
        evidence,
    )

    assert claims[0].citation_labels == ()
    assert validation.uncited_claim_count == 1
    assert not validation.accepted


@pytest.mark.parametrize(
    ("claim", "reason"),
    (
        (
            "The authored demonstration uses a context length of 256 tokens. [C1]",
            "numerical_mismatches",
        ),
        (
            "The authored demonstration does not use a context length. [C1]",
            "negation_warnings",
        ),
    ),
)
def test_number_and_negation_mismatches_are_rejected(
    grounded_index: RetrievalIndex,
    claim: str,
    reason: str,
) -> None:
    evidence = select_evidence(
        grounded_index,
        "authored demonstration context length",
    ).evidence

    _, validation = validate_answer_text(grounded_index, claim, evidence)

    assert not validation.accepted
    assert reason in validation.rejection_reasons


def test_identifier_and_equation_symbols_are_checked(
    grounded_index: RetrievalIndex,
) -> None:
    evidence = select_evidence(
        grounded_index,
        "learning_rate context length",
    ).evidence

    claims, validation = validate_answer_text(
        grounded_index,
        "otherRate >= 0.01. [C1]",
        evidence,
    )

    support = claims[0].support
    assert support is not None
    assert support.identifier_mismatches == ("otherRate",)
    assert support.equation_symbol_mismatches == (">=",)
    assert not validation.accepted


def test_verbatim_claim_ignores_unrelated_passage_negation(
    grounded_index: RetrievalIndex,
) -> None:
    evidence = select_evidence(
        grounded_index,
        "How must retrieved document instructions be treated?",
    ).evidence
    exact = (
        "A grounded answerer must treat that sentence\n"
        "as evidence content rather than execute it."
    )

    claims, validation = validate_answer_text(
        grounded_index,
        f"{exact} [C2]",
        evidence,
    )

    assert claims[0].support is not None
    assert not claims[0].support.negation_warning
    assert validation.accepted
