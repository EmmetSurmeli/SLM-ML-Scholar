"""Audience-neutral answer targets, deterministic renderers, and style grading."""

from __future__ import annotations

import re
from collections.abc import Iterable

from localml_scholar.answering import GroundedAnswer, format_inline_citation
from localml_scholar.answering.citations import strip_inline_citations
from localml_scholar.evaluation.schemas import (
    AudienceGrade,
    BenchmarkQuestion,
    CitedAnswerPoint,
    StructuredAnswerTarget,
)
from localml_scholar.retrieval import tokenize_lexically

_JARGON = frozenset(
    {
        "ablation",
        "autoregressive",
        "backpropagation",
        "covariate",
        "distributional",
        "embedding",
        "entropy",
        "gradient",
        "hyperparameter",
        "inference",
        "latent",
        "likelihood",
        "logits",
        "normalization",
        "objective",
        "optimization",
        "parameterization",
        "regularization",
        "representation",
        "stochastic",
        "tensor",
        "tokenization",
        "variance",
    }
)
_MECHANISM_MARKERS = frozenset(
    {
        "allows",
        "because",
        "blocks",
        "by",
        "enables",
        "prevents",
        "so",
        "through",
        "using",
        "uses",
    }
)
_QUALIFIERS = frozenset(
    {
        "although",
        "however",
        "limited",
        "may",
        "might",
        "only",
        "suggests",
        "under",
        "unless",
        "within",
    }
)
_LIMITATION_MARKERS = frozenset(
    {"limitation", "limited", "unknown", "unclear", "not", "cannot", "future"}
)
_DEFINITION_PATTERN = re.compile(
    r"\b(?:means|refers to|is called|is defined as|in other words)\b",
    re.IGNORECASE,
)
_EQUATION_PATTERN = re.compile(r"(?:\b[A-Za-z]\s*=|[∑∏√]|\\frac|\\begin\{|\$\$)")


def _point_text(point: CitedAnswerPoint) -> str:
    citation = format_inline_citation(point.citations)
    return f"{strip_inline_citations(point.text).strip()} {citation}"


def structured_target_from_answer(answer: GroundedAnswer) -> StructuredAnswerTarget:
    """Build one factual target from supported, cited answer claims."""
    if not isinstance(answer, GroundedAnswer):
        raise TypeError("answer must be a GroundedAnswer.")
    if answer.abstained and not answer.evidence:
        raise ValueError("An evidence-free abstention has no structured target.")
    claims = tuple(
        claim
        for claim in answer.claims
        if claim.substantive and claim.supported and claim.citation_labels
    )
    if not claims:
        raise ValueError("A structured target requires one supported cited claim.")
    points = tuple(
        CitedAnswerPoint(
            text=strip_inline_citations(claim.text).strip(),
            citations=claim.citation_labels,
        )
        for claim in claims
    )
    return StructuredAnswerTarget(
        core_answer=points[0],
        supporting_points=points[1:],
        unresolved_items=(
            ()
            if answer.validation.accepted
            else tuple(answer.validation.rejection_reasons)
        ),
    )


def render_beginner_answer(target: StructuredAnswerTarget) -> str:
    """Render a short factual answer without introducing a new analogy or claim."""
    if not isinstance(target, StructuredAnswerTarget):
        raise TypeError("target must be StructuredAnswerTarget.")
    lines = ["In simple terms:", _point_text(target.core_answer)]
    for point in target.supporting_points[:2]:
        lines.append(_point_text(point))
    if target.qualifications:
        lines.extend(
            ["Important qualification:", _point_text(target.qualifications[0])]
        )
    if target.unresolved_items:
        lines.append("Some details remain unresolved from the supplied paper.")
    return "\n\n".join(lines)


def render_undergraduate_answer(target: StructuredAnswerTarget) -> str:
    """Render core mechanism and relevant technical detail for an ML student."""
    if not isinstance(target, StructuredAnswerTarget):
        raise TypeError("target must be StructuredAnswerTarget.")
    sections: list[tuple[str, tuple[CitedAnswerPoint, ...]]] = [
        ("Answer", (target.core_answer,)),
        ("Mechanism and evidence", target.supporting_points),
        ("Relevant equations", target.equations),
        ("Assumptions and qualifications", target.assumptions + target.qualifications),
        ("Limitations", target.limitations),
    ]
    rendered = [
        f"{heading}:\n" + "\n".join(f"- {_point_text(point)}" for point in points)
        for heading, points in sections
        if points
    ]
    if target.unresolved_items:
        rendered.append(
            "Unresolved from the supplied evidence:\n- "
            + "\n- ".join(target.unresolved_items)
        )
    return "\n\n".join(rendered)


def render_researcher_answer(target: StructuredAnswerTarget) -> str:
    """Render the complete cited target with assumptions and evidential limits."""
    if not isinstance(target, StructuredAnswerTarget):
        raise TypeError("target must be StructuredAnswerTarget.")
    sections: list[tuple[str, tuple[CitedAnswerPoint, ...]]] = [
        ("Core answer", (target.core_answer,)),
        ("Supporting evidence", target.supporting_points),
        ("Equations", target.equations),
        ("Assumptions", target.assumptions),
        ("Qualifications", target.qualifications),
        ("Limitations and what is not established", target.limitations),
    ]
    rendered = [
        f"{heading}:\n" + "\n".join(f"- {_point_text(point)}" for point in points)
        for heading, points in sections
        if points
    ]
    if target.unresolved_items:
        rendered.append("Unresolved items:\n- " + "\n- ".join(target.unresolved_items))
    return "\n\n".join(rendered)


def render_for_audience(
    target: StructuredAnswerTarget,
    audience_level: str,
) -> str:
    """Dispatch to the deterministic trusted renderer."""
    renderers = {
        "beginner": render_beginner_answer,
        "undergraduate": render_undergraduate_answer,
        "researcher": render_researcher_answer,
    }
    try:
        renderer = renderers[audience_level]
    except KeyError as error:
        raise ValueError("Unknown audience level.") from error
    return renderer(target)


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", strip_inline_citations(text))
        if item.strip()
    )


def _contains_any(terms: set[str], candidates: Iterable[str]) -> bool:
    return bool(terms & set(candidates))


def grade_audience(
    question: BenchmarkQuestion,
    answer_text: str,
) -> AudienceGrade:
    """Grade transparent level-specific signals; this is not semantic proof."""
    if not isinstance(question, BenchmarkQuestion):
        raise TypeError("question must be BenchmarkQuestion.")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("answer_text must contain non-whitespace text.")
    terms = tokenize_lexically(answer_text)
    term_set = set(terms)
    sentences = _sentences(answer_text)
    average_sentence_words = (
        0.0
        if not sentences
        else sum(len(tokenize_lexically(item)) for item in sentences) / len(sentences)
    )
    jargon_density = sum(term in _JARGON for term in terms) / max(1, len(terms))
    equation_count = len(_EQUATION_PATTERN.findall(answer_text))
    definition_present = float(bool(_DEFINITION_PATTERN.search(answer_text)))
    mechanism_present = float(_contains_any(term_set, _MECHANISM_MARKERS))
    qualification_present = float(_contains_any(term_set, _QUALIFIERS))
    limitation_present = float(_contains_any(term_set, _LIMITATION_MARKERS))
    reasons: list[str] = []

    if question.audience_level == "beginner":
        signals = [
            float(jargon_density <= 0.12),
            float(average_sentence_words <= 24.0),
            float(equation_count <= 1),
            definition_present,
            mechanism_present,
        ]
        if jargon_density > 0.12:
            reasons.append("beginner_jargon_density_high")
        if average_sentence_words > 24.0:
            reasons.append("beginner_sentences_long")
        if not definition_present:
            reasons.append("beginner_definition_missing")
    elif question.audience_level == "undergraduate":
        expected_equation = question.question_type in {"equation", "notation"}
        signals = [
            mechanism_present,
            float(bool(term_set & _JARGON)),
            float(not expected_equation or equation_count > 0),
            float(average_sentence_words <= 32.0),
        ]
        if not mechanism_present:
            reasons.append("undergraduate_mechanism_missing")
        if expected_equation and equation_count == 0:
            reasons.append("undergraduate_expected_equation_missing")
    else:
        signals = [
            mechanism_present,
            qualification_present,
            limitation_present,
            float(bool(question.required_concepts)),
            float(average_sentence_words >= 8.0),
        ]
        if not qualification_present:
            reasons.append("researcher_qualification_missing")
        if not limitation_present:
            reasons.append("researcher_limitations_missing")
    appropriateness = sum(signals) / len(signals)
    borderline = 0.4 <= appropriateness < 0.85
    requires_review = (
        borderline
        or question.question_type
        in {"historical_impact", "synthesis", "interpretation"}
        or question.answerability == "ambiguous"
    )
    if requires_review:
        reasons.append("human_review_required")
    return AudienceGrade(
        audience_level=question.audience_level,
        appropriateness=appropriateness,
        jargon_density=jargon_density,
        average_sentence_words=average_sentence_words,
        equation_count=equation_count,
        definition_present=definition_present,
        mechanism_present=mechanism_present,
        qualification_present=qualification_present,
        limitation_present=limitation_present,
        reasons=tuple(dict.fromkeys(reasons)),
        requires_human_review=requires_review,
    )


def factual_basis_is_preserved(
    target: StructuredAnswerTarget,
    rendered_answers: Iterable[str],
) -> bool:
    """Check that every rendering retains the core answer and known citations."""
    if not isinstance(target, StructuredAnswerTarget):
        raise TypeError("target must be StructuredAnswerTarget.")
    core_terms = set(tokenize_lexically(target.core_answer.text))
    required_citations = set(target.core_answer.citations)
    for rendered in rendered_answers:
        if not isinstance(rendered, str):
            raise TypeError("rendered_answers must contain strings.")
        if not core_terms <= set(tokenize_lexically(rendered)):
            return False
        if not all(f"[{label}]" in rendered for label in required_citations):
            return False
    return True
