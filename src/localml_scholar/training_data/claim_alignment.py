"""Evidence-to-claim alignment and citation-first answer construction.

This module deliberately avoids free-form generation.  It turns structured
reviewer facts or local evidence sentences into small claims, validates those
claims, plans an answer, and renders only the approved plan.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from localml_scholar.training_data.provenance import content_sha256

CLAIM_POLICY_VERSION = "1.0"

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NUMBER = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?%?",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[(C[1-9]\d*)\]")
_UNIT = re.compile(
    r"\b(?:seconds?|minutes?|hours?|days?|parameters?|examples?|samples?|"
    r"tokens?|images?|epochs?|steps?|gpus?|cpus?|gb|mb|bleu|accuracy|"
    r"perplexity|percent|%)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "hundred",
    "thousand",
    "million",
    "billion",
}
_CONNECTIVE_SENTENCES = {
    "the indexed paper evidence supports the following answer",
    "the available paper evidence supports only the following partial answer",
    "the indexed paper evidence is insufficient to answer this question",
}

_QUESTION_MARKERS: dict[str, set[str]] = {
    "metadata": {"author", "authors", "title", "written", "paper"},
    "who": {"author", "authors", "person", "people", "written"},
    "method": {"method", "model", "approach", "algorithm", "framework"},
    "architecture": {
        "architecture",
        "attention",
        "component",
        "decoder",
        "encoder",
        "layer",
        "mask",
        "masked",
        "masking",
        "model",
        "network",
        "softmax",
    },
    "motivation": {"motivation", "problem", "goal", "aim", "because"},
    "why": {"reason", "because", "motivation", "problem", "why"},
    "how": {"step", "process", "train", "compute", "algorithm", "how"},
    "dataset": {"dataset", "data", "corpus", "benchmark"},
    "experiment": {"experiment", "dataset", "training", "evaluation", "setup"},
    "result": {"result", "score", "accuracy", "bleu", "perplexity", "performance"},
    "reproduction": {
        "optimizer",
        "learning",
        "batch",
        "minibatch",
        "dataset",
        "architecture",
        "hyperparameter",
    },
    "limitation": {"limitation", "disadvantage", "failure", "cannot", "requires"},
    "historical_impact": {"impact", "influence", "historical", "later"},
}


class SupportStatus(str, Enum):
    """How a claim is grounded."""

    EXPLICIT = "explicit"
    INFERRED_VALID = "inferred_valid"
    EXTERNAL_BACKGROUND = "external_background"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class ClaimRelevance(str, Enum):
    """How much a claim contributes to the requested answer."""

    DIRECT = "direct"
    SUPPORTING = "supporting"
    OPTIONAL = "optional"
    IRRELEVANT = "irrelevant"


class Answerability(str, Enum):
    """Whether the approved claims can answer the question."""

    SUFFICIENT = "sufficient"
    PARTIALLY_SUFFICIENT = "partially_sufficient"
    INSUFFICIENT = "insufficient"
    EXTERNAL_SOURCE_REQUIRED = "external_source_required"


class RepairResult(str, Enum):
    """Outcome of one failure-specific repair."""

    FIXED = "fixed"
    UNCHANGED = "unchanged"
    WORSENED = "worsened"
    INTRODUCED_NEW_FAILURE = "introduced_new_failure"


@dataclass(frozen=True)
class SupportedClaim:
    """One atomic claim with its evidence and policy decisions."""

    claim_id: str
    normalized_claim: str
    claim_type: str
    source_type: str
    evidence_ids: tuple[str, ...]
    citation_labels: tuple[str, ...]
    support_status: SupportStatus
    confidence: float
    required_for_answer: bool
    qualifiers: tuple[str, ...] = ()
    relevance: ClaimRelevance = ClaimRelevance.OPTIONAL
    validation_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.normalized_claim.strip():
            raise ValueError("claim_id and normalized_claim must contain text.")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("claim confidence must be finite and in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "normalized_claim": self.normalized_claim,
            "claim_type": self.claim_type,
            "source_type": self.source_type,
            "evidence_ids": list(self.evidence_ids),
            "citation_labels": list(self.citation_labels),
            "support_status": self.support_status.value,
            "confidence": self.confidence,
            "required_for_answer": self.required_for_answer,
            "qualifiers": list(self.qualifiers),
            "relevance": self.relevance.value,
            "validation_failures": list(self.validation_failures),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SupportedClaim:
        """Load a claim from a persisted 1.2.5 representation."""
        return cls(
            claim_id=str(value["claim_id"]),
            normalized_claim=str(value["normalized_claim"]),
            claim_type=str(value["claim_type"]),
            source_type=str(value["source_type"]),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", [])),
            citation_labels=tuple(
                str(item) for item in value.get("citation_labels", [])
            ),
            support_status=SupportStatus(str(value["support_status"])),
            confidence=float(value["confidence"]),
            required_for_answer=bool(value["required_for_answer"]),
            qualifiers=tuple(str(item) for item in value.get("qualifiers", [])),
            relevance=ClaimRelevance(str(value.get("relevance", "optional"))),
            validation_failures=tuple(
                str(item) for item in value.get("validation_failures", [])
            ),
        )


@dataclass(frozen=True)
class AnswerPlan:
    """Inspectable list of claim IDs that the composer may use."""

    direct_claim_ids: tuple[str, ...]
    supporting_claim_ids: tuple[str, ...]
    qualification_claim_ids: tuple[str, ...]
    inference_claim_ids: tuple[str, ...]
    omitted_claim_ids: tuple[str, ...]
    answerability: Answerability
    citation_plan: tuple[tuple[str, tuple[str, ...]], ...]
    missing_required_concepts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_claim_ids": list(self.direct_claim_ids),
            "supporting_claim_ids": list(self.supporting_claim_ids),
            "qualification_claim_ids": list(self.qualification_claim_ids),
            "inference_claim_ids": list(self.inference_claim_ids),
            "omitted_claim_ids": list(self.omitted_claim_ids),
            "answerability": self.answerability.value,
            "citation_plan": [
                {"claim_id": claim_id, "citation_labels": list(labels)}
                for claim_id, labels in self.citation_plan
            ],
            "missing_required_concepts": list(self.missing_required_concepts),
        }


@dataclass(frozen=True)
class AnswerSentence:
    """One rendered sentence and its exact claim/evidence trace."""

    text: str
    claim_ids: tuple[str, ...]
    citation_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "claim_ids": list(self.claim_ids),
            "citation_labels": list(self.citation_labels),
        }


@dataclass(frozen=True)
class ClaimGraph:
    """Complete evidence-first planning artifact for one answer."""

    claims: tuple[SupportedClaim, ...]
    plan: AnswerPlan
    sentences: tuple[AnswerSentence, ...]
    answer_text: str
    unsupported_language: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": CLAIM_POLICY_VERSION,
            "claims": [item.to_dict() for item in self.claims],
            "answer_plan": self.plan.to_dict(),
            "answer_sentences": [item.to_dict() for item in self.sentences],
            "answer_text": self.answer_text,
            "unsupported_language": [dict(item) for item in self.unsupported_language],
            "claim_count": len(self.claims),
            "supported_claim_count": sum(
                item.support_status
                in {SupportStatus.EXPLICIT, SupportStatus.INFERRED_VALID}
                for item in self.claims
            ),
            "unsupported_claim_count": sum(
                item.support_status == SupportStatus.UNSUPPORTED for item in self.claims
            ),
            "uncited_claim_count": sum(
                item.required_for_answer and not item.citation_labels
                for item in self.claims
            ),
        }


def _text(item: Mapping[str, Any]) -> str:
    for key in ("selected_text", "text", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _terms(text: str) -> set[str]:
    return {item.casefold() for item in _WORD.findall(text)} - _STOP_WORDS


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", _CITATION.sub("", text)).strip(" \t-*•.:#")


def _claim_type(text: str) -> str:
    lower = text.casefold()
    if _NUMBER.search(text):
        return "numerical"
    if any(symbol in text for symbol in ("=", "∑", "∂", "√", "\\(")):
        return "equation"
    if any(term in lower for term in ("because", "therefore", "causes", "leads to")):
        return "causal"
    if any(term in lower for term in ("better", "worse", "faster", "than")):
        return "comparison"
    if any(term in lower for term in ("limitation", "cannot", "fails", "drawback")):
        return "limitation"
    return "factual"


def _atomic_parts(text: str) -> tuple[str, ...]:
    """Split only at strong clause boundaries; never invent linking prose."""
    parts: list[str] = []
    for sentence in _SENTENCE.split(text):
        if sentence.lstrip().startswith("#"):
            continue
        for clause in re.split(r"\s*;\s*|\s+,\s+(?=(?:while|whereas|but)\b)", sentence):
            cleaned = _plain(clause)
            if cleaned:
                parts.append(cleaned)
    return tuple(parts)


def _aliases(evidence: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in evidence:
        label = item.get("label")
        if not isinstance(label, str):
            continue
        for key in ("label", "evidence_id", "stable_evidence_id", "chunk_id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                result[value] = label
    return result


def _source_status(provenance: str) -> SupportStatus:
    if provenance == "paper_explicit":
        return SupportStatus.EXPLICIT
    if provenance == "mathematical_inference":
        return SupportStatus.INFERRED_VALID
    if provenance in {"external_background", "external_knowledge"}:
        return SupportStatus.EXTERNAL_BACKGROUND
    if provenance == "uncertain":
        return SupportStatus.UNCERTAIN
    return SupportStatus.UNSUPPORTED


def classify_claim_relevance(
    question: str, question_type: str, claim: str
) -> ClaimRelevance:
    """Apply a conservative lexical and question-type relevance gate."""
    question_terms = _terms(question)
    claim_terms = _terms(claim)
    markers = _QUESTION_MARKERS.get(question_type, set())
    if question_type in {"historical_impact", "external_context"}:
        markers |= _QUESTION_MARKERS["historical_impact"]
    if question_terms & claim_terms:
        return ClaimRelevance.DIRECT
    if question_type in {"who", "metadata"} and len(_named_entities(claim)) >= 2:
        return ClaimRelevance.DIRECT
    if question_type in {"dataset", "experiment"} and _named_entities(claim):
        return ClaimRelevance.DIRECT
    if question_type == "result" and _NUMBER.search(claim):
        return ClaimRelevance.DIRECT
    if markers & claim_terms:
        return ClaimRelevance.DIRECT
    if (
        len(claim_terms) >= 2
        and question_terms
        and len(question_terms & claim_terms) >= 1
    ):
        return ClaimRelevance.SUPPORTING
    return ClaimRelevance.IRRELEVANT


def extract_candidate_claims(
    question: str,
    question_type: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    structured_target: Mapping[str, Any] | None = None,
) -> tuple[SupportedClaim, ...]:
    """Extract atomic candidate claims from reviewer facts or local passages.

    Structured facts are the preferred Codex-assisted path.  The deterministic
    fallback uses only verbatim evidence sentences and is intentionally narrow.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain text.")
    if not isinstance(question_type, str) or not question_type.strip():
        raise ValueError("question_type must contain text.")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence of mappings.")
    if not all(isinstance(item, Mapping) for item in evidence):
        raise TypeError("Every evidence item must be a mapping.")
    aliases = _aliases(evidence)
    rows: list[tuple[str, str, tuple[str, ...], tuple[str, ...], float]] = []
    if structured_target:
        for category in (
            "facts",
            "equations",
            "derivation_steps",
            "assumptions",
            "qualifications",
            "limitations",
        ):
            values = structured_target.get(category, [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for value in values:
                if not isinstance(value, Mapping):
                    continue
                text = value.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                provenance = str(value.get("provenance", "uncertain"))
                labels = tuple(
                    dict.fromkeys(
                        aliases[item]
                        for item in value.get("citation_ids", [])
                        if isinstance(item, str) and item in aliases
                    )
                )
                qualifiers = (
                    ("inferred from cited paper evidence",)
                    if provenance == "mathematical_inference"
                    else ()
                )
                confidence = float(value.get("confidence", 0.0))
                for part in _atomic_parts(text):
                    rows.append((part, provenance, labels, qualifiers, confidence))
    if not rows:
        for item in evidence:
            label = item.get("label")
            evidence_id = item.get("stable_evidence_id", item.get("evidence_id"))
            if not isinstance(label, str) or not isinstance(evidence_id, str):
                continue
            for part in _atomic_parts(_text(item)):
                relevance = classify_claim_relevance(question, question_type, part)
                if relevance != ClaimRelevance.IRRELEVANT:
                    rows.append((part, "paper_explicit", (label,), (), 1.0))
    claims = []
    for position, (text, provenance, labels, qualifiers, confidence) in enumerate(rows):
        identity = content_sha256(
            {"question": question, "text": text, "position": position}
        )[:20]
        by_label = {
            str(item.get("label")): str(
                item.get("stable_evidence_id", item.get("evidence_id", ""))
            )
            for item in evidence
        }
        claims.append(
            SupportedClaim(
                claim_id=f"claim_{identity}",
                normalized_claim=text,
                claim_type=_claim_type(text),
                source_type=provenance,
                evidence_ids=tuple(
                    by_label[label] for label in labels if label in by_label
                ),
                citation_labels=labels,
                support_status=_source_status(provenance),
                confidence=max(0.0, min(1.0, confidence)),
                required_for_answer=True,
                qualifiers=qualifiers,
                relevance=classify_claim_relevance(question, question_type, text),
            )
        )
    return tuple(claims)


def validate_numerical_alignment(
    claim: str, passages: Sequence[str]
) -> tuple[bool, tuple[str, ...]]:
    """Require exact values, compatible units, and nearby context."""
    numbers = tuple(_NUMBER.findall(claim))
    if not numbers:
        return True, ()
    joined = "\n".join(passages)
    failures = []
    for number in numbers:
        if number not in _NUMBER.findall(joined):
            failures.append(f"numeric_value_missing:{number}")
            continue
        claim_match = next(
            match for match in _NUMBER.finditer(claim) if match.group(0) == number
        )
        passage_match = next(
            match for match in _NUMBER.finditer(joined) if match.group(0) == number
        )
        claim_window = claim[max(0, claim_match.start() - 60) : claim_match.end() + 60]
        passage_window = joined[
            max(0, passage_match.start() - 60) : passage_match.end() + 60
        ]
        claim_units = {item.casefold() for item in _UNIT.findall(claim_window)}
        passage_units = {item.casefold() for item in _UNIT.findall(passage_window)}
        if claim_units and not claim_units <= passage_units:
            failures.append(f"numeric_unit_mismatch:{number}")
        context = (_terms(claim_window) - set(number)) & _terms(passage_window)
        missing_entities = _named_entities(claim_window) - _named_entities(
            passage_window
        )
        if not context or missing_entities:
            failures.append(f"numeric_context_mismatch:{number}")
    return not failures, tuple(failures)


def _named_entities(text: str) -> set[str]:
    known = {
        item.casefold()
        for item in re.findall(
            r"\b(?:Adam|SGD|MNIST|CIFAR-?10|CIFAR-?100|ImageNet|BLEU|Transformer|"
            r"BERT|GAN|ReLU|GELU|RMSProp)\b",
            text,
            flags=re.IGNORECASE,
        )
    }
    tokens = list(re.finditer(r"\b[A-Z][A-Za-z0-9+.-]*\b", text))
    entities = set(known)
    for index, match in enumerate(tokens):
        token = match.group(0)
        letters = "".join(character for character in token if character.isalpha())
        sentence_prefix = text[: match.start()]
        sentence_initial = not sentence_prefix.strip() or bool(
            re.search(r"(?:[.!?]\s+|\n)\s*$", sentence_prefix)
        )
        acronym = (
            bool(letters)
            and letters.isupper()
            and (len(letters) > 1 or not sentence_initial)
        )
        internal_capital = any(character.isupper() for character in token[1:])
        identifier = any(character.isdigit() for character in token)
        named_subject = sentence_initial and bool(
            re.match(
                r"\s+(?:authored|created|designed|developed|founded|introduced|"
                r"invented|proposed|wrote)\b",
                text[match.end() :],
                flags=re.IGNORECASE,
            )
        )
        adjacent_title = (
            index > 0
            and tokens[index - 1].end() < match.start()
            and not text[tokens[index - 1].end() : match.start()].strip()
            and tokens[index - 1].group(0)[:1].isupper()
            and tokens[index - 1].group(0)[1:].islower()
            and token[1:].islower()
        )
        if acronym or internal_capital or identifier or named_subject or adjacent_title:
            entities.add(token.casefold())
    generic = {
        "a",
        "an",
        "architecture",
        "dataset",
        "method",
        "model",
        "result",
        "the",
        "this",
        "training",
    }
    return entities - generic - _NUMBER_WORDS


def validate_entity_alignment(
    claim: str, passages: Sequence[str]
) -> tuple[bool, tuple[str, ...]]:
    """Reject named entities absent from all cited passages."""
    claim_entities = _named_entities(claim)
    passage_entities = _named_entities("\n".join(passages))
    missing = sorted(claim_entities - passage_entities - {"the", "this"})
    return not missing, tuple(f"entity_missing:{item}" for item in missing)


def validate_supported_claim(
    claim: SupportedClaim,
    evidence: Sequence[Mapping[str, Any]],
) -> SupportedClaim:
    """Validate one claim against every evidence item it declares as required."""
    by_label = {str(item.get("label")): item for item in evidence}
    failures: list[str] = []
    passages = []
    for label in claim.citation_labels:
        item = by_label.get(label)
        if item is None:
            failures.append(f"unresolved_citation:{label}")
        else:
            passages.append(_text(item))
    if claim.support_status in {SupportStatus.EXPLICIT, SupportStatus.INFERRED_VALID}:
        if not claim.citation_labels:
            failures.append("missing_citation")
        if any(not passage for passage in passages):
            failures.append("empty_evidence")
        numeric_ok, numeric_failures = validate_numerical_alignment(
            claim.normalized_claim, passages
        )
        entity_ok, entity_failures = validate_entity_alignment(
            claim.normalized_claim, passages
        )
        failures.extend(numeric_failures)
        failures.extend(entity_failures)
        claim_terms = _terms(claim.normalized_claim)
        passage_terms = _terms("\n".join(passages))
        coverage = (
            len(claim_terms & passage_terms) / len(claim_terms) if claim_terms else 0
        )
        if passages and coverage < 0.35:
            failures.append("insufficient_lexical_support")
        if (
            claim.support_status == SupportStatus.INFERRED_VALID
            and not claim.qualifiers
        ):
            failures.append("unlabelled_inference")
        if not numeric_ok or not entity_ok:
            pass
    status = claim.support_status
    if failures and status in {SupportStatus.EXPLICIT, SupportStatus.INFERRED_VALID}:
        status = SupportStatus.UNSUPPORTED
    confidence = claim.confidence if not failures else min(claim.confidence, 0.49)
    return replace(
        claim,
        support_status=status,
        confidence=confidence,
        validation_failures=tuple(dict.fromkeys(failures)),
    )


def validate_claims(
    claims: Sequence[SupportedClaim], evidence: Sequence[Mapping[str, Any]]
) -> tuple[SupportedClaim, ...]:
    """Validate a deterministic sequence of claims without changing ordering."""
    return tuple(validate_supported_claim(item, evidence) for item in claims)


def build_answer_plan(
    claims: Sequence[SupportedClaim],
    *,
    question_type: str,
    required_concepts: Sequence[str | Sequence[str]] = (),
) -> AnswerPlan:
    """Select only approved, relevant claims before any prose is rendered."""
    approved = {
        item.claim_id: item
        for item in claims
        if item.support_status in {SupportStatus.EXPLICIT, SupportStatus.INFERRED_VALID}
        and item.relevance in {ClaimRelevance.DIRECT, ClaimRelevance.SUPPORTING}
    }
    direct = tuple(
        item.claim_id
        for item in claims
        if item.claim_id in approved and item.relevance == ClaimRelevance.DIRECT
    )
    supporting = tuple(
        item.claim_id
        for item in claims
        if item.claim_id in approved and item.relevance == ClaimRelevance.SUPPORTING
    )
    inference = tuple(
        item.claim_id
        for item in claims
        if item.claim_id in approved
        and item.support_status == SupportStatus.INFERRED_VALID
    )
    qualification = tuple(
        item.claim_id
        for item in claims
        if item.claim_id in approved and item.claim_type == "qualification"
    )
    selected = set((*direct, *supporting, *qualification, *inference))
    omitted = tuple(item.claim_id for item in claims if item.claim_id not in selected)
    all_text = " ".join(approved[item].normalized_claim for item in approved)
    missing = []
    for concept in required_concepts:
        aliases = (concept,) if isinstance(concept, str) else tuple(concept)
        if aliases and not any(_terms(alias) <= _terms(all_text) for alias in aliases):
            missing.append(str(aliases[0]))
    external_only = (
        any(item.support_status == SupportStatus.EXTERNAL_BACKGROUND for item in claims)
        and not approved
    )
    if external_only or (question_type == "historical_impact" and not approved):
        answerability = Answerability.EXTERNAL_SOURCE_REQUIRED
    elif direct and not missing:
        answerability = Answerability.SUFFICIENT
    elif direct or supporting:
        answerability = Answerability.PARTIALLY_SUFFICIENT
    else:
        answerability = Answerability.INSUFFICIENT
    ordered = tuple(dict.fromkeys((*direct, *supporting, *qualification, *inference)))
    return AnswerPlan(
        direct_claim_ids=direct,
        supporting_claim_ids=supporting,
        qualification_claim_ids=qualification,
        inference_claim_ids=inference,
        omitted_claim_ids=omitted,
        answerability=answerability,
        citation_plan=tuple((item, approved[item].citation_labels) for item in ordered),
        missing_required_concepts=tuple(missing),
    )


def compose_answer(
    claims: Sequence[SupportedClaim], plan: AnswerPlan
) -> tuple[str, tuple[AnswerSentence, ...]]:
    """Render only planned claims; citations are attached before rendering."""
    by_id = {item.claim_id: item for item in claims}
    if plan.answerability == Answerability.INSUFFICIENT:
        text = "The indexed paper evidence is insufficient to answer this question."
        return text, (AnswerSentence(text, (), ()),)
    if plan.answerability == Answerability.EXTERNAL_SOURCE_REQUIRED:
        text = (
            "The indexed paper evidence cannot establish this historical or external "
            "claim; an external source is required."
        )
        return text, (AnswerSentence(text, (), ()),)
    sentences: list[AnswerSentence] = []
    if plan.answerability == Answerability.PARTIALLY_SUFFICIENT:
        prefix = (
            "The available paper evidence supports only the following partial answer:"
        )
        sentences.append(AnswerSentence(prefix, (), ()))
    for claim_id, labels in plan.citation_plan:
        claim = by_id[claim_id]
        text = claim.normalized_claim.rstrip(" .") + "."
        if labels:
            text += " " + " ".join(f"[{label}]" for label in labels)
        sentences.append(AnswerSentence(text, (claim_id,), labels))
    return " ".join(item.text for item in sentences), tuple(sentences)


def detect_unsupported_language(
    answer_text: str,
    sentences: Sequence[AnswerSentence],
    claims: Sequence[SupportedClaim],
) -> tuple[dict[str, Any], ...]:
    """Detect rendered facts that have no exact approved-claim trace."""
    by_id = {item.claim_id: item for item in claims}
    failures: list[dict[str, Any]] = []
    if answer_text.strip() != " ".join(item.text for item in sentences).strip():
        failures.append({"category": "unplanned_answer_text", "text": answer_text})
    for sentence in sentences:
        plain = _plain(sentence.text).casefold()
        if not sentence.claim_ids:
            if plain not in _CONNECTIVE_SENTENCES and not plain.startswith(
                "the indexed paper evidence cannot establish"
            ):
                failures.append(
                    {"category": "untraced_sentence", "text": sentence.text}
                )
            continue
        expected = " ".join(
            by_id[claim_id].normalized_claim for claim_id in sentence.claim_ids
        )
        if _plain(expected).casefold() != plain:
            categories = []
            if set(_NUMBER.findall(plain)) - set(_NUMBER.findall(expected)):
                categories.append("new_number")
            if _named_entities(plain) - _named_entities(expected):
                categories.append("new_named_entity")
            if any(term in plain for term in ("because", "causes", "therefore")):
                categories.append("new_causal_claim")
            if any(term in plain for term in ("better", "faster", "superior", "than")):
                categories.append("new_comparison_claim")
            failures.append(
                {
                    "category": categories or ["claim_text_changed"],
                    "text": sentence.text,
                    "claim_ids": list(sentence.claim_ids),
                }
            )
    return tuple(failures)


def build_claim_graph(
    question: str,
    question_type: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    structured_target: Mapping[str, Any] | None = None,
    required_concepts: Sequence[str | Sequence[str]] = (),
) -> ClaimGraph:
    """Execute extraction, validation, planning, and citation-first composition."""
    candidates = extract_candidate_claims(
        question,
        question_type,
        evidence,
        structured_target=structured_target,
    )
    claims = validate_claims(candidates, evidence)
    plan = build_answer_plan(
        claims,
        question_type=question_type,
        required_concepts=required_concepts,
    )
    answer_text, sentences = compose_answer(claims, plan)
    unsupported = detect_unsupported_language(answer_text, sentences, claims)
    return ClaimGraph(claims, plan, sentences, answer_text, unsupported)


def claim_graph_metrics(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate completeness and traceability metrics for a persisted graph."""
    claims = graph.get("claims", [])
    planned = {
        item
        for key in ("direct_claim_ids", "supporting_claim_ids")
        for item in graph.get("answer_plan", {}).get(key, [])
    }
    supported = [
        item
        for item in claims
        if item.get("support_status") in {"explicit", "inferred_valid"}
    ]
    cited = [item for item in supported if item.get("citation_labels")]
    sentences = graph.get("answer_sentences", [])
    substantive = [item for item in sentences if item.get("claim_ids")]
    traced = [
        item
        for item in substantive
        if set(item.get("claim_ids", [])) <= planned and item.get("citation_labels")
    ]
    return {
        "claim_count": len(claims),
        "supported_claim_count": len(supported),
        "unsupported_claim_count": sum(
            item.get("support_status") == "unsupported" for item in claims
        ),
        "uncited_claim_count": len(supported) - len(cited),
        "claim_citation_completeness": len(cited) / len(supported)
        if supported
        else 0.0,
        "evidence_to_claim_alignment": len(supported) / len(claims) if claims else 0.0,
        "sentence_to_claim_traceability": len(traced) / len(substantive)
        if substantive
        else 0.0,
        "unsupported_language_count": len(graph.get("unsupported_language", [])),
        "answerability": graph.get("answer_plan", {}).get(
            "answerability", "insufficient"
        ),
    }


def repair_claim_graph(
    graph: ClaimGraph,
    *,
    evidence: Sequence[Mapping[str, Any]],
    question_type: str,
    required_concepts: Sequence[str | Sequence[str]] = (),
) -> tuple[ClaimGraph, dict[str, Any]]:
    """Remove unsupported claims, remap moved citations, and recompose safely."""
    before = claim_graph_metrics(graph.to_dict())
    labels = {str(item.get("label")) for item in evidence}
    repaired = []
    repair_types: set[str] = set()
    for claim in graph.claims:
        if claim.support_status == SupportStatus.UNSUPPORTED:
            repair_types.add("unsupported_claim_deletion")
            continue
        current = tuple(label for label in claim.citation_labels if label in labels)
        if current != claim.citation_labels:
            repair_types.add("citation_remapping")
            best_label = None
            best_overlap = 0
            for item in evidence:
                overlap = len(_terms(claim.normalized_claim) & _terms(_text(item)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = item.get("label")
            current = (str(best_label),) if best_label and best_overlap else ()
        repaired.append(replace(claim, citation_labels=current))
    validated = validate_claims(repaired, evidence)
    plan = build_answer_plan(
        validated,
        question_type=question_type,
        required_concepts=required_concepts,
    )
    answer_text, sentences = compose_answer(validated, plan)
    unsupported = detect_unsupported_language(answer_text, sentences, validated)
    result = ClaimGraph(validated, plan, sentences, answer_text, unsupported)
    after = claim_graph_metrics(result.to_dict())
    hard_before = (
        before["unsupported_claim_count"] + before["unsupported_language_count"]
    )
    hard_after = after["unsupported_claim_count"] + after["unsupported_language_count"]
    if hard_after == 0 and hard_before > 0 or hard_after < hard_before:
        outcome = RepairResult.FIXED
    elif hard_after > hard_before and hard_before == 0:
        outcome = RepairResult.INTRODUCED_NEW_FAILURE
    elif hard_after > hard_before:
        outcome = RepairResult.WORSENED
    else:
        outcome = RepairResult.UNCHANGED
    return result, {
        "repair_types": sorted(repair_types) or ["recomposition"],
        "outcome": outcome.value,
        "before": before,
        "after": after,
        "introduced_new_failure": outcome == RepairResult.INTRODUCED_NEW_FAILURE,
    }


def diagnostic_claim_trace(
    question: str, evidence: Sequence[Mapping[str, Any]], graph: Mapping[str, Any]
) -> str:
    """Return a readable question → evidence → claim → answer trace."""
    lines = ["Question", question, "", "Evidence"]
    for item in evidence:
        lines.append(f"{item.get('label')} {_text(item)}")
    lines.extend(["", "Validated claims"])
    for claim in graph.get("claims", []):
        labels = ", ".join(claim.get("citation_labels", [])) or "no citation"
        lines.append(
            f"{claim.get('claim_id')} {claim.get('normalized_claim')} → {labels} "
            f"({claim.get('support_status')})"
        )
    plan = graph.get("answer_plan", {})
    selected = [
        *plan.get("direct_claim_ids", []),
        *plan.get("supporting_claim_ids", []),
    ]
    lines.extend(
        [
            "",
            "Answer plan",
            " + ".join(selected) or plan.get("answerability", "insufficient"),
            "",
            "Final answer",
            str(graph.get("answer_text", "")),
        ]
    )
    return "\n".join(lines)
