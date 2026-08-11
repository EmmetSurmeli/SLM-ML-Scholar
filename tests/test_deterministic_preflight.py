"""Authored regressions for the Milestone 12A.6 deterministic preflight."""

from __future__ import annotations

import hashlib

import pytest

from localml_scholar.answering import (
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
    assess_evidence_sufficiency,
    select_evidence,
)
from localml_scholar.retrieval import (
    RetrievalIndex,
    infer_scholarly_headings,
    ingest_markdown,
    ingest_plain_text,
    normalize_query_terms,
)
from localml_scholar.retrieval.query import question_concepts
from localml_scholar.review_app.autonomous_curation import AutonomousCorpusCurator
from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data.autonomous import AutonomousCurationConfig
from localml_scholar.training_data.claim_alignment import validate_entity_alignment
from localml_scholar.training_data.preflight import (
    DeterministicPreflightCache,
    IngestionHealthConfig,
    paper_ingestion_health,
    pipeline_self_test,
    question_topic_eligible,
    rebuild_index_section_structure,
    topic_signals,
)
from localml_scholar.training_data.questions import generate_paper_questions


def _paper(*, ablation: bool = True) -> str:
    ablation_text = (
        "\n## Ablation\nThe ablation removes one layer and reduces accuracy.\n"
        if ablation
        else ""
    )
    return f"""# Controlled Paper

## Abstract
We propose a neural network architecture for classification.

## Method
The architecture has two encoder layers and one decoder module.
{ablation_text}
## Complexity
Runtime complexity is quadratic and requires 10 FLOPs per item.

## Experiments
Training uses Adam, batch size 16, and learning rate 0.001.

## Limitations
The evaluation uses one dataset.

## References
One reference appears here.
"""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Abstract", "Abstract"),
        ("1 Introduction", "Introduction"),
        ("3.2 Training", "Training"),
        ("REFERENCES", "References"),
    ],
)
def test_heading_inference_patterns(line: str, expected: str) -> None:
    headings = infer_scholarly_headings(f"{line}\nBody text follows.\n")
    assert any(item.title == expected for item in headings)


def test_heading_inference_suppresses_duplicate_and_sentence_false_positive() -> None:
    headings = infer_scholarly_headings(
        "1 Introduction\nBody\n1 Introduction\n"
        "2 categories. In all, there are many examples.\n"
    )
    assert [item.title for item in headings] == ["Introduction"]


def test_truly_untitled_content_remains_uninferred() -> None:
    assert infer_scholarly_headings("ordinary prose without a heading\n") == ()


def test_health_accepts_structured_paper() -> None:
    document = ingest_markdown(_paper(), source="healthy.md")
    health = paper_ingestion_health(document)
    assert health.healthy_for_question_generation
    assert health.titled_section_fraction == 1.0


def test_health_rejects_unstructured_and_near_empty_extraction() -> None:
    unstructured = ingest_plain_text(
        "ordinary extracted text with no recognizable scholarly sections",
        source="unstructured.txt",
    )
    short = ingest_plain_text("tiny", source="short.txt")
    assert not paper_ingestion_health(unstructured).healthy_for_question_generation
    assert (
        "section_structure_low_confidence"
        in paper_ingestion_health(unstructured).extraction_warnings
    )
    assert (
        "empty_or_near_empty_extraction"
        in paper_ingestion_health(short).extraction_warnings
    )


def test_health_threshold_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="lie in"):
        IngestionHealthConfig(maximum_untitled_fraction=1.1)


def test_topic_signals_and_eligibility_follow_source_content() -> None:
    present = _paper()
    absent = _paper(ablation=False)
    signals = topic_signals(present)
    assert signals["ablation"]
    assert signals["architecture"]
    assert signals["complexity"]
    assert signals["reproduction"]
    assert signals["limitation"]
    assert question_topic_eligible(
        "ablation",
        "What does the ablation establish?",
        paper_text=present,
        headings=("Ablation",),
    )
    assert not question_topic_eligible(
        "ablation", "What does the ablation establish?", paper_text=absent
    )


def test_topic_aware_question_generation_suppresses_absent_ablation() -> None:
    present = generate_paper_questions(
        "paper-present",
        "Present",
        count=80,
        paper_text=_paper(),
        section_titles=("Architecture", "Ablation", "Complexity", "Limitations"),
    )
    absent = generate_paper_questions(
        "paper-absent", "Absent", count=80, paper_text=_paper(ablation=False)
    )
    assert any(item.question_type == "ablation" for item in present)
    assert not any(item.question_type == "ablation" for item in absent)


def test_architecture_and_ablation_require_structural_topic_signals() -> None:
    generic_text = (
        "The network has layers. A variant is trained without dropout, but the "
        "paper contains no component study."
    )
    assert not question_topic_eligible(
        "architecture",
        "What is the architecture?",
        paper_text=generic_text,
        headings=("Method", "Experiments"),
    )
    assert not question_topic_eligible(
        "ablation",
        "What does the ablation establish?",
        paper_text=generic_text,
        headings=("Method", "Experiments"),
    )
    assert question_topic_eligible(
        "architecture",
        "What is the architecture?",
        paper_text=generic_text,
        headings=("Model Architecture",),
    )


def test_query_normalization_removes_stopwords_and_preserves_technical_terms() -> None:
    assert normalize_query_terms("What does the ablation establish?") == (
        "ablation",
        "establish",
    )
    assert normalize_query_terms("How does RMSNorm compare to LayerNorm?") == (
        "rmsnorm",
        "compare",
        "layernorm",
    )
    assert normalize_query_terms("What is the paper's main idea?") == ("method",)
    assert normalize_query_terms("What is the strongest reason to be skeptical?") == (
        "limitations",
    )


@pytest.mark.parametrize(
    ("question", "essential", "intent"),
    [
        ("What optimizer is used?", ("optimizer",), "lookup"),
        ("How does attention work?", ("attention", "work"), "mechanism"),
        ("What does the ablation establish?", ("ablation",), "ablation"),
        (
            "What accuracy is reported at 10 steps?",
            ("accuracy", "10"),
            "numerical",
        ),
    ],
)
def test_essential_concepts(
    question: str, essential: tuple[str, ...], intent: str
) -> None:
    concepts = question_concepts(question)
    assert concepts.essential_terms == essential
    assert concepts.question_intent == intent


def test_sufficiency_requires_essential_terms_and_compatible_section() -> None:
    document = ingest_markdown(_paper(), source="sufficiency.md")
    index = RetrievalIndex.build((document,))
    selection = select_evidence(
        index,
        "What does the ablation establish?",
        config=EvidenceSelectionConfig(retrieval_method="bm25"),
    )
    passing = assess_evidence_sufficiency(
        "What does the ablation establish?",
        selection.evidence,
        expected_sections=("Ablation",),
    )
    wrong_section = assess_evidence_sufficiency(
        "What does the ablation establish?",
        selection.evidence,
        expected_sections=("References",),
    )
    assert passing.sufficient
    assert not wrong_section.sufficient
    assert "evidence_section_topic_mismatch" in wrong_section.reasons


def test_sufficiency_treats_batch_and_minibatch_as_equivalent() -> None:
    document = ingest_markdown(
        "# Training\n\nThe minibatch size was set to 128 during optimization.",
        source="minibatch.md",
    )
    index = RetrievalIndex.build((document,))
    selection = select_evidence(
        index,
        "What batch size was used?",
        config=EvidenceSelectionConfig(retrieval_method="bm25"),
    )

    sufficiency = assess_evidence_sufficiency(
        "What batch size was used?",
        selection.evidence,
        expected_sections=("Training",),
    )

    assert sufficiency.sufficient
    assert sufficiency.query_term_coverage == 1.0
    assert sufficiency.matched_query_terms == ("batch", "size")


def test_causal_masking_sufficiency_recognizes_autoregressive_evidence() -> None:
    document = ingest_markdown(
        (
            "# Architecture\n\nMasking prevents attention to subsequent positions "
            "and preserves the autoregressive property."
        ),
        source="causal-mask.md",
    )
    index = RetrievalIndex.build((document,))
    selection = select_evidence(
        index,
        "What does causal masking do?",
        config=EvidenceSelectionConfig(retrieval_method="bm25"),
    )

    sufficiency = assess_evidence_sufficiency(
        "What does causal masking do?",
        selection.evidence,
        expected_sections=("Architecture",),
    )

    assert sufficiency.sufficient
    assert sufficiency.query_term_coverage == 1.0
    assert sufficiency.matched_query_terms == ("causal", "masking")


def test_pipeline_applies_expected_sections_during_sufficiency() -> None:
    document = ingest_markdown(_paper(), source="section-aware.md")
    pipeline = GroundedAnswerPipeline(RetrievalIndex.build((document,)))
    accepted = pipeline.answer(
        "What does the ablation establish?", expected_sections=("Ablation",)
    )
    rejected = pipeline.answer(
        "What does the ablation establish?", expected_sections=("References",)
    )
    assert not accepted.abstained
    assert rejected.abstained
    assert "evidence_section_topic_mismatch" in rejected.sufficiency.reasons


def test_generic_only_retrieval_abstains_without_codex() -> None:
    document = ingest_markdown(_paper(ablation=False), source="no-ablation.md")
    answer = GroundedAnswerPipeline(RetrievalIndex.build((document,))).answer(
        "What does the ablation establish?"
    )
    assert answer.abstained
    assert answer.metadata["grounded_abstention"]["citations_required"] is False


@pytest.mark.parametrize("word", ["Five", "Three", "Eleven", "Hundred", "Billion"])
def test_number_words_are_not_named_entities(word: str) -> None:
    valid, failures = validate_entity_alignment(
        f"{word} layers are used.", ["The architecture uses layers."]
    )
    assert valid
    assert not failures


def test_real_named_entity_still_requires_source_support() -> None:
    valid, failures = validate_entity_alignment(
        "Geoffrey designed the model.", ["The model has two layers."]
    )
    assert not valid
    assert "entity_missing:geoffrey" in failures


def test_section_rebuild_changes_only_unstructured_index() -> None:
    document = ingest_plain_text(
        "Abstract\nA controlled paper.\n1 Introduction\nThe method is introduced.\n",
        source="resection.txt",
    )
    index = RetrievalIndex.build((document,))
    rebuilt, report = rebuild_index_section_structure(index)
    assert report["documents_changed"] == 1
    assert rebuilt.index_sha256 != index.index_sha256
    assert [item.heading for item in rebuilt.documents[0].sections] == [
        "Abstract",
        "Introduction",
    ]


def test_preflight_cache_reuses_exact_hash_and_invalidates_changed_source() -> None:
    first_hash = hashlib.sha256(b"first").hexdigest()
    second_hash = hashlib.sha256(b"second").hexdigest()
    cache = DeterministicPreflightCache()
    cache.put(first_hash, {"healthy": True})
    assert cache.get(first_hash) == {"healthy": True}
    assert cache.get(second_hash) is None


def test_fast_pipeline_self_test_passes_without_codex() -> None:
    report = pipeline_self_test()
    assert report["passed"]
    assert report["codex_calls"] == 0
    assert all(report["checks"].values())


def test_single_candidate_failure_is_recorded_and_next_candidate_continues(
    tmp_path, monkeypatch
) -> None:
    service = ReviewService(tmp_path)
    paper = service.add_paper(filename="paper.md", payload=_paper().encode())
    calls = 0

    def fail_once(self, candidate, config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("candidate construction exploded")
        return {
            "curation_record_id": f"local-{calls}",
            "question_id": candidate["question_id"],
            "paper_ids": candidate["paper_ids"],
            "question": candidate["question"],
            "question_type": candidate["question_type"],
            "status": "insufficient_evidence",
            "terminal_reasons": ["authored_local_stop"],
            "answer": {},
        }

    monkeypatch.setattr(AutonomousCorpusCurator, "_curate_candidate", fail_once)
    run = service.start_autonomous_curation(
        paper_ids=(paper["document_id"],),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            include_multi_turn=False,
        ),
    )
    assert run["records"][0]["status"] == "construction_failed"
    assert run["records"][1]["status"] == "insufficient_evidence"
    assert run["status"] == "completed"


def test_repeated_candidate_failure_triggers_systemic_stop(
    tmp_path, monkeypatch
) -> None:
    service = ReviewService(tmp_path)
    paper = service.add_paper(filename="paper.md", payload=_paper().encode())

    def always_fail(self, candidate, config):
        raise ValueError("same construction defect")

    monkeypatch.setattr(AutonomousCorpusCurator, "_curate_candidate", always_fail)
    run = service.start_autonomous_curation(
        paper_ids=(paper["document_id"],),
        config=AutonomousCurationConfig(
            questions_per_paper=40,
            include_multi_turn=False,
            maximum_repeated_systemic_errors=3,
        ),
    )
    assert run["status"] == "suspended"
    assert run["stage"] == "safety_stop"
    assert any("systemic_candidate_failure" in item for item in run["errors"])
