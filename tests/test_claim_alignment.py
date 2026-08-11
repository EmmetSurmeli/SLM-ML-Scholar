"""Authored regressions for evidence-to-claim alignment."""

from __future__ import annotations

from typing import Any

import pytest

from localml_scholar.training_data.claim_alignment import (
    Answerability,
    ClaimRelevance,
    RepairResult,
    SupportStatus,
    build_answer_plan,
    build_claim_graph,
    claim_graph_metrics,
    classify_claim_relevance,
    detect_unsupported_language,
    diagnostic_claim_trace,
    extract_candidate_claims,
    repair_claim_graph,
    validate_entity_alignment,
    validate_numerical_alignment,
    validate_supported_claim,
)
from localml_scholar.training_data.cli import (
    _finished_controlled_diagnostics,
    _parser,
)


def _evidence(
    label: str = "C1",
    *,
    text: str = "The model uses Adam with a learning rate of 0.001.",
    chunk_id: str = "chunk-1",
) -> dict[str, Any]:
    return {
        "label": label,
        "evidence_id": chunk_id,
        "stable_evidence_id": f"stable-{chunk_id}",
        "chunk_id": chunk_id,
        "document_id": "paper-a",
        "text": text,
    }


def _fact(
    text: str,
    *,
    citations: tuple[str, ...] = ("C1",),
    provenance: str = "paper_explicit",
) -> dict[str, Any]:
    return {
        "text": text,
        "provenance": provenance,
        "citation_ids": list(citations),
        "confidence": 0.99,
    }


def _target(*facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "facts": list(facts),
        "equations": [],
        "derivation_steps": [],
        "assumptions": [],
        "qualifications": [],
        "limitations": [],
        "unresolved_items": [],
        "prohibited_claims": [],
    }


def test_claim_extraction_single_fact_and_atomic_clause_split() -> None:
    evidence = [_evidence(text="Adam is used. Training uses 0.001.")]
    claims = extract_candidate_claims(
        "How is the model trained?",
        "how",
        evidence,
        structured_target=_target(
            _fact("Adam is used; training uses a learning rate of 0.001.")
        ),
    )
    assert len(claims) == 2
    assert claims[0].citation_labels == ("C1",)
    assert claims[1].claim_type == "numerical"


def test_claim_extraction_rejects_irrelevant_fallback() -> None:
    claims = extract_candidate_claims(
        "Who wrote the paper?",
        "who",
        [_evidence(text="Each training step took 0.4 seconds.")],
    )
    assert claims == ()


def test_explicit_and_inferred_claim_support() -> None:
    evidence = [_evidence()]
    explicit = extract_candidate_claims(
        "Which optimizer is used?",
        "method",
        evidence,
        structured_target=_target(_fact("The model uses Adam.")),
    )[0]
    assert (
        validate_supported_claim(explicit, evidence).support_status
        == SupportStatus.EXPLICIT
    )
    inferred = extract_candidate_claims(
        "How is optimization performed?",
        "how",
        evidence,
        structured_target=_target(
            _fact(
                "The update is an optimization step.",
                provenance="mathematical_inference",
            )
        ),
    )[0]
    assert inferred.qualifiers


def test_external_background_and_unsupported_claims_are_not_planned() -> None:
    evidence = [_evidence()]
    claims = extract_candidate_claims(
        "What was the historical impact?",
        "historical_impact",
        evidence,
        structured_target=_target(
            _fact(
                "The method later transformed the field.",
                provenance="external_background",
            )
        ),
    )
    plan = build_answer_plan(claims, question_type="historical_impact")
    assert plan.answerability == Answerability.EXTERNAL_SOURCE_REQUIRED
    assert not plan.direct_claim_ids


@pytest.mark.parametrize(
    ("claim", "passage", "failure"),
    [
        ("BLEU is 28.4.", "BLEU is 28.4.", None),
        (
            "Training took 12 hours.",
            "Training took 12 seconds.",
            "numeric_unit_mismatch",
        ),
        (
            "Model B obtained 28.4 BLEU.",
            "Model A obtained 28.4 BLEU.",
            "numeric_context_mismatch",
        ),
    ],
)
def test_numerical_alignment(claim: str, passage: str, failure: str | None) -> None:
    valid, failures = validate_numerical_alignment(claim, [passage])
    assert valid is (failure is None)
    if failure:
        assert any(item.startswith(failure) for item in failures)


def test_entity_alignment_author_dataset_optimizer_model_and_metric() -> None:
    valid, failures = validate_entity_alignment(
        "Adam is evaluated on MNIST with BLEU.",
        ["Adam is evaluated on MNIST with BLEU."],
    )
    assert valid
    assert failures == ()
    valid, failures = validate_entity_alignment(
        "Adam is evaluated on CIFAR-10.",
        ["Adam is evaluated on MNIST."],
    )
    assert not valid
    assert any("cifar-10" in item for item in failures)


def test_sentence_initial_technical_words_are_not_named_entities() -> None:
    valid, failures = validate_entity_alignment(
        (
            "Causal masking prevents subsequent attention. Combined with shifted "
            "embeddings, Softmax inputs for illegal connections are negative."
        ),
        [
            "Masking prevents subsequent attention. Softmax inputs for illegal "
            "connections are set to negative infinity."
        ],
    )

    assert valid
    assert failures == ()


def test_multi_evidence_claim_requires_all_citations() -> None:
    evidence = [
        _evidence(text="Recurrence prevents full sequence parallelization."),
        _evidence(
            "C2",
            text="Self-attention permits parallel computation during training.",
            chunk_id="chunk-2",
        ),
    ]
    graph = build_claim_graph(
        "Why does the method use self-attention?",
        "why",
        evidence,
        structured_target=_target(
            _fact(
                "Recurrence prevents parallelization, while self-attention "
                "permits parallel computation.",
                citations=("C1", "C2"),
            )
        ),
    )
    assert graph.claims[0].citation_labels == ("C1", "C2")
    assert "[C1] [C2]" in graph.answer_text


def test_answer_plan_sufficient_partial_and_insufficient() -> None:
    evidence = [_evidence()]
    sufficient = build_claim_graph(
        "Which method optimizer is used?",
        "method",
        evidence,
        structured_target=_target(_fact("The method uses Adam.")),
    )
    assert sufficient.plan.answerability == Answerability.SUFFICIENT
    partial = build_claim_graph(
        "Which method and dataset are used?",
        "method",
        evidence,
        structured_target=_target(_fact("The model uses Adam.")),
        required_concepts=("Adam", "MNIST"),
    )
    assert partial.plan.answerability == Answerability.PARTIALLY_SUFFICIENT
    insufficient = build_claim_graph(
        "Who wrote the paper?", "who", evidence, structured_target=_target()
    )
    assert insufficient.plan.answerability == Answerability.INSUFFICIENT


def test_citation_first_composition_preserves_exact_claim_text() -> None:
    graph = build_claim_graph(
        "Which optimizer is used?",
        "method",
        [_evidence()],
        structured_target=_target(_fact("The model uses Adam.")),
    )
    assert graph.answer_text == "The model uses Adam. [C1]"
    assert graph.sentences[0].claim_ids == (graph.claims[0].claim_id,)
    assert not graph.unsupported_language


def test_reproduction_batch_question_accepts_minibatch_wording() -> None:
    graph = build_claim_graph(
        "What batch size was used?",
        "reproduction",
        [_evidence(text="The minibatch size was set to 128.")],
        structured_target=_target(_fact("The minibatch size was set to 128.")),
    )

    assert graph.plan.answerability == Answerability.SUFFICIENT
    assert graph.answer_text == "The minibatch size was set to 128. [C1]"
    assert "partial answer" not in graph.answer_text


def test_direct_architecture_claim_does_not_require_generic_category_word() -> None:
    graph = build_claim_graph(
        "What does causal masking do?",
        "architecture",
        [
            _evidence(
                text=(
                    "Masking prevents a position from attending to subsequent "
                    "positions."
                )
            )
        ],
        structured_target=_target(
            _fact("Masking prevents a position from attending to subsequent positions.")
        ),
    )

    assert graph.plan.answerability == Answerability.SUFFICIENT
    assert graph.answer_text == (
        "Masking prevents a position from attending to subsequent positions. [C1]"
    )
    assert "partial answer" not in graph.answer_text


def test_causal_masking_answerer_target_remains_direct_and_supported() -> None:
    evidence = [
        _evidence(
            text=(
                "We modify decoder self-attention to prevent positions from "
                "attending to subsequent positions. This masking, combined with "
                "the one-position output-embedding offset, restricts predictions "
                "to known earlier outputs."
            )
        ),
        _evidence(
            "C2",
            text=(
                "We preserve the auto-regressive property by masking illegal "
                "connections, setting their softmax inputs to negative infinity."
            ),
            chunk_id="chunk-2",
        ),
    ]
    graph = build_claim_graph(
        "What does causal masking do?",
        "architecture",
        evidence,
        structured_target=_target(
            _fact(
                "Causal masking prevents decoder positions from attending to "
                "subsequent positions."
            ),
            _fact(
                "Softmax inputs for illegal connections are set to negative infinity.",
                citations=("C2",),
            ),
        ),
    )

    assert graph.plan.answerability == Answerability.SUFFICIENT
    assert len(graph.plan.direct_claim_ids) == 2
    assert all(claim.support_status == SupportStatus.EXPLICIT for claim in graph.claims)


def test_unsupported_second_clause_is_omitted() -> None:
    graph = build_claim_graph(
        "Which optimizer is used?",
        "method",
        [_evidence()],
        structured_target=_target(
            _fact("The model uses Adam."),
            _fact("It always converges faster."),
        ),
    )
    assert "Adam" in graph.answer_text
    assert "always converges" not in graph.answer_text
    assert any(
        item.support_status == SupportStatus.UNSUPPORTED for item in graph.claims
    )


def test_uncited_limitation_and_unlabelled_inference_are_rejected() -> None:
    evidence = [_evidence(text="The method may fail when gradients vanish.")]
    limitation = extract_candidate_claims(
        "What are the limitations?",
        "limitation",
        evidence,
        structured_target=_target(
            _fact("The method may fail when gradients vanish.", citations=())
        ),
    )[0]
    assert validate_supported_claim(limitation, evidence).support_status == (
        SupportStatus.UNSUPPORTED
    )
    inferred = extract_candidate_claims(
        "Why can training fail?",
        "why",
        evidence,
        structured_target=_target(
            _fact(
                "The mechanism must cause instability.",
                provenance="mathematical_inference",
            )
        ),
    )[0]
    inferred = type(inferred)(**{**inferred.__dict__, "qualifiers": ()})
    checked = validate_supported_claim(inferred, evidence)
    assert "unlabelled_inference" in checked.validation_failures


def test_unplanned_rewrite_detects_new_number_entity_and_comparison() -> None:
    graph = build_claim_graph(
        "Which optimizer is used?",
        "method",
        [_evidence()],
        structured_target=_target(_fact("The model uses Adam.")),
    )
    sentence = graph.sentences[0]
    changed = type(sentence)(
        "The model uses Adam and is 2x faster than SGD. [C1]",
        sentence.claim_ids,
        sentence.citation_labels,
    )
    failures = detect_unsupported_language(changed.text, (changed,), graph.claims)
    categories = set(failures[0]["category"])
    assert {"new_number", "new_named_entity", "new_comparison_claim"} <= categories


def test_answer_repair_that_introduces_new_claim_is_rejected() -> None:
    graph = build_claim_graph(
        "Which optimizer is used?",
        "method",
        [_evidence()],
        structured_target=_target(_fact("The model uses Adam.")),
    )
    sentence = graph.sentences[0]
    rewritten = type(sentence)(
        "The model uses Adam because it always converges. [C1]",
        sentence.claim_ids,
        sentence.citation_labels,
    )
    failures = detect_unsupported_language(rewritten.text, (rewritten,), graph.claims)
    assert failures
    assert "new_causal_claim" in failures[0]["category"]


def test_failure_specific_repair_removes_unsupported_claim() -> None:
    graph = build_claim_graph(
        "Which optimizer is used?",
        "method",
        [_evidence()],
        structured_target=_target(
            _fact("The model uses Adam."),
            _fact("It always converges."),
        ),
    )
    repaired, diagnostic = repair_claim_graph(
        graph, evidence=[_evidence()], question_type="method"
    )
    assert diagnostic["outcome"] == RepairResult.FIXED.value
    assert "unsupported_claim_deletion" in diagnostic["repair_types"]
    assert all(
        item.support_status != SupportStatus.UNSUPPORTED for item in repaired.claims
    )


def test_citation_is_remapped_after_evidence_replacement() -> None:
    original = build_claim_graph(
        "Which optimizer is used?",
        "method",
        [_evidence()],
        structured_target=_target(_fact("The model uses Adam.")),
    )
    replacement = [_evidence("C2", chunk_id="chunk-2")]
    repaired, diagnostic = repair_claim_graph(
        original, evidence=replacement, question_type="method"
    )
    assert "citation_remapping" in diagnostic["repair_types"]
    assert repaired.claims[0].citation_labels == ("C2",)


def test_abstention_does_not_add_unsupported_explanation() -> None:
    graph = build_claim_graph(
        "Who wrote the paper?", "who", [_evidence()], structured_target=_target()
    )
    assert graph.answer_text == (
        "The indexed paper evidence is insufficient to answer this question."
    )
    assert not graph.unsupported_language


def test_soft_wording_difference_is_not_part_of_claim_graph() -> None:
    relevance = classify_claim_relevance(
        "Which optimizer is used?", "method", "The model uses Adam."
    )
    assert relevance == ClaimRelevance.DIRECT


def test_diagnostic_trace_and_metrics_are_inspectable() -> None:
    evidence = [_evidence()]
    graph = build_claim_graph(
        "Which optimizer is used?",
        "method",
        evidence,
        structured_target=_target(_fact("The model uses Adam.")),
    )
    metrics = claim_graph_metrics(graph.to_dict())
    assert metrics["claim_citation_completeness"] == 1.0
    assert metrics["sentence_to_claim_traceability"] == 1.0
    trace = diagnostic_claim_trace(
        "Which optimizer is used?", evidence, graph.to_dict()
    )
    assert "Validated claims" in trace
    assert graph.claims[0].claim_id in trace


def test_invalid_claim_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="question"):
        extract_candidate_claims("", "method", [_evidence()])
    with pytest.raises(TypeError, match="evidence"):
        extract_candidate_claims("Question", "method", "not evidence")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        ["claim-audit", "--run", "run-1"],
        ["repair-report", "--run", "run-1"],
        ["claim-trace", "--run", "run-1", "--candidate", "question-1"],
    ],
)
def test_claim_diagnostic_cli_commands_parse(arguments: list[str]) -> None:
    parsed = _parser().parse_args(arguments)
    assert parsed.command == arguments[0]


def test_readiness_ignores_invalid_and_incomplete_diagnostics() -> None:
    runs = [
        {
            "run_id": "invalid",
            "diagnostic": {
                "controlled": True,
                "valid_for_readiness": False,
                "count": 50,
            },
            "records": [{} for _ in range(50)],
        },
        {
            "run_id": "incomplete",
            "diagnostic": {
                "controlled": True,
                "valid_for_readiness": True,
                "count": 50,
            },
            "records": [{}],
        },
        {
            "run_id": "finished",
            "diagnostic": {
                "controlled": True,
                "valid_for_readiness": True,
                "count": 50,
            },
            "records": [{} for _ in range(50)],
        },
    ]

    class Curator:
        def list_runs(self):
            return runs

    class Service:
        def _autonomous_curator(self):
            return Curator()

    eligible = _finished_controlled_diagnostics(Service())  # type: ignore[arg-type]
    assert [item["run_id"] for item in eligible] == ["finished"]
