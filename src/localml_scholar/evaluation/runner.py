"""Deterministic stage-wise benchmark evaluation runner."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import fmean

from localml_scholar.answering import (
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
)
from localml_scholar.evaluation.answer_grading import (
    grade_answer,
    grade_citations,
    grade_concepts,
    grade_sufficiency,
)
from localml_scholar.evaluation.audience import (
    factual_basis_is_preserved,
    grade_audience,
    render_for_audience,
    structured_target_from_answer,
)
from localml_scholar.evaluation.failures import (
    categorize_failures,
    is_automatic_pass,
    root_cause_attribution,
)
from localml_scholar.evaluation.retrieval_grading import grade_retrieval
from localml_scholar.evaluation.schemas import (
    Benchmark,
    EvaluationConfig,
    EvaluationRun,
    QuestionEvaluation,
    stable_run_id,
)
from localml_scholar.retrieval import RetrievalIndex, SearchFilters


def aggregate_evaluation_metrics(
    results: tuple[QuestionEvaluation, ...],
) -> dict[str, float]:
    """Macro-average stage metrics with explicit zero-denominator behavior."""
    if not isinstance(results, tuple) or not all(
        isinstance(item, QuestionEvaluation) for item in results
    ):
        raise TypeError("results must contain QuestionEvaluation objects.")
    if not results:
        return {
            "question_count": 0.0,
            "automatic_pass_rate": 0.0,
        }

    def mean(values) -> float:
        materialized = tuple(float(item) for item in values)
        return 0.0 if not materialized else fmean(materialized)

    metrics = {
        "question_count": float(len(results)),
        "paper_count": float(len({item.paper_id for item in results})),
        "retrieval.recall_at_1": mean(item.retrieval.recall_at_1 for item in results),
        "retrieval.recall_at_3": mean(item.retrieval.recall_at_3 for item in results),
        "retrieval.recall_at_5": mean(item.retrieval.recall_at_5 for item in results),
        "retrieval.mrr": mean(item.retrieval.reciprocal_rank for item in results),
        "retrieval.expected_section_hit_rate": mean(
            item.retrieval.expected_section_hit for item in results
        ),
        "retrieval.title_page_metadata_hit_rate": mean(
            item.retrieval.title_page_hit
            for item in results
            if item.question_type == "metadata"
        ),
        "retrieval.motivation_source_hit_rate": mean(
            item.retrieval.motivation_source_hit
            for item in results
            if item.question_type == "motivation"
        ),
        "retrieval.forbidden_section_rate": mean(
            item.retrieval.forbidden_section_rate for item in results
        ),
        "retrieval.boilerplate_rate": mean(
            item.retrieval.boilerplate_rate for item in results
        ),
        "sufficiency.accuracy": mean(
            item.sufficiency.correct for item in results if item.sufficiency is not None
        ),
        "sufficiency.false_answer_rate": mean(
            item.sufficiency.false_answer
            for item in results
            if item.sufficiency is not None
        ),
        "sufficiency.false_abstention_rate": mean(
            item.sufficiency.false_abstention
            for item in results
            if item.sufficiency is not None
        ),
        "sufficiency.external_context_recognition": mean(
            item.sufficiency.external_context_recognized
            for item in results
            if item.sufficiency is not None
        ),
        "answer.relevance": mean(
            item.answer.relevance for item in results if item.answer is not None
        ),
        "answer.required_concept_recall": mean(
            item.concepts.required_recall
            for item in results
            if item.concepts is not None
        ),
        "answer.completeness": mean(
            item.answer.completeness for item in results if item.answer is not None
        ),
        "answer.prohibited_claim_rate": mean(
            float(item.answer.prohibited_claim_count > 0)
            for item in results
            if item.answer is not None
        ),
        "answer.false_premise_acceptance": mean(
            item.answer.false_premise_accepted
            for item in results
            if item.answer is not None
        ),
        "answer.numerical_accuracy": mean(
            item.answer.numerical_accuracy
            for item in results
            if item.answer is not None
        ),
        "answer.answerability_accuracy": mean(
            item.answer.answerability_correct
            for item in results
            if item.answer is not None
        ),
        "citation.validity": mean(
            item.citations.syntax_valid * item.citations.existence_valid
            for item in results
            if item.citations is not None
        ),
        "citation.support": mean(
            item.citations.support_rate
            for item in results
            if item.citations is not None
        ),
        "citation.relevance": mean(
            item.citations.relevance_rate
            for item in results
            if item.citations is not None
        ),
        "citation.precision": mean(
            item.citations.precision for item in results if item.citations is not None
        ),
        "citation.recall": mean(
            item.citations.recall for item in results if item.citations is not None
        ),
        "citation.coverage": mean(
            item.citations.coverage for item in results if item.citations is not None
        ),
        "citation.wrong_section_rate": mean(
            float(item.citations.wrong_section_count > 0)
            for item in results
            if item.citations is not None
        ),
        "audience.beginner_appropriateness": mean(
            item.audience.appropriateness
            for item in results
            if item.audience is not None and item.audience_level == "beginner"
        ),
        "audience.undergraduate_appropriateness": mean(
            item.audience.appropriateness
            for item in results
            if item.audience is not None and item.audience_level == "undergraduate"
        ),
        "audience.researcher_appropriateness": mean(
            item.audience.appropriateness
            for item in results
            if item.audience is not None and item.audience_level == "researcher"
        ),
        "audience.factual_consistency": mean(
            float(
                item.system_answer is not None
                and item.system_answer.get("audience_factual_consistency", False)
            )
            for item in results
            if item.system_answer is not None
        ),
        "automatic_pass_rate": mean(item.automatic_pass for item in results),
    }
    failures = Counter(
        category for item in results for category in item.failure_categories
    )
    metrics.update(
        {
            f"failure.{category}": float(count)
            for category, count in sorted(failures.items())
        }
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Aggregate evaluation produced non-finite metrics.")
    return metrics


def grouped_metrics(
    results: tuple[QuestionEvaluation, ...],
    *,
    key: str,
) -> dict[str, dict[str, float]]:
    """Aggregate by paper or question type for report generation."""
    if key not in {"paper_id", "question_type", "audience_level"}:
        raise ValueError("key must be paper_id, question_type, or audience_level.")
    groups: dict[str, list[QuestionEvaluation]] = defaultdict(list)
    for result in results:
        groups[getattr(result, key)].append(result)
    return {
        label: aggregate_evaluation_metrics(tuple(values))
        for label, values in sorted(groups.items())
    }


class EvaluationRunner:
    """Run retrieval and answering as independently graded deterministic stages."""

    def __init__(
        self,
        benchmark: Benchmark,
        index: RetrievalIndex,
        configuration: EvaluationConfig,
        *,
        answering_pipeline: GroundedAnswerPipeline | None = None,
    ) -> None:
        if not isinstance(benchmark, Benchmark):
            raise TypeError("benchmark must be Benchmark.")
        if not isinstance(index, RetrievalIndex):
            raise TypeError("index must be RetrievalIndex.")
        if not isinstance(configuration, EvaluationConfig):
            raise TypeError("configuration must be EvaluationConfig.")
        benchmark.validate_against_index(index)
        if not benchmark.approved_questions:
            raise ValueError(
                "Evaluation requires human-approved or human-edited questions."
            )
        self.benchmark = benchmark
        self.index = index
        self.configuration = configuration
        self.answering_pipeline = answering_pipeline
        if answering_pipeline is not None and not isinstance(
            answering_pipeline, GroundedAnswerPipeline
        ):
            raise TypeError("answering_pipeline must be GroundedAnswerPipeline.")
        if configuration.mode in {
            "generative",
            "generative_with_extractive_fallback",
        } and (
            answering_pipeline is None or answering_pipeline.generative_answerer is None
        ):
            raise ValueError(
                "Generative evaluation requires an explicit local answering pipeline."
            )

    def _pipeline(self) -> GroundedAnswerPipeline:
        if self.answering_pipeline is not None:
            return self.answering_pipeline
        evidence_top_k = min(4, self.configuration.top_k)
        return GroundedAnswerPipeline(
            self.index,
            evidence_config=EvidenceSelectionConfig(
                retrieval_method=self.configuration.retrieval_method,
                retrieval_top_k=self.configuration.top_k,
                evidence_top_k=evidence_top_k,
            ),
        )

    def _evaluate_question(self, question, pipeline) -> QuestionEvaluation:
        filters = SearchFilters(document_id=question.paper_id)
        results = self.index.search(
            question.question,
            method=self.configuration.retrieval_method,
            top_k=self.configuration.top_k,
            filters=filters,
        )
        retrieval = grade_retrieval(
            question,
            results,
            index=self.index,
            k=self.configuration.top_k,
        )
        if self.configuration.mode == "retrieval_only":
            categories = categorize_failures(
                question,
                retrieval,
                answer_object=None,
                sufficiency=None,
                answer=None,
                concepts=None,
                citations=None,
                audience=None,
            )
            cause = root_cause_attribution(categories)
            return QuestionEvaluation(
                question_id=question.question_id,
                paper_id=question.paper_id,
                question_type=question.question_type,
                audience_level=question.audience_level,
                retrieval_results=tuple(item.to_dict() for item in results),
                system_answer=None,
                retrieval=retrieval,
                sufficiency=None,
                answer=None,
                concepts=None,
                citations=None,
                audience=None,
                failure_categories=categories,
                root_cause=cause,
                automatic_pass=is_automatic_pass(categories),
                automated_confidence=(
                    "low" if question.answerability == "ambiguous" else cause.confidence
                ),
            )
        answer_object = pipeline.answer(
            question.question,
            method=self.configuration.mode,
            top_k=self.configuration.top_k,
            filters=filters,
        )
        concepts = grade_concepts(question, answer_object)
        sufficiency = grade_sufficiency(question, answer_object)
        answer = grade_answer(question, answer_object, concepts)
        citations = grade_citations(question, answer_object, index=self.index)
        system_answer = answer_object.to_dict()
        audience = None
        try:
            target = structured_target_from_answer(answer_object)
        except ValueError:
            target = None
        if target is not None:
            renderings = {
                level: render_for_audience(target, level)
                for level in ("beginner", "undergraduate", "researcher")
            }
            rendered = renderings[question.audience_level]
            audience = grade_audience(question, rendered)
            system_answer["structured_answer_target"] = target.to_dict()
            system_answer["audience_renderings"] = renderings
            system_answer["evaluated_answer_text"] = rendered
            system_answer["audience_factual_consistency"] = factual_basis_is_preserved(
                target,
                renderings.values(),
            )
        else:
            audience = grade_audience(question, answer_object.answer_text)
            system_answer["evaluated_answer_text"] = answer_object.answer_text
            system_answer["audience_factual_consistency"] = True
        categories = categorize_failures(
            question,
            retrieval,
            answer_object=answer_object,
            sufficiency=sufficiency,
            answer=answer,
            concepts=concepts,
            citations=citations,
            audience=audience,
        )
        cause = root_cause_attribution(categories)
        confidence = (
            "low"
            if audience.requires_human_review or question.answerability == "ambiguous"
            else cause.confidence
        )
        return QuestionEvaluation(
            question_id=question.question_id,
            paper_id=question.paper_id,
            question_type=question.question_type,
            audience_level=question.audience_level,
            retrieval_results=tuple(item.to_dict() for item in results),
            system_answer=system_answer,
            retrieval=retrieval,
            sufficiency=sufficiency,
            answer=answer,
            concepts=concepts,
            citations=citations,
            audience=audience,
            failure_categories=categories,
            root_cause=cause,
            automatic_pass=is_automatic_pass(categories),
            automated_confidence=confidence,
        )

    def run(
        self,
        *,
        existing_run: EvaluationRun | None = None,
    ) -> EvaluationRun:
        """Run or resume exact question IDs without recomputing completed records."""
        run_id = stable_run_id(
            self.benchmark,
            self.index,
            self.configuration,
        )
        existing: dict[str, QuestionEvaluation] = {}
        if existing_run is not None:
            if not isinstance(existing_run, EvaluationRun):
                raise TypeError("existing_run must be EvaluationRun.")
            if (
                existing_run.run_id != run_id
                or existing_run.benchmark_sha256 != self.benchmark.benchmark_sha256
                or existing_run.index_sha256 != self.index.index_sha256
                or existing_run.configuration != self.configuration
            ):
                raise ValueError("Existing evaluation run is incompatible.")
            existing = {
                item.question_id: item for item in existing_run.question_results
            }
        pipeline = self._pipeline()
        results = tuple(
            existing.get(question.question_id)
            or self._evaluate_question(question, pipeline)
            for question in sorted(
                self.benchmark.approved_questions,
                key=lambda item: item.question_id,
            )
        )
        return EvaluationRun(
            run_id=run_id,
            benchmark_sha256=self.benchmark.benchmark_sha256,
            index_sha256=self.index.index_sha256,
            configuration=self.configuration,
            question_results=results,
            aggregate_metrics=aggregate_evaluation_metrics(results),
        )
