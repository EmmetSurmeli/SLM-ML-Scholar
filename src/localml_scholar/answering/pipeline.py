"""Top-level retrieval-to-answer pipeline with explicit validation and fallback."""

from __future__ import annotations

from dataclasses import replace

from localml_scholar.answering.context import build_grounded_context
from localml_scholar.answering.evidence import (
    EvidenceSelectionConfig,
    assess_evidence_sufficiency,
    select_evidence,
)
from localml_scholar.answering.extractive import (
    ExtractiveAnswerConfig,
    ExtractiveAnswerer,
)
from localml_scholar.answering.generative import GroundedGenerativeAnswerer
from localml_scholar.answering.models import (
    CitationBinding,
    GroundedAbstention,
    GroundedAnswer,
)
from localml_scholar.answering.validation import (
    AnswerAcceptanceConfig,
    validate_answer_text,
)
from localml_scholar.retrieval import RetrievalIndex, SearchFilters

ABSTENTION_TEXT = (
    "I could not find enough support in the indexed documents to answer this question."
)


def _bindings(evidence):
    return tuple(
        CitationBinding(
            label=item.label,
            evidence_id=item.evidence_id,
            citation=item.citation,
        )
        for item in evidence
    )


class GroundedAnswerPipeline:
    """Compose retrieval, evidence controls, answer methods, and validation."""

    def __init__(
        self,
        index: RetrievalIndex,
        *,
        evidence_config: EvidenceSelectionConfig | None = None,
        extractive_answerer: ExtractiveAnswerer | None = None,
        generative_answerer: GroundedGenerativeAnswerer | None = None,
        acceptance_config: AnswerAcceptanceConfig | None = None,
    ) -> None:
        if not isinstance(index, RetrievalIndex):
            raise TypeError("index must be a RetrievalIndex.")
        self.index = index
        self.evidence_config = evidence_config or EvidenceSelectionConfig()
        self.extractive_answerer = extractive_answerer or ExtractiveAnswerer()
        self.generative_answerer = generative_answerer
        self.acceptance_config = acceptance_config or AnswerAcceptanceConfig()
        if not isinstance(self.evidence_config, EvidenceSelectionConfig):
            raise TypeError("evidence_config must be EvidenceSelectionConfig.")
        if not isinstance(self.extractive_answerer, ExtractiveAnswerer):
            raise TypeError("extractive_answerer must be ExtractiveAnswerer.")
        if self.generative_answerer is not None and not isinstance(
            self.generative_answerer, GroundedGenerativeAnswerer
        ):
            raise TypeError(
                "generative_answerer must be None or GroundedGenerativeAnswerer."
            )
        if not isinstance(self.acceptance_config, AnswerAcceptanceConfig):
            raise TypeError("acceptance_config must be AnswerAcceptanceConfig.")

    def _metadata(
        self,
        selection,
        evidence,
        *,
        evidence_config: EvidenceSelectionConfig | None = None,
        extractive_config: ExtractiveAnswerConfig | None = None,
        context=None,
    ) -> dict:
        resolved_config = evidence_config or self.evidence_config
        resolved_extractive = extractive_config or self.extractive_answerer.config
        metadata = {
            "index_sha256": self.index.index_sha256,
            "corpus_sha256": self.index.corpus_sha256,
            "retrieval_method": resolved_config.retrieval_method,
            "evidence_selection_config": resolved_config.to_dict(),
            "extractive_answer_config": resolved_extractive.to_dict(),
            "acceptance_config": self.acceptance_config.to_dict(),
            "retrieval_results": [
                {
                    "rank": result.rank,
                    "score": result.score,
                    "retrieval_method": result.retrieval_method,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "matched_terms": list(result.matched_terms),
                    "semantic_query_terms": list(result.semantic_query_terms),
                    "scoring_details": result.scoring_details,
                }
                for result in selection.retrieval_results
            ],
            "suppressed_chunk_ids": list(selection.suppressed_chunk_ids),
            "selected_evidence_ids": [item.evidence_id for item in evidence],
        }
        if context is not None:
            metadata["context"] = {
                "prompt": context.prompt,
                "prompt_token_count": context.prompt_token_count,
                "maximum_context_tokens": context.maximum_context_tokens,
                "generation_allowance": context.generation_allowance,
                "removed_evidence_ids": list(context.removed_evidence_ids),
            }
        return metadata

    def _abstention(
        self,
        question,
        method,
        evidence,
        sufficiency,
        selection,
        *,
        metadata=None,
    ) -> GroundedAnswer:
        claims, validation = validate_answer_text(
            self.index,
            ABSTENTION_TEXT,
            evidence,
            config=self.acceptance_config,
            abstained=True,
        )
        if claims:
            raise RuntimeError("The deterministic abstention must not create claims.")
        resolved_metadata = metadata or self._metadata(selection, evidence)
        reason_code = (
            sufficiency.reasons[-1] if sufficiency.reasons else "insufficient_evidence"
        )
        resolved_metadata = {
            **resolved_metadata,
            "grounded_abstention": GroundedAbstention(
                reason_code=reason_code,
                evidence_attempt_summary=(
                    f"Retrieved {len(selection.retrieval_results)} passages; "
                    f"selected {len(evidence)} and matched "
                    f"{len(sufficiency.matched_query_terms)} essential terms."
                ),
                citations_required=False,
                supporting_evidence_ids=tuple(item.evidence_id for item in evidence),
            ).to_dict(),
        }
        return GroundedAnswer(
            question=question,
            method=method,
            answer_text=ABSTENTION_TEXT,
            raw_generated_text=None,
            processed_generated_text=None,
            claims=claims,
            evidence=evidence,
            citations=_bindings(evidence),
            sufficiency=sufficiency,
            abstained=True,
            abstention_reason=";".join(sufficiency.reasons),
            validation=validation,
            fallback_used=False,
            fallback_reason=None,
            metadata=resolved_metadata,
        )

    def _extractive(
        self,
        question,
        method,
        evidence,
        sufficiency,
        selection,
        *,
        answerer=None,
        raw_generated_text=None,
        processed_generated_text=None,
        fallback_used=False,
        fallback_reason=None,
        metadata=None,
    ) -> GroundedAnswer:
        resolved_answerer = answerer or self.extractive_answerer
        try:
            result = resolved_answerer.answer(question, evidence)
        except ValueError as error:
            if str(error) != "No source sentence fits the extractive answer budget.":
                raise
            abstention_sufficiency = replace(
                sufficiency,
                sufficient=False,
                reasons=tuple(
                    dict.fromkeys(
                        (*sufficiency.reasons, "no_extractable_source_sentence")
                    )
                ),
            )
            abstention_metadata = metadata or self._metadata(selection, evidence)
            abstention_metadata = {
                **abstention_metadata,
                "answer_construction_abstention": {
                    "reason": "no_extractable_source_sentence",
                    "extractive_config": resolved_answerer.config.to_dict(),
                },
            }
            return self._abstention(
                question,
                method,
                evidence,
                abstention_sufficiency,
                selection,
                metadata=abstention_metadata,
            )
        claims, validation = validate_answer_text(
            self.index,
            result.answer_text,
            evidence,
            config=self.acceptance_config,
        )
        if not validation.accepted:
            failed = replace(
                sufficiency,
                sufficient=False,
                reasons=tuple(
                    dict.fromkeys(
                        (*sufficiency.reasons, "extractive_grounding_validation_failed")
                    )
                ),
            )
            failed_metadata = {
                **(metadata or self._metadata(selection, evidence)),
                "answer_construction_failure": {
                    "stage": "extractive_validation",
                    "rejection_reasons": list(validation.rejection_reasons),
                    "recoverable": True,
                },
            }
            return self._abstention(
                question,
                method,
                evidence,
                failed,
                selection,
                metadata=failed_metadata,
            )
        return GroundedAnswer(
            question=question,
            method=method,
            answer_text=result.answer_text,
            raw_generated_text=raw_generated_text,
            processed_generated_text=processed_generated_text,
            claims=claims,
            evidence=evidence,
            citations=_bindings(evidence),
            sufficiency=sufficiency,
            abstained=False,
            abstention_reason=None,
            validation=validation,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            metadata=metadata or self._metadata(selection, evidence),
        )

    def answer(
        self,
        question: str,
        *,
        method: str = "extractive",
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        expected_sections: tuple[str, ...] = (),
    ) -> GroundedAnswer:
        """Return a fully structured answer or deterministic abstention."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must contain non-whitespace text.")
        if not isinstance(expected_sections, tuple) or not all(
            isinstance(item, str) and item.strip() for item in expected_sections
        ):
            raise ValueError("expected_sections must be a tuple of non-empty strings.")
        if method not in {
            "top_passage",
            "extractive",
            "generative",
            "generative_with_extractive_fallback",
        }:
            raise ValueError("Unknown answer method.")
        evidence_config = self.evidence_config
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError("top_k must be None or an integer.")
            if top_k <= 0:
                raise ValueError("top_k must be positive.")
            evidence_config = replace(
                evidence_config,
                retrieval_top_k=top_k,
                evidence_top_k=min(evidence_config.evidence_top_k, top_k),
            )
        generative = method in {
            "generative",
            "generative_with_extractive_fallback",
        }
        if generative and self.generative_answerer is None:
            raise ValueError(
                "Generative answering requires an explicit local model checkpoint."
            )
        tokenizer = (
            None
            if self.generative_answerer is None or not generative
            else self.generative_answerer.tokenizer
        )
        extractive_config = (
            replace(self.extractive_answerer.config, strategy="top_passage")
            if method == "top_passage"
            else self.extractive_answerer.config
        )
        selection = select_evidence(
            self.index,
            question,
            config=evidence_config,
            filters=filters,
            tokenizer=tokenizer,
        )
        evidence = selection.evidence
        context = None
        if generative and evidence:
            generator = self.generative_answerer
            context = build_grounded_context(
                self.index,
                question,
                evidence,
                tokenizer=generator.tokenizer,
                maximum_context_tokens=generator.model.config.maximum_context_length,
                generation_allowance=generator.config.maximum_new_tokens,
            )
            evidence = context.evidence
        sufficiency = assess_evidence_sufficiency(
            question,
            evidence,
            config=evidence_config,
            expected_sections=expected_sections,
        )
        metadata = self._metadata(
            selection,
            evidence,
            evidence_config=evidence_config,
            extractive_config=extractive_config,
            context=context,
        )
        if generative:
            generator = self.generative_answerer
            metadata["generation_request"] = {
                "config": generator.config.to_dict(),
                "checkpoint_sha256": generator.checkpoint_sha256,
                "checkpoint_path": generator.checkpoint_path,
                "model_configuration": generator.model.config.to_dict(),
                "parameter_count": generator.model.parameter_count,
                "tokenizer_type": generator.tokenizer.tokenizer_type,
                "tokenizer_state_sha256": generator.tokenizer.state_hash(),
            }
        if not sufficiency.sufficient:
            return self._abstention(
                question,
                method,
                evidence,
                sufficiency,
                selection,
                metadata=metadata,
            )
        if method == "top_passage":
            top_answerer = ExtractiveAnswerer(extractive_config)
            return self._extractive(
                question,
                method,
                evidence,
                sufficiency,
                selection,
                answerer=top_answerer,
                metadata=metadata,
            )
        if method == "extractive":
            return self._extractive(
                question,
                method,
                evidence,
                sufficiency,
                selection,
                metadata=metadata,
            )
        generator = self.generative_answerer
        generation = generator.generate(context)
        processed = generation.processed_text
        answer_text = processed or "The local model produced no usable text."
        claims, validation = validate_answer_text(
            self.index,
            answer_text,
            evidence,
            config=self.acceptance_config,
        )
        metadata["generation"] = {
            "config": generator.config.to_dict(),
            "generated_token_ids": list(generation.generated_token_ids),
            "checkpoint_sha256": generator.checkpoint_sha256,
            "checkpoint_path": generator.checkpoint_path,
            "tokenizer_type": generator.tokenizer.tokenizer_type,
            "tokenizer_state_sha256": generator.tokenizer.state_hash(),
            "stopped_on_delimiter": generation.stopped_on_delimiter,
        }
        if method == "generative_with_extractive_fallback" and not validation.accepted:
            metadata["rejected_generative_validation"] = validation.to_dict()
            reason = ";".join(validation.rejection_reasons)
            return self._extractive(
                question,
                method,
                evidence,
                sufficiency,
                selection,
                raw_generated_text=generation.raw_text,
                processed_generated_text=generation.processed_text,
                fallback_used=True,
                fallback_reason=reason,
                metadata=metadata,
            )
        return GroundedAnswer(
            question=question,
            method=method,
            answer_text=answer_text,
            raw_generated_text=generation.raw_text,
            processed_generated_text=generation.processed_text,
            claims=claims,
            evidence=evidence,
            citations=_bindings(evidence),
            sufficiency=sufficiency,
            abstained=False,
            abstention_reason=None,
            validation=validation,
            fallback_used=False,
            fallback_reason=None,
            metadata=metadata,
        )
