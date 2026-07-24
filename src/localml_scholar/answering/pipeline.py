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
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
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
            metadata=metadata or self._metadata(selection, evidence),
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
        result = resolved_answerer.answer(question, evidence)
        claims, validation = validate_answer_text(
            self.index,
            result.answer_text,
            evidence,
            config=self.acceptance_config,
        )
        if not validation.accepted:
            raise RuntimeError(
                "The deterministic extractive baseline failed its own grounding "
                f"invariants: {validation.rejection_reasons}."
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
    ) -> GroundedAnswer:
        """Return a fully structured answer or deterministic abstention."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must contain non-whitespace text.")
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
