from __future__ import annotations

import numpy as np
import pytest

from localml_scholar.answering import (
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
    GroundedGeneration,
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
)
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.retrieval import RetrievalIndex, SemanticRetrievalConfig
from localml_scholar.tokenizer import ByteTokenizer


def _enriched(index: RetrievalIndex) -> RetrievalIndex:
    rank = min(4, len(index.chunks), len(index.vocabulary))
    return index.enrich_semantic(SemanticRetrievalConfig(dimensions=rank))


def _answerer() -> GroundedGenerativeAnswerer:
    tokenizer = ByteTokenizer()
    model = TransformerLanguageModel(
        TransformerConfig(
            vocabulary_size=tokenizer.vocabulary_size,
            maximum_context_length=2500,
            model_dimension=4,
            number_of_layers=1,
            number_of_heads=1,
            key_dimension=2,
            value_dimension=2,
            feed_forward_dimension=8,
            dtype=np.float64,
            seed=7,
        )
    )
    return GroundedGenerativeAnswerer(
        model,
        tokenizer,
        config=GroundedGenerationConfig(maximum_new_tokens=4, greedy=True),
    )


@pytest.mark.parametrize(
    "retriever",
    ["semantic", "hybrid", "hybrid_reranked"],
)
def test_extractive_answer_preserves_citations_for_every_new_method(
    grounded_index: RetrievalIndex,
    retriever: str,
) -> None:
    index = _enriched(grounded_index)
    pipeline = GroundedAnswerPipeline(
        index,
        evidence_config=EvidenceSelectionConfig(retrieval_method=retriever),
    )

    answer = pipeline.answer(
        "How does a decoder prevent future token leakage?",
        method="extractive",
    )

    assert not answer.abstained
    assert answer.validation.accepted
    assert answer.metadata["retrieval_method"] == retriever
    assert answer.metadata["evidence_selection_config"]["hybrid_config"]
    assert answer.evidence
    assert all(item.retrieval_method == retriever for item in answer.evidence)
    assert all(item.citation.chunk_id == item.chunk_id for item in answer.evidence)
    assert all(item.index_sha256 == index.index_sha256 for item in answer.evidence)


def test_generative_semantic_path_uses_same_validation_without_substitution(
    grounded_index: RetrievalIndex,
    monkeypatch,
) -> None:
    index = _enriched(grounded_index)
    generator = _answerer()
    monkeypatch.setattr(
        GroundedGenerativeAnswerer,
        "generate",
        lambda _self, _context: GroundedGeneration(
            raw_text="Unsupported generated statement. [C1]",
            processed_text="Unsupported generated statement. [C1]",
            generated_token_ids=(1, 2),
            prompt_token_count=10,
            stopped_on_delimiter=False,
        ),
    )
    answer = GroundedAnswerPipeline(
        index,
        evidence_config=EvidenceSelectionConfig(retrieval_method="semantic"),
        generative_answerer=generator,
    ).answer(
        "How does a decoder prevent future token leakage?",
        method="generative",
    )

    assert answer.metadata["retrieval_method"] == "semantic"
    assert not answer.validation.accepted
    assert not answer.fallback_used
    assert all(item.retrieval_method == "semantic" for item in answer.evidence)


def test_missing_semantic_index_raises_without_lexical_substitution(
    grounded_index: RetrievalIndex,
) -> None:
    pipeline = GroundedAnswerPipeline(
        grounded_index,
        evidence_config=EvidenceSelectionConfig(retrieval_method="semantic"),
    )

    with pytest.raises(ValueError, match="enrich"):
        pipeline.answer("future token leakage")
