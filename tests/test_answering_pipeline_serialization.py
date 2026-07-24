from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from localml_scholar.answering import (
    ABSTENTION_TEXT,
    GroundedAnswerPipeline,
    GroundedGeneration,
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
    load_grounded_answer,
    save_grounded_answer,
)
from localml_scholar.answering.models import EvidenceItem, evidence_set_hash
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.tokenizer import ByteTokenizer


def _answerer(maximum_new_tokens: int = 4) -> GroundedGenerativeAnswerer:
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
            seed=2,
        )
    )
    return GroundedGenerativeAnswerer(
        model,
        tokenizer,
        config=GroundedGenerationConfig(
            maximum_new_tokens=maximum_new_tokens,
            greedy=True,
        ),
    )


def test_pipeline_abstains_without_generation(
    grounded_index: RetrievalIndex,
    monkeypatch,
) -> None:
    answerer = _answerer()

    def fail(_self, _context):
        raise AssertionError("generation must not run")

    monkeypatch.setattr(GroundedGenerativeAnswerer, "generate", fail)
    answer = GroundedAnswerPipeline(
        grounded_index,
        generative_answerer=answerer,
    ).answer(
        "What superconducting quantum topology is used?",
        method="generative",
    )

    assert answer.abstained
    assert answer.answer_text == ABSTENTION_TEXT
    assert answer.validation.accepted
    assert answer.claims == ()
    assert answer.metadata["generation_request"]["parameter_count"] > 0
    assert answer.metadata["generation_request"]["tokenizer_type"] == "byte"


def test_generative_mode_requires_explicit_model(
    grounded_index: RetrievalIndex,
) -> None:
    pipeline = GroundedAnswerPipeline(grounded_index)
    with pytest.raises(ValueError, match="explicit local model checkpoint"):
        pipeline.answer("causal mask", method="generative")


def test_pipeline_records_effective_method_and_top_k_configuration(
    grounded_index: RetrievalIndex,
) -> None:
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a decoder prevent leakage?",
        method="top_passage",
        top_k=2,
    )

    assert answer.metadata["evidence_selection_config"]["retrieval_top_k"] == 2
    assert answer.metadata["evidence_selection_config"]["evidence_top_k"] == 2
    assert answer.metadata["extractive_answer_config"]["strategy"] == "top_passage"


def test_invalid_generation_is_retained_and_rejected(
    grounded_index: RetrievalIndex,
    monkeypatch,
) -> None:
    answerer = _answerer()
    generated = GroundedGeneration(
        raw_text="The answer is seven. [C99] ",
        processed_text="The answer is seven. [C99]",
        generated_token_ids=(1, 2, 3),
        prompt_token_count=100,
        stopped_on_delimiter=False,
    )
    monkeypatch.setattr(
        GroundedGenerativeAnswerer,
        "generate",
        lambda _self, _context: generated,
    )

    answer = GroundedAnswerPipeline(
        grounded_index,
        generative_answerer=answerer,
    ).answer("How does a causal mask work?", method="generative")

    assert not answer.validation.accepted
    assert answer.raw_generated_text == generated.raw_text
    assert answer.processed_generated_text == generated.processed_text
    assert answer.validation.unknown_citation_labels == ("C99",)
    assert not answer.fallback_used


def test_invalid_generation_uses_explicit_extractive_fallback(
    grounded_index: RetrievalIndex,
    monkeypatch,
) -> None:
    answerer = _answerer()
    generated = GroundedGeneration(
        raw_text="Invented. [C99]",
        processed_text="Invented. [C99]",
        generated_token_ids=(1,),
        prompt_token_count=100,
        stopped_on_delimiter=False,
    )
    monkeypatch.setattr(
        GroundedGenerativeAnswerer,
        "generate",
        lambda _self, _context: generated,
    )

    answer = GroundedAnswerPipeline(
        grounded_index,
        generative_answerer=answerer,
    ).answer(
        "How does a causal mask prevent leakage?",
        method="generative_with_extractive_fallback",
    )

    assert answer.fallback_used
    assert answer.validation.accepted
    assert answer.raw_generated_text == generated.raw_text
    assert answer.answer_text.startswith("The indexed sources state:")
    assert "unknown_citations" in answer.fallback_reason
    assert "rejected_generative_validation" in answer.metadata


def test_answer_artifact_exact_round_trip_and_index_revalidation(
    grounded_index: RetrievalIndex,
    tmp_path: Path,
) -> None:
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a decoder prevent future token leakage?"
    )
    path = save_grounded_answer(tmp_path / "answer.json", answer)

    loaded = load_grounded_answer(path, index=grounded_index)

    assert loaded == answer
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_answer_artifact_rejects_invented_source_metadata(
    grounded_index: RetrievalIndex,
    tmp_path: Path,
) -> None:
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a decoder prevent leakage?"
    )
    path = save_grounded_answer(tmp_path / "answer.json", answer)
    state = json.loads(path.read_text(encoding="utf-8"))
    evidence = state["answer"]["evidence"][0]
    evidence["source_name"] = "invented.md"
    evidence["citation"]["source_name"] = "invented.md"
    evidence["citation"]["display"] = evidence["citation"]["display"].replace(
        "attention.md",
        "invented.md",
    )
    binding = state["answer"]["citations"][0]
    binding["citation"]["source_name"] = "invented.md"
    items = tuple(EvidenceItem.from_dict(item) for item in state["answer"]["evidence"])
    state["answer"]["validation"]["evidence_hash"] = evidence_set_hash(items)
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="revalidate"):
        load_grounded_answer(path, index=grounded_index)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda state: state.update(answer_format_version=999), "Unsupported"),
        (lambda state: state.pop("artifact_type"), "keys are malformed"),
        (
            lambda state: state["answer"]["evidence"][0].update(
                selected_text="altered"
            ),
            "character_count|selected_text_sha256",
        ),
    ),
)
def test_answer_artifact_rejects_malformed_state(
    grounded_index: RetrievalIndex,
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    answer = GroundedAnswerPipeline(grounded_index).answer(
        "How does a decoder prevent leakage?"
    )
    path = save_grounded_answer(tmp_path / "answer.json", answer)
    state = json.loads(path.read_text(encoding="utf-8"))
    mutation(state)
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_grounded_answer(path, index=grounded_index)


def test_checkpoint_loading_preserves_explicit_identity(tmp_path: Path) -> None:
    answerer = _answerer(maximum_new_tokens=1)
    checkpoint = answerer.model.save_checkpoint(
        tmp_path / "model.npz",
        tokenizer=answerer.tokenizer,
    )

    loaded = GroundedGenerativeAnswerer.from_checkpoint(
        checkpoint,
        config=answerer.config,
    )

    assert loaded.checkpoint_path == str(checkpoint)
    assert loaded.checkpoint_sha256 is not None
    assert loaded.model.state_dict().keys() == answerer.model.state_dict().keys()
    assert all(
        np.array_equal(loaded.model.state_dict()[name], values)
        for name, values in answerer.model.state_dict().items()
    )
