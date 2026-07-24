from __future__ import annotations

import numpy as np
import pytest

import localml_scholar.answering.generative as generative_module
from localml_scholar.answering import (
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
    build_grounded_context,
    select_evidence,
)
from localml_scholar.answering.segmentation import (
    segment_answer_claims,
    segment_source_text,
)
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.tokenizer import ByteTokenizer


def test_source_segmentation_preserves_decimal_abbreviation_code_and_unicode() -> None:
    text = (
        "Dr. Example used 0.01. Next sentence!\n\n"
        "```python\nx = 1.0\n```\n"
        "Unicode question？"
    )

    spans = segment_source_text(text)

    assert [span.text for span in spans] == [
        "Dr. Example used 0.01.",
        "Next sentence!",
        "```python\nx = 1.0\n```",
        "Unicode question？",
    ]
    assert all(
        text[span.start_character : span.end_character] == span.text for span in spans
    )


def test_claim_segmentation_excludes_boilerplate_headings_and_citation_only() -> None:
    text = (
        "The indexed sources state:\n"
        "# Sources\n"
        "- Supported claim. [C1]\n"
        "[C2]\n"
        "I could not find enough support in the indexed documents to answer "
        "this question."
    )

    claims = segment_answer_claims(text)

    assert claims == ("Supported claim. [C1]",)


def test_generation_configuration_validation() -> None:
    with pytest.raises(ValueError):
        GroundedGenerationConfig(maximum_new_tokens=0)
    with pytest.raises(ValueError):
        GroundedGenerationConfig(temperature=float("nan"))
    with pytest.raises(ValueError):
        GroundedGenerationConfig(top_k=0)
    with pytest.raises(ValueError):
        GroundedGenerationConfig(decode_errors="ignore")


def test_grounded_generation_is_new_tokens_only_and_delimiter_bounded(
    grounded_index: RetrievalIndex,
    monkeypatch,
) -> None:
    tokenizer = ByteTokenizer()
    model = TransformerLanguageModel(
        TransformerConfig(
            vocabulary_size=256,
            maximum_context_length=2500,
            model_dimension=4,
            number_of_layers=1,
            number_of_heads=1,
            key_dimension=2,
            value_dimension=2,
            feed_forward_dimension=8,
            seed=4,
        )
    )
    evidence = select_evidence(
        grounded_index,
        "causal mask",
        tokenizer=tokenizer,
    ).evidence
    context = build_grounded_context(
        grounded_index,
        "causal mask",
        evidence,
        tokenizer=tokenizer,
        maximum_context_tokens=2500,
        generation_allowance=8,
    )
    suffix = np.asarray(list(b"ok\nEND\xffx"), dtype=np.int64)

    def fake_generate(_model, prompt, **_kwargs):
        return np.concatenate((prompt, suffix[None, :]), axis=1)

    monkeypatch.setattr(
        generative_module,
        "generate_transformer_ids",
        fake_generate,
    )
    answerer = GroundedGenerativeAnswerer(
        model,
        tokenizer,
        config=GroundedGenerationConfig(
            maximum_new_tokens=8,
            decoded_stop_delimiter="\nEND",
        ),
    )

    generated = answerer.generate(context)

    assert generated.raw_text == "ok\nEND�x"
    assert generated.processed_text == "ok"
    assert generated.stopped_on_delimiter
    assert generated.generated_token_ids == tuple(int(value) for value in suffix)
    assert model.training
    assert not model.has_pending_cache()
