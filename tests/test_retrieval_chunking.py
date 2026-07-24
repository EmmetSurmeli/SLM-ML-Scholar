import pytest

from localml_scholar.retrieval import (
    ChunkingConfig,
    chunk_document,
    ingest_markdown,
    ingest_pdf_text,
    ingest_plain_text,
    validate_chunk_coverage,
)
from localml_scholar.retrieval.documents import PageText


def _config(overlap: int = 10) -> ChunkingConfig:
    return ChunkingConfig(
        target_characters=45,
        maximum_characters=60,
        overlap_characters=overlap,
        minimum_characters=15,
    )


@pytest.mark.parametrize("overlap", [0, 10])
def test_chunking_has_exact_slices_complete_coverage_and_stable_ids(
    overlap: int,
) -> None:
    text = (
        "First sentence has useful content. Second sentence follows here. "
        "Third sentence continues the paragraph.\n\n"
        "A second paragraph includes Unicode λ and café. Final sentence."
    )
    document = ingest_plain_text(text, source="long.txt")
    config = _config(overlap)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert first == second
    assert all(
        chunk.text == text[chunk.start_character : chunk.end_character]
        for chunk in first
    )
    assert all(len(chunk.text) <= config.maximum_characters for chunk in first)
    validate_chunk_coverage(document, first, config)
    for left, right in zip(first, first[1:], strict=False):
        assert left.end_character - right.start_character == overlap
        assert left.text[-overlap:] == right.text[:overlap] if overlap else True


def test_chunking_respects_markdown_sections_and_heading_paths() -> None:
    text = "# One\n\nShort one.\n## Two\n\nShort two.\n"
    document = ingest_markdown(text, source="sections.md")
    chunks = chunk_document(document, _config())

    assert len(chunks) == 2
    assert chunks[0].section_id != chunks[1].section_id
    assert chunks[0].heading_path == ("One",)
    assert chunks[1].heading_path == ("One", "Two")
    assert chunks[0].end_character == chunks[1].start_character


def test_long_paragraph_uses_hard_maximum_without_gaps() -> None:
    text = "x" * 157
    document = ingest_plain_text(text, source="long.txt")
    config = ChunkingConfig(
        target_characters=30,
        maximum_characters=40,
        overlap_characters=5,
        minimum_characters=10,
    )

    chunks = chunk_document(document, config)

    assert max(len(chunk.text) for chunk in chunks) <= 40
    validate_chunk_coverage(document, chunks, config)


def test_overlap_larger_than_minimum_still_makes_forward_progress() -> None:
    text = (
        "Early.\n\n"
        "later words continue through enough source text to require several "
        "deterministic chunks without moving a start offset backward"
    )
    document = ingest_plain_text(text, source="overlap.txt")
    config = ChunkingConfig(
        target_characters=40,
        maximum_characters=50,
        overlap_characters=20,
        minimum_characters=5,
    )

    chunks = chunk_document(document, config)

    assert all(
        right.start_character > left.start_character
        for left, right in zip(chunks, chunks[1:], strict=False)
    )
    validate_chunk_coverage(document, chunks, config)


def test_code_fence_is_kept_whole_when_it_fits_hard_maximum() -> None:
    text = (
        "# Code\n\n"
        "Intro words before code.\n\n"
        "```python\n"
        "value = query_key_score\n"
        "mask = future_position\n"
        "```\n\n"
        "Words after code explain the result in detail."
    )
    document = ingest_markdown(text, source="code.md")
    config = ChunkingConfig(
        target_characters=50,
        maximum_characters=100,
        overlap_characters=5,
        minimum_characters=15,
    )

    chunks = chunk_document(document, config)
    fence_start = text.index("```python")
    fence_end = text.index("```", fence_start + 3) + 4

    assert not any(
        fence_start < chunk.end_character < fence_end for chunk in chunks[:-1]
    )


def test_page_ranges_survive_chunking() -> None:
    document = ingest_pdf_text(
        [PageText(4, "Page four text."), PageText(5, "Page five text.")],
        source="paper.pdf",
    )
    chunks = chunk_document(document, _config())

    assert [chunk.page_start for chunk in chunks] == [4, 5]
    assert [chunk.page_end for chunk in chunks] == [4, 5]


def test_configuration_change_changes_chunk_and_index_identity() -> None:
    document = ingest_plain_text(
        "Sentence one. Sentence two. Sentence three. Sentence four. " * 3,
        source="identity.txt",
    )
    first = chunk_document(document, _config(0))
    second = chunk_document(document, _config(5))

    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in second]
    assert first[0].configuration_sha256 != second[0].configuration_sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_characters": 0},
        {"target_characters": 20, "maximum_characters": 10},
        {"target_characters": 20, "overlap_characters": 20},
        {"target_characters": 20, "minimum_characters": 21},
        {"respect_sections": False},
    ],
)
def test_chunking_configuration_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChunkingConfig(**kwargs)
