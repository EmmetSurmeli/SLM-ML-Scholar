from pathlib import Path

import pytest

from localml_scholar.retrieval import (
    IndexConfig,
    PageText,
    RetrievalIndex,
    ingest_file,
    ingest_files,
    ingest_markdown,
    ingest_pdf_text,
    ingest_plain_text,
)


def test_plain_text_preserves_exact_source_and_metadata() -> None:
    text = "First line.\n\nSecond  line with café.\n"
    document = ingest_plain_text(
        text,
        source="notes/example.txt",
        title="Example",
        metadata={
            "authors": ["Project Author"],
            "publication_year": 2026,
            "logical_collection": "tests",
            "tags": ["local", "retrieval"],
        },
    )

    assert document.text == text
    assert document.source_name == "example.txt"
    assert document.media_type == "text/plain"
    assert document.title == "Example"
    assert document.byte_length == len(text.encode("utf-8"))
    assert document.character_length == len(text)
    assert document.sections[0].text == text
    assert document.sections[0].start_line == 1
    assert document.sections[0].end_line == 3
    assert document.metadata["user"]["authors"] == ["Project Author"]


def test_markdown_detects_hierarchy_but_not_fenced_headings() -> None:
    text = (
        "preamble\n\n"
        "# Model\n\n"
        "intro\n"
        "## Attention\n\n"
        "details\n"
        "```text\n"
        "# not a heading\n"
        "```\n"
        "### Mask\n\n"
        "causal\n"
        "## Loss\n\n"
        "cross entropy\n"
    )

    document = ingest_markdown(text, source="paper.md")

    assert document.title == "Model"
    assert [section.heading for section in document.sections] == [
        None,
        "Model",
        "Attention",
        "Mask",
        "Loss",
    ]
    assert [section.heading_path for section in document.sections] == [
        (),
        ("Model",),
        ("Model", "Attention"),
        ("Model", "Attention", "Mask"),
        ("Model", "Loss"),
    ]
    assert "".join(section.text for section in document.sections) == text
    assert "# not a heading" in document.sections[2].text


def test_document_identity_distinguishes_content_and_path_changes() -> None:
    original = ingest_plain_text("same text", source="a.txt")
    same = ingest_plain_text("same text", source="a.txt")
    changed_content = ingest_plain_text("changed text", source="a.txt")
    changed_path = ingest_plain_text("same text", source="b.txt")

    assert original.document_id == same.document_id
    assert original.document_id != changed_content.document_id
    assert original.document_id != changed_path.document_id
    assert original.content_sha256 == changed_path.content_sha256


def test_pdf_text_adapter_preserves_pages_and_reports_empty_pages() -> None:
    document = ingest_pdf_text(
        [
            PageText(1, "First page."),
            PageText(2, ""),
            PageText(3, "Third page."),
        ],
        source="paper.pdf",
        title="Paper",
    )

    assert document.text == "First page.\nThird page."
    assert [section.page_start for section in document.sections] == [1, 3]
    assert document.metadata["inferred"]["page_count"] == 3
    assert document.metadata["inferred"]["empty_pages"] == [2]
    assert "".join(section.text for section in document.sections) == document.text
    results = RetrievalIndex.build([document]).search("third")
    assert results[0].page_start == 3
    assert results[0].citation.format() == "[Paper, p. 3]"


def test_pdf_adapter_rejects_invalid_page_order_and_all_empty() -> None:
    with pytest.raises(ValueError, match="increasing"):
        ingest_pdf_text(
            [PageText(2, "two"), PageText(1, "one")],
            source="bad.pdf",
        )
    with pytest.raises(ValueError, match="some extracted text"):
        ingest_pdf_text([PageText(1, "")], source="empty.pdf")


def test_file_ingestion_is_strict_utf8_and_extension_scoped(tmp_path: Path) -> None:
    markdown = tmp_path / "source.md"
    markdown.write_text("# Heading\n\nUnicode λ.\n", encoding="utf-8")
    malformed = tmp_path / "bad.txt"
    malformed.write_bytes(b"\xff")
    unsupported = tmp_path / "source.pdf"
    unsupported.write_bytes(b"not parsed")

    assert ingest_file(markdown).sections[0].heading == "Heading"
    with pytest.raises(ValueError, match="valid UTF-8"):
        ingest_file(malformed)
    assert "\ufffd" in ingest_file(malformed, errors="replace").text
    with pytest.raises(ValueError, match="extensions"):
        ingest_file(unsupported)


def test_file_collection_order_and_duplicate_path_validation(tmp_path: Path) -> None:
    later = tmp_path / "z.txt"
    earlier = tmp_path / "a.txt"
    later.write_text("later source", encoding="utf-8")
    earlier.write_text("earlier source", encoding="utf-8")

    documents = ingest_files([later, earlier])

    assert [document.source_name for document in documents] == ["a.txt", "z.txt"]
    with pytest.raises(ValueError, match="Duplicate source"):
        ingest_files([earlier, earlier])


def test_index_duplicate_source_and_content_policies() -> None:
    first = ingest_plain_text("shared content", source="one.txt")
    duplicate_content = ingest_plain_text("shared content", source="two.txt")
    changed_same_source = ingest_plain_text("different content", source="one.txt")

    with pytest.raises(ValueError, match="Duplicate document content"):
        RetrievalIndex.build([first, duplicate_content])
    allowed = RetrievalIndex.build(
        [first, duplicate_content],
        index_config=IndexConfig(allow_duplicate_content=True),
    )
    assert len(allowed.documents) == 2
    with pytest.raises(ValueError, match="source paths"):
        RetrievalIndex.build([first, changed_same_source])


@pytest.mark.parametrize(
    "metadata",
    [
        {1: "integer key"},
        {"tuple": ("not", "JSON array")},
        {"not_finite": float("nan")},
    ],
)
def test_ingestion_rejects_malformed_metadata(metadata: dict) -> None:
    with pytest.raises(ValueError, match="JSON"):
        ingest_plain_text("text", source="source.txt", metadata=metadata)


@pytest.mark.parametrize("text", ["", None])
def test_empty_or_non_string_documents_fail(text) -> None:
    with pytest.raises((TypeError, ValueError)):
        ingest_plain_text(text, source="source.txt")
