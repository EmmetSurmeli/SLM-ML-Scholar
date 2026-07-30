"""Exact source-range construction and validation for scholarly evidence."""

from __future__ import annotations

import hashlib

from localml_scholar.retrieval import Document, Section
from localml_scholar.scholarly.models import SourceCitation


def section_for_range(document: Document, start: int, end: int) -> Section:
    """Return the unique section containing a non-empty document range."""
    if not isinstance(document, Document):
        raise TypeError("document must be a Document.")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError("Source offsets must be integers.")
    if not 0 <= start < end <= len(document.text):
        raise ValueError("Source range lies outside the document.")
    matches = tuple(
        section
        for section in document.sections
        if section.start_character <= start and end <= section.end_character
    )
    if len(matches) != 1:
        raise ValueError("Source range must lie inside exactly one section.")
    return matches[0]


def citation_for_range(
    document: Document,
    start: int,
    end: int,
) -> SourceCitation:
    """Build an exact citation without rewriting or expanding its source range."""
    section = section_for_range(document, start, end)
    source_text = document.text[start:end]
    return SourceCitation(
        document_id=document.document_id,
        section_id=section.section_id,
        source_name=document.source_name,
        title=document.title,
        heading_path=section.heading_path,
        start_character=start,
        end_character=end,
        start_line=document.text.count("\n", 0, start) + 1,
        end_line=document.text.count("\n", 0, end - 1) + 1,
        page_start=section.page_start,
        page_end=section.page_end,
        source_text_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
    )


def validate_source_citation(
    document: Document,
    citation: SourceCitation,
    source_text: str | None = None,
) -> None:
    """Reject missing sources, range drift, metadata drift, and hash drift."""
    if citation.document_id != document.document_id:
        raise ValueError("Citation document is not the supplied source document.")
    section = section_for_range(
        document,
        citation.start_character,
        citation.end_character,
    )
    if citation.section_id != section.section_id:
        raise ValueError("Citation section does not match its character range.")
    exact = document.text[citation.start_character : citation.end_character]
    digest = hashlib.sha256(exact.encode("utf-8")).hexdigest()
    if digest != citation.source_text_sha256:
        raise ValueError("Citation source hash does not match the document.")
    if source_text is not None and source_text != exact:
        raise ValueError("Cited source text is not an exact document substring.")
    expected = citation_for_range(
        document,
        citation.start_character,
        citation.end_character,
    )
    if citation != expected:
        raise ValueError("Citation metadata does not match the canonical source.")
