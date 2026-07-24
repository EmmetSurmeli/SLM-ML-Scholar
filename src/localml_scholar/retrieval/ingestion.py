"""Strict local text, Markdown, and pre-extracted PDF-page ingestion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from localml_scholar.retrieval.documents import (
    Document,
    PageText,
    Section,
    canonical_json,
    normalize_source_identifier,
    sha256_text,
    stable_identifier,
)

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_SUPPORTED_ERRORS = {"strict", "replace"}


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _metadata(
    user_metadata: Mapping[str, Any] | None,
    inferred_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    user = {} if user_metadata is None else dict(user_metadata)
    inferred = dict(inferred_metadata)
    canonical_json(user)
    canonical_json(inferred)
    return {"user": user, "inferred": inferred}


def _document_shell(
    *,
    source: str | Path,
    media_type: str,
    text: str,
    title: str | None,
    user_metadata: Mapping[str, Any] | None,
    parser_identifier: str,
    section_specs: list[dict[str, Any]],
    inferred_metadata_extra: Mapping[str, Any] | None = None,
) -> Document:
    if not isinstance(text, str) or not text:
        raise ValueError("Document text must be non-empty.")
    source_path = normalize_source_identifier(source)
    source_name = Path(source_path).name or source_path
    content_hash = sha256_text(text)
    document_id = stable_identifier("doc", source_path, content_hash)
    sections: list[Section] = []
    for ordinal, spec in enumerate(section_specs):
        start = spec["start_character"]
        end = spec["end_character"]
        section_id = stable_identifier(
            "sec",
            document_id,
            ordinal,
            start,
            end,
            spec["heading_path"],
        )
        sections.append(
            Section(
                section_id=section_id,
                document_id=document_id,
                ordinal=ordinal,
                heading=spec["heading"],
                heading_path=tuple(spec["heading_path"]),
                level=spec["level"],
                text=text[start:end],
                start_character=start,
                end_character=end,
                start_line=_line_number(text, start),
                end_line=_line_number(text, max(start, end - 1)),
                page_start=spec.get("page_start"),
                page_end=spec.get("page_end"),
            )
        )
    inferred = {
        "source_filename": source_name,
        "media_type": media_type,
        "document_hash": content_hash,
        "parser_identifier": parser_identifier,
        "section_count": len(sections),
    }
    if inferred_metadata_extra is not None:
        inferred.update(inferred_metadata_extra)
    return Document(
        document_id=document_id,
        source_path=source_path,
        source_name=source_name,
        media_type=media_type,
        title=title,
        text=text,
        content_sha256=content_hash,
        byte_length=len(text.encode("utf-8")),
        character_length=len(text),
        metadata=_metadata(user_metadata, inferred),
        sections=tuple(sections),
        parser_identifier=parser_identifier,
    )


def ingest_plain_text(
    text: str,
    *,
    source: str | Path,
    title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    """Ingest exact non-empty plain text as one root section."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text:
        raise ValueError("Plain-text documents must be non-empty.")
    return _document_shell(
        source=source,
        media_type="text/plain",
        text=text,
        title=title,
        user_metadata=metadata,
        parser_identifier="plain_text_v1",
        section_specs=[
            {
                "heading": None,
                "heading_path": (),
                "level": None,
                "start_character": 0,
                "end_character": len(text),
            }
        ],
    )


def _markdown_headings(text: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_marker: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker_match = re.match(r"(`{3,}|~{3,})", stripped)
        if marker_match:
            marker = marker_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
        elif not in_fence:
            match = _ATX_HEADING.match(line.rstrip("\r\n"))
            if match:
                headings.append((offset, len(match.group(1)), match.group(2).strip()))
        offset += len(line)
    return headings


def ingest_markdown(
    text: str,
    *,
    source: str | Path,
    title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    """Ingest Markdown with deterministic ATX-heading section boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text:
        raise ValueError("Markdown documents must be non-empty.")
    headings = _markdown_headings(text)
    boundaries = [offset for offset, _, _ in headings]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))
    heading_by_offset = {
        offset: (level, heading) for offset, level, heading in headings
    }
    hierarchy: list[str] = []
    levels: list[int] = []
    specs: list[dict[str, Any]] = []
    inferred_title = title
    for index, start in enumerate(boundaries[:-1]):
        end = boundaries[index + 1]
        heading_info = heading_by_offset.get(start)
        if heading_info is None:
            heading = None
            level = None
            path: tuple[str, ...] = ()
        else:
            level, heading = heading_info
            while levels and levels[-1] >= level:
                levels.pop()
                hierarchy.pop()
            levels.append(level)
            hierarchy.append(heading)
            path = tuple(hierarchy)
            if inferred_title is None and level == 1:
                inferred_title = heading
        if start != end:
            specs.append(
                {
                    "heading": heading,
                    "heading_path": path,
                    "level": level,
                    "start_character": start,
                    "end_character": end,
                }
            )
    return _document_shell(
        source=source,
        media_type="text/markdown",
        text=text,
        title=inferred_title,
        user_metadata=metadata,
        parser_identifier="markdown_atx_v1",
        section_specs=specs,
    )


def ingest_pdf_text(
    pages: Sequence[PageText],
    *,
    source: str | Path,
    title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    """Ingest externally extracted PDF page text without performing PDF parsing."""
    if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
        raise TypeError("pages must be a sequence of PageText objects.")
    materialized = list(pages)
    if not materialized:
        raise ValueError("PDF-derived input must contain at least one page.")
    if not all(isinstance(page, PageText) for page in materialized):
        raise TypeError("Every PDF-derived page must be PageText.")
    numbers = [page.page_number for page in materialized]
    if numbers != sorted(set(numbers)):
        raise ValueError("PDF-derived page numbers must be unique and increasing.")
    nonempty_pages = [page for page in materialized if page.text]
    if not nonempty_pages:
        raise ValueError("PDF-derived input must contain some extracted text.")
    # Empty extracted pages contain no source characters. Keep their numbers in
    # metadata without creating artificial newline-only searchable sections.
    text = "\n".join(page.text for page in nonempty_pages)
    specs: list[dict[str, Any]] = []
    cursor = 0
    for index, page in enumerate(nonempty_pages):
        end = cursor + len(page.text) + (1 if index < len(nonempty_pages) - 1 else 0)
        specs.append(
            {
                "heading": None,
                "heading_path": (),
                "level": None,
                "start_character": cursor,
                "end_character": end,
                "page_start": page.page_number,
                "page_end": page.page_number,
            }
        )
        cursor = end
    return _document_shell(
        source=source,
        media_type="application/pdf-derived-text",
        text=text,
        title=title,
        user_metadata=metadata,
        parser_identifier="external_page_text_v1",
        section_specs=specs,
        inferred_metadata_extra={
            "page_count": len(materialized),
            "empty_pages": [page.page_number for page in materialized if not page.text],
            "extraction_policy": "externally_supplied_page_text",
        },
    )


def ingest_file(
    path: str | Path,
    *,
    errors: str = "strict",
    title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    """Read one local `.txt` or Markdown file with an explicit UTF-8 policy."""
    source = Path(path)
    if errors not in _SUPPORTED_ERRORS:
        raise ValueError("errors must be 'strict' or 'replace'.")
    try:
        raw = source.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document does not exist: {source}") from None
    try:
        text = raw.decode("utf-8", errors=errors)
    except UnicodeDecodeError as error:
        raise ValueError(f"Document is not valid UTF-8: {source}") from error
    suffix = source.suffix.casefold()
    if suffix == ".txt":
        return ingest_plain_text(
            text,
            source=source,
            title=title,
            metadata=metadata,
        )
    if suffix in {".md", ".markdown"}:
        return ingest_markdown(
            text,
            source=source,
            title=title,
            metadata=metadata,
        )
    raise ValueError("Supported document extensions are .txt, .md, and .markdown.")


def ingest_files(paths: Sequence[str | Path]) -> tuple[Document, ...]:
    """Ingest explicitly supplied files in normalized deterministic path order."""
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("paths must be a sequence.")
    normalized = sorted(
        (normalize_source_identifier(path), Path(path)) for path in paths
    )
    if not normalized:
        raise ValueError("At least one document path is required.")
    source_ids = [item[0] for item in normalized]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source paths are not allowed.")
    return tuple(ingest_file(path) for _, path in normalized)
