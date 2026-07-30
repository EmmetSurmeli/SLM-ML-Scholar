"""Canonical paper metadata and reference extraction."""

from __future__ import annotations

import re
from typing import Any

from localml_scholar.retrieval import Document
from localml_scholar.retrieval.documents import stable_identifier
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.models import (
    Paper,
    PaperSection,
    ReferenceEntry,
    ScholarlyEvidence,
)
from localml_scholar.scholarly.source import citation_for_range

_YEAR = re.compile(r"\b(?:18|19|20|21)\d{2}\b")


def _evidence(
    document: Document,
    category: str,
    value: Any,
    start: int,
    end: int,
    method: str,
    confidence: str = "high",
) -> ScholarlyEvidence:
    source = document.text[start:end]
    normalized = value.casefold().strip() if isinstance(value, str) else value
    return ScholarlyEvidence(
        evidence_id=stable_identifier(
            "ev", document.document_id, category, start, end, value
        ),
        category=category,
        value=value,
        normalized_value=normalized,
        citation=citation_for_range(document, start, end),
        source_text=source,
        extraction_method=method,
        confidence=confidence,
    )


def _exact_metadata_value(
    document: Document,
    category: str,
    value: object,
) -> ScholarlyEvidence | None:
    if isinstance(value, str) and value.strip():
        start = document.text.find(value)
        if start >= 0:
            return _evidence(
                document,
                category,
                value,
                start,
                start + len(value),
                "validated_user_metadata_exact_source_match",
            )
    if (
        category == "authors"
        and isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
    ):
        joined = ", ".join(value)
        start = document.text.find(joined)
        if start >= 0:
            return _evidence(
                document,
                category,
                value,
                start,
                start + len(joined),
                "validated_user_metadata_exact_source_match",
            )
    if category == "year" and isinstance(value, int):
        raw = str(value)
        start = document.text.find(raw)
        if start >= 0:
            return _evidence(
                document,
                category,
                value,
                start,
                start + len(raw),
                "validated_user_metadata_exact_source_match",
            )
    return None


def _explicit_label(
    document: Document,
    category: str,
    pattern: str,
    *,
    transform=lambda value: value.strip(),
) -> ScholarlyEvidence | None:
    match = re.search(pattern, document.text[:4000], flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        return None
    value = transform(match.group("value"))
    start, end = match.span("value")
    return _evidence(
        document,
        category,
        value,
        start,
        end,
        "explicit_source_label",
    )


def extract_metadata(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[dict[str, ScholarlyEvidence | None], tuple[str, ...]]:
    """Extract only source-verifiable metadata using a disclosed priority order."""
    user = document.metadata.get("user", {})
    if not isinstance(user, dict):
        user = {}
    warnings: list[str] = []
    fields: dict[str, ScholarlyEvidence | None] = {}
    for category in ("title", "authors", "year", "venue", "keywords", "identifier"):
        supplied = user.get(category)
        fields[category] = _exact_metadata_value(document, category, supplied)
        if supplied is not None and fields[category] is None:
            warnings.append(f"user_metadata_not_present_in_source:{category}")

    if fields["title"] is None and document.title:
        start = document.text.find(document.title)
        if start >= 0:
            fields["title"] = _evidence(
                document,
                "title",
                document.title,
                start,
                start + len(document.title),
                "canonical_document_title_exact_source_match",
            )
    if fields["title"] is None:
        first_nonempty = next(
            (
                (offset, line.strip("# \t\r\n"))
                for offset, line in _line_offsets(document.text)
                if line.strip("# \t\r\n")
            ),
            None,
        )
        if first_nonempty is not None:
            offset, title = first_nonempty
            start = document.text.find(title, offset)
            fields["title"] = _evidence(
                document,
                "title",
                title,
                start,
                start + len(title),
                "first_nonempty_source_line",
                "low",
            )

    if fields["authors"] is None:
        fields["authors"] = _explicit_label(
            document,
            "authors",
            r"^\s*(?:authors?|by)\s*:\s*(?P<value>[^\n]+)$",
            transform=lambda value: [
                item.strip() for item in re.split(r",|;|\band\b", value) if item.strip()
            ],
        )
    if fields["year"] is None:
        fields["year"] = _explicit_label(
            document,
            "year",
            r"^\s*(?:year|published)\s*:\s*(?P<value>(?:18|19|20|21)\d{2})\s*$",
            transform=int,
        )
    if fields["venue"] is None:
        fields["venue"] = _explicit_label(
            document,
            "venue",
            r"^\s*venue\s*:\s*(?P<value>[^\n]+)$",
        )
    if fields["keywords"] is None:
        fields["keywords"] = _explicit_label(
            document,
            "keywords",
            r"^\s*keywords?\s*:\s*(?P<value>[^\n]+)$",
            transform=lambda value: [
                item.strip() for item in value.split(",") if item.strip()
            ],
        )
    if fields["identifier"] is None:
        fields["identifier"] = _explicit_label(
            document,
            "identifier",
            r"^\s*(?:doi|identifier)\s*:\s*(?P<value>[^\s]+)\s*$",
        )

    abstract_sections = tuple(
        section for section in sections if "abstract" in section.roles
    )
    fields["abstract"] = None
    if abstract_sections:
        source_section = next(
            section
            for section in document.sections
            if section.section_id == abstract_sections[0].section_id
        )
        fields["abstract"] = _evidence(
            document,
            "abstract",
            _without_heading(source_section.text),
            source_section.start_character,
            source_section.end_character,
            "section_role_abstract",
        )
    return fields, tuple(warnings)


def _line_offsets(text: str):
    offset = 0
    for line in text.splitlines(keepends=True):
        yield offset, line
        offset += len(line)


def _without_heading(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def extract_references(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[ReferenceEntry, ...]:
    """Parse separate reference lines conservatively without external lookup."""
    reference_ids = {
        section.section_id for section in sections if "references" in section.roles
    }
    entries: list[ReferenceEntry] = []
    for section in document.sections:
        if section.section_id not in reference_ids:
            continue
        for local_offset, line in _line_offsets(section.text):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            start = section.start_character + local_offset + line.find(raw)
            end = start + len(raw)
            year_match = _YEAR.search(raw)
            identifier_match = re.search(r"\b10\.\d{4,9}/\S+\b", raw)
            numbered = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", raw)
            warnings: list[str] = []
            authors: tuple[str, ...] = ()
            title: str | None = None
            venue: str | None = None
            if year_match is not None:
                local_year = re.search(r"(?:19|20)\d{2}", numbered)
            else:
                local_year = None
            if local_year is not None:
                before_year = numbered[: local_year.start()].rstrip(" .")
                pieces = [
                    item.strip()
                    for item in re.split(r"\.\s+", before_year)
                    if item.strip()
                ]
                author_text = ". ".join(pieces[:-1])
                title = pieces[-1] if len(pieces) >= 2 else None
                authors = tuple(
                    item.strip()
                    for item in re.split(r",|;|\s+and\s+", author_text)
                    if item.strip()
                )
                venue = numbered[local_year.end() :].strip(" .") or None
            if not authors or title is None:
                authors = ()
                title = None
                venue = None
                warnings.append("reference_fields_not_reliably_separable")
            entries.append(
                ReferenceEntry(
                    reference_id=stable_identifier(
                        "ref", document.document_id, start, end, raw
                    ),
                    raw_text=raw,
                    authors=authors,
                    title=title,
                    year=None if year_match is None else int(year_match.group()),
                    venue=venue,
                    identifier=(
                        None if identifier_match is None else identifier_match.group()
                    ),
                    citation=citation_for_range(document, start, end),
                    parse_warnings=tuple(warnings),
                )
            )
    return tuple(entries)


def build_paper(
    document: Document,
    sections: tuple[PaperSection, ...],
    references: tuple[ReferenceEntry, ...],
    config: ScholarlyConfig,
) -> tuple[Paper, tuple[str, ...]]:
    """Build a canonical paper that never invents absent metadata."""
    fields, warnings = extract_metadata(document, sections)
    paper = Paper(
        paper_id=stable_identifier(
            "paper", document.document_id, document.content_sha256, config.state_hash()
        ),
        document_id=document.document_id,
        title=fields["title"],
        authors=fields["authors"],
        year=fields["year"],
        venue=fields["venue"],
        abstract=fields["abstract"],
        keywords=fields["keywords"],
        identifier=fields["identifier"],
        sections=sections,
        references=references,
        source_hash=document.content_sha256,
        analysis_config_sha256=config.state_hash(),
    )
    return paper, warnings
