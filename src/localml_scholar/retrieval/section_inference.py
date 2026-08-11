"""Deterministic scholarly-heading inference for extracted local text."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from localml_scholar.retrieval.documents import Document, Section, stable_identifier

MAJOR_SECTION_ALIASES: dict[str, str] = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "method": "Method",
    "methods": "Methods",
    "methodology": "Methodology",
    "model": "Model",
    "model architecture": "Model Architecture",
    "architecture": "Architecture",
    "algorithm": "Algorithm",
    "approach": "Approach",
    "training": "Training",
    "implementation": "Implementation",
    "implementation details": "Implementation Details",
    "data": "Data",
    "dataset": "Dataset",
    "datasets": "Datasets",
    "experiments": "Experiments",
    "experimental setup": "Experimental Setup",
    "results": "Results",
    "analysis": "Analysis",
    "evaluation": "Evaluation",
    "ablation": "Ablation",
    "ablation study": "Ablation Study",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
    "appendix": "Appendix",
    "front matter": "Front Matter",
    "title": "Front Matter",
}
_NUMBERED = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>[A-Za-z][^\n]{1,100}?)\s*$"
)


@dataclass(frozen=True)
class InferredHeading:
    """One inferred heading boundary."""

    offset: int
    title: str
    level: int
    confidence: float


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .:\t")


def _normalized_heading(value: str) -> str:
    value = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", value)
    value = re.sub(r"\s+\d+\s*$", "", value)
    return re.sub(r"[^a-z ]", "", value.casefold()).strip()


def canonical_section_role(title: str) -> str | None:
    """Return the canonical major role contained in a conservative heading."""
    if not isinstance(title, str):
        raise TypeError("title must be a string.")
    normalized = _normalized_heading(title)
    if normalized in MAJOR_SECTION_ALIASES:
        return MAJOR_SECTION_ALIASES[normalized]
    for alias, canonical in sorted(
        MAJOR_SECTION_ALIASES.items(), key=lambda item: -len(item[0])
    ):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical
    return None


def section_topics_compatible(
    expected: tuple[str, ...], observed: tuple[str, ...]
) -> bool:
    """Return whether any observed heading matches an expected title or role."""
    if not isinstance(expected, tuple) or not isinstance(observed, tuple):
        raise TypeError("expected and observed must be tuples.")
    if not all(
        isinstance(item, str) and item.strip() for item in (*expected, *observed)
    ):
        raise ValueError("Section headings must contain non-whitespace text.")
    if not expected:
        return True
    expected_normalized = {_normalized_heading(item) for item in expected}
    observed_normalized = {_normalized_heading(item) for item in observed}
    if expected_normalized & observed_normalized:
        return True
    if any(
        left in right or right in left
        for left in expected_normalized
        for right in observed_normalized
    ):
        return True
    expected_roles = {
        role for item in expected for role in (canonical_section_role(item),) if role
    }
    observed_roles = {
        role for item in observed for role in (canonical_section_role(item),) if role
    }
    return bool(expected_roles & observed_roles)


def _looks_like_heading_title(title: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
    if not words or len(words) > 14:
        return False
    if "." in title or any(
        symbol in title for symbol in ("=", "∑", "∏", "∈", "≤", "≥", "→")
    ):
        return False
    if not re.search(r"[aeiouy]", title, flags=re.IGNORECASE):
        return False
    content = [
        word
        for word in words
        if word.casefold() not in {"a", "an", "and", "of", "the", "to"}
    ]
    titled = sum(word[0].isupper() for word in content)
    normalized = _normalized_heading(title)
    direct_role = normalized in MAJOR_SECTION_ALIASES
    short_role = canonical_section_role(title) is not None and len(words) <= 8
    return direct_role or short_role or titled / max(1, len(content)) >= 0.5


def infer_scholarly_headings(text: str) -> tuple[InferredHeading, ...]:
    """Infer conservative numbered, named, and uppercase scholarly headings."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    headings: list[InferredHeading] = []
    offset = 0
    seen: set[str] = set()
    in_references = False
    for raw in text.splitlines(keepends=True):
        line = _clean(raw)
        candidate: tuple[str, int, float] | None = None
        numbered = _NUMBERED.match(line)
        if numbered:
            title = _clean(numbered.group("title"))
            if not title.endswith((".", ";", ",")) and _looks_like_heading_title(title):
                candidate = (
                    title,
                    min(6, numbered.group("number").count(".") + 1),
                    0.95,
                )
        if candidate is None:
            normalized = _normalized_heading(line)
            if normalized in MAJOR_SECTION_ALIASES:
                candidate = (MAJOR_SECTION_ALIASES[normalized], 1, 0.98)
        if candidate is None and 2 <= len(line) <= 80:
            words = line.split()
            letters = [char for char in line if char.isalpha()]
            if (
                letters
                and 2 <= len(words) <= 10
                and line.upper() == line
                and not line.endswith((".", ",", ";"))
                and sum(char.isdigit() for char in line) <= 4
                and _looks_like_heading_title(line)
            ):
                candidate = (line.title(), 1, 0.72)
        if candidate is not None:
            title, level, confidence = candidate
            role = canonical_section_role(title)
            if in_references and role != "Appendix":
                candidate = None
            elif role == "References":
                in_references = True
            elif role == "Appendix":
                in_references = False
        if candidate is not None:
            title, level, confidence = candidate
            key = _normalized_heading(title)
            if key not in seen:
                headings.append(InferredHeading(offset, title, level, confidence))
                seen.add(key)
        offset += len(raw)
    return tuple(headings)


def rebuild_document_sections(document: Document) -> Document:
    """Replace untitled page slices with inferred scholarly sections when possible."""
    if not isinstance(document, Document):
        raise TypeError("document must be a Document.")
    titled_fraction = sum(
        section.heading is not None for section in document.sections
    ) / len(document.sections)
    inferred_parser = document.parser_identifier.endswith("+scholarly_sections_v1")
    if titled_fraction >= 0.75 and not inferred_parser:
        return document
    headings = infer_scholarly_headings(document.text)
    if not headings:
        return document
    boundaries = [item.offset for item in headings]
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(document.text))
    by_offset = {item.offset: item for item in headings}
    old_page_spans = [
        (
            section.start_character,
            section.end_character,
            section.page_start,
            section.page_end,
        )
        for section in document.sections
        if section.page_start is not None and section.page_end is not None
    ]
    hierarchy: list[str] = []
    levels: list[int] = []
    sections: list[Section] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if start >= end:
            continue
        inferred = by_offset.get(start)
        if inferred is None:
            heading, level, path = "Front Matter", 1, ("Front Matter",)
            hierarchy, levels = [heading], [level]
        else:
            heading, level = inferred.title, inferred.level
            while levels and levels[-1] >= level:
                levels.pop()
                hierarchy.pop()
            hierarchy.append(heading)
            levels.append(level)
            path = tuple(hierarchy)
        page_ranges = [
            (page_start, page_end)
            for span_start, span_end, page_start, page_end in old_page_spans
            if span_start < end and span_end > start
        ]
        page_start = min(item[0] for item in page_ranges) if page_ranges else None
        page_end = max(item[1] for item in page_ranges) if page_ranges else None
        ordinal = len(sections)
        sections.append(
            Section(
                section_id=stable_identifier(
                    "sec", document.document_id, ordinal, start, end, path
                ),
                document_id=document.document_id,
                ordinal=ordinal,
                heading=heading,
                heading_path=path,
                level=level,
                text=document.text[start:end],
                start_character=start,
                end_character=end,
                start_line=document.text.count("\n", 0, start) + 1,
                end_line=document.text.count("\n", 0, max(start, end - 1)) + 1,
                page_start=page_start,
                page_end=page_end,
            )
        )
    inferred_metadata = dict(document.metadata.get("inferred", {}))
    inferred_metadata.update(
        {
            "heading_inference": "deterministic_scholarly_lines_v1",
            "heading_count": len(headings),
            "section_structure_low_confidence": False,
        }
    )
    metadata = {
        **document.metadata,
        "inferred": inferred_metadata,
    }
    return replace(
        document,
        sections=tuple(sections),
        metadata=metadata,
        parser_identifier=(
            document.parser_identifier
            if inferred_parser
            else f"{document.parser_identifier}+scholarly_sections_v1"
        ),
    )
