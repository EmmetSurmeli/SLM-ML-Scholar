"""Transparent sentence-like segmentation for sources and answer claims."""

from __future__ import annotations

import re
from dataclasses import dataclass

from localml_scholar.answering.citations import (
    parse_inline_citations,
    strip_inline_citations,
)

_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "vs.",
    "etc.",
}
_TERMINATORS = ".!?。！？"
_BOILERPLATE = {
    "the indexed sources state:",
    "sources:",
    "bibliography:",
    "i could not find enough support in the indexed documents to answer this question.",
}


@dataclass(frozen=True)
class SentenceSpan:
    """One exact source substring and its half-open character span."""

    text: str
    start_character: int
    end_character: int
    kind: str = "prose"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Sentence text must be non-empty.")
        if not 0 <= self.start_character < self.end_character:
            raise ValueError("Sentence offsets must be non-empty and ordered.")
        if self.kind not in {"prose", "bullet", "code", "heading"}:
            raise ValueError("Unknown sentence span kind.")


def _is_decimal(text: str, position: int) -> bool:
    return (
        text[position] == "."
        and position > 0
        and position + 1 < len(text)
        and text[position - 1].isdigit()
        and text[position + 1].isdigit()
    )


def _is_abbreviation(text: str, position: int) -> bool:
    prefix = text[: position + 1].casefold()
    return any(prefix.endswith(value) for value in _ABBREVIATIONS)


def _trimmed_span(text: str, start: int, end: int, kind: str) -> SentenceSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    return SentenceSpan(
        text=text[start:end], start_character=start, end_character=end, kind=kind
    )


def _split_prose_region(
    text: str,
    start: int,
    end: int,
    *,
    kind: str,
) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    sentence_start = start
    position = start
    while position < end:
        character = text[position]
        if (
            character in _TERMINATORS
            and not _is_decimal(text, position)
            and not _is_abbreviation(text, position)
        ):
            boundary = position + 1
            cursor = boundary
            while cursor < end and text[cursor] in " \t":
                cursor += 1
            citation_match = re.match(
                r"(?:\[C[1-9]\d*(?:\s*,\s*C[1-9]\d*)*\][ \t]*)+",
                text[cursor:end],
            )
            if citation_match:
                boundary = cursor + citation_match.end()
            span = _trimmed_span(text, sentence_start, boundary, kind)
            if span is not None:
                spans.append(span)
            sentence_start = boundary
            position = boundary
            continue
        position += 1
    span = _trimmed_span(text, sentence_start, end, kind)
    if span is not None:
        spans.append(span)
    return spans


def segment_source_text(text: str) -> tuple[SentenceSpan, ...]:
    """Segment source text while preserving fenced code as an exact block."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text:
        return ()
    spans: list[SentenceSpan] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    prose_start: int | None = None
    in_fence = False
    fence_start = 0
    fence_marker: str | None = None

    def flush_prose(end: int) -> None:
        nonlocal prose_start
        if prose_start is not None:
            spans.extend(_split_prose_region(text, prose_start, end, kind="prose"))
            prose_start = None

    for line in lines:
        stripped = line.lstrip()
        fence = re.match(r"(`{3,}|~{3,})", stripped)
        if in_fence:
            if fence and fence.group(1)[0] == fence_marker:
                fence_end = offset + len(line)
                span = _trimmed_span(text, fence_start, fence_end, "code")
                if span is not None:
                    spans.append(span)
                in_fence = False
                fence_marker = None
            offset += len(line)
            continue
        if fence:
            flush_prose(offset)
            in_fence = True
            fence_start = offset
            fence_marker = fence.group(1)[0]
            offset += len(line)
            continue
        stripped_line = line.strip()
        if not stripped_line:
            flush_prose(offset)
        elif re.match(r"^#{1,6}\s+", stripped_line):
            flush_prose(offset)
            span = _trimmed_span(text, offset, offset + len(line), "heading")
            if span is not None:
                spans.append(span)
        elif re.match(r"^(?:[-*+]|\d+[.)])\s+", stripped_line):
            flush_prose(offset)
            span = _trimmed_span(text, offset, offset + len(line), "bullet")
            if span is not None:
                spans.append(span)
        elif prose_start is None:
            prose_start = offset
        offset += len(line)
    if in_fence:
        span = _trimmed_span(text, fence_start, len(text), "code")
        if span is not None:
            spans.append(span)
    else:
        flush_prose(len(text))
    return tuple(sorted(spans, key=lambda item: item.start_character))


def segment_answer_claims(text: str) -> tuple[str, ...]:
    """Return sentence-like answer units, excluding headings and boilerplate."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    claims: list[str] = []
    bullet_groups: list[str] = []
    prose_lines: list[str] = []
    current_bullet: list[str] | None = None

    def flush_bullet() -> None:
        nonlocal current_bullet
        if current_bullet is not None:
            bullet_groups.append("\n".join(current_bullet).strip())
            current_bullet = None

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(?:[-*+]|\d+[.)])\s+", stripped):
            flush_bullet()
            current_bullet = [stripped]
        elif current_bullet is not None:
            citation_only = bool(parse_inline_citations(stripped, strict=False)) and (
                not strip_inline_citations(stripped).strip()
            )
            if (
                not stripped
                or citation_only
                or stripped.casefold() in _BOILERPLATE
                or re.match(r"^#{1,6}\s+", stripped)
            ):
                flush_bullet()
                prose_lines.append(line)
            else:
                current_bullet.append(line)
        else:
            prose_lines.append(line)
    flush_bullet()
    candidates = list(bullet_groups)
    prose = "\n".join(prose_lines)
    candidates.extend(
        span.text for span in segment_source_text(prose) if span.kind != "heading"
    )
    for raw_candidate in candidates:
        candidate = raw_candidate.strip()
        if candidate.casefold() in _BOILERPLATE:
            continue
        if candidate.casefold().startswith(("sources:", "bibliography:")):
            continue
        without_citations = strip_inline_citations(candidate).strip()
        if without_citations.casefold() in _BOILERPLATE:
            continue
        candidate = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", candidate).strip()
        if not candidate:
            continue
        occurrences = parse_inline_citations(candidate, strict=False)
        if occurrences and not strip_inline_citations(candidate).strip():
            continue
        claims.append(candidate)
    return tuple(claims)
