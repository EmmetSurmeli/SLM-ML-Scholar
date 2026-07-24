"""Deterministic source-offset-preserving character chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from localml_scholar.retrieval.documents import (
    Chunk,
    Document,
    canonical_json,
    sha256_text,
    stable_identifier,
)
from localml_scholar.retrieval.text import LexicalTokenizerConfig, tokenize_lexically


@dataclass(frozen=True)
class ChunkingConfig:
    """Character chunk targets and an exact fixed overlap."""

    target_characters: int = 600
    maximum_characters: int = 900
    overlap_characters: int = 100
    minimum_characters: int = 80
    respect_sections: bool = True
    respect_paragraphs: bool = True

    def __post_init__(self) -> None:
        for name in (
            "target_characters",
            "maximum_characters",
            "overlap_characters",
            "minimum_characters",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.target_characters <= 0 or self.maximum_characters <= 0:
            raise ValueError("Target and maximum character counts must be positive.")
        if self.target_characters > self.maximum_characters:
            raise ValueError("target_characters cannot exceed maximum_characters.")
        if not 0 <= self.overlap_characters < self.target_characters:
            raise ValueError("overlap_characters must lie in [0, target_characters).")
        if not 1 <= self.minimum_characters <= self.target_characters:
            raise ValueError("minimum_characters must lie in [1, target_characters].")
        if not isinstance(self.respect_sections, bool) or not isinstance(
            self.respect_paragraphs, bool
        ):
            raise TypeError("Chunk boundary flags must be boolean.")
        if not self.respect_sections:
            raise ValueError("Milestone 8 requires respect_sections=True.")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> ChunkingConfig:
        if not isinstance(state, dict) or set(state) != set(cls.__dataclass_fields__):
            raise ValueError("Chunking configuration is malformed.")
        return cls(**state)

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()


def _candidate_boundary(
    positions: list[int],
    *,
    minimum: int,
    target: int,
    maximum: int,
) -> int | None:
    valid = [value for value in positions if minimum <= value <= maximum]
    before = [value for value in valid if value <= target]
    return max(before) if before else (min(valid) if valid else None)


def _fence_spans(text: str, base: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    opened: tuple[str, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)[0]
            if opened is None:
                opened = (marker, base + offset)
            elif opened[0] == marker:
                spans.append((opened[1], base + offset + len(line)))
                opened = None
        offset += len(line)
    if opened is not None:
        spans.append((opened[1], base + len(text)))
    return spans


def _select_end(
    document_text: str,
    start: int,
    region_end: int,
    config: ChunkingConfig,
    fence_spans: list[tuple[int, int]],
) -> int:
    if region_end - start <= config.maximum_characters:
        return region_end
    # A split chunk must be longer than its overlap so the next start advances.
    minimum = min(
        start + max(config.minimum_characters, config.overlap_characters + 1),
        region_end,
    )
    target = min(start + config.target_characters, region_end)
    maximum = min(start + config.maximum_characters, region_end)
    window = document_text[start:maximum]
    groups: list[list[int]] = []
    if config.respect_paragraphs:
        groups.append(
            [start + match.end() for match in re.finditer(r"\n[ \t]*\n", window)]
        )
    groups.extend(
        [
            [
                start + match.end()
                for match in re.finditer(r"[.!?](?:[\"')\]]*)\s+", window)
            ],
            [start + match.end() for match in re.finditer(r"\s+", window)],
        ]
    )
    selected = None
    for positions in groups:
        selected = _candidate_boundary(
            positions,
            minimum=minimum,
            target=target,
            maximum=maximum,
        )
        if selected is not None:
            break
    if selected is None:
        selected = maximum
    for fence_start, fence_end in fence_spans:
        if fence_start < selected < fence_end:
            if fence_start >= minimum:
                selected = fence_start
            elif fence_end <= maximum:
                selected = fence_end
            else:
                selected = maximum
            break
    if selected <= start:
        raise RuntimeError("Chunk boundary selection failed to make progress.")
    return selected


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
    lexical_config: LexicalTokenizerConfig | None = None,
) -> tuple[Chunk, ...]:
    """Chunk each section while preserving exact document slices and overlap."""
    if not isinstance(document, Document):
        raise TypeError("document must be a Document.")
    resolved = config or ChunkingConfig()
    lexical = lexical_config or LexicalTokenizerConfig()
    if not isinstance(resolved, ChunkingConfig):
        raise TypeError("config must be ChunkingConfig.")
    configuration_hash = resolved.state_hash()
    chunks: list[Chunk] = []
    for section in document.sections:
        start = section.start_character
        fences = _fence_spans(section.text, section.start_character)
        while start < section.end_character:
            end = _select_end(
                document.text,
                start,
                section.end_character,
                resolved,
                fences,
            )
            text = document.text[start:end]
            term_count = len(tokenize_lexically(text, lexical))
            chunk_id = stable_identifier(
                "chk",
                document.document_id,
                section.section_id,
                start,
                end,
                configuration_hash,
                sha256_text(text),
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    section_id=section.section_id,
                    ordinal=len(chunks),
                    text=text,
                    start_character=start,
                    end_character=end,
                    start_line=_line_number(document.text, start),
                    end_line=_line_number(document.text, end - 1),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    heading_path=section.heading_path,
                    term_count=term_count,
                    content_sha256=sha256_text(text),
                    configuration_sha256=configuration_hash,
                )
            )
            if end == section.end_character:
                break
            start = end - resolved.overlap_characters
    validate_chunk_coverage(document, chunks, resolved)
    return tuple(chunks)


def validate_chunk_coverage(
    document: Document,
    chunks: tuple[Chunk, ...] | list[Chunk],
    config: ChunkingConfig,
) -> None:
    """Validate exact slices, order, overlap, bounds, and complete coverage."""
    if not chunks:
        raise ValueError("At least one chunk is required.")
    cursor = 0
    previous: Chunk | None = None
    for ordinal, chunk in enumerate(chunks):
        if chunk.document_id != document.document_id or chunk.ordinal != ordinal:
            raise ValueError("Chunk document linkage or order is inconsistent.")
        if chunk.end_character > len(document.text):
            raise ValueError("Chunk extends outside the document.")
        if document.text[chunk.start_character : chunk.end_character] != chunk.text:
            raise ValueError("Chunk text does not equal its source slice.")
        if len(chunk.text) > config.maximum_characters:
            raise ValueError("Chunk exceeds maximum_characters.")
        if chunk.start_character > cursor:
            raise ValueError("Chunk coverage contains an unexplained gap.")
        if previous is not None and previous.section_id == chunk.section_id:
            overlap = previous.end_character - chunk.start_character
            if overlap != config.overlap_characters:
                raise ValueError("Adjacent same-section chunks have wrong overlap.")
            if previous.text[-overlap:] != chunk.text[:overlap] if overlap else False:
                raise ValueError("Chunk overlap text is inconsistent.")
        cursor = max(cursor, chunk.end_character)
        previous = chunk
    if cursor != len(document.text):
        raise ValueError("Chunks do not cover the complete document.")
