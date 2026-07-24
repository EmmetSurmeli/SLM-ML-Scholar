"""Canonical immutable retrieval documents, sections, chunks, and citations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonical_json(value: object) -> str:
    """Return deterministic JSON after rejecting lossy JSON conversions."""
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Value must be exactly JSON-serializable.") from error
    if json.loads(serialized) != value:
        raise ValueError(
            "Value must use exact JSON arrays, objects with string keys, and scalars."
        )
    return serialized


def sha256_text(text: str) -> str:
    """Hash exact UTF-8 text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_identifier(prefix: str, *parts: object) -> str:
    """Create a compact deterministic identifier from canonical components."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string.")

    def identity_value(value: object) -> object:
        if isinstance(value, tuple):
            return [identity_value(item) for item in value]
        if isinstance(value, list):
            return [identity_value(item) for item in value]
        if isinstance(value, dict):
            return {key: identity_value(item) for key, item in value.items()}
        return value

    digest = hashlib.sha256(
        canonical_json(identity_value(list(parts))).encode("utf-8")
    ).hexdigest()
    return f"{prefix}_{digest[:24]}"


def normalize_source_identifier(source: str | Path) -> str:
    """Normalize a logical source without resolving it against this machine."""
    if isinstance(source, Path):
        value = source.as_posix()
    elif isinstance(source, str):
        value = source.replace("\\", "/")
    else:
        raise TypeError("source must be a string or Path.")
    value = value.strip()
    if not value:
        raise ValueError("source must be non-empty.")
    while "//" in value:
        value = value.replace("//", "/")
    return value


def _optional_string(value: str | None, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be None or a non-empty string.")
    return value


def _positive_optional_integer(value: int | None, name: str) -> int | None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be None or an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True)
class PageText:
    """Externally extracted text for one known PDF page."""

    page_number: int
    text: str

    def __post_init__(self) -> None:
        _positive_optional_integer(self.page_number, "page_number")
        if not isinstance(self.text, str):
            raise TypeError("Page text must be a string.")


@dataclass(frozen=True)
class Section:
    """One exact ordered source slice with optional heading and page metadata."""

    section_id: str
    document_id: str
    ordinal: int
    heading: str | None
    heading_path: tuple[str, ...]
    level: int | None
    text: str
    start_character: int
    end_character: int
    start_line: int
    end_line: int
    page_start: int | None = None
    page_end: int | None = None

    def __post_init__(self) -> None:
        if not self.section_id or not self.document_id:
            raise ValueError("Section and document IDs must be non-empty.")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("Section ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("Section ordinal must be non-negative.")
        _optional_string(self.heading, "heading")
        if not isinstance(self.heading_path, tuple) or not all(
            isinstance(part, str) and part for part in self.heading_path
        ):
            raise ValueError("heading_path must be a tuple of non-empty strings.")
        if self.level is not None:
            if isinstance(self.level, bool) or not isinstance(self.level, int):
                raise TypeError("Section level must be None or an integer.")
            if not 1 <= self.level <= 6:
                raise ValueError("Section level must lie in [1, 6].")
        if not isinstance(self.text, str):
            raise TypeError("Section text must be a string.")
        if not 0 <= self.start_character < self.end_character:
            raise ValueError("Section character offsets must be non-empty and ordered.")
        if not 1 <= self.start_line <= self.end_line:
            raise ValueError("Section line ranges must be positive and ordered.")
        page_start = _positive_optional_integer(self.page_start, "page_start")
        page_end = _positive_optional_integer(self.page_end, "page_end")
        if (page_start is None) != (page_end is None):
            raise ValueError("Section page range must be fully present or absent.")
        if page_start is not None and page_end is not None and page_end < page_start:
            raise ValueError("Section page range must be ordered.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "document_id": self.document_id,
            "ordinal": self.ordinal,
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "level": self.level,
            "text": self.text,
            "start_character": self.start_character,
            "end_character": self.end_character,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Section:
        expected = {
            "section_id",
            "document_id",
            "ordinal",
            "heading",
            "heading_path",
            "level",
            "text",
            "start_character",
            "end_character",
            "start_line",
            "end_line",
            "page_start",
            "page_end",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Section state keys are malformed.")
        values = dict(state)
        path = values["heading_path"]
        if not isinstance(path, list):
            raise ValueError("Serialized heading_path must be a list.")
        values["heading_path"] = tuple(path)
        return cls(**values)


@dataclass(frozen=True)
class Document:
    """Canonical local document with exact original text and ordered sections."""

    document_id: str
    source_path: str
    source_name: str
    media_type: str
    title: str | None
    text: str
    content_sha256: str
    byte_length: int
    character_length: int
    metadata: dict[str, Any]
    sections: tuple[Section, ...]
    parser_identifier: str
    ingestion_version: int = 1

    def __post_init__(self) -> None:
        for name in ("document_id", "source_path", "source_name", "media_type"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        _optional_string(self.title, "title")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Document text must be a non-empty string.")
        if self.content_sha256 != sha256_text(self.text):
            raise ValueError("Document content_sha256 does not match its text.")
        if self.byte_length != len(self.text.encode("utf-8")):
            raise ValueError("Document byte_length does not match its text.")
        if self.character_length != len(self.text):
            raise ValueError("Document character_length does not match its text.")
        if not isinstance(self.metadata, dict):
            raise TypeError("Document metadata must be a dictionary.")
        canonical_json(self.metadata)
        if not isinstance(self.sections, tuple) or not self.sections:
            raise ValueError("Document must contain at least one section.")
        if not isinstance(self.parser_identifier, str) or not self.parser_identifier:
            raise ValueError("parser_identifier must be non-empty.")
        if self.ingestion_version != 1:
            raise ValueError("Unsupported document ingestion version.")
        cursor = 0
        for ordinal, section in enumerate(self.sections):
            if section.document_id != self.document_id or section.ordinal != ordinal:
                raise ValueError("Document sections have inconsistent linkage/order.")
            if section.start_character != cursor:
                raise ValueError("Document sections must cover text without gaps.")
            if (
                self.text[section.start_character : section.end_character]
                != section.text
            ):
                raise ValueError(
                    "Section text does not equal its document source slice."
                )
            cursor = section.end_character
        if cursor != len(self.text):
            raise ValueError("Document sections do not cover the complete source text.")

    @property
    def logical_collection(self) -> str | None:
        value = self.metadata.get("user", {}).get("logical_collection")
        return value if isinstance(value, str) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "media_type": self.media_type,
            "title": self.title,
            "text": self.text,
            "content_sha256": self.content_sha256,
            "byte_length": self.byte_length,
            "character_length": self.character_length,
            "metadata": self.metadata,
            "sections": [section.to_dict() for section in self.sections],
            "parser_identifier": self.parser_identifier,
            "ingestion_version": self.ingestion_version,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Document:
        expected = {
            "document_id",
            "source_path",
            "source_name",
            "media_type",
            "title",
            "text",
            "content_sha256",
            "byte_length",
            "character_length",
            "metadata",
            "sections",
            "parser_identifier",
            "ingestion_version",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Document state keys are malformed.")
        values = dict(state)
        sections = values["sections"]
        if not isinstance(sections, list):
            raise ValueError("Serialized document sections must be a list.")
        values["sections"] = tuple(Section.from_dict(item) for item in sections)
        return cls(**values)


@dataclass(frozen=True)
class Chunk:
    """One exact searchable source slice."""

    chunk_id: str
    document_id: str
    section_id: str
    ordinal: int
    text: str
    start_character: int
    end_character: int
    start_line: int
    end_line: int
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]
    term_count: int
    content_sha256: str
    configuration_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.document_id or not self.section_id:
            raise ValueError("Chunk identifiers must be non-empty.")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("Chunk ordinal must be an integer.")
        if self.ordinal < 0:
            raise ValueError("Chunk ordinal must be non-negative.")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Chunk text must be non-empty.")
        if not 0 <= self.start_character < self.end_character:
            raise ValueError("Chunk character offsets must be non-empty and ordered.")
        if not 1 <= self.start_line <= self.end_line:
            raise ValueError("Chunk line range must be positive and ordered.")
        if self.content_sha256 != sha256_text(self.text):
            raise ValueError("Chunk content hash does not match its text.")
        if isinstance(self.term_count, bool) or not isinstance(self.term_count, int):
            raise TypeError("Chunk term_count must be an integer.")
        if self.term_count < 0:
            raise ValueError("Chunk term_count must be non-negative.")
        if (
            not isinstance(self.configuration_sha256, str)
            or len(self.configuration_sha256) != 64
        ):
            raise ValueError("Chunk configuration_sha256 must be a SHA-256 digest.")
        if not isinstance(self.heading_path, tuple) or not all(
            isinstance(part, str) and part for part in self.heading_path
        ):
            raise ValueError("Chunk heading_path must contain non-empty strings.")
        page_start = _positive_optional_integer(self.page_start, "page_start")
        page_end = _positive_optional_integer(self.page_end, "page_end")
        if (page_start is None) != (page_end is None):
            raise ValueError("Chunk page range must be fully present or absent.")
        if page_start is not None and page_end is not None and page_end < page_start:
            raise ValueError("Chunk page range must be ordered.")
        if not isinstance(self.metadata, dict):
            raise TypeError("Chunk metadata must be a dictionary.")
        canonical_json(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["heading_path"] = list(self.heading_path)
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Chunk:
        expected = set(cls.__dataclass_fields__)
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Chunk state keys are malformed.")
        values = dict(state)
        path = values["heading_path"]
        if not isinstance(path, list):
            raise ValueError("Serialized chunk heading_path must be a list.")
        values["heading_path"] = tuple(path)
        return cls(**values)


@dataclass(frozen=True)
class Citation:
    """Structured deterministic pointer to one exact retrieved chunk."""

    document_id: str
    source_name: str
    title: str | None
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    start_line: int
    end_line: int
    chunk_id: str

    def __post_init__(self) -> None:
        for name in ("document_id", "source_name", "chunk_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        _optional_string(self.title, "title")
        if not isinstance(self.heading_path, tuple) or not all(
            isinstance(part, str) and part for part in self.heading_path
        ):
            raise ValueError("Citation heading_path must contain non-empty strings.")
        page_start = _positive_optional_integer(self.page_start, "page_start")
        page_end = _positive_optional_integer(self.page_end, "page_end")
        if (page_start is None) != (page_end is None):
            raise ValueError("Citation page range must be fully present or absent.")
        if page_start is not None and page_end is not None and page_end < page_start:
            raise ValueError("Citation page range must be ordered.")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.start_line, self.end_line)
        ):
            raise TypeError("Citation line ranges must be integers.")
        if not 1 <= self.start_line <= self.end_line:
            raise ValueError("Citation line ranges must be positive and ordered.")

    def format(self) -> str:
        """Format only source fields that are actually known."""
        label = self.title or self.source_name
        locations: list[str] = []
        if self.page_start is not None and self.page_end is not None:
            if self.page_start == self.page_end:
                locations.append(f"p. {self.page_start}")
            else:
                locations.append(f"pp. {self.page_start}–{self.page_end}")
        elif self.start_line == self.end_line:
            locations.append(f"line {self.start_line}")
        else:
            locations.append(f"lines {self.start_line}–{self.end_line}")
        if self.heading_path:
            locations.append("§ " + " › ".join(self.heading_path))
        return f"[{label}, {', '.join(locations)}]"

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["heading_path"] = list(self.heading_path)
        state["display"] = self.format()
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Citation:
        expected = set(cls.__dataclass_fields__) | {"display"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Citation state keys are malformed.")
        values = dict(state)
        display = values.pop("display")
        path = values["heading_path"]
        if not isinstance(path, list):
            raise ValueError("Serialized citation heading_path must be a list.")
        values["heading_path"] = tuple(path)
        citation = cls(**values)
        if display != citation.format():
            raise ValueError("Serialized citation display is inconsistent.")
        return citation
