"""Validated immutable models for citation-preserving scholarly artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from localml_scholar.retrieval.documents import canonical_json, stable_identifier

_CONFIDENCE = {"high", "medium", "low"}
_VALIDATION = {"validated", "ambiguous", "unresolved", "conflicting", "not_found"}
_ROLES = {
    "title",
    "abstract",
    "introduction",
    "background",
    "related_work",
    "methodology",
    "theory",
    "algorithm",
    "data",
    "experiments",
    "results",
    "ablation",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    "appendix",
    "unknown",
}


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _optional(value: object, name: str) -> str | None:
    if value is not None:
        return _nonempty(value, name)
    return None


def _tuple_of(value: object, expected: type, name: str) -> tuple:
    if not isinstance(value, tuple) or not all(
        isinstance(item, expected) for item in value
    ):
        raise TypeError(f"{name} must be a tuple of {expected.__name__} objects.")
    return value


def _serialize(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SourceCitation:
    """An exact character and line range in one canonical source document."""

    document_id: str
    section_id: str
    source_name: str
    title: str | None
    heading_path: tuple[str, ...]
    start_character: int
    end_character: int
    start_line: int
    end_line: int
    page_start: int | None
    page_end: int | None
    source_text_sha256: str

    def __post_init__(self) -> None:
        for name in ("document_id", "section_id", "source_name"):
            _nonempty(getattr(self, name), name)
        _optional(self.title, "title")
        if not isinstance(self.heading_path, tuple) or not all(
            isinstance(part, str) and part for part in self.heading_path
        ):
            raise ValueError("heading_path must contain non-empty strings.")
        if not 0 <= self.start_character < self.end_character:
            raise ValueError("Citation character range must be non-empty and ordered.")
        if not 1 <= self.start_line <= self.end_line:
            raise ValueError("Citation line range must be positive and ordered.")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("Citation page range must be fully present or absent.")
        if self.page_start is not None and (
            self.page_start <= 0 or self.page_end < self.page_start
        ):
            raise ValueError("Citation page range must be positive and ordered.")
        if (
            not isinstance(self.source_text_sha256, str)
            or len(self.source_text_sha256) != 64
        ):
            raise ValueError("source_text_sha256 must be a SHA-256 digest.")

    def format(self) -> str:
        """Format only source locations that are actually available."""
        label = self.title or self.source_name
        if self.page_start is not None:
            location = (
                f"p. {self.page_start}"
                if self.page_start == self.page_end
                else f"pp. {self.page_start}–{self.page_end}"
            )
        else:
            location = (
                f"line {self.start_line}"
                if self.start_line == self.end_line
                else f"lines {self.start_line}–{self.end_line}"
            )
        heading = (
            "" if not self.heading_path else ", § " + " › ".join(self.heading_path)
        )
        return f"[{label}, {location}{heading}]"

    def to_dict(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["heading_path"] = list(self.heading_path)
        state["display"] = self.format()
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> SourceCitation:
        expected = set(cls.__dataclass_fields__) | {"display"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Source citation state is malformed.")
        values = dict(state)
        display = values.pop("display")
        path = values["heading_path"]
        if not isinstance(path, list):
            raise ValueError("Serialized heading_path must be a list.")
        values["heading_path"] = tuple(path)
        citation = cls(**values)
        if display != citation.format():
            raise ValueError("Serialized citation display is inconsistent.")
        return citation


@dataclass(frozen=True)
class ScholarlyEvidence:
    """One structured value supported by an exact source slice."""

    evidence_id: str
    category: str
    value: Any
    normalized_value: Any
    citation: SourceCitation
    source_text: str
    extraction_method: str
    confidence: str
    validation: str = "validated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "category", "source_text", "extraction_method"):
            _nonempty(getattr(self, name), name)
        if self.confidence not in _CONFIDENCE:
            raise ValueError(f"confidence must be one of {sorted(_CONFIDENCE)}.")
        if self.validation not in _VALIDATION:
            raise ValueError(f"validation must be one of {sorted(_VALIDATION)}.")
        canonical_json(_serialize(self.value))
        canonical_json(_serialize(self.normalized_value))
        canonical_json(self.metadata)
        digest = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if digest != self.citation.source_text_sha256:
            raise ValueError("Evidence source_text does not match its citation hash.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "category": self.category,
            "value": _serialize(self.value),
            "normalized_value": _serialize(self.normalized_value),
            "citation": self.citation.to_dict(),
            "source_text": self.source_text,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "validation": self.validation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ScholarlyEvidence:
        if not isinstance(state, Mapping) or set(state) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Scholarly evidence state is malformed.")
        values = dict(state)
        values["citation"] = SourceCitation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class PaperSection:
    """One source section with deterministic scholarly-role labels."""

    section_id: str
    heading: str | None
    roles: tuple[str, ...]
    reasons: tuple[str, ...]
    confidence: str
    citation: SourceCitation

    def __post_init__(self) -> None:
        _nonempty(self.section_id, "section_id")
        _optional(self.heading, "heading")
        if not self.roles or any(role not in _ROLES for role in self.roles):
            raise ValueError("roles must contain recognized scholarly roles.")
        if not isinstance(self.reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.reasons
        ):
            raise ValueError("reasons must contain non-empty strings.")
        if self.confidence not in _CONFIDENCE:
            raise ValueError("Invalid section confidence.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "roles": list(self.roles),
            "reasons": list(self.reasons),
            "confidence": self.confidence,
            "citation": self.citation.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> PaperSection:
        values = dict(state)
        values["roles"] = tuple(values["roles"])
        values["reasons"] = tuple(values["reasons"])
        values["citation"] = SourceCitation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class EquationBlock:
    """One equation-like block detected from extracted source text."""

    equation_id: str
    document_id: str
    section_id: str
    raw_text: str
    normalized_text: str
    equation_number: str | None
    citation: SourceCitation
    detection_method: str

    def __post_init__(self) -> None:
        for name in (
            "equation_id",
            "document_id",
            "section_id",
            "raw_text",
            "normalized_text",
            "detection_method",
        ):
            _nonempty(getattr(self, name), name)
        _optional(self.equation_number, "equation_number")
        if self.document_id != self.citation.document_id:
            raise ValueError("Equation document linkage is inconsistent.")
        if self.section_id != self.citation.section_id:
            raise ValueError("Equation section linkage is inconsistent.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in vars(self).items() if key != "citation"},
            "citation": self.citation.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EquationBlock:
        values = dict(state)
        values["citation"] = SourceCitation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class DefinitionCandidate:
    """A transparent textual candidate for one symbol definition."""

    symbol: str
    defining_text: str
    citation: SourceCitation
    distance_characters: int
    pattern: str
    confidence: str
    section_role: str

    def __post_init__(self) -> None:
        for name in ("symbol", "defining_text", "pattern", "section_role"):
            _nonempty(getattr(self, name), name)
        if self.distance_characters < 0:
            raise ValueError("distance_characters must be non-negative.")
        if self.confidence not in _CONFIDENCE:
            raise ValueError("Invalid definition confidence.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in vars(self).items() if key != "citation"},
            "citation": self.citation.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> DefinitionCandidate:
        values = dict(state)
        values["citation"] = SourceCitation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class NotationEntry:
    """A symbol, all cited occurrences, and zero or more definition candidates."""

    symbol_id: str
    raw_symbol: str
    normalized_symbol: str
    symbol_type: str
    occurrences: tuple[SourceCitation, ...]
    definition_candidates: tuple[DefinitionCandidate, ...]
    selected_definition: DefinitionCandidate | None
    ambiguity: str | None

    def __post_init__(self) -> None:
        for name in ("symbol_id", "raw_symbol", "normalized_symbol", "symbol_type"):
            _nonempty(getattr(self, name), name)
        _tuple_of(self.occurrences, SourceCitation, "occurrences")
        if not self.occurrences:
            raise ValueError("Notation entries require at least one occurrence.")
        _tuple_of(
            self.definition_candidates,
            DefinitionCandidate,
            "definition_candidates",
        )
        if self.selected_definition is not None and (
            self.selected_definition not in self.definition_candidates
        ):
            raise ValueError("Selected definition must be a retained candidate.")
        _optional(self.ambiguity, "ambiguity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "raw_symbol": self.raw_symbol,
            "normalized_symbol": self.normalized_symbol,
            "symbol_type": self.symbol_type,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "definition_candidates": [
                item.to_dict() for item in self.definition_candidates
            ],
            "selected_definition": (
                None
                if self.selected_definition is None
                else self.selected_definition.to_dict()
            ),
            "ambiguity": self.ambiguity,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> NotationEntry:
        values = dict(state)
        values["occurrences"] = tuple(
            SourceCitation.from_dict(item) for item in values["occurrences"]
        )
        values["definition_candidates"] = tuple(
            DefinitionCandidate.from_dict(item)
            for item in values["definition_candidates"]
        )
        selected = values["selected_definition"]
        if selected is not None:
            reconstructed = DefinitionCandidate.from_dict(selected)
            values["selected_definition"] = next(
                (
                    item
                    for item in values["definition_candidates"]
                    if item == reconstructed
                ),
                reconstructed,
            )
        return cls(**values)


@dataclass(frozen=True)
class EquationAnalysis:
    """Symbols, cited definitions, and unresolved items for one equation."""

    equation_id: str
    symbols: tuple[str, ...]
    definitions: tuple[DefinitionCandidate, ...]
    unresolved_symbols: tuple[str, ...]
    defined_later_symbols: tuple[str, ...]
    related_text: tuple[ScholarlyEvidence, ...]

    def __post_init__(self) -> None:
        _nonempty(self.equation_id, "equation_id")
        _tuple_of(self.definitions, DefinitionCandidate, "definitions")
        _tuple_of(self.related_text, ScholarlyEvidence, "related_text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "symbols": list(self.symbols),
            "definitions": [item.to_dict() for item in self.definitions],
            "unresolved_symbols": list(self.unresolved_symbols),
            "defined_later_symbols": list(self.defined_later_symbols),
            "related_text": [item.to_dict() for item in self.related_text],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EquationAnalysis:
        values = dict(state)
        values["symbols"] = tuple(values["symbols"])
        values["definitions"] = tuple(
            DefinitionCandidate.from_dict(item) for item in values["definitions"]
        )
        values["unresolved_symbols"] = tuple(values["unresolved_symbols"])
        values["defined_later_symbols"] = tuple(values["defined_later_symbols"])
        values["related_text"] = tuple(
            ScholarlyEvidence.from_dict(item) for item in values["related_text"]
        )
        return cls(**values)


@dataclass(frozen=True)
class ExtractedTable:
    """A conservatively parsed table that preserves its complete raw source."""

    table_id: str
    caption: str | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    citation: SourceCitation
    raw_text: str
    parsing_method: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.table_id, "table_id")
        _optional(self.caption, "caption")
        if not self.headers or not all(isinstance(item, str) for item in self.headers):
            raise ValueError("Table headers must be a non-empty tuple of strings.")
        if not isinstance(self.rows, tuple) or not all(
            isinstance(row, tuple) and all(isinstance(cell, str) for cell in row)
            for row in self.rows
        ):
            raise TypeError("Table rows must be tuples of string tuples.")
        if any(len(row) != len(self.headers) for row in self.rows):
            raise ValueError("Every table row must match the header width.")
        _nonempty(self.raw_text, "raw_text")
        _nonempty(self.parsing_method, "parsing_method")

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "caption": self.caption,
            "headers": list(self.headers),
            "rows": [list(row) for row in self.rows],
            "citation": self.citation.to_dict(),
            "raw_text": self.raw_text,
            "parsing_method": self.parsing_method,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ExtractedTable:
        values = dict(state)
        values["headers"] = tuple(values["headers"])
        values["rows"] = tuple(tuple(row) for row in values["rows"])
        values["warnings"] = tuple(values["warnings"])
        values["citation"] = SourceCitation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class Procedure:
    """An explicitly enumerated source procedure with cited ordered steps."""

    procedure_id: str
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    steps: tuple[ScholarlyEvidence, ...]
    citation: SourceCitation
    related_equation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.procedure_id, "procedure_id")
        _nonempty(self.name, "name")
        _tuple_of(self.steps, ScholarlyEvidence, "steps")
        if not self.steps:
            raise ValueError("Procedure requires at least one explicit step.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "steps": [item.to_dict() for item in self.steps],
            "citation": self.citation.to_dict(),
            "related_equation_ids": list(self.related_equation_ids),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Procedure:
        values = dict(state)
        values["inputs"] = tuple(values["inputs"])
        values["outputs"] = tuple(values["outputs"])
        values["steps"] = tuple(
            ScholarlyEvidence.from_dict(item) for item in values["steps"]
        )
        values["citation"] = SourceCitation.from_dict(values["citation"])
        values["related_equation_ids"] = tuple(values["related_equation_ids"])
        return cls(**values)


@dataclass(frozen=True)
class ReferenceEntry:
    """A conservative reference entry with optional parsed fields."""

    reference_id: str
    raw_text: str
    authors: tuple[str, ...]
    title: str | None
    year: int | None
    venue: str | None
    identifier: str | None
    citation: SourceCitation
    parse_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, "reference_id")
        _nonempty(self.raw_text, "raw_text")
        _optional(self.title, "title")
        _optional(self.venue, "venue")
        _optional(self.identifier, "identifier")
        if self.year is not None and not 1000 <= self.year <= 9999:
            raise ValueError("Reference year must contain four digits.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in vars(self).items()
                if key not in {"citation", "authors", "parse_warnings"}
            },
            "authors": list(self.authors),
            "citation": self.citation.to_dict(),
            "parse_warnings": list(self.parse_warnings),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ReferenceEntry:
        values = dict(state)
        values["authors"] = tuple(values["authors"])
        values["citation"] = SourceCitation.from_dict(values["citation"])
        values["parse_warnings"] = tuple(values["parse_warnings"])
        return cls(**values)


@dataclass(frozen=True)
class ExperimentRecord:
    """A source-scoped experiment composed only from cited fields."""

    experiment_id: str
    name: str
    purpose: ScholarlyEvidence | None
    datasets: tuple[ScholarlyEvidence, ...]
    methods: tuple[ScholarlyEvidence, ...]
    baselines: tuple[ScholarlyEvidence, ...]
    metrics: tuple[ScholarlyEvidence, ...]
    hyperparameters: tuple[ScholarlyEvidence, ...]
    results: tuple[ScholarlyEvidence, ...]
    ablations: tuple[ScholarlyEvidence, ...]
    citation: SourceCitation

    def __post_init__(self) -> None:
        _nonempty(self.experiment_id, "experiment_id")
        _nonempty(self.name, "name")
        for name in (
            "datasets",
            "methods",
            "baselines",
            "metrics",
            "hyperparameters",
            "results",
            "ablations",
        ):
            _tuple_of(getattr(self, name), ScholarlyEvidence, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "purpose": None if self.purpose is None else self.purpose.to_dict(),
            **{
                name: [item.to_dict() for item in getattr(self, name)]
                for name in (
                    "datasets",
                    "methods",
                    "baselines",
                    "metrics",
                    "hyperparameters",
                    "results",
                    "ablations",
                )
            },
            "citation": self.citation.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ExperimentRecord:
        values = dict(state)
        purpose = values["purpose"]
        values["purpose"] = (
            None if purpose is None else ScholarlyEvidence.from_dict(purpose)
        )
        for name in (
            "datasets",
            "methods",
            "baselines",
            "metrics",
            "hyperparameters",
            "results",
            "ablations",
        ):
            values[name] = tuple(
                ScholarlyEvidence.from_dict(item) for item in values[name]
            )
        values["citation"] = SourceCitation.from_dict(values["citation"])
        return cls(**values)


@dataclass(frozen=True)
class Paper:
    """Canonical source-linked paper identity and ordered scholarly sections."""

    paper_id: str
    document_id: str
    title: ScholarlyEvidence | None
    authors: ScholarlyEvidence | None
    year: ScholarlyEvidence | None
    venue: ScholarlyEvidence | None
    abstract: ScholarlyEvidence | None
    keywords: ScholarlyEvidence | None
    identifier: ScholarlyEvidence | None
    sections: tuple[PaperSection, ...]
    references: tuple[ReferenceEntry, ...]
    source_hash: str
    analysis_config_sha256: str
    analysis_version: int = 1

    def __post_init__(self) -> None:
        _nonempty(self.paper_id, "paper_id")
        _nonempty(self.document_id, "document_id")
        _tuple_of(self.sections, PaperSection, "sections")
        _tuple_of(self.references, ReferenceEntry, "references")
        for name in ("source_hash", "analysis_config_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest.")
        if self.analysis_version != 1:
            raise ValueError("Unsupported paper analysis version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            **{
                name: (
                    None
                    if getattr(self, name) is None
                    else getattr(self, name).to_dict()
                )
                for name in (
                    "title",
                    "authors",
                    "year",
                    "venue",
                    "abstract",
                    "keywords",
                    "identifier",
                )
            },
            "sections": [item.to_dict() for item in self.sections],
            "references": [item.to_dict() for item in self.references],
            "source_hash": self.source_hash,
            "analysis_config_sha256": self.analysis_config_sha256,
            "analysis_version": self.analysis_version,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Paper:
        values = dict(state)
        for name in (
            "title",
            "authors",
            "year",
            "venue",
            "abstract",
            "keywords",
            "identifier",
        ):
            value = values[name]
            values[name] = None if value is None else ScholarlyEvidence.from_dict(value)
        values["sections"] = tuple(
            PaperSection.from_dict(item) for item in values["sections"]
        )
        values["references"] = tuple(
            ReferenceEntry.from_dict(item) for item in values["references"]
        )
        return cls(**values)


@dataclass(frozen=True)
class PaperAnalysis:
    """Complete deterministic extraction for one canonical paper."""

    paper: Paper
    equations: tuple[EquationBlock, ...]
    equation_analyses: tuple[EquationAnalysis, ...]
    notation: tuple[NotationEntry, ...]
    assumptions: tuple[ScholarlyEvidence, ...]
    claims: tuple[ScholarlyEvidence, ...]
    methodology: tuple[ScholarlyEvidence, ...]
    procedures: tuple[Procedure, ...]
    datasets: tuple[ScholarlyEvidence, ...]
    metrics: tuple[ScholarlyEvidence, ...]
    baselines: tuple[ScholarlyEvidence, ...]
    hyperparameters: tuple[ScholarlyEvidence, ...]
    experiments: tuple[ExperimentRecord, ...]
    results: tuple[ScholarlyEvidence, ...]
    tables: tuple[ExtractedTable, ...]
    ablations: tuple[ScholarlyEvidence, ...]
    limitations: tuple[ScholarlyEvidence, ...]
    in_text_references: tuple[ScholarlyEvidence, ...]
    unresolved_symbols: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _tuple_of(self.equations, EquationBlock, "equations")
        _tuple_of(self.equation_analyses, EquationAnalysis, "equation_analyses")
        _tuple_of(self.notation, NotationEntry, "notation")
        for name in (
            "assumptions",
            "claims",
            "methodology",
            "datasets",
            "metrics",
            "baselines",
            "hyperparameters",
            "results",
            "ablations",
            "limitations",
            "in_text_references",
        ):
            _tuple_of(getattr(self, name), ScholarlyEvidence, name)
        _tuple_of(self.procedures, Procedure, "procedures")
        _tuple_of(self.experiments, ExperimentRecord, "experiments")
        _tuple_of(self.tables, ExtractedTable, "tables")

    @property
    def analysis_id(self) -> str:
        return stable_identifier(
            "analysis",
            self.paper.paper_id,
            self.paper.analysis_config_sha256,
            self.paper.source_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "paper": self.paper.to_dict(),
            "equations": [item.to_dict() for item in self.equations],
            "equation_analyses": [item.to_dict() for item in self.equation_analyses],
            "notation": [item.to_dict() for item in self.notation],
            **{
                name: [item.to_dict() for item in getattr(self, name)]
                for name in (
                    "assumptions",
                    "claims",
                    "methodology",
                    "datasets",
                    "metrics",
                    "baselines",
                    "hyperparameters",
                    "results",
                    "ablations",
                    "limitations",
                    "in_text_references",
                )
            },
            "procedures": [item.to_dict() for item in self.procedures],
            "experiments": [item.to_dict() for item in self.experiments],
            "tables": [item.to_dict() for item in self.tables],
            "unresolved_symbols": list(self.unresolved_symbols),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> PaperAnalysis:
        expected = set(cls.__dataclass_fields__) | {"analysis_id"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Paper analysis state is malformed.")
        values = dict(state)
        analysis_id = values.pop("analysis_id")
        values["paper"] = Paper.from_dict(values["paper"])
        values["equations"] = tuple(
            EquationBlock.from_dict(item) for item in values["equations"]
        )
        values["equation_analyses"] = tuple(
            EquationAnalysis.from_dict(item) for item in values["equation_analyses"]
        )
        values["notation"] = tuple(
            NotationEntry.from_dict(item) for item in values["notation"]
        )
        for name in (
            "assumptions",
            "claims",
            "methodology",
            "datasets",
            "metrics",
            "baselines",
            "hyperparameters",
            "results",
            "ablations",
            "limitations",
            "in_text_references",
        ):
            values[name] = tuple(
                ScholarlyEvidence.from_dict(item) for item in values[name]
            )
        values["procedures"] = tuple(
            Procedure.from_dict(item) for item in values["procedures"]
        )
        values["experiments"] = tuple(
            ExperimentRecord.from_dict(item) for item in values["experiments"]
        )
        values["tables"] = tuple(
            ExtractedTable.from_dict(item) for item in values["tables"]
        )
        values["unresolved_symbols"] = tuple(values["unresolved_symbols"])
        values["warnings"] = tuple(values["warnings"])
        result = cls(**values)
        if result.analysis_id != analysis_id:
            raise ValueError("Serialized analysis identity is inconsistent.")
        return result


@dataclass(frozen=True)
class ChecklistItem:
    """One reproduction requirement and its cited extraction status."""

    section: str
    item: str
    status: str
    values: tuple[ScholarlyEvidence, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.section, "section")
        _nonempty(self.item, "item")
        if self.status not in {"found", "ambiguous", "conflicting", "not_found"}:
            raise ValueError("Invalid checklist status.")
        _tuple_of(self.values, ScholarlyEvidence, "values")
        if self.status == "not_found" and self.values:
            raise ValueError("not_found checklist items cannot contain values.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "item": self.item,
            "status": self.status,
            "values": [value.to_dict() for value in self.values],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RiskFlag:
    """A deterministic document-completeness observation."""

    risk_id: str
    reason: str
    absence_scope: str
    checklist_section: str
    severity: str
    citations: tuple[SourceCitation, ...] = ()

    def __post_init__(self) -> None:
        for name in ("risk_id", "reason", "absence_scope", "checklist_section"):
            _nonempty(getattr(self, name), name)
        if self.severity not in {"low", "medium", "high"}:
            raise ValueError("Invalid risk severity.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in vars(self).items() if key != "citations"},
            "citations": [item.to_dict() for item in self.citations],
        }


@dataclass(frozen=True)
class ReproductionChecklist:
    """Citation-preserving reproduction fields and transparent risk flags."""

    paper_id: str
    items: tuple[ChecklistItem, ...]
    risk_flags: tuple[RiskFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "items": [item.to_dict() for item in self.items],
            "risk_flags": [item.to_dict() for item in self.risk_flags],
        }


@dataclass(frozen=True)
class SummaryField:
    """One structured summary field containing only cited extracted values."""

    name: str
    status: str
    evidence: tuple[ScholarlyEvidence, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        if self.status not in {"found", "missing", "ambiguous", "conflicting"}:
            raise ValueError("Invalid summary-field status.")
        _tuple_of(self.evidence, ScholarlyEvidence, "evidence")
        if self.status == "missing" and self.evidence:
            raise ValueError("Missing summary fields cannot contain evidence.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": [item.to_dict() for item in self.evidence],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StructuredSummary:
    """A deterministic field summary plus extraction-completeness diagnostics."""

    paper_id: str
    fields: tuple[SummaryField, ...]
    completeness: dict[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.paper_id, "paper_id")
        _tuple_of(self.fields, SummaryField, "fields")
        canonical_json(self.completeness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "fields": [item.to_dict() for item in self.fields],
            "completeness": self.completeness,
        }


@dataclass(frozen=True)
class ComparisonDimension:
    """One cross-paper field with dual-sided cited evidence."""

    name: str
    values_by_paper: dict[str, tuple[ScholarlyEvidence, ...]]
    relationship: str
    comparable: bool
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        if self.relationship not in {"shared", "different", "missing", "incomparable"}:
            raise ValueError("Invalid comparison relationship.")
        if not isinstance(self.comparable, bool):
            raise TypeError("comparable must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values_by_paper": {
                paper_id: [item.to_dict() for item in values]
                for paper_id, values in sorted(self.values_by_paper.items())
            },
            "relationship": self.relationship,
            "comparable": self.comparable,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PaperComparison:
    """A structured comparison that never ranks incomparable results."""

    paper_ids: tuple[str, ...]
    dimensions: tuple[ComparisonDimension, ...]
    false_superiority_claim_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_ids": list(self.paper_ids),
            "dimensions": [item.to_dict() for item in self.dimensions],
            "false_superiority_claim_count": self.false_superiority_claim_count,
        }


@dataclass(frozen=True)
class ResearchGapCandidate:
    """A source-based planning candidate with an explicit novelty caution."""

    gap_id: str
    gap_type: str
    statement: str
    source_basis: str
    citations: tuple[SourceCitation, ...]
    system_inference: bool
    confidence: str
    cautions: tuple[str, ...]
    question_template: str | None

    def __post_init__(self) -> None:
        for name in ("gap_id", "gap_type", "statement", "source_basis"):
            _nonempty(getattr(self, name), name)
        if not self.citations:
            raise ValueError("Research-gap candidates require cited source basis.")
        if self.confidence not in _CONFIDENCE:
            raise ValueError("Invalid gap confidence.")
        if not any("novel" in caution.casefold() for caution in self.cautions):
            raise ValueError("Gap candidates must carry an explicit novelty caution.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in vars(self).items()
                if key not in {"citations", "cautions"}
            },
            "citations": [item.to_dict() for item in self.citations],
            "cautions": list(self.cautions),
        }


@dataclass(frozen=True)
class ScholarlySearchResult:
    """An equation-aware reranking wrapper around an unchanged retrieval result."""

    base_result: dict[str, Any]
    scholarly_score: float
    equation_signals: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.base_result, dict):
            raise TypeError("base_result must be serialized retrieval result state.")
        canonical_json(self.base_result)
        if (
            isinstance(self.scholarly_score, bool)
            or not isinstance(self.scholarly_score, (int, float))
            or not 0.0 <= float(self.scholarly_score) <= 1.0
        ):
            raise ValueError("scholarly_score must lie in [0, 1].")
        canonical_json(self.equation_signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_result": self.base_result,
            "scholarly_score": float(self.scholarly_score),
            "equation_signals": self.equation_signals,
        }
