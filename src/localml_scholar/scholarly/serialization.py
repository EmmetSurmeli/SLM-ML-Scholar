"""Versioned atomic JSON persistence for scholarly artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localml_scholar._version import __version__
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.retrieval.documents import canonical_json
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.models import PaperAnalysis, SourceCitation
from localml_scholar.scholarly.source import validate_source_citation
from localml_scholar.serialization import atomic_write_text

SCHOLARLY_ARTIFACT_FORMAT_VERSION = 1
_ARTIFACT_TYPES = {
    "paper_analysis",
    "paper_analysis_section",
    "notation_glossary",
    "equation_analysis",
    "methodology",
    "experiments",
    "structured_summary",
    "reproduction_checklist",
    "paper_comparison",
    "research_gap_worksheet",
}


def _payload(value: object) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_payload(item) for item in value]
    if isinstance(value, list):
        return [_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _payload(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ScholarlyArtifact:
    """One self-identifying artifact tied to an immutable retrieval index."""

    artifact_type: str
    index_sha256: str
    document_hashes: dict[str, str]
    analysis_configuration: dict[str, Any]
    package_version: str
    payload: dict[str, Any] | list[Any]
    artifact_format_version: int = SCHOLARLY_ARTIFACT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.artifact_type not in _ARTIFACT_TYPES:
            raise ValueError("Unsupported scholarly artifact type.")
        if self.artifact_format_version != SCHOLARLY_ARTIFACT_FORMAT_VERSION:
            raise ValueError("Unsupported scholarly artifact format version.")
        if not isinstance(self.index_sha256, str) or len(self.index_sha256) != 64:
            raise ValueError("index_sha256 must be a SHA-256 digest.")
        if not self.document_hashes or any(
            not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
            for key, value in self.document_hashes.items()
        ):
            raise ValueError("document_hashes must map IDs to SHA-256 digests.")
        if not isinstance(self.package_version, str) or not self.package_version:
            raise ValueError("Artifact package_version must be a non-empty string.")
        canonical_json(self.analysis_configuration)
        canonical_json(self.payload)

    def _identity_state(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "index_sha256": self.index_sha256,
            "document_hashes": self.document_hashes,
            "analysis_configuration": self.analysis_configuration,
            "package_version": self.package_version,
            "payload": self.payload,
            "artifact_format_version": self.artifact_format_version,
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self._identity_state()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_state(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ScholarlyArtifact:
        expected = set(cls.__dataclass_fields__) | {"artifact_sha256"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Scholarly artifact state is malformed.")
        values = dict(state)
        digest = values.pop("artifact_sha256")
        artifact = cls(**values)
        if digest != artifact.artifact_sha256:
            raise ValueError("Scholarly artifact hash is inconsistent.")
        return artifact


def create_artifact(
    value: object,
    *,
    artifact_type: str,
    index: RetrievalIndex,
    document_ids: tuple[str, ...],
    config: ScholarlyConfig,
) -> ScholarlyArtifact:
    """Create an artifact only for documents present in the supplied index."""
    documents = {document.document_id: document for document in index.documents}
    if not document_ids or any(item not in documents for item in document_ids):
        raise ValueError("Every artifact document must exist in the index.")
    payload = _payload(value)
    if not isinstance(payload, (dict, list)):
        raise TypeError(
            "Scholarly artifact payload must serialize to an object or list."
        )
    artifact = ScholarlyArtifact(
        artifact_type=artifact_type,
        index_sha256=index.index_sha256,
        document_hashes={
            item: documents[item].content_sha256 for item in sorted(document_ids)
        },
        analysis_configuration=config.to_dict(),
        package_version=__version__,
        payload=payload,
    )
    validate_artifact(artifact, index)
    return artifact


def save_artifact(artifact: ScholarlyArtifact, path: str | Path) -> Path:
    """Atomically persist canonical artifact JSON."""
    if not isinstance(artifact, ScholarlyArtifact):
        raise TypeError("artifact must be a ScholarlyArtifact.")
    return atomic_write_text(
        path,
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def load_artifact(
    path: str | Path,
    *,
    index: RetrievalIndex,
) -> ScholarlyArtifact:
    """Parse, reconstruct, hash-check, and source-check an artifact."""
    source = Path(path)
    try:
        state = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scholarly artifact does not exist: {source}"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Scholarly artifact is not valid UTF-8 JSON.") from error
    artifact = ScholarlyArtifact.from_dict(state)
    validate_artifact(artifact, index)
    return artifact


def validate_artifact(
    artifact: ScholarlyArtifact,
    index: RetrievalIndex,
) -> None:
    """Validate index identity, document hashes, and embedded citations."""
    if artifact.index_sha256 != index.index_sha256:
        raise ValueError("Scholarly artifact index hash does not match.")
    documents = {document.document_id: document for document in index.documents}
    for document_id, digest in artifact.document_hashes.items():
        document = documents.get(document_id)
        if document is None:
            raise ValueError("Scholarly artifact references a missing source document.")
        if document.content_sha256 != digest:
            raise ValueError("Scholarly artifact source hash does not match.")
    expected_config = ScholarlyConfig.from_dict(artifact.analysis_configuration)
    if expected_config.to_dict() != artifact.analysis_configuration:
        raise ValueError("Scholarly artifact configuration is not canonical.")
    for citation_state in _citation_states(artifact.payload):
        citation = SourceCitation.from_dict(citation_state)
        document = documents.get(citation.document_id)
        if document is None:
            raise ValueError("Artifact citation references a missing document.")
        validate_source_citation(document, citation)


def _citation_states(value: object):
    if isinstance(value, dict):
        required = {
            "document_id",
            "section_id",
            "start_character",
            "end_character",
            "source_text_sha256",
            "display",
        }
        if required <= set(value):
            yield value
        for item in value.values():
            yield from _citation_states(item)
    elif isinstance(value, list):
        for item in value:
            yield from _citation_states(item)


def save_analysis(
    analysis: PaperAnalysis,
    path: str | Path,
    *,
    index: RetrievalIndex,
    config: ScholarlyConfig,
) -> Path:
    """Save one full paper analysis."""
    artifact = create_artifact(
        analysis,
        artifact_type="paper_analysis",
        index=index,
        document_ids=(analysis.paper.document_id,),
        config=config,
    )
    return save_artifact(artifact, path)


def load_analysis(
    path: str | Path,
    *,
    index: RetrievalIndex,
) -> PaperAnalysis:
    """Load a validated full paper analysis."""
    artifact = load_artifact(path, index=index)
    if artifact.artifact_type != "paper_analysis" or not isinstance(
        artifact.payload, dict
    ):
        raise ValueError("Artifact does not contain a paper analysis.")
    return PaperAnalysis.from_dict(artifact.payload)
