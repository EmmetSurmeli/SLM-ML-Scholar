"""Atomic, hash-checked JSON persistence for evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from localml_scholar._version import __version__
from localml_scholar.evaluation.schemas import (
    EVALUATION_FORMAT_VERSION,
    Benchmark,
    CorrectionExample,
    EvaluationRun,
    HumanReviewRecord,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.retrieval.documents import canonical_json
from localml_scholar.serialization import atomic_write_text

_ARTIFACT_TYPES = {
    "benchmark",
    "evaluation_run",
    "review_queue",
    "human_reviews",
    "correction_dataset",
    "comparison_report",
}


def _envelope(artifact_type: str, payload: object) -> dict[str, Any]:
    if artifact_type not in _ARTIFACT_TYPES:
        raise ValueError("Unknown evaluation artifact type.")
    canonical = canonical_json(payload)
    return {
        "artifact_type": artifact_type,
        "format_version": EVALUATION_FORMAT_VERSION,
        "package_version": __version__,
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _save(path: str | Path, artifact_type: str, payload: object) -> Path:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("Evaluation artifact paths must end with .json.")
    return atomic_write_text(
        destination,
        json.dumps(
            _envelope(artifact_type, payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _load(path: str | Path, artifact_type: str) -> object:
    source = Path(path)
    try:
        state = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Evaluation artifact does not exist: {source}"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation artifact is not valid UTF-8 JSON.") from error
    expected = {
        "artifact_type",
        "format_version",
        "package_version",
        "payload",
        "payload_sha256",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("Evaluation artifact envelope is malformed.")
    if state["artifact_type"] != artifact_type:
        raise ValueError("Evaluation artifact type does not match.")
    if state["format_version"] != EVALUATION_FORMAT_VERSION:
        raise ValueError("Evaluation artifact format version is unsupported.")
    if (
        not isinstance(state["package_version"], str)
        or not state["package_version"].strip()
    ):
        raise ValueError("Evaluation artifact package version is malformed.")
    try:
        digest = hashlib.sha256(
            canonical_json(state["payload"]).encode("utf-8")
        ).hexdigest()
    except ValueError as error:
        raise ValueError(
            "Evaluation artifact payload is not canonical JSON."
        ) from error
    if digest != state["payload_sha256"]:
        raise ValueError("Evaluation artifact payload hash is inconsistent.")
    return state["payload"]


def save_benchmark(benchmark: Benchmark, path: str | Path) -> Path:
    """Atomically save proposed, reviewed, or approved benchmark state."""
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must be Benchmark.")
    return _save(path, "benchmark", benchmark.to_dict())


def load_benchmark(
    path: str | Path,
    *,
    index: RetrievalIndex | None = None,
) -> Benchmark:
    """Load a benchmark and optionally reject stale source/index identities."""
    payload = _load(path, "benchmark")
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark artifact payload must be an object.")
    benchmark = Benchmark.from_dict(payload)
    if index is not None:
        benchmark.validate_against_index(index)
    return benchmark


def save_evaluation_run(run: EvaluationRun, path: str | Path) -> Path:
    if not isinstance(run, EvaluationRun):
        raise TypeError("run must be EvaluationRun.")
    return _save(path, "evaluation_run", run.to_dict())


def load_evaluation_run(path: str | Path) -> EvaluationRun:
    payload = _load(path, "evaluation_run")
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation-run payload must be an object.")
    return EvaluationRun.from_dict(payload)


def save_review_records(
    records: tuple[HumanReviewRecord, ...],
    path: str | Path,
    *,
    artifact_type: str = "review_queue",
) -> Path:
    if artifact_type not in {"review_queue", "human_reviews"}:
        raise ValueError("Review artifact_type must be review_queue or human_reviews.")
    if not isinstance(records, tuple) or not all(
        isinstance(item, HumanReviewRecord) for item in records
    ):
        raise TypeError("records must contain HumanReviewRecord objects.")
    identifiers = [item.review_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Review IDs must be unique.")
    return _save(path, artifact_type, [item.to_dict() for item in records])


def load_review_records(
    path: str | Path,
    *,
    artifact_type: str = "review_queue",
) -> tuple[HumanReviewRecord, ...]:
    if artifact_type not in {"review_queue", "human_reviews"}:
        raise ValueError("Review artifact_type must be review_queue or human_reviews.")
    payload = _load(path, artifact_type)
    if not isinstance(payload, list):
        raise ValueError("Review artifact payload must be a list.")
    records = tuple(HumanReviewRecord.from_dict(item) for item in payload)
    if len({item.review_id for item in records}) != len(records):
        raise ValueError("Review artifact contains duplicate review IDs.")
    return records


def save_corrections(
    examples: tuple[CorrectionExample, ...],
    path: str | Path,
) -> Path:
    if not isinstance(examples, tuple) or not all(
        isinstance(item, CorrectionExample) for item in examples
    ):
        raise TypeError("examples must contain CorrectionExample objects.")
    return _save(path, "correction_dataset", [item.to_dict() for item in examples])


def load_corrections(path: str | Path) -> tuple[CorrectionExample, ...]:
    payload = _load(path, "correction_dataset")
    if not isinstance(payload, list):
        raise ValueError("Correction dataset payload must be a list.")
    return tuple(CorrectionExample.from_dict(item) for item in payload)


def save_comparison_report(report: dict[str, Any], path: str | Path) -> Path:
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")
    canonical_json(report)
    return _save(path, "comparison_report", report)


def load_comparison_report(path: str | Path) -> dict[str, Any]:
    payload = _load(path, "comparison_report")
    if not isinstance(payload, dict):
        raise ValueError("Comparison report payload must be an object.")
    return payload
