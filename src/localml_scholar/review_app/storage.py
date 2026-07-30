"""Repository-local, atomic persistence for the review application."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Atomically replace ``path`` with exact bytes on the same filesystem."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, value: object) -> Path:
    """Atomically write deterministic, finite JSON."""
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Review application state must be valid finite JSON."
        ) from error
    return atomic_write_bytes(path, payload + b"\n")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    """Load a persisted list of JSON objects, or return an empty new collection."""
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"State file is not valid UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"State file is not valid JSON: {path}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"State file must contain a JSON list of objects: {path}")
    return value
