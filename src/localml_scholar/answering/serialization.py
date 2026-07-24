"""Versioned atomic JSON persistence for complete grounded answers."""

from __future__ import annotations

import json
from pathlib import Path

from localml_scholar._version import __version__
from localml_scholar.answering.models import GroundedAnswer
from localml_scholar.answering.validation import (
    AnswerAcceptanceConfig,
    validate_answer_text,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.serialization import atomic_write_text

ANSWER_FORMAT_VERSION = 1


def answer_artifact_state(answer: GroundedAnswer) -> dict:
    """Return the exact versioned, deterministic answer artifact state."""
    if not isinstance(answer, GroundedAnswer):
        raise TypeError("answer must be a GroundedAnswer.")
    return {
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "package_version": __version__,
        "artifact_type": "grounded_answer",
        "answer": answer.to_dict(),
    }


def save_grounded_answer(
    path: str | Path,
    answer: GroundedAnswer,
) -> Path:
    """Atomically save one non-pickle grounded-answer JSON artifact."""
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("Grounded answer path must end with .json.")
    payload = json.dumps(
        answer_artifact_state(answer),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return atomic_write_text(destination, payload + "\n")


def load_grounded_answer(
    path: str | Path,
    *,
    index: RetrievalIndex | None = None,
) -> GroundedAnswer:
    """Transactionally load and optionally revalidate against an exact index."""
    source = Path(path)
    try:
        state = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Grounded answer artifact does not exist: {source}"
        ) from None
    except UnicodeDecodeError as error:
        raise ValueError("Grounded answer artifact is not valid UTF-8.") from error
    except json.JSONDecodeError as error:
        raise ValueError("Grounded answer artifact is not valid JSON.") from error
    expected = {
        "answer_format_version",
        "package_version",
        "artifact_type",
        "answer",
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("Grounded answer artifact keys are malformed.")
    if state["answer_format_version"] != ANSWER_FORMAT_VERSION:
        raise ValueError("Unsupported grounded answer format version.")
    if state["package_version"] != __version__:
        raise ValueError("Grounded answer package version is incompatible.")
    if state["artifact_type"] != "grounded_answer":
        raise ValueError("Grounded answer artifact type is incompatible.")
    answer = GroundedAnswer.from_dict(state["answer"])
    if index is not None:
        if not isinstance(index, RetrievalIndex):
            raise TypeError("index must be None or a RetrievalIndex.")
        serialized_config = answer.metadata.get("acceptance_config")
        config = (
            AnswerAcceptanceConfig()
            if serialized_config is None
            else AnswerAcceptanceConfig.from_dict(serialized_config)
        )
        claims, validation = validate_answer_text(
            index,
            answer.answer_text,
            answer.evidence,
            config=config,
            abstained=answer.abstained,
        )
        if claims != answer.claims or validation != answer.validation:
            raise ValueError(
                "Grounded answer does not revalidate against the supplied index."
            )
    return answer
