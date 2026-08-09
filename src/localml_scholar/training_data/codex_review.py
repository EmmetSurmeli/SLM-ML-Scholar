"""Schema-constrained Codex review passes for autonomous corpus curation."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from localml_scholar._version import __version__
from localml_scholar.training_data.provenance import content_sha256

PASS_NAMES = (
    "answerer",
    "evidence_critic",
    "answer_critic",
    "citation_critic",
    "final_adjudicator",
)
DECISIONS = {"accept", "repair", "reject", "uncertain"}
SCORE_NAMES = (
    "answer_correctness",
    "evidence_relevance",
    "factual_support",
    "completeness",
    "citation_support",
    "citation_relevance",
    "instruction_following",
)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a sequence of non-empty strings.")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True)
class CodexReview:
    """One structured result produced by a focused Codex review pass."""

    decision: str
    confidence: float
    answer_correctness: float
    evidence_relevance: float
    factual_support: float
    completeness: float
    citation_support: float
    citation_relevance: float
    instruction_following: float
    unsupported_claims: tuple[str, ...] = ()
    missing_concepts: tuple[str, ...] = ()
    required_corrections: tuple[str, ...] = ()
    corrected_evidence_ids: tuple[str, ...] = ()
    corrected_target: dict[str, Any] | None = None
    corrected_answer: str | None = None
    abstention_required: bool = False
    uncertainty_reasons: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}.")
        for name in ("confidence", *SCORE_NAMES):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, float(value))
        for name in (
            "unsupported_claims",
            "missing_concepts",
            "required_corrections",
            "corrected_evidence_ids",
            "uncertainty_reasons",
        ):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _string_tuple(value, name))
            elif not isinstance(value, (tuple, list)):
                raise TypeError(f"{name} must be a sequence.")
            else:
                object.__setattr__(self, name, ())
        if self.corrected_target is not None and not isinstance(
            self.corrected_target, dict
        ):
            raise TypeError("corrected_target must be a dictionary or None.")
        if self.corrected_answer is not None and (
            not isinstance(self.corrected_answer, str)
            or not self.corrected_answer.strip()
        ):
            raise ValueError("corrected_answer must contain text or be None.")
        if not isinstance(self.abstention_required, bool):
            raise TypeError("abstention_required must be boolean.")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must contain non-whitespace text.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            **{name: getattr(self, name) for name in SCORE_NAMES},
            "unsupported_claims": list(self.unsupported_claims),
            "missing_concepts": list(self.missing_concepts),
            "required_corrections": list(self.required_corrections),
            "corrected_evidence_ids": list(self.corrected_evidence_ids),
            "corrected_target": self.corrected_target,
            "corrected_answer": self.corrected_answer,
            "abstention_required": self.abstention_required,
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CodexReview:
        if not isinstance(value, dict):
            raise TypeError("Codex review output must be a JSON object.")
        expected = {
            "decision",
            "confidence",
            *SCORE_NAMES,
            "unsupported_claims",
            "missing_concepts",
            "required_corrections",
            "corrected_evidence_ids",
            "corrected_target",
            "corrected_answer",
            "abstention_required",
            "uncertainty_reasons",
            "rationale",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(
                f"Malformed Codex review keys; missing={missing}, extra={extra}."
            )
        return cls(**value)


@dataclass(frozen=True)
class CodexReviewPass:
    """A review result plus immutable reviewer and input/output identities."""

    pass_name: str
    reviewer_system: str
    reviewer_version: str
    input_hash: str
    output_hash: str
    result: CodexReview

    def __post_init__(self) -> None:
        if self.pass_name not in PASS_NAMES:
            raise ValueError(f"pass_name must be one of {PASS_NAMES}.")
        for name in (
            "reviewer_system",
            "reviewer_version",
            "input_hash",
            "output_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must contain text.")
        if not isinstance(self.result, CodexReview):
            raise TypeError("result must be CodexReview.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_name": self.pass_name,
            "reviewer_system": self.reviewer_system,
            "reviewer_version": self.reviewer_version,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "result": self.result.to_dict(),
        }


class CodexReviewProvider(Protocol):
    """Minimal injectable interface used by the autonomous curator."""

    @property
    def identity(self) -> tuple[str, str]: ...

    def available(self) -> bool: ...

    def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview: ...


def codex_review_json_schema() -> dict[str, Any]:
    """Return the strict JSON schema passed to ``codex exec``."""
    fact = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "provenance", "citation_ids", "confidence"],
        "properties": {
            "text": {"type": "string"},
            "provenance": {
                "type": "string",
                "enum": [
                    "paper_explicit",
                    "mathematical_inference",
                    "external_background",
                    "external_knowledge",
                    "uncertain",
                ],
            },
            "citation_ids": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    target_properties = {
        name: {"type": "array", "items": fact}
        for name in (
            "facts",
            "equations",
            "derivation_steps",
            "assumptions",
            "qualifications",
            "limitations",
        )
    } | {
        "unresolved_items": {"type": "array", "items": {"type": "string"}},
        "prohibited_claims": {"type": "array", "items": {"type": "string"}},
    }
    properties: dict[str, Any] = {
        "decision": {"type": "string", "enum": sorted(DECISIONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        **{
            name: {"type": "number", "minimum": 0, "maximum": 1} for name in SCORE_NAMES
        },
        **{
            name: {"type": "array", "items": {"type": "string"}}
            for name in (
                "unsupported_claims",
                "missing_concepts",
                "required_corrections",
                "corrected_evidence_ids",
                "uncertainty_reasons",
            )
        },
        "corrected_target": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": list(target_properties),
            "properties": target_properties,
        },
        "corrected_answer": {"type": ["string", "null"]},
        "abstention_required": {"type": "boolean"},
        "rationale": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


_PASS_INSTRUCTIONS = {
    "answerer": (
        "Build the answer evidence-first. Select supported facts, construct the "
        "target, then write and cite the answer. Never invent missing facts."
    ),
    "evidence_critic": (
        "Judge only whether the raw evidence can answer the question and support "
        "the intended facts. Ignore prose quality and all prior confidence."
    ),
    "answer_critic": (
        "Judge correctness, relevance, completeness, unsupported claims, and "
        "instruction following. Ignore retrieval scores and prior decisions."
    ),
    "citation_critic": (
        "Verify every claim-to-source mapping. Do not infer or predict the final "
        "acceptance decision."
    ),
    "final_adjudicator": (
        "Review the raw evidence and focused critic results. Accept only when all "
        "required checks pass; otherwise repair, reject, or mark uncertain."
    ),
}


def blind_payload(pass_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the deliberately restricted view for one focused review pass."""
    if pass_name not in PASS_NAMES:
        raise ValueError(f"pass_name must be one of {PASS_NAMES}.")
    common = {
        "paper_metadata": payload.get("paper_metadata"),
        "question": payload.get("question"),
        "conversation_context": payload.get("conversation_context", []),
        "instruction_profile": payload.get("instruction_profile", {}),
        "source_passages": payload.get("source_passages", []),
    }
    if pass_name == "evidence_critic":
        return {**common, "structured_target": payload.get("structured_target", {})}
    if pass_name == "answer_critic":
        passages = [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "score",
                    "rank",
                    "retrieval_method",
                    "scoring_details",
                    "term_contributions",
                }
            }
            for item in payload.get("source_passages", [])
            if isinstance(item, dict)
        ]
        return {
            **{**common, "source_passages": passages},
            "answer": payload.get("answer"),
            "structured_target": payload.get("structured_target", {}),
            "known_failure_labels": payload.get("known_failure_labels", []),
        }
    if pass_name == "citation_critic":
        return {
            "paper_metadata": payload.get("paper_metadata"),
            "question": payload.get("question"),
            "answer": payload.get("answer"),
            "source_passages": payload.get("source_passages", []),
            "citation_mappings": payload.get("citation_mappings", []),
        }
    if pass_name == "final_adjudicator":
        return {
            **common,
            "answer": payload.get("answer"),
            "citation_mappings": payload.get("citation_mappings", []),
            "structured_target": payload.get("structured_target", {}),
            "deterministic_review": payload.get("deterministic_review", {}),
            "critic_results": payload.get("critic_results", []),
            "repair_history": payload.get("repair_history", []),
        }
    return {
        **common,
        "answer": payload.get("answer"),
        "citation_mappings": payload.get("citation_mappings", []),
        "structured_target": payload.get("structured_target", {}),
        "required_corrections": payload.get("required_corrections", []),
    }


def execute_review_pass(
    provider: CodexReviewProvider, pass_name: str, payload: dict[str, Any]
) -> CodexReviewPass:
    """Execute and provenance-wrap one blinded review pass."""
    if not provider.available():
        raise RuntimeError("Codex reviewer service is unavailable.")
    blinded = blind_payload(pass_name, payload)
    result = provider.review(pass_name, blinded)
    if not isinstance(result, CodexReview):
        raise TypeError("Codex review provider returned an invalid result type.")
    system, version = provider.identity
    return CodexReviewPass(
        pass_name=pass_name,
        reviewer_system=system,
        reviewer_version=version,
        input_hash=content_sha256(blinded),
        output_hash=content_sha256(result.to_dict()),
        result=result,
    )


class CodexCLIReviewProvider:
    """Invoke the installed Codex CLI with a strict structured-output schema."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        executable: str = "codex",
        timeout_seconds: int = 300,
    ) -> None:
        root = Path(repository_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository_root must be a directory: {root}")
        if not isinstance(timeout_seconds, int) or timeout_seconds < 10:
            raise ValueError("timeout_seconds must be an integer of at least 10.")
        self.repository_root = root
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    @property
    def identity(self) -> tuple[str, str]:
        return "openai_codex_cli", __version__

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def review(self, pass_name: str, payload: dict[str, Any]) -> CodexReview:
        if pass_name not in PASS_NAMES:
            raise ValueError(f"pass_name must be one of {PASS_NAMES}.")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary.")
        prompt = (
            "You are one focused stage in a local research-paper dataset curation "
            "pipeline. Do not browse or use facts outside the supplied passages. "
            f"Your role is {pass_name}. {_PASS_INSTRUCTIONS[pass_name]} "
            "Return only the requested schema.\n\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        with tempfile.TemporaryDirectory(prefix="localml_codex_review_") as directory:
            schema_path = Path(directory) / "schema.json"
            output_path = Path(directory) / "result.json"
            schema_path.write_text(
                json.dumps(codex_review_json_schema(), sort_keys=True), encoding="utf-8"
            )
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--ask-for-approval",
                "never",
                "--cd",
                str(self.repository_root),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                prompt,
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"Codex {pass_name} review exceeded {self.timeout_seconds} seconds."
                ) from error
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"Codex {pass_name} review failed: {message[:1000]}")
            try:
                value = json.loads(output_path.read_text(encoding="utf-8"))
            except (
                FileNotFoundError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:
                raise ValueError(
                    f"Codex {pass_name} review returned malformed JSON."
                ) from error
        return CodexReview.from_dict(value)
