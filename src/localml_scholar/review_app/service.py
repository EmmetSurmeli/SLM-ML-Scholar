"""Application service for local ingestion, questioning, analysis, and feedback."""

from __future__ import annotations

import re
import threading
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from localml_scholar.answering import GroundedAnswerPipeline
from localml_scholar.retrieval import (
    PageText,
    RetrievalIndex,
    SearchFilters,
    ingest_markdown,
    ingest_pdf_text,
    ingest_plain_text,
)
from localml_scholar.review_app.automation import (
    propose_automatic_failure_review,
    propose_automatic_review,
    summarize_automatic_reviews,
)
from localml_scholar.review_app.storage import (
    atomic_write_bytes,
    atomic_write_json,
    load_json_list,
    load_json_object,
)
from localml_scholar.scholarly import ScholarlyAnalysisPipeline
from localml_scholar.training_data import (
    AutoReviewPolicy,
    ConversationContext,
    ConversationTurn,
    GroundedInstructionExample,
    PaperAcquisitionItem,
    QuestionCandidate,
    assign_paper_splits,
    build_dataset,
    calibration_report,
    dataset_report,
    generate_paper_questions,
    generate_prompt_variations,
    infer_instruction_profile,
    propose_correction,
    review_interaction_second_pass,
    save_dataset,
    select_audit_sample,
    select_calibration_sample,
)
from localml_scholar.training_data import (
    approve_correction as approve_correction_example,
)
from localml_scholar.training_data.diversity import (
    diversity_metrics,
    diversity_warnings,
    progress_status,
    review_priority,
)
from localml_scholar.training_data.provenance import content_sha256
from localml_scholar.training_data.schemas import REVIEW_LABELS

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
_VERDICTS = {
    "correct",
    "partial",
    "partially_correct",
    "incorrect",
    "should_abstain",
    "benchmark_problem",
}
_AUDIENCE_LEVELS = {"beginner", "undergraduate", "researcher"}
_LEGACY_AUDIENCE_LEVELS = {
    "phd_researcher_professor": "researcher",
    "high_school_beginner": "beginner",
}
_ISSUE_CATEGORIES = {
    "wrong_answer",
    "missing_evidence",
    "incorrect_citation",
    "missed_equation",
    "missing_method_detail",
    "extraction_error",
    "too_advanced",
    "too_basic",
    "missing_prerequisite",
    "unclear_explanation",
    "unsupported_claim",
    "wrong_evidence",
    "style_mismatch",
    "comparison_incomplete",
    "other",
}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _identifier(prefix: str) -> str:
    compact_time = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{compact_time}_{uuid.uuid4().hex[:10]}"


def _nonempty_text(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-whitespace text.")
    cleaned = value.strip()
    if len(cleaned) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters.")
    return cleaned


def _safe_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise TypeError("filename must be a string.")
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("filename must identify one local file.")
    cleaned = _SAFE_FILENAME.sub("_", name)
    if cleaned.startswith("."):
        cleaned = f"paper{cleaned}"
    suffix = Path(cleaned).suffix.casefold()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError("Supported paper files are .pdf, .txt, .md, and .markdown.")
    return cleaned


def _audience_level(value: object) -> str:
    if isinstance(value, str):
        value = _LEGACY_AUDIENCE_LEVELS.get(value, value)
    if not isinstance(value, str) or value not in _AUDIENCE_LEVELS:
        raise ValueError(f"audience_level must be one of {sorted(_AUDIENCE_LEVELS)}.")
    return value


class ReviewService:
    """Own all mutable state for one repository-local review workspace.

    Feedback is stored for later human/Codex inspection. Saving feedback does not
    retrain a model or alter an answer automatically.
    """

    def __init__(self, repository_root: str | Path) -> None:
        root = Path(repository_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository_root must be an existing directory: {root}")
        self.repository_root = root
        self.upload_directory = root / "data" / "raw" / "review_app"
        self.output_directory = root / "outputs" / "review_app"
        self.index_path = self.output_directory / "index.json"
        self.interactions_path = self.output_directory / "interactions.json"
        self.feedback_path = self.output_directory / "feedback.json"
        self.questions_path = self.output_directory / "question_candidates.json"
        self.corrections_path = self.output_directory / "corrections.json"
        self.dataset_path = self.output_directory / "grounded_instruction_dataset.json"
        self.sessions_path = self.output_directory / "opt_in_sessions.json"
        self.automatic_reviews_path = (
            self.output_directory / "automatic_review_batches.json"
        )
        self.review_policy_path = self.output_directory / "review_policy.json"
        self.audit_queue_path = self.output_directory / "audit_queue.json"
        self.calibration_sample_path = self.output_directory / "calibration_sample.json"
        self.calibration_pairs_path = self.output_directory / "calibration_pairs.json"
        self.historical_reruns_path = (
            self.output_directory / "historical_review_reruns.json"
        )
        self.acquisition_queue_path = (
            self.output_directory / "paper_acquisition_queue.json"
        )
        self._sessions: dict[str, ConversationContext] = {}
        self._lock = threading.RLock()

    def _load_index(self, *, required: bool = True) -> RetrievalIndex | None:
        if not self.index_path.exists():
            if required:
                raise ValueError("No papers are indexed yet. Add a paper first.")
            return None
        return RetrievalIndex.load(self.index_path)

    def _document(self, document_id: str):
        document_id = _nonempty_text(document_id, "document_id", maximum=200)
        index = self._load_index()
        matches = tuple(
            document
            for document in index.documents
            if document.document_id == document_id
        )
        if len(matches) != 1:
            raise ValueError(f"Unknown document_id: {document_id}")
        return index, matches[0]

    @staticmethod
    def _extract_pdf(payload: bytes) -> tuple[tuple[PageText, ...], str | None]:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                'PDF support requires: python -m pip install -e ".[app]"'
            ) from error
        try:
            reader = PdfReader(BytesIO(payload))
        except Exception as error:
            raise ValueError("The uploaded file is not a readable PDF.") from error
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")
        pages: list[PageText] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as error:
                raise ValueError(
                    f"Text extraction failed on PDF page {page_number}."
                ) from error
            pages.append(PageText(page_number=page_number, text=text))
        if not pages:
            raise ValueError("The uploaded PDF contains no pages.")
        if not any(page.text.strip() for page in pages):
            raise ValueError(
                "The PDF contains no extractable text. Scanned PDFs need OCR first."
            )
        metadata_title = reader.metadata.title if reader.metadata is not None else None
        title = metadata_title.strip() if isinstance(metadata_title, str) else None
        return tuple(pages), title or None

    def add_paper(
        self,
        *,
        filename: str,
        payload: bytes,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Persist and index one PDF, UTF-8 text file, or Markdown paper."""
        safe_name = _safe_filename(filename)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes.")
        if not payload:
            raise ValueError("The uploaded paper is empty.")
        if len(payload) > 30 * 1024 * 1024:
            raise ValueError("Paper uploads are limited to 30 MiB.")
        resolved_title = (
            None
            if title is None or not title.strip()
            else _nonempty_text(title, "title", maximum=500)
        )
        relative_source = (Path("data") / "raw" / "review_app" / safe_name).as_posix()
        suffix = Path(safe_name).suffix.casefold()
        if suffix == ".pdf":
            pages, pdf_title = self._extract_pdf(payload)
            document = ingest_pdf_text(
                pages,
                source=relative_source,
                title=resolved_title or pdf_title or Path(safe_name).stem,
                metadata={"review_app_upload": True},
            )
        else:
            try:
                text = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "Text and Markdown uploads must be valid UTF-8."
                ) from error
            if not text.strip():
                raise ValueError("The uploaded paper contains no non-whitespace text.")
            ingest = (
                ingest_markdown if suffix in {".md", ".markdown"} else ingest_plain_text
            )
            document_title = (
                resolved_title
                if suffix in {".md", ".markdown"}
                else resolved_title or Path(safe_name).stem
            )
            document = ingest(
                text,
                source=relative_source,
                title=document_title,
                metadata={"review_app_upload": True},
            )

        with self._lock:
            existing = self._load_index(required=False)
            documents = [] if existing is None else list(existing.documents)
            documents = [
                item for item in documents if item.source_path != relative_source
            ]
            documents.append(document)
            new_index = RetrievalIndex.build(documents)
            destination = self.upload_directory / safe_name
            atomic_write_bytes(destination, payload)
            new_index.save(self.index_path)
        return self.paper_summary(document.document_id)

    def paper_summary(self, document_id: str) -> dict[str, Any]:
        """Return compact metadata for one indexed paper."""
        index, document = self._document(document_id)
        chunk_count = sum(
            chunk.document_id == document.document_id for chunk in index.chunks
        )
        page_count = document.metadata.get("inferred", {}).get("page_count")
        questions = load_json_list(self.questions_path)
        interactions = load_json_list(self.interactions_path)
        corrections = load_json_list(self.corrections_path)
        return {
            "document_id": document.document_id,
            "title": document.title or document.source_name,
            "source_name": document.source_name,
            "source_path": document.source_path,
            "media_type": document.media_type,
            "page_count": page_count,
            "section_count": len(document.sections),
            "chunk_count": chunk_count,
            "character_count": document.character_length,
            "benchmark_question_count": sum(
                document_id in item.get("paper_ids", []) for item in questions
            ),
            "reviewed_answer_count": sum(
                document_id
                in item.get(
                    "paper_ids",
                    [item.get("document_id")]
                    if item.get("document_id") is not None
                    else [],
                )
                for item in interactions
            ),
            "correction_count": sum(
                document_id in item.get("paper_ids", []) for item in corrections
            ),
        }

    def list_papers(self) -> list[dict[str, Any]]:
        """List indexed papers in deterministic title/source order."""
        with self._lock:
            index = self._load_index(required=False)
            if index is None:
                return []
            summaries = [
                self.paper_summary(document.document_id) for document in index.documents
            ]
        return sorted(
            summaries,
            key=lambda item: (item["title"].casefold(), item["source_name"].casefold()),
        )

    def _load_candidates(self) -> list[QuestionCandidate]:
        return [
            QuestionCandidate.from_dict(item)
            for item in load_json_list(self.questions_path)
        ]

    def _save_candidates(self, items: list[QuestionCandidate]) -> None:
        atomic_write_json(
            self.questions_path,
            [
                item.to_dict()
                for item in sorted(items, key=lambda item: item.question_id)
            ],
        )

    def _load_corrections(self) -> list[GroundedInstructionExample]:
        return [
            GroundedInstructionExample.from_dict(item)
            for item in load_json_list(self.corrections_path)
        ]

    def _save_corrections(self, items: list[GroundedInstructionExample]) -> None:
        atomic_write_json(
            self.corrections_path,
            [
                item.to_dict()
                for item in sorted(items, key=lambda item: item.example_id)
            ],
        )

    def _load_automatic_review_batches(self) -> list[dict[str, Any]]:
        batches = load_json_list(self.automatic_reviews_path)
        for batch in batches:
            if not isinstance(batch.get("batch_id"), str) or not isinstance(
                batch.get("reviews"), list
            ):
                raise ValueError("Automatic review state contains a malformed batch.")
        return batches

    def _save_automatic_review_batches(self, batches: list[dict[str, Any]]) -> None:
        atomic_write_json(self.automatic_reviews_path, batches)

    def _persist_automatic_review_batch(self, batch: dict[str, Any]) -> None:
        with self._lock:
            batches = self._load_automatic_review_batches()
            for position, item in enumerate(batches):
                if item["batch_id"] == batch["batch_id"]:
                    batches[position] = batch
                    self._save_automatic_review_batches(batches)
                    return
        raise RuntimeError(f"Automatic review batch disappeared: {batch['batch_id']}")

    def _review_policy_state(self) -> dict[str, Any]:
        return load_json_object(
            self.review_policy_path,
            default={
                "approval_threshold": 0.95,
                "explicit_enable": False,
                "calibration_records": [],
            },
        )

    def _all_automatic_reviews(self) -> list[dict[str, Any]]:
        """Return original reviews plus append-only reruns in stable order."""
        originals = [
            review
            for batch in self._load_automatic_review_batches()
            for review in batch.get("reviews", [])
        ]
        reruns = load_json_list(self.historical_reruns_path)
        latest_by_original: dict[str, dict[str, Any]] = {}
        for rerun in reruns:
            original_id = rerun.get("original_review_id")
            review = rerun.get("new_review")
            if isinstance(original_id, str) and isinstance(review, dict):
                latest_by_original[original_id] = review
        return [
            latest_by_original.get(item.get("review_id"), item) for item in originals
        ]

    def _calibration_records(self) -> list[dict[str, Any]]:
        """Load 1.2.2 pairs and legacy policy records without double counting."""
        legacy = self._review_policy_state().get("calibration_records", [])
        if not isinstance(legacy, list):
            raise ValueError("review_policy calibration_records must be a list.")
        pairs = load_json_list(self.calibration_pairs_path)
        by_review = {
            item.get("review_id"): item
            for item in legacy
            if isinstance(item.get("review_id"), str)
        }
        for item in pairs:
            review_id = item.get("review_id")
            if isinstance(review_id, str) and item.get("status") == "finalized":
                by_review[review_id] = item
        return list(by_review.values())

    def _calibration_integrity(self) -> dict[str, int]:
        """Compute integrity failures for native 1.2.2 calibration pairs."""
        pairs = [
            item
            for item in load_json_list(self.calibration_pairs_path)
            if item.get("status") == "finalized"
        ]
        source_hash_errors = 0
        provenance_errors = 0
        test_leakage_errors = 0
        identities: list[str] = []
        for pair in pairs:
            snapshot = pair.get("reviewed_snapshot", {})
            second = snapshot.get("second_pass", {})
            provenance = second.get("provenance", {})
            source_hashes = provenance.get("source_hashes")
            if not isinstance(source_hashes, list) or not source_hashes:
                source_hash_errors += 1
            if provenance.get("circular_warnings"):
                provenance_errors += 1
            metadata = snapshot.get("metadata", {})
            if isinstance(metadata, dict) and (
                metadata.get("test_only") or metadata.get("test_only_paper_ids")
            ):
                test_leakage_errors += 1
            identities.append(
                content_sha256(
                    {
                        "question": snapshot.get("question"),
                        "answer": snapshot.get("answer", {}).get("answer_text"),
                        "paper_ids": snapshot.get("paper_ids", []),
                    }
                )
            )
        duplicate_errors = len(identities) - len(set(identities))
        return {
            "source_hash_errors": source_hash_errors,
            "test_leakage_errors": test_leakage_errors,
            "provenance_errors": provenance_errors,
            "duplicate_errors": duplicate_errors,
        }

    def _calibration(self) -> dict[str, Any]:
        state = self._review_policy_state()
        integrity = self._calibration_integrity()
        for name, value in state.get("integrity", {}).items():
            if (
                name in integrity
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                integrity[name] += value
        return calibration_report(
            self._calibration_records(),
            explicit_enable=bool(state.get("explicit_enable", False)),
            integrity=integrity,
            approval_threshold=float(state.get("approval_threshold", 0.95)),
        )

    def _auto_review_policy(self) -> AutoReviewPolicy:
        state = self._review_policy_state()
        calibration = self._calibration()
        return AutoReviewPolicy(
            approval_threshold=state.get("approval_threshold", 0.95),
            calibration_state=calibration["state"],
        )

    def _record_human_outcome(
        self,
        *,
        example_id: str,
        human_approved: bool,
        reviewer: str,
    ) -> None:
        """Attach an immutable human outcome to its originating auto review."""
        batches = self._load_automatic_review_batches()
        matched_review: dict[str, Any] | None = None
        for batch in batches:
            for review in batch.get("reviews", []):
                if review.get("correction_example_id") == example_id:
                    if matched_review is not None:
                        raise ValueError(
                            "Multiple automatic reviews reference example "
                            f"{example_id}."
                        )
                    matched_review = review
        if matched_review is None:
            return
        matched_review["review_status"] = (
            "human_approved" if human_approved else "human_rejected"
        )
        matched_review["human_outcome"] = {
            "approved": human_approved,
            "reviewer": reviewer,
            "recorded_at": _timestamp(),
        }
        self._save_automatic_review_batches(batches)
        second_pass = matched_review.get("second_pass", {})
        confidence = second_pass.get(
            "confidence", matched_review.get("proposed_confidence", 0.0)
        )
        policy_state = self._review_policy_state()
        records = policy_state.setdefault("calibration_records", [])
        record = {
            "review_id": matched_review["review_id"],
            "confidence": confidence,
            "automated_approved": bool(
                second_pass.get(
                    "would_approve_if_enabled",
                    second_pass.get("review_status") == "codex_approved",
                )
            ),
            "human_approved": human_approved,
            "reviewer": reviewer,
            "recorded_at": _timestamp(),
        }
        records = [
            item for item in records if item.get("review_id") != record["review_id"]
        ]
        records.append(record)
        policy_state["calibration_records"] = records
        atomic_write_json(self.review_policy_path, policy_state)
        audit_queue = load_json_object(
            self.audit_queue_path,
            default={"items": [], "selected_count": 0, "population_count": 0},
        )
        changed = False
        for item in audit_queue.get("items", []):
            if item.get("example_id") == matched_review["review_id"]:
                item["status"] = "audited_pass" if human_approved else "overturned"
                item["audited_by"] = reviewer
                item["audited_at"] = _timestamp()
                changed = True
        if changed:
            atomic_write_json(self.audit_queue_path, audit_queue)

    def _materialize_codex_approval(self, review: dict[str, Any]) -> None:
        """Create a provenance-preserving Codex-approved dataset candidate."""
        second_pass = review.get("second_pass", {})
        if second_pass.get("review_status") != "codex_approved":
            return
        failures = []
        provenance = second_pass.get("provenance", {})
        if not provenance.get("source_hashes"):
            failures.append("source_hashes_missing")
        if provenance.get("circular_warnings"):
            failures.append("circular_provenance")
        candidates = {item.question_id: item for item in self._load_candidates()}
        source_candidate = candidates.get(review.get("question_id"))
        if source_candidate is not None and (
            source_candidate.metadata.get("test_only")
            or source_candidate.metadata.get("test_only_paper_ids")
        ):
            failures.append("test_only_leakage")
        identity = content_sha256(
            {
                "question": review.get("question"),
                "answer": review.get("answer", {}).get("answer_text"),
                "paper_ids": review.get("paper_ids", []),
            }
        )
        for correction in self._load_corrections():
            correction_identity = content_sha256(
                {
                    "question": correction.turns[-1].content,
                    "answer": correction.final_answer,
                    "paper_ids": list(correction.paper_ids),
                }
            )
            if correction_identity == identity:
                failures.append("duplicate_training_candidate")
                break
        if failures:
            second_pass["review_status"] = "needs_human_review"
            second_pass["human_review_route"] = "approval_integrity_failure"
            second_pass["approval_integrity_failures"] = failures
            review["review_status"] = "needs_human_review"
            review["decision"] = "pending_user_review"
            return
        interaction_id = review.get("interaction_id")
        if not isinstance(interaction_id, str):
            raise ValueError("A Codex-approved review must identify its interaction.")
        correction = self.review_interaction(
            interaction_id=interaction_id,
            review_label="correct",
            corrected_answer=review["answer"]["answer_text"],
            required_facts=tuple(review.get("proposed_required_facts", [])),
            prohibited_claims=tuple(review.get("proposed_prohibited_claims", [])),
            replacement_evidence_ids=tuple(review.get("proposed_evidence_ids", [])),
            notes="Confidence-gated Codex approval; not human gold.",
        )
        with self._lock:
            corrections = self._load_corrections()
            positions = [
                position
                for position, item in enumerate(corrections)
                if item.example_id == correction["example_id"]
            ]
            if len(positions) != 1:
                raise RuntimeError("Codex-approved correction was not persisted.")
            candidate = corrections[positions[0]]
            updated = replace(
                candidate,
                review_status="codex_approved",
                metadata={
                    **candidate.metadata,
                    "codex_approved": True,
                    "human_approved": False,
                    "audit_status": "pending",
                    "approval_provenance": second_pass["provenance"],
                    "second_pass": second_pass,
                },
            )
            corrections[positions[0]] = updated
            self._save_corrections(corrections)
        review["correction_example_id"] = updated.example_id
        review["decision"] = "codex_approved"

    def create_session(
        self,
        *,
        selected_paper_ids: tuple[str, ...] = (),
        preferences: dict[str, Any] | None = None,
        persist_preferences: bool = False,
    ) -> dict[str, Any]:
        """Create local conversation state; disk persistence is explicit opt-in."""
        if not isinstance(selected_paper_ids, tuple):
            raise TypeError("selected_paper_ids must be a tuple.")
        if preferences is not None and not isinstance(preferences, dict):
            raise TypeError("preferences must be a dictionary.")
        known = {paper["document_id"] for paper in self.list_papers()}
        unknown = set(selected_paper_ids) - known
        if unknown:
            raise ValueError(f"Unknown selected paper IDs: {sorted(unknown)}.")
        session = ConversationContext(
            session_id=_identifier("session"),
            selected_paper_ids=selected_paper_ids,
            preferences={} if preferences is None else dict(preferences),
            persist_preferences=persist_preferences,
        )
        with self._lock:
            self._sessions[session.session_id] = session
            if persist_preferences:
                persisted = load_json_list(self.sessions_path)
                persisted.append(session.to_dict())
                atomic_write_json(self.sessions_path, persisted)
        return session.to_dict()

    def get_session(self, session_id: str) -> ConversationContext:
        """Return one in-memory session without silently loading old preferences."""
        session_id = _nonempty_text(session_id, "session_id", maximum=200)
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ValueError(f"Unknown or expired session_id: {session_id}") from None

    def update_session(
        self,
        session_id: str,
        *,
        selected_paper_ids: tuple[str, ...] | None = None,
        preferences: dict[str, Any] | None = None,
        persist_preferences: bool | None = None,
    ) -> dict[str, Any]:
        """Update explicit session preferences and selected local papers."""
        with self._lock:
            current = self.get_session(session_id)
            selected = (
                current.selected_paper_ids
                if selected_paper_ids is None
                else selected_paper_ids
            )
            known = {paper["document_id"] for paper in self.list_papers()}
            unknown = set(selected) - known
            if unknown:
                raise ValueError(f"Unknown selected paper IDs: {sorted(unknown)}.")
            updated = replace(
                current,
                selected_paper_ids=selected,
                preferences=current.preferences if preferences is None else preferences,
                persist_preferences=(
                    current.persist_preferences
                    if persist_preferences is None
                    else persist_preferences
                ),
            )
            self._sessions[session_id] = updated
        return updated.to_dict()

    def list_questions(
        self,
        *,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List candidate questions in high-value review priority order."""
        with self._lock:
            items = self._load_candidates()
        if paper_id is not None:
            items = [item for item in items if paper_id in item.paper_ids]
        states = [item.to_dict() for item in items]
        return sorted(states, key=review_priority)

    def generate_questions(
        self,
        *,
        paper_id: str,
        count: int = 60,
    ) -> list[dict[str, Any]]:
        """Generate and persist proposed-only candidates for one indexed paper."""
        with self._lock:
            _index, document = self._document(paper_id)
            generated = generate_paper_questions(
                paper_id,
                document.title or document.source_name,
                count=count,
                section_titles=tuple(
                    section.heading or "Untitled section"
                    for section in document.sections
                ),
            )
            existing = {item.question_id: item for item in self._load_candidates()}
            existing.update({item.question_id: item for item in generated})
            self._save_candidates(list(existing.values()))
        return [item.to_dict() for item in generated]

    def add_question(
        self,
        *,
        question: str,
        paper_ids: tuple[str, ...],
        question_type: str = "user_authored",
    ) -> dict[str, Any]:
        """Add an arbitrary human-authored question as an unapproved candidate."""
        cleaned = _nonempty_text(question, "question", maximum=4000)
        if not isinstance(paper_ids, tuple) or not paper_ids:
            raise ValueError("paper_ids must be a non-empty tuple.")
        known = {paper["document_id"] for paper in self.list_papers()}
        unknown = set(paper_ids) - known
        if unknown:
            raise ValueError(f"Unknown selected paper IDs: {sorted(unknown)}.")
        candidate = QuestionCandidate.create(
            paper_ids=paper_ids,
            question=cleaned,
            question_type=question_type,
            metadata={"candidate_only": True, "source": "human_authored"},
        )
        with self._lock:
            items = {item.question_id: item for item in self._load_candidates()}
            items[candidate.question_id] = candidate
            self._save_candidates(list(items.values()))
        return candidate.to_dict()

    def search_evidence(
        self,
        *,
        query: str,
        paper_ids: tuple[str, ...],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return exact alternative chunks for deliberate evidence replacement."""
        query = _nonempty_text(query, "query", maximum=4000)
        if not isinstance(paper_ids, tuple) or not paper_ids:
            raise ValueError("paper_ids must be a non-empty tuple.")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 30
        ):
            raise ValueError("top_k must be an integer in [1, 30].")
        with self._lock:
            index = self._load_index()
            documents = {document.document_id: document for document in index.documents}
            unknown = set(paper_ids) - set(documents)
            if unknown:
                raise ValueError(f"Unknown selected paper IDs: {sorted(unknown)}.")
            selected_index = RetrievalIndex.build(
                [documents[paper_id] for paper_id in paper_ids]
            )
            results = selected_index.search(query, method="bm25", top_k=top_k)
        return [item.to_dict() for item in results]

    def propose_question_variations(self, question_id: str) -> list[dict[str, Any]]:
        """Create candidate-only phrasings linked to one reviewed target."""
        question_id = _nonempty_text(question_id, "question_id", maximum=200)
        with self._lock:
            items = self._load_candidates()
            matches = [item for item in items if item.question_id == question_id]
            if len(matches) != 1:
                raise ValueError(f"Unknown question_id: {question_id}")
            variations = generate_prompt_variations(matches[0])
            by_id = {item.question_id: item for item in items}
            by_id.update({item.question_id: item for item in variations})
            self._save_candidates(list(by_id.values()))
        return [item.to_dict() for item in variations]

    def create_calibration_sample(
        self, *, target_count: int = 50, seed: int = 42
    ) -> dict[str, Any]:
        """Persist a deterministic, coverage-seeking calibration work queue."""
        with self._lock:
            result = select_calibration_sample(
                self._all_automatic_reviews(), target_count=target_count, seed=seed
            )
            result.update(
                {
                    "sample_id": _identifier("calibration_sample"),
                    "created_at": _timestamp(),
                    "format_version": 1,
                }
            )
            atomic_write_json(self.calibration_sample_path, result)
        return result

    def _review_by_id(self, review_id: str) -> dict[str, Any]:
        matches = [
            item
            for item in self._all_automatic_reviews()
            if item.get("review_id") == review_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown review_id: {review_id}")
        return matches[0]

    def calibration_cards(self) -> list[dict[str, Any]]:
        """Return sampled review snapshots enriched with human decision state."""
        sample = load_json_object(
            self.calibration_sample_path,
            default={"review_ids": [], "items": [], "coverage_gaps": []},
        )
        decisions = {
            item.get("review_id"): item
            for item in load_json_list(self.calibration_pairs_path)
        }
        papers = {item["document_id"]: item for item in self.list_papers()}
        cards = []
        for review_id in sample.get("review_ids", []):
            try:
                review = deepcopy(self._review_by_id(review_id))
            except ValueError:
                cards.append(
                    {
                        "review_id": review_id,
                        "unavailable": True,
                        "error": "The linked review is no longer available.",
                    }
                )
                continue
            review["paper_metadata"] = [
                papers[paper_id]
                for paper_id in review.get("paper_ids", [])
                if paper_id in papers
            ]
            review["calibration_decision"] = decisions.get(review_id)
            second = review.get("second_pass", {})
            review["root_cause"] = sorted(
                {
                    gate
                    for result in second.get("reviewer_results", [])
                    for gate, passed in result.get("gates", {}).items()
                    if not passed
                }
            )
            cards.append(review)
        return cards

    def rerun_historical_reviews(
        self, *, review_ids: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Append linked modern reruns while preserving original review snapshots."""
        if review_ids is not None and (
            not isinstance(review_ids, tuple)
            or not all(isinstance(item, str) and item.strip() for item in review_ids)
        ):
            raise TypeError("review_ids must be None or a tuple of non-empty strings.")
        originals = [
            review
            for batch in self._load_automatic_review_batches()
            for review in batch.get("reviews", [])
        ]
        requested = set(review_ids or (item.get("review_id") for item in originals))
        candidates = {item.question_id: item for item in self._load_candidates()}
        appended = []
        with self._lock:
            records = load_json_list(self.historical_reruns_path)
            for original in originals:
                original_id = original.get("review_id")
                if original_id not in requested:
                    continue
                candidate = candidates.get(original.get("question_id"))
                if candidate is None:
                    continue
                rerun_batch_id = _identifier("historical_rerun")
                try:
                    interaction = self.run_question(candidate.question_id)
                    new_review = propose_automatic_review(
                        interaction,
                        candidate,
                        batch_id=rerun_batch_id,
                        policy=self._auto_review_policy(),
                    )
                except Exception as error:
                    new_review = propose_automatic_failure_review(
                        candidate,
                        batch_id=rerun_batch_id,
                        error=error,
                        policy=self._auto_review_policy(),
                    )
                new_review["review_id"] = original_id
                record = {
                    "rerun_id": rerun_batch_id,
                    "original_review_id": original_id,
                    "original_snapshot": deepcopy(original),
                    "original_snapshot_hash": content_sha256(original),
                    "new_review": new_review,
                    "new_snapshot_hash": content_sha256(new_review),
                    "created_at": _timestamp(),
                    "non_destructive": True,
                }
                records.append(record)
                appended.append(record)
            missing = requested - {item.get("original_review_id") for item in appended}
            if missing:
                raise ValueError(
                    f"Historical review IDs could not be rerun: {sorted(missing)}"
                )
            atomic_write_json(self.historical_reruns_path, records)
        return {"rerun_count": len(appended), "reruns": appended}

    def record_calibration_decision(
        self,
        *,
        review_id: str,
        action: str,
        reviewer: str,
        edits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one human calibration label; never approve it for training."""
        review_id = _nonempty_text(review_id, "review_id", maximum=200)
        reviewer = _nonempty_text(reviewer, "reviewer", maximum=500)
        actions = {
            "approve_auto": None,
            "override_correct": "correct",
            "override_partial": "partial",
            "override_incorrect": "incorrect",
            "override_should_abstain": "should_abstain",
            "benchmark_problem": "benchmark_problem",
            "skip": None,
        }
        if action not in actions:
            raise ValueError(f"action must be one of {sorted(actions)}.")
        if edits is not None and not isinstance(edits, dict):
            raise TypeError("edits must be an object or None.")
        with self._lock:
            reviews = {
                item.get("review_id"): item for item in self._all_automatic_reviews()
            }
            if review_id not in reviews:
                raise ValueError(f"Unknown review_id: {review_id}")
            existing = load_json_list(self.calibration_pairs_path)
            if any(
                item.get("review_id") == review_id and item.get("status") == "finalized"
                for item in existing
            ):
                raise ValueError(
                    "This calibration review already has a finalized decision."
                )
            review = deepcopy(reviews[review_id])
            original_snapshot = deepcopy(review)
            before_hash = content_sha256(review)
            allowed_edits = {
                "answer_text",
                "evidence_ids",
                "required_facts",
                "prohibited_claims",
                "structured_target",
                "citations",
            }
            unknown = set(edits or {}) - allowed_edits
            if unknown:
                raise ValueError(
                    f"Unsupported calibration edit fields: {sorted(unknown)}"
                )
            if edits:
                if "answer_text" in edits:
                    review.setdefault("answer", {})["answer_text"] = _nonempty_text(
                        edits["answer_text"], "answer_text", maximum=100_000
                    )
                for field in (
                    "required_facts",
                    "prohibited_claims",
                    "evidence_ids",
                    "citations",
                ):
                    if field in edits and not isinstance(edits[field], list):
                        raise TypeError(f"{field} must be a list.")
                    if field in edits and not all(
                        isinstance(value, str) and value.strip()
                        for value in edits[field]
                    ):
                        raise ValueError(f"{field} values must be non-empty strings.")
                if "required_facts" in edits:
                    review["proposed_required_facts"] = edits["required_facts"]
                if "prohibited_claims" in edits:
                    review["proposed_prohibited_claims"] = edits["prohibited_claims"]
                if "evidence_ids" in edits:
                    review["proposed_evidence_ids"] = edits["evidence_ids"]
                if "structured_target" in edits:
                    if not isinstance(edits["structured_target"], dict):
                        raise TypeError("structured_target must be an object.")
                    review["structured_target"] = edits["structured_target"]
                if "citations" in edits:
                    review["citations"] = edits["citations"]
                candidate_matches = [
                    item
                    for item in self._load_candidates()
                    if item.question_id == review.get("question_id")
                ]
                candidate = (
                    candidate_matches[0]
                    if len(candidate_matches) == 1
                    else QuestionCandidate.create(
                        paper_ids=tuple(review.get("paper_ids", [])),
                        question=review.get("question", "Calibration question"),
                        question_type=review.get("question_type", "user_authored"),
                        required_concepts=tuple(
                            review.get("proposed_required_facts", [])
                        ),
                        prohibited_claims=tuple(
                            review.get("proposed_prohibited_claims", [])
                        ),
                        metadata={"reconstructed_for_calibration": True},
                    )
                )
                revalidation = review_interaction_second_pass(
                    {
                        "paper_ids": list(review.get("paper_ids", [])),
                        "question": review.get("question", ""),
                        "answer": review.get("answer", {}),
                        "diagnostics": review.get("diagnostics", {}),
                    },
                    candidate,
                    policy=self._auto_review_policy(),
                    corrected_answer=review.get("answer", {}).get("answer_text"),
                ).to_dict()
                review["edit_revalidation"] = revalidation
            proposed_label = review.get("proposed_label", "partial")
            human_label = (
                proposed_label if action == "approve_auto" else actions[action]
            )
            if action == "skip":
                pair = {
                    "pair_id": _identifier("calibration_skip"),
                    "review_id": review_id,
                    "status": "skipped",
                    "reviewer": reviewer,
                    "recorded_at": _timestamp(),
                    "training_approved": False,
                }
            else:
                if human_label not in REVIEW_LABELS:
                    raise ValueError("Calibration decision produced an invalid label.")
                second = review.get("second_pass", {})
                automatic = bool(
                    second.get(
                        "would_approve_if_enabled",
                        second.get("review_status") == "codex_approved",
                    )
                )
                confidence = float(
                    second.get("confidence", review.get("proposed_confidence", 0.0))
                )
                failures = sorted(
                    {
                        gate
                        for result in second.get("reviewer_results", [])
                        for gate, passed in result.get("gates", {}).items()
                        if not passed
                    }
                )
                pair = {
                    "pair_id": _identifier("calibration_pair"),
                    "review_id": review_id,
                    "status": "finalized",
                    "action": action,
                    "reviewer": reviewer,
                    "recorded_at": _timestamp(),
                    "confidence": confidence,
                    "automated_label": proposed_label,
                    "human_label": human_label,
                    "automated_approved": automatic,
                    "human_approved": human_label == "correct",
                    "question_type": review.get("question_type", "unknown"),
                    "paper_ids": list(review.get("paper_ids", [])),
                    "failure_categories": failures,
                    "mandatory_human_categories": list(
                        second.get("mandatory_human_categories", [])
                    ),
                    "reviewer_profile": ",".join(
                        item.get("reviewer_profile", "unknown")
                        for item in second.get("reviewer_results", [])
                    )
                    or "unknown",
                    "original_snapshot_hash": before_hash,
                    "original_snapshot": original_snapshot,
                    "reviewed_snapshot": review,
                    "reviewed_snapshot_hash": content_sha256(review),
                    "edited": bool(edits),
                    "correction_revalidated": bool(edits),
                    "revalidation": review.get("edit_revalidation"),
                    "edits": edits or {},
                    "training_approved": False,
                    "training_example_id": None,
                }
            existing.append(pair)
            atomic_write_json(self.calibration_pairs_path, existing)
        return pair

    def approve_calibration_for_training(
        self, *, pair_id: str, reviewer: str
    ) -> dict[str, Any]:
        """Separately approve an eligible finalized calibration pair for export."""
        pair_id = _nonempty_text(pair_id, "pair_id", maximum=200)
        reviewer = _nonempty_text(reviewer, "reviewer", maximum=500)
        with self._lock:
            pairs = load_json_list(self.calibration_pairs_path)
            matches = [item for item in pairs if item.get("pair_id") == pair_id]
            if len(matches) != 1:
                raise ValueError(f"Unknown pair_id: {pair_id}")
            pair = matches[0]
            if pair.get("status") != "finalized":
                raise ValueError(
                    "Only finalized calibration pairs can enter training review."
                )
            if pair.get("training_approved"):
                raise ValueError(
                    "This calibration pair already entered training approval."
                )
            if pair.get("human_label") in {"benchmark_problem", "should_abstain"}:
                raise ValueError("This label is not eligible for a correction target.")
            snapshot = pair.get("reviewed_snapshot", {})
            interaction_id = snapshot.get("interaction_id")
            if not isinstance(interaction_id, str):
                raise ValueError(
                    "The calibration snapshot has no reviewable interaction."
                )
            correction = self.review_interaction(
                interaction_id=interaction_id,
                review_label=pair["human_label"],
                corrected_answer=snapshot.get("answer", {}).get("answer_text"),
                required_facts=tuple(snapshot.get("proposed_required_facts", [])),
                prohibited_claims=tuple(snapshot.get("proposed_prohibited_claims", [])),
                replacement_evidence_ids=tuple(
                    snapshot.get("proposed_evidence_ids", [])
                ),
                notes=f"Calibration pair {pair_id}; separate human training approval.",
            )
            approved = self.approve_correction(
                example_id=correction["example_id"], reviewer=reviewer
            )
            pair["training_approved"] = True
            pair["training_example_id"] = approved["example_id"]
            pair["training_approved_at"] = _timestamp()
            atomic_write_json(self.calibration_pairs_path, pairs)
        return pair

    def add_acquisition_item(self, **values: Any) -> dict[str, Any]:
        """Add a local paper suggestion; this method never fetches the paper."""
        item = PaperAcquisitionItem(**values)
        with self._lock:
            items = load_json_list(self.acquisition_queue_path)
            if any(existing.get("item_id") == item.item_id for existing in items):
                raise ValueError("This paper is already in the acquisition queue.")
            state = {**item.to_dict(), "created_at": _timestamp()}
            items.append(state)
            atomic_write_json(self.acquisition_queue_path, items)
        return state

    def update_acquisition_item(self, *, item_id: str, status: str) -> dict[str, Any]:
        """Update queue status without downloading or opening a network resource."""
        item_id = _nonempty_text(item_id, "item_id", maximum=200)
        if status not in {"suggested", "obtained", "declined"}:
            raise ValueError("status must be suggested, obtained, or declined.")
        with self._lock:
            items = load_json_list(self.acquisition_queue_path)
            matches = [item for item in items if item.get("item_id") == item_id]
            if len(matches) != 1:
                raise ValueError(f"Unknown acquisition item: {item_id}")
            matches[0]["status"] = status
            matches[0]["updated_at"] = _timestamp()
            atomic_write_json(self.acquisition_queue_path, items)
        return matches[0]

    def state(self) -> dict[str, Any]:
        """Return the complete lightweight browser bootstrap state."""
        with self._lock:
            feedback = load_json_list(self.feedback_path)
            interactions = load_json_list(self.interactions_path)
            papers = self.list_papers()
            questions = self._load_candidates()
            corrections = self._load_corrections()
            automatic_batches = self._load_automatic_review_batches()
        approved = tuple(
            item for item in corrections if item.review_status == "human_approved"
        )
        metrics = diversity_metrics(approved)
        all_reviews = [
            review for batch in automatic_batches for review in batch.get("reviews", [])
        ]
        second_pass_counts: dict[str, int] = {}
        confidence_values = []
        for review in all_reviews:
            second_pass = review.get("second_pass", {})
            status = second_pass.get("review_status", "not_run")
            second_pass_counts[status] = second_pass_counts.get(status, 0) + 1
            confidence = second_pass.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(
                confidence, bool
            ):
                confidence_values.append(float(confidence))
        calibration = self._calibration()
        calibration_records = self._calibration_records()
        false_approvals = sum(
            item.get("automated_approved") is True
            and item.get("human_approved") is False
            for item in calibration_records
        )
        false_rejections = sum(
            item.get("automated_approved") is False
            and item.get("human_approved") is True
            for item in calibration_records
        )
        failed_gate_counts: dict[str, int] = {}
        for review in all_reviews:
            for reviewer in review.get("second_pass", {}).get("reviewer_results", []):
                for gate, passed in reviewer.get("gates", {}).items():
                    if not passed:
                        failed_gate_counts[gate] = failed_gate_counts.get(gate, 0) + 1
        audit_queue = load_json_object(
            self.audit_queue_path,
            default={"items": [], "selected_count": 0, "population_count": 0},
        )
        paper_splits = (
            assign_paper_splits(
                tuple(paper["document_id"] for paper in papers), seed=42
            )
            if papers
            else {}
        )
        normalized_interactions = []
        for interaction in interactions:
            normalized = dict(interaction)
            if not isinstance(normalized.get("paper_ids"), list):
                legacy_document_id = normalized.get("document_id")
                normalized["paper_ids"] = (
                    [legacy_document_id]
                    if isinstance(legacy_document_id, str)
                    and legacy_document_id.strip()
                    else []
                )
            normalized_interactions.append(normalized)
        return {
            "papers": papers,
            "feedback": list(reversed(feedback[-50:])),
            "interactions": list(reversed(normalized_interactions[-50:])),
            "questions": [
                item.to_dict()
                for item in sorted(
                    questions, key=lambda item: review_priority(item.to_dict())
                )
            ],
            "corrections": [
                {
                    **item.to_dict(),
                    "effective_trust_status": (
                        "audited_codex_approved"
                        if item.review_status == "codex_approved"
                        and item.metadata.get("audit_status")
                        in {"human_confirmed", "passed"}
                        else item.review_status
                    ),
                }
                for item in corrections[-50:]
            ],
            "interaction_count": len(interactions),
            "feedback_count": len(feedback),
            "question_count": len(questions),
            "approved_question_count": sum(
                item.review_status == "human_approved" for item in questions
            ),
            "correction_count": len(corrections),
            "approved_example_count": len(approved),
            "automatic_review_batches": list(reversed(automatic_batches[-5:])),
            "automatic_review_batch_count": len(automatic_batches),
            "second_pass_metrics": {
                "status_counts": dict(sorted(second_pass_counts.items())),
                "review_count": len(all_reviews),
                "mean_confidence": (
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else None
                ),
                "reviewers_are_independent": False,
                "codex_approval_rate": second_pass_counts.get("codex_approved", 0)
                / len(all_reviews)
                if all_reviews
                else 0.0,
                "codex_rejection_rate": second_pass_counts.get("codex_rejected", 0)
                / len(all_reviews)
                if all_reviews
                else 0.0,
                "human_route_rate": second_pass_counts.get("needs_human_review", 0)
                / len(all_reviews)
                if all_reviews
                else 0.0,
                "false_approval_count": false_approvals,
                "false_rejection_count": false_rejections,
                "failure_type_distribution": dict(sorted(failed_gate_counts.items())),
            },
            "calibration": calibration,
            "calibration_sample": load_json_object(
                self.calibration_sample_path,
                default={
                    "review_ids": [],
                    "items": [],
                    "selected_count": 0,
                    "coverage_gaps": [],
                    "warnings": [],
                },
            ),
            "calibration_cards": self.calibration_cards(),
            "calibration_pairs": load_json_list(self.calibration_pairs_path),
            "audit_queue": audit_queue,
            "paper_acquisition_queue": load_json_list(self.acquisition_queue_path),
            "paper_splits": paper_splits,
            "dataset_metrics": metrics,
            "dataset_warnings": list(diversity_warnings(metrics)),
            "progress": progress_status(len(approved)),
            "storage": {
                "uploads": str(self.upload_directory),
                "index": str(self.index_path),
                "interactions": str(self.interactions_path),
                "feedback": str(self.feedback_path),
                "questions": str(self.questions_path),
                "corrections": str(self.corrections_path),
                "dataset": str(self.dataset_path),
                "automatic_reviews": str(self.automatic_reviews_path),
                "review_policy": str(self.review_policy_path),
                "audit_queue": str(self.audit_queue_path),
                "calibration_sample": str(self.calibration_sample_path),
                "calibration_pairs": str(self.calibration_pairs_path),
                "historical_reruns": str(self.historical_reruns_path),
                "paper_acquisition_queue": str(self.acquisition_queue_path),
            },
            "privacy": (
                "Files and review records stay on this machine. "
                "Nothing is uploaded to a cloud service."
            ),
        }

    def analyze(self, document_id: str) -> dict[str, Any]:
        """Return deterministic scholarly artifacts and exact extracted source."""
        with self._lock:
            index, document = self._document(document_id)
            pipeline = ScholarlyAnalysisPipeline(index)
            analysis = pipeline.analyze_paper(document_id)
            summary = pipeline.summarize_paper(document_id)
            checklist = pipeline.build_reproduction_checklist(document_id)
        return {
            "paper": self.paper_summary(document_id),
            "analysis": analysis.to_dict(),
            "summary": summary.to_dict(),
            "checklist": checklist.to_dict(),
            "source": {
                "text": document.text,
                "sections": [section.to_dict() for section in document.sections],
            },
        }

    def ask(
        self,
        *,
        question: str,
        document_id: str | None = None,
        document_ids: tuple[str, ...] | None = None,
        audience_level: str | None = None,
        session_id: str | None = None,
        instruction_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer selected local papers with adaptive presentation metadata."""
        cleaned_question = _nonempty_text(question, "question", maximum=4000)
        if audience_level is not None:
            audience_level = _audience_level(audience_level)
        if document_id is not None:
            document_id = _nonempty_text(document_id, "document_id", maximum=200)
        if document_ids is not None and (
            not isinstance(document_ids, tuple)
            or not document_ids
            or not all(isinstance(item, str) and item.strip() for item in document_ids)
        ):
            raise ValueError("document_ids must be a non-empty tuple of paper IDs.")
        if document_id is not None and document_ids is not None:
            raise ValueError("Use document_id or document_ids, not both.")
        with self._lock:
            index = self._load_index()
            session = None if session_id is None else self.get_session(session_id)
            selected = (
                (document_id,)
                if document_id is not None
                else document_ids
                if document_ids is not None
                else session.selected_paper_ids
                if session is not None and session.selected_paper_ids
                else tuple(document.document_id for document in index.documents)
            )
            known = {document.document_id for document in index.documents}
            unknown = set(selected) - known
            if unknown:
                if document_id is not None:
                    raise ValueError(f"Unknown document_id: {document_id}")
                raise ValueError(f"Unknown selected paper IDs: {sorted(unknown)}.")
            recent_turns = () if session is None else session.turns
            profile = infer_instruction_profile(
                cleaned_question,
                recent_turns=recent_turns,
                stored_preferences=None if session is None else session.preferences,
                explicit_overrides=instruction_overrides,
            )
            resolved_audience = (
                audience_level or profile.canonical_audience or "undergraduate"
            )
            if len(selected) == 1:
                answer_index = index
                filters = SearchFilters(document_id=selected[0])
            elif set(selected) == known:
                answer_index = index
                filters = None
            else:
                answer_index = RetrievalIndex.build(
                    [
                        document
                        for document in index.documents
                        if document.document_id in selected
                    ]
                )
                filters = None
            answer = GroundedAnswerPipeline(answer_index).answer(
                cleaned_question,
                method="extractive",
                top_k=8,
                filters=filters,
            )
            answer_state = answer.to_dict()
            evidence_papers = {
                item["document_id"] for item in answer_state.get("evidence", [])
            }
            missing_comparison_papers = sorted(set(selected) - evidence_papers)
            comparison_requested = profile.include_comparison or len(selected) > 1
            validation = answer_state.get("validation", {})
            sufficiency = answer_state.get("sufficiency", {})
            failures = []
            if answer_state.get("abstained"):
                failures.append("abstained")
            if not evidence_papers:
                failures.append("no_evidence")
            if not validation.get("citations_valid", True):
                failures.append("citation_validation_failure")
            if validation.get("unsupported_claim_count", 0):
                failures.append("unsupported_claim")
            if sufficiency.get("query_term_coverage", 1.0) < 0.5:
                failures.append("low_query_term_coverage")
            comparison_incomplete = comparison_requested and (
                len(selected) < 2 or bool(missing_comparison_papers)
            )
            if comparison_incomplete:
                failures.append("comparison_incomplete")
            record = {
                "interaction_id": _identifier("interaction"),
                "created_at": _timestamp(),
                "session_id": session_id,
                "document_id": selected[0] if len(selected) == 1 else None,
                "paper_ids": list(selected),
                "question": cleaned_question,
                "audience_level": resolved_audience,
                "instruction_profile": profile.to_dict(),
                "answer": answer_state,
                "evidence_selection": {
                    "selected_paper_ids": list(selected),
                    "evidence_paper_ids": sorted(evidence_papers),
                    "independent_of_instruction_profile": True,
                },
                "comparison": {
                    "requested": comparison_requested,
                    "complete": not comparison_incomplete,
                    "missing_evidence_from_paper_ids": missing_comparison_papers,
                    "missing_source_count": (
                        1 if comparison_requested and len(selected) < 2 else 0
                    ),
                    "warning": (
                        "Comparison evidence is incomplete; no claims about missing "
                        "sources should be fabricated."
                        if comparison_incomplete
                        else None
                    ),
                },
                "diagnostics": {
                    "automatic_only": True,
                    "accepted": validation.get("accepted", False),
                    "citation_coverage": validation.get("citation_coverage", 0.0),
                    "query_term_coverage": sufficiency.get("query_term_coverage", 0.0),
                    "failure_categories": failures,
                    "human_review_required": True,
                },
                "conversation_turns": [turn.to_dict() for turn in recent_turns],
            }
            interactions = load_json_list(self.interactions_path)
            interactions.append(record)
            atomic_write_json(self.interactions_path, interactions)
            if session is not None:
                updated_turns = session.turns + (
                    ConversationTurn(
                        "user", cleaned_question, record["interaction_id"]
                    ),
                    ConversationTurn(
                        "assistant",
                        answer_state["answer_text"],
                        record["interaction_id"],
                    ),
                )
                self._sessions[session.session_id] = replace(
                    session,
                    selected_paper_ids=tuple(selected),
                    turns=updated_turns,
                )
        return record

    def save_feedback(
        self,
        *,
        interaction_id: str,
        verdict: str,
        issue_categories: list[str] | tuple[str, ...],
        notes: str = "",
        corrected_answer: str = "",
        audience_level: str | None = None,
    ) -> dict[str, Any]:
        """Save feedback with the immutable question/answer snapshot it reviews."""
        interaction_id = _nonempty_text(
            interaction_id,
            "interaction_id",
            maximum=200,
        )
        if verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VERDICTS)}.")
        if not isinstance(issue_categories, (list, tuple)) or not all(
            isinstance(item, str) for item in issue_categories
        ):
            raise TypeError("issue_categories must be a list of strings.")
        categories = tuple(dict.fromkeys(issue_categories))
        unknown = set(categories) - _ISSUE_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown feedback categories: {sorted(unknown)}.")
        if verdict != "correct" and not categories:
            raise ValueError("Non-correct feedback must select at least one issue.")
        if not isinstance(notes, str) or len(notes) > 10_000:
            raise ValueError("notes must be a string of at most 10000 characters.")
        if not isinstance(corrected_answer, str) or len(corrected_answer) > 20_000:
            raise ValueError(
                "corrected_answer must be a string of at most 20000 characters."
            )
        with self._lock:
            interactions = load_json_list(self.interactions_path)
            matches = [
                item
                for item in interactions
                if item.get("interaction_id") == interaction_id
            ]
            if len(matches) != 1:
                raise ValueError(f"Unknown interaction_id: {interaction_id}")
            interaction = matches[0]
            resolved_audience = _audience_level(
                interaction.get("audience_level", "undergraduate")
                if audience_level is None
                else audience_level
            )
            record = {
                "feedback_id": _identifier("feedback"),
                "created_at": _timestamp(),
                "status": "pending_codex_review",
                "audience_level": resolved_audience,
                "verdict": verdict,
                "issue_categories": list(categories),
                "notes": notes.strip(),
                "corrected_answer": corrected_answer.strip(),
                "interaction": interaction,
            }
            feedback = load_json_list(self.feedback_path)
            feedback.append(record)
            atomic_write_json(self.feedback_path, feedback)
        return record

    def review_question(
        self,
        *,
        question_id: str,
        review_status: str,
        required_concepts: tuple[str, ...] | None = None,
        prohibited_claims: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Approve, reject, or edit one generated benchmark candidate."""
        question_id = _nonempty_text(question_id, "question_id", maximum=200)
        if review_status not in {
            "proposed",
            "human_approved",
            "human_rejected",
            "ambiguous",
            "benchmark_problem",
        }:
            raise ValueError(
                "review_status must be proposed, human_approved, human_rejected, "
                "ambiguous, or benchmark_problem."
            )
        with self._lock:
            items = self._load_candidates()
            positions = [
                position
                for position, item in enumerate(items)
                if item.question_id == question_id
            ]
            if len(positions) != 1:
                raise ValueError(f"Unknown question_id: {question_id}")
            current = items[positions[0]]
            updated = replace(
                current,
                review_status=review_status,
                required_concepts=(
                    current.required_concepts
                    if required_concepts is None
                    else required_concepts
                ),
                prohibited_claims=(
                    current.prohibited_claims
                    if prohibited_claims is None
                    else prohibited_claims
                ),
                metadata={
                    **current.metadata,
                    "human_review_performed": True,
                },
            )
            items[positions[0]] = updated
            self._save_candidates(items)
        return updated.to_dict()

    def run_question(
        self,
        question_id: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one proposed or reviewed question through the trusted baseline."""
        question_id = _nonempty_text(question_id, "question_id", maximum=200)
        with self._lock:
            matches = [
                item
                for item in self._load_candidates()
                if item.question_id == question_id
            ]
        if len(matches) != 1:
            raise ValueError(f"Unknown question_id: {question_id}")
        candidate = matches[0]
        result = self.ask(
            question=candidate.question,
            document_ids=candidate.paper_ids,
            audience_level=candidate.canonical_audience,
            session_id=session_id,
        )
        result["question_id"] = candidate.question_id
        result["question_type"] = candidate.question_type
        result["parent_question_id"] = candidate.parent_question_id
        with self._lock:
            interactions = load_json_list(self.interactions_path)
            for position, item in enumerate(interactions):
                if item.get("interaction_id") == result["interaction_id"]:
                    interactions[position] = result
                    break
            else:
                raise RuntimeError("New interaction was not persisted.")
            atomic_write_json(self.interactions_path, interactions)
        return result

    def _complete_automatic_review_batch(
        self,
        batch: dict[str, Any],
        candidates: list[QuestionCandidate],
    ) -> dict[str, Any]:
        """Run remaining candidates while preserving per-question failures."""
        policy = self._auto_review_policy()
        for candidate in candidates:
            current_batches = self._load_automatic_review_batches()
            current = next(
                item
                for item in current_batches
                if item["batch_id"] == batch["batch_id"]
            )
            if current.get("status") == "stop_requested":
                batch.update(current)
                batch["status"] = "stopped"
                batch["completed_at"] = _timestamp()
                batch["summary"] = summarize_automatic_reviews(batch["reviews"])
                self._persist_automatic_review_batch(batch)
                return batch
            try:
                interaction = self.run_question(candidate.question_id)
                review = propose_automatic_review(
                    interaction,
                    candidate,
                    batch_id=batch["batch_id"],
                    policy=policy,
                )
            except Exception as error:
                review = propose_automatic_failure_review(
                    candidate,
                    batch_id=batch["batch_id"],
                    error=error,
                    policy=policy,
                )
            self._materialize_codex_approval(review)
            batch["reviews"].append(review)
            batch["summary"] = summarize_automatic_reviews(batch["reviews"])
            self._persist_automatic_review_batch(batch)

        batch["status"] = "awaiting_user_review"
        batch["completed_at"] = _timestamp()
        batch.pop("error", None)
        batch["summary"] = summarize_automatic_reviews(batch["reviews"])
        self._persist_automatic_review_batch(batch)
        return batch

    def run_automatic_review_batch(
        self,
        *,
        paper_ids: tuple[str, ...],
        question_ids: tuple[str, ...] | None = None,
        generate_if_empty: bool = True,
        generated_question_count: int = 60,
        uncertain_only: bool = False,
    ) -> dict[str, Any]:
        """Run selected questions and create cautious, editable review drafts."""
        if (
            not isinstance(paper_ids, tuple)
            or not paper_ids
            or not all(isinstance(item, str) and item.strip() for item in paper_ids)
        ):
            raise ValueError("paper_ids must be a non-empty tuple of paper IDs.")
        if question_ids is not None and (
            not isinstance(question_ids, tuple)
            or not question_ids
            or not all(isinstance(item, str) and item.strip() for item in question_ids)
        ):
            raise ValueError("question_ids must be None or a non-empty tuple.")
        if not isinstance(uncertain_only, bool):
            raise TypeError("uncertain_only must be boolean.")
        known_papers = {paper["document_id"] for paper in self.list_papers()}
        unknown_papers = set(paper_ids) - known_papers
        if unknown_papers:
            raise ValueError(f"Unknown selected paper IDs: {sorted(unknown_papers)}.")
        with self._lock:
            candidates = [
                item
                for item in self._load_candidates()
                if item.review_status not in {"rejected", "human_rejected"}
                and set(item.paper_ids) <= set(paper_ids)
            ]
        if not candidates and generate_if_empty:
            if len(paper_ids) != 1:
                raise ValueError(
                    "Automatic generation requires one paper; add multi-paper "
                    "questions explicitly first."
                )
            self.generate_questions(
                paper_id=paper_ids[0], count=generated_question_count
            )
            with self._lock:
                candidates = [
                    item
                    for item in self._load_candidates()
                    if item.review_status not in {"rejected", "human_rejected"}
                    and set(item.paper_ids) <= set(paper_ids)
                ]
        if question_ids is not None:
            requested = set(question_ids)
            candidates = [item for item in candidates if item.question_id in requested]
            missing = requested - {item.question_id for item in candidates}
            if missing:
                raise ValueError(
                    f"Unknown or out-of-scope question IDs: {sorted(missing)}."
                )
        if uncertain_only:
            uncertain_ids = {
                review.get("question_id")
                for batch in self._load_automatic_review_batches()
                for review in batch.get("reviews", [])
                if review.get("second_pass", {}).get("review_status")
                in {"needs_human_review", "ambiguous"}
            }
            candidates = [
                item for item in candidates if item.question_id in uncertain_ids
            ]
        if not candidates:
            raise ValueError("No non-rejected questions exist for this paper scope.")
        candidates = sorted(candidates, key=lambda item: item.question_id)
        if len(candidates) > 200:
            raise ValueError("Automatic review batches are limited to 200 questions.")

        batch_id = _identifier("auto_batch")
        batch = {
            "batch_id": batch_id,
            "created_at": _timestamp(),
            "completed_at": None,
            "paper_ids": list(paper_ids),
            "question_ids": [item.question_id for item in candidates],
            "status": "running",
            "reviewer_type": "deterministic_local_first_pass",
            "semantic_judge_used": False,
            "human_confirmation_required": True,
            "approval_threshold": self._auto_review_policy().approval_threshold,
            "calibration_state": self._auto_review_policy().calibration_state,
            "uncertain_only": uncertain_only,
            "reviews": [],
            "summary": {},
        }
        with self._lock:
            batches = self._load_automatic_review_batches()
            batches.append(batch)
            self._save_automatic_review_batches(batches)

        return self._complete_automatic_review_batch(batch, candidates)

    def resume_automatic_review_batch(self, batch_id: str) -> dict[str, Any]:
        """Resume the unprocessed portion of an older failed batch."""
        batch_id = _nonempty_text(batch_id, "batch_id", maximum=200)
        with self._lock:
            batches = self._load_automatic_review_batches()
            matches = [item for item in batches if item["batch_id"] == batch_id]
            if len(matches) != 1:
                raise ValueError(f"Unknown batch_id: {batch_id}")
            batch = matches[0]
            if batch["status"] not in {"failed", "stopped"}:
                raise ValueError("Only a failed or stopped batch can be resumed.")
            reviewed = {item.get("question_id") for item in batch["reviews"]}
            remaining_ids = [
                item for item in batch["question_ids"] if item not in reviewed
            ]
            candidates_by_id = {
                item.question_id: item for item in self._load_candidates()
            }
            missing = set(remaining_ids) - set(candidates_by_id)
            if missing:
                raise ValueError(
                    "Cannot resume because question candidates are missing: "
                    f"{sorted(missing)}."
                )
            candidates = [candidates_by_id[item] for item in remaining_ids]
            batch["status"] = "running"
            batch.pop("error", None)
            for position, item in enumerate(batches):
                if item["batch_id"] == batch_id:
                    batches[position] = batch
                    break
            self._save_automatic_review_batches(batches)
        return self._complete_automatic_review_batch(batch, candidates)

    def stop_automatic_review_batch(self, batch_id: str) -> dict[str, Any]:
        """Request a running batch to stop after its current question."""
        batch_id = _nonempty_text(batch_id, "batch_id", maximum=200)
        with self._lock:
            batches = self._load_automatic_review_batches()
            matches = [item for item in batches if item["batch_id"] == batch_id]
            if len(matches) != 1:
                raise ValueError(f"Unknown batch_id: {batch_id}")
            batch = matches[0]
            if batch["status"] != "running":
                raise ValueError("Only a running batch can be stopped.")
            batch["status"] = "stop_requested"
            batch["stop_requested_at"] = _timestamp()
            self._save_automatic_review_batches(batches)
        return batch

    def create_audit_sample(
        self,
        *,
        sample_fraction: float = 0.10,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Create and persist a deterministic risk-aware audit queue."""
        with self._lock:
            reviews = [
                {
                    "review_id": review["review_id"],
                    "confidence": review.get("second_pass", {}).get(
                        "confidence", review.get("proposed_confidence", 0.0)
                    ),
                    "mandatory_human_categories": review.get("second_pass", {}).get(
                        "mandatory_human_categories", []
                    ),
                    "novel_failure": bool(review.get("novel_failure", False)),
                }
                for batch in self._load_automatic_review_batches()
                for review in batch.get("reviews", [])
            ]
            result = select_audit_sample(
                reviews,
                sample_fraction=sample_fraction,
                seed=seed,
                approval_threshold=self._auto_review_policy().approval_threshold,
            )
            result["created_at"] = _timestamp()
            atomic_write_json(self.audit_queue_path, result)
        return result

    def rerun_automatic_review(self, review_id: str) -> dict[str, Any]:
        """Rerun one failed or uncertain item without changing other decisions."""
        review_id = _nonempty_text(review_id, "review_id", maximum=200)
        with self._lock:
            batches = self._load_automatic_review_batches()
            matches = [
                (batch, position, review)
                for batch in batches
                for position, review in enumerate(batch.get("reviews", []))
                if review.get("review_id") == review_id
            ]
            if len(matches) != 1:
                raise ValueError(f"Unknown review_id: {review_id}")
            batch, position, previous = matches[0]
            candidate_matches = [
                item
                for item in self._load_candidates()
                if item.question_id == previous.get("question_id")
            ]
            if len(candidate_matches) != 1:
                raise ValueError("The review's question candidate is unavailable.")
            candidate = candidate_matches[0]
        try:
            interaction = self.run_question(candidate.question_id)
            review = propose_automatic_review(
                interaction,
                candidate,
                batch_id=batch["batch_id"],
                policy=self._auto_review_policy(),
            )
        except Exception as error:
            review = propose_automatic_failure_review(
                candidate,
                batch_id=batch["batch_id"],
                error=error,
                policy=self._auto_review_policy(),
            )
        review["rerun_count"] = int(previous.get("rerun_count", 0)) + 1
        review["rerun_at"] = _timestamp()
        batch["reviews"][position] = review
        batch["summary"] = summarize_automatic_reviews(batch["reviews"])
        self._persist_automatic_review_batch(batch)
        return review

    def set_auto_approval_enabled(self, *, enabled: bool) -> dict[str, Any]:
        """Explicitly enable approval only after calibration metrics qualify."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean.")
        with self._lock:
            state = self._review_policy_state()
            if enabled:
                eligible = calibration_report(
                    self._calibration_records(),
                    explicit_enable=False,
                    integrity=state.get("integrity", {}),
                    approval_threshold=float(state.get("approval_threshold", 0.95)),
                )
                if eligible["state"] != "calibration_active":
                    raise ValueError(
                        "Automatic approval cannot be enabled until calibration "
                        "meets every policy threshold."
                    )
            state["explicit_enable"] = enabled
            state["updated_at"] = _timestamp()
            atomic_write_json(self.review_policy_path, state)
        return self._calibration()

    def bulk_auto_review(self, *, eligible_only: bool = True) -> dict[str, Any]:
        """Run pending candidates only after explicit calibration activation."""
        if eligible_only is not True:
            raise ValueError(
                "Bulk review requires eligible_only=True in production mode."
            )
        calibration = self._calibration()
        if calibration["state"] != "auto_approval_enabled":
            raise ValueError(
                "Bulk automatic approval is locked: "
                + "; ".join(calibration["reasons"])
            )
        previous_question_ids = {
            review.get("question_id")
            for batch in self._load_automatic_review_batches()
            for review in batch.get("reviews", [])
        }
        candidates = [
            item
            for item in self._load_candidates()
            if item.review_status == "proposed"
            and item.question_id not in previous_question_ids
            and not bool(item.metadata.get("test_only", False))
        ]
        if not candidates:
            raise ValueError(
                "No eligible pending questions are available for bulk review."
            )
        if len(candidates) > 200:
            candidates = candidates[:200]
        paper_ids = tuple(
            sorted({paper for item in candidates for paper in item.paper_ids})
        )
        batch = self.run_automatic_review_batch(
            paper_ids=paper_ids,
            question_ids=tuple(item.question_id for item in candidates),
            generate_if_empty=False,
        )
        audit = self.create_audit_sample(sample_fraction=0.10, seed=42)
        return {
            "batch": batch,
            "audit": audit,
            "eligible_only": True,
            "human_audit_still_required": True,
        }

    def finalize_automatic_review_batch(
        self,
        *,
        batch_id: str,
        reviewer: str,
        decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Save explicitly accepted/edited drafts as the user's review records."""
        batch_id = _nonempty_text(batch_id, "batch_id", maximum=200)
        reviewer = _nonempty_text(reviewer, "reviewer", maximum=500)
        if (
            not isinstance(decisions, list)
            or not decisions
            or not all(isinstance(item, dict) for item in decisions)
        ):
            raise ValueError("decisions must be a non-empty list of objects.")
        with self._lock:
            batches = self._load_automatic_review_batches()
            positions = [
                position
                for position, item in enumerate(batches)
                if item["batch_id"] == batch_id
            ]
            if len(positions) != 1:
                raise ValueError(f"Unknown batch_id: {batch_id}")
            batch = batches[positions[0]]
            if batch["status"] not in {"awaiting_user_review", "partially_saved"}:
                raise ValueError(
                    "Only a completed batch awaiting user review can be saved."
                )
            reviews = {item["review_id"]: item for item in batch["reviews"]}
            decision_ids = [item.get("review_id") for item in decisions]
            if any(not isinstance(item, str) for item in decision_ids):
                raise ValueError("Every decision requires a review_id.")
            if len(decision_ids) != len(set(decision_ids)):
                raise ValueError("Duplicate review decisions are not allowed.")
            unknown = set(decision_ids) - set(reviews)
            if unknown:
                raise ValueError(f"Unknown automatic review IDs: {sorted(unknown)}.")

        normalized_decisions: list[dict[str, Any]] = []
        for decision in decisions:
            review = reviews[decision["review_id"]]
            if review.get("decision") in {
                "saved_as_user_review",
                "excluded_by_user",
            }:
                continue
            accepted = decision.get("accepted", True)
            if not isinstance(accepted, bool):
                raise TypeError("decision accepted must be a boolean.")
            if not accepted:
                normalized_decisions.append({"review": review, "accepted": False})
                continue
            if not bool(review.get("saveable", True)):
                raise ValueError(
                    "A failed answer attempt cannot be saved as a user review; "
                    "exclude it and rerun the question after fixing the error."
                )
            label = decision.get("review_label", review["proposed_label"])
            if not isinstance(label, str) or label not in REVIEW_LABELS:
                raise ValueError(
                    f"review_label must be one of {sorted(REVIEW_LABELS)}."
                )
            corrected_answer = decision.get(
                "corrected_answer", review["proposed_corrected_answer"]
            )
            corrected_answer = _nonempty_text(
                corrected_answer, "corrected_answer", maximum=50_000
            )
            list_fields = {
                "required_facts": decision.get(
                    "required_facts", review["proposed_required_facts"]
                ),
                "prohibited_claims": decision.get(
                    "prohibited_claims", review["proposed_prohibited_claims"]
                ),
                "evidence_ids": decision.get(
                    "evidence_ids", review["proposed_evidence_ids"]
                ),
            }
            for field_name, value in list_fields.items():
                if not isinstance(value, list) or not all(
                    isinstance(item, str) and item.strip() for item in value
                ):
                    raise TypeError(
                        f"decision {field_name} must be a list of non-empty strings."
                    )
            normalized_decisions.append(
                {
                    "review": review,
                    "accepted": True,
                    "label": label,
                    "corrected_answer": corrected_answer,
                    "required_facts": tuple(list_fields["required_facts"]),
                    "prohibited_claims": tuple(list_fields["prohibited_claims"]),
                    "evidence_ids": tuple(list_fields["evidence_ids"]),
                }
            )

        saved = 0
        excluded = 0
        for decision in normalized_decisions:
            review = decision["review"]
            if not decision["accepted"]:
                review["decision"] = "excluded_by_user"
                review["decided_by"] = reviewer
                excluded += 1
                continue
            label = decision["label"]
            corrected_answer = decision["corrected_answer"]
            required_facts = decision["required_facts"]
            prohibited_claims = decision["prohibited_claims"]
            evidence_ids = decision["evidence_ids"]
            correction = self.review_interaction(
                interaction_id=review["interaction_id"],
                review_label=label,
                corrected_answer=corrected_answer,
                required_facts=required_facts,
                prohibited_claims=prohibited_claims,
                replacement_evidence_ids=evidence_ids,
                notes=(
                    "Accepted from deterministic automatic-review draft by "
                    f"{reviewer}. Original confidence: "
                    f"{review['proposed_confidence']:.2f}."
                ),
            )
            review["decision"] = "saved_as_user_review"
            review["decided_by"] = reviewer
            review["final_label"] = label
            review["final_corrected_answer"] = corrected_answer
            review["final_required_facts"] = list(required_facts)
            review["final_prohibited_claims"] = list(prohibited_claims)
            review["final_evidence_ids"] = list(evidence_ids)
            review["correction_example_id"] = correction["example_id"]
            review["user_edited"] = any(
                (
                    label != review["proposed_label"],
                    corrected_answer != review["proposed_corrected_answer"],
                    list(required_facts) != review["proposed_required_facts"],
                    list(prohibited_claims) != review["proposed_prohibited_claims"],
                    list(evidence_ids) != review["proposed_evidence_ids"],
                )
            )
            saved += 1

        batch["reviews"] = list(reviews.values())
        batch["summary"] = summarize_automatic_reviews(batch["reviews"])
        if all(
            item["decision"] in {"saved_as_user_review", "excluded_by_user"}
            for item in batch["reviews"]
        ):
            batch["status"] = "saved"
            batch["saved_at"] = _timestamp()
        else:
            batch["status"] = "partially_saved"
        batch["last_decided_by"] = reviewer
        with self._lock:
            batches = self._load_automatic_review_batches()
            for position, item in enumerate(batches):
                if item["batch_id"] == batch_id:
                    batches[position] = batch
                    break
            self._save_automatic_review_batches(batches)
        return {
            "batch": batch,
            "saved_review_count": saved,
            "excluded_count": excluded,
            "corrections_remain_proposed": True,
        }

    def review_interaction(
        self,
        *,
        interaction_id: str,
        review_label: str,
        corrected_answer: str | None = None,
        required_facts: tuple[str, ...] = (),
        prohibited_claims: tuple[str, ...] = (),
        replacement_evidence_ids: tuple[str, ...] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Create a correction suggestion from a human-labelled interaction."""
        interaction_id = _nonempty_text(interaction_id, "interaction_id", maximum=200)
        with self._lock:
            interactions = load_json_list(self.interactions_path)
            matches = [
                item
                for item in interactions
                if item.get("interaction_id") == interaction_id
            ]
            if len(matches) != 1:
                raise ValueError(f"Unknown interaction_id: {interaction_id}")
            snapshot = dict(matches[0])
            answer = dict(snapshot["answer"])
            evidence = list(answer.get("evidence", []))
            if replacement_evidence_ids is not None:
                if not isinstance(replacement_evidence_ids, tuple):
                    raise TypeError("replacement_evidence_ids must be a tuple.")
                by_id = {}
                for item in evidence:
                    for key in (
                        item.get("evidence_id"),
                        item.get("chunk_id"),
                        item.get("label"),
                    ):
                        if isinstance(key, str):
                            by_id[key] = item
                missing = set(replacement_evidence_ids) - set(by_id)
                if missing:
                    index = self._load_index()
                    selected_papers = set(snapshot.get("paper_ids", []))
                    documents = {item.document_id: item for item in index.documents}
                    chunks = {
                        item.chunk_id: item
                        for item in index.chunks
                        if item.document_id in selected_papers
                    }
                    for chunk_id in tuple(missing):
                        chunk = chunks.get(chunk_id)
                        if chunk is None:
                            continue
                        document = documents[chunk.document_id]
                        by_id[chunk_id] = {
                            "evidence_id": chunk.chunk_id,
                            "label": "replacement",
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "source_name": document.source_name,
                            "title": document.title,
                            "heading_path": list(chunk.heading_path),
                            "selected_text": chunk.text,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "replacement_from_exact_index_chunk": True,
                        }
                        missing.remove(chunk_id)
                if missing:
                    raise ValueError(
                        f"Unknown replacement evidence IDs: {sorted(missing)}."
                    )
                answer["evidence"] = []
                for position, item_id in enumerate(replacement_evidence_ids, start=1):
                    replacement = dict(by_id[item_id])
                    replacement["label"] = f"C{position}"
                    answer["evidence"].append(replacement)
                snapshot["answer"] = answer
            candidate = propose_correction(
                snapshot,
                review_label=review_label,
                corrected_answer=corrected_answer,
                required_facts=required_facts,
                prohibited_claims=prohibited_claims,
                notes=notes,
            )
            corrections = self._load_corrections()
            by_id = {item.example_id: item for item in corrections}
            by_id[candidate.example_id] = candidate
            self._save_corrections(list(by_id.values()))
        return candidate.to_dict()

    def approve_correction(
        self,
        *,
        example_id: str,
        reviewer: str,
        final_answer: str | None = None,
    ) -> dict[str, Any]:
        """Approve one inspected correction, optionally with a final answer edit."""
        example_id = _nonempty_text(example_id, "example_id", maximum=200)
        with self._lock:
            corrections = self._load_corrections()
            positions = [
                position
                for position, item in enumerate(corrections)
                if item.example_id == example_id
            ]
            if len(positions) != 1:
                raise ValueError(f"Unknown example_id: {example_id}")
            candidate = corrections[positions[0]]
            if final_answer is not None:
                candidate = replace(
                    candidate,
                    final_answer=_nonempty_text(
                        final_answer, "final_answer", maximum=50_000
                    ),
                )
            approved = approve_correction_example(candidate, reviewer=reviewer)
            corrections[positions[0]] = approved
            self._save_corrections(corrections)
            self._record_human_outcome(
                example_id=approved.example_id,
                human_approved=True,
                reviewer=reviewer.strip(),
            )
            interactions = load_json_list(self.interactions_path)
            source = next(
                (
                    item
                    for item in interactions
                    if item.get("interaction_id") == approved.source_interaction_id
                ),
                None,
            )
            if source is not None:
                source_session_id = source.get("session_id")
                if source_session_id in self._sessions:
                    session = self._sessions[source_session_id]
                    self._sessions[source_session_id] = replace(
                        session,
                        turns=session.turns
                        + (
                            ConversationTurn(
                                "assistant",
                                f"Human-approved correction: {approved.final_answer}",
                                approved.source_interaction_id,
                            ),
                        ),
                    )
        return approved.to_dict()

    def edit_correction(
        self,
        *,
        example_id: str,
        final_answer: str,
    ) -> dict[str, Any]:
        """Edit a proposed correction without approving it."""
        example_id = _nonempty_text(example_id, "example_id", maximum=200)
        cleaned_answer = _nonempty_text(final_answer, "final_answer", maximum=50_000)
        with self._lock:
            corrections = self._load_corrections()
            positions = [
                position
                for position, item in enumerate(corrections)
                if item.example_id == example_id
            ]
            if len(positions) != 1:
                raise ValueError(f"Unknown example_id: {example_id}")
            current = corrections[positions[0]]
            if current.review_status != "proposed":
                raise ValueError("Only proposed corrections can be edited.")
            updated = replace(current, final_answer=cleaned_answer)
            corrections[positions[0]] = updated
            self._save_corrections(corrections)
        return updated.to_dict()

    def audit_codex_approval(
        self,
        *,
        example_id: str,
        reviewer: str,
        passed: bool,
    ) -> dict[str, Any]:
        """Record an explicit human audit without relabeling Codex as human gold."""
        example_id = _nonempty_text(example_id, "example_id", maximum=200)
        reviewer = _nonempty_text(reviewer, "reviewer", maximum=500)
        if not isinstance(passed, bool):
            raise TypeError("passed must be boolean.")
        with self._lock:
            corrections = self._load_corrections()
            positions = [
                position
                for position, item in enumerate(corrections)
                if item.example_id == example_id
            ]
            if len(positions) != 1:
                raise ValueError(f"Unknown example_id: {example_id}")
            current = corrections[positions[0]]
            if current.review_status != "codex_approved":
                raise ValueError("Only a Codex-approved example can be audited here.")
            updated = replace(
                current,
                review_status="codex_approved" if passed else "human_rejected",
                metadata={
                    **current.metadata,
                    "audit_status": "human_confirmed" if passed else "overturned",
                    "audited_by": reviewer,
                    "audited_at": _timestamp(),
                },
            )
            corrections[positions[0]] = updated
            self._save_corrections(corrections)
            self._record_human_outcome(
                example_id=example_id,
                human_approved=passed,
                reviewer=reviewer,
            )
        return updated.to_dict()

    def reject_correction(
        self,
        *,
        example_id: str,
        reviewer: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Explicitly reject one automatic correction suggestion."""
        example_id = _nonempty_text(example_id, "example_id", maximum=200)
        reviewer = _nonempty_text(reviewer, "reviewer", maximum=500)
        if not isinstance(reason, str) or len(reason) > 10_000:
            raise ValueError("reason must be a string of at most 10000 characters.")
        with self._lock:
            corrections = self._load_corrections()
            positions = [
                position
                for position, item in enumerate(corrections)
                if item.example_id == example_id
            ]
            if len(positions) != 1:
                raise ValueError(f"Unknown example_id: {example_id}")
            current = corrections[positions[0]]
            if current.review_status not in {"proposed", "codex_approved"}:
                raise ValueError(
                    "Only proposed or Codex-approved corrections can be rejected."
                )
            updated = replace(
                current,
                review_status="human_rejected",
                metadata={
                    **current.metadata,
                    "rejected_by": reviewer,
                    "rejection_reason": reason.strip(),
                },
            )
            corrections[positions[0]] = updated
            self._save_corrections(corrections)
            self._record_human_outcome(
                example_id=updated.example_id,
                human_approved=False,
                reviewer=reviewer,
            )
        return updated.to_dict()

    def export_training_dataset(
        self,
        *,
        output: str | Path | None = None,
        seed: int = 0,
        manual_paper_splits: dict[str, str] | None = None,
        trust_tier: str = "human-and-audited",
    ) -> dict[str, Any]:
        """Export one explicit trust tier with paper-level split protection."""
        with self._lock:
            corrections = tuple(self._load_corrections())
            dataset = build_dataset(
                corrections,
                dataset_version="1.2.2",
                seed=seed,
                manual_paper_splits=manual_paper_splits,
                trust_tier=trust_tier,
            )
            destination = self.dataset_path if output is None else Path(output)
            save_dataset(dataset, destination)
        return {
            "output": str(destination),
            "dataset": dataset.to_dict(),
            "report": dataset_report(dataset),
            "trust_tier": trust_tier,
        }
