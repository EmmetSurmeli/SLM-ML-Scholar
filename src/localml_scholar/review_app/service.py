"""Application service for local ingestion, questioning, analysis, and feedback."""

from __future__ import annotations

import re
import threading
import uuid
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
from localml_scholar.review_app.storage import (
    atomic_write_bytes,
    atomic_write_json,
    load_json_list,
)
from localml_scholar.scholarly import ScholarlyAnalysisPipeline

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
_VERDICTS = {"correct", "partially_correct", "incorrect"}
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

    def state(self) -> dict[str, Any]:
        """Return the complete lightweight browser bootstrap state."""
        with self._lock:
            feedback = load_json_list(self.feedback_path)
            interactions = load_json_list(self.interactions_path)
            papers = self.list_papers()
        return {
            "papers": papers,
            "feedback": list(reversed(feedback[-50:])),
            "interaction_count": len(interactions),
            "feedback_count": len(feedback),
            "storage": {
                "uploads": str(self.upload_directory),
                "index": str(self.index_path),
                "interactions": str(self.interactions_path),
                "feedback": str(self.feedback_path),
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
        audience_level: str = "undergraduate",
    ) -> dict[str, Any]:
        """Answer from cited local evidence and persist a reviewable snapshot."""
        cleaned_question = _nonempty_text(question, "question", maximum=4000)
        audience_level = _audience_level(audience_level)
        if document_id is not None:
            document_id = _nonempty_text(document_id, "document_id", maximum=200)
        with self._lock:
            index = self._load_index()
            if document_id is not None and not any(
                document.document_id == document_id for document in index.documents
            ):
                raise ValueError(f"Unknown document_id: {document_id}")
            filters = (
                None if document_id is None else SearchFilters(document_id=document_id)
            )
            answer = GroundedAnswerPipeline(index).answer(
                cleaned_question,
                method="extractive",
                top_k=8,
                filters=filters,
            )
            record = {
                "interaction_id": _identifier("interaction"),
                "created_at": _timestamp(),
                "document_id": document_id,
                "question": cleaned_question,
                "audience_level": audience_level,
                "answer": answer.to_dict(),
            }
            interactions = load_json_list(self.interactions_path)
            interactions.append(record)
            atomic_write_json(self.interactions_path, interactions)
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
