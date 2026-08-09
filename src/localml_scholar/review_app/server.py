"""Localhost HTTP server for the LocalML Scholar review lab."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from localml_scholar.review_app.service import ReviewService

_STATIC_DIRECTORY = Path(__file__).parent / "static"
_MAX_REQUEST_BYTES = 30 * 1024 * 1024


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Serve the static application and its small same-origin JSON API."""

    service: ReviewService
    server_version = "LocalMLScholarReview/1.0"

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _read_body(self) -> bytes:
        value = self.headers.get("Content-Length")
        if value is None:
            raise ValueError("Content-Length is required.")
        try:
            length = int(value)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer.") from error
        if length <= 0:
            raise ValueError("Request body must not be empty.")
        if length > _MAX_REQUEST_BYTES:
            raise ValueError("Request body exceeds the 30 MiB limit.")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("Request body ended unexpectedly.")
        return payload

    def _read_json(self) -> dict[str, Any]:
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise ValueError("Content-Type must be application/json.")
        try:
            value = json.loads(self._read_body().decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("JSON request must be valid UTF-8.") from error
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(value, dict):
            raise ValueError("JSON request must be an object.")
        return value

    def _static(self, filename: str) -> None:
        path = _STATIC_DIRECTORY / filename
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/state":
                self._json(self.service.state())
                return
            if parsed.path == "/api/questions":
                query = parse_qs(parsed.query)
                self._json(
                    self.service.list_questions(
                        paper_id=query.get("paper_id", [None])[0]
                    )
                )
                return
            if parsed.path.startswith("/api/papers/") and parsed.path.endswith(
                "/analysis"
            ):
                document_id = unquote(
                    parsed.path.removeprefix("/api/papers/").removesuffix("/analysis")
                ).strip("/")
                self._json(self.service.analyze(document_id))
                return
            if parsed.path in {"/", "/index.html"}:
                self._static("index.html")
                return
            if parsed.path in {"/app.js", "/styles.css"}:
                self._static(parsed.path.removeprefix("/"))
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found.")
        except (TypeError, ValueError, RuntimeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/papers":
                filename_header = self.headers.get("X-Filename")
                if filename_header is None:
                    raise ValueError("X-Filename is required.")
                query = parse_qs(parsed.query)
                title = query.get("title", [None])[0]
                result = self.service.add_paper(
                    filename=unquote(filename_header),
                    payload=self._read_body(),
                    title=title,
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/sessions":
                request = self._read_json()
                result = self.service.create_session(
                    selected_paper_ids=tuple(request.get("selected_paper_ids", [])),
                    preferences=request.get("preferences"),
                    persist_preferences=request.get("persist_preferences", False),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/sessions/"):
                session_id = unquote(parsed.path.removeprefix("/api/sessions/")).strip(
                    "/"
                )
                request = self._read_json()
                selected = request.get("selected_paper_ids")
                result = self.service.update_session(
                    session_id,
                    selected_paper_ids=(None if selected is None else tuple(selected)),
                    preferences=request.get("preferences"),
                    persist_preferences=request.get("persist_preferences"),
                )
                self._json(result)
                return
            if parsed.path.startswith("/api/papers/") and parsed.path.endswith(
                "/questions/generate"
            ):
                paper_id = unquote(
                    parsed.path.removeprefix("/api/papers/").removesuffix(
                        "/questions/generate"
                    )
                ).strip("/")
                request = self._read_json()
                result = self.service.generate_questions(
                    paper_id=paper_id,
                    count=request.get("count", 60),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/questions/manual":
                request = self._read_json()
                result = self.service.add_question(
                    question=request.get("question"),
                    paper_ids=tuple(request.get("paper_ids", [])),
                    question_type=request.get("question_type", "user_authored"),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/evidence/search":
                request = self._read_json()
                result = self.service.search_evidence(
                    query=request.get("query"),
                    paper_ids=tuple(request.get("paper_ids", [])),
                    top_k=request.get("top_k", 10),
                )
                self._json(result)
                return
            if parsed.path == "/api/automation/run":
                request = self._read_json()
                question_ids = request.get("question_ids")
                result = self.service.run_automatic_review_batch(
                    paper_ids=tuple(request.get("paper_ids", [])),
                    question_ids=(
                        None if question_ids is None else tuple(question_ids)
                    ),
                    generate_if_empty=request.get("generate_if_empty", True),
                    generated_question_count=request.get(
                        "generated_question_count", 60
                    ),
                    uncertain_only=request.get("uncertain_only", False),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/calibration/sample":
                request = self._read_json()
                self._json(
                    self.service.create_calibration_sample(
                        target_count=request.get("target_count", 50),
                        seed=request.get("seed", 42),
                    ),
                    HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/calibration/rerun-historical":
                request = self._read_json()
                review_ids = request.get("review_ids")
                self._json(
                    self.service.rerun_historical_reviews(
                        review_ids=None if review_ids is None else tuple(review_ids)
                    ),
                    HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith(
                "/api/calibration/reviews/"
            ) and parsed.path.endswith("/decision"):
                review_id = unquote(
                    parsed.path.removeprefix("/api/calibration/reviews/").removesuffix(
                        "/decision"
                    )
                ).strip("/")
                request = self._read_json()
                self._json(
                    self.service.record_calibration_decision(
                        review_id=review_id,
                        action=request.get("action"),
                        reviewer=request.get("reviewer"),
                        edits=request.get("edits"),
                    ),
                    HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith(
                "/api/calibration/pairs/"
            ) and parsed.path.endswith("/approve-training"):
                pair_id = unquote(
                    parsed.path.removeprefix("/api/calibration/pairs/").removesuffix(
                        "/approve-training"
                    )
                ).strip("/")
                request = self._read_json()
                self._json(
                    self.service.approve_calibration_for_training(
                        pair_id=pair_id, reviewer=request.get("reviewer")
                    )
                )
                return
            if parsed.path == "/api/calibration/bulk-auto-review":
                request = self._read_json()
                self._json(
                    self.service.bulk_auto_review(
                        eligible_only=request.get("eligible_only", True)
                    ),
                    HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/acquisition":
                request = self._read_json()
                self._json(
                    self.service.add_acquisition_item(
                        title=request.get("title"),
                        doi=request.get("doi"),
                        arxiv_id=request.get("arxiv_id"),
                        citation=request.get("citation"),
                        reason=request.get("reason"),
                        category=request.get("category"),
                    ),
                    HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith("/api/acquisition/"):
                item_id = unquote(parsed.path.removeprefix("/api/acquisition/")).strip(
                    "/"
                )
                request = self._read_json()
                self._json(
                    self.service.update_acquisition_item(
                        item_id=item_id, status=request.get("status")
                    )
                )
                return
            if parsed.path == "/api/automation/audit-sample":
                request = self._read_json()
                result = self.service.create_audit_sample(
                    sample_fraction=request.get("sample_fraction", 0.10),
                    seed=request.get("seed", 42),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/automation/enable":
                request = self._read_json()
                self._json(
                    self.service.set_auto_approval_enabled(
                        enabled=request.get("enabled")
                    )
                )
                return
            if parsed.path.startswith(
                "/api/automation/reviews/"
            ) and parsed.path.endswith("/rerun"):
                review_id = unquote(
                    parsed.path.removeprefix("/api/automation/reviews/").removesuffix(
                        "/rerun"
                    )
                ).strip("/")
                self._read_json()
                self._json(self.service.rerun_automatic_review(review_id))
                return
            if parsed.path.startswith(
                "/api/automation/batches/"
            ) and parsed.path.endswith("/finalize"):
                batch_id = unquote(
                    parsed.path.removeprefix("/api/automation/batches/").removesuffix(
                        "/finalize"
                    )
                ).strip("/")
                request = self._read_json()
                result = self.service.finalize_automatic_review_batch(
                    batch_id=batch_id,
                    reviewer=request.get("reviewer"),
                    decisions=request.get("decisions"),
                )
                self._json(result)
                return
            if parsed.path.startswith(
                "/api/automation/batches/"
            ) and parsed.path.endswith("/resume"):
                batch_id = unquote(
                    parsed.path.removeprefix("/api/automation/batches/").removesuffix(
                        "/resume"
                    )
                ).strip("/")
                result = self.service.resume_automatic_review_batch(batch_id)
                self._json(result)
                return
            if parsed.path.startswith(
                "/api/automation/batches/"
            ) and parsed.path.endswith("/stop"):
                batch_id = unquote(
                    parsed.path.removeprefix("/api/automation/batches/").removesuffix(
                        "/stop"
                    )
                ).strip("/")
                self._read_json()
                self._json(self.service.stop_automatic_review_batch(batch_id))
                return
            if parsed.path.startswith("/api/questions/") and parsed.path.endswith(
                "/run"
            ):
                question_id = unquote(
                    parsed.path.removeprefix("/api/questions/").removesuffix("/run")
                ).strip("/")
                request = self._read_json()
                result = self.service.run_question(
                    question_id,
                    session_id=request.get("session_id"),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/questions/") and parsed.path.endswith(
                "/review"
            ):
                question_id = unquote(
                    parsed.path.removeprefix("/api/questions/").removesuffix("/review")
                ).strip("/")
                request = self._read_json()
                concepts = request.get("required_concepts")
                prohibited = request.get("prohibited_claims")
                result = self.service.review_question(
                    question_id=question_id,
                    review_status=request.get("review_status"),
                    required_concepts=(None if concepts is None else tuple(concepts)),
                    prohibited_claims=(
                        None if prohibited is None else tuple(prohibited)
                    ),
                )
                self._json(result)
                return
            if parsed.path.startswith("/api/questions/") and parsed.path.endswith(
                "/variations"
            ):
                question_id = unquote(
                    parsed.path.removeprefix("/api/questions/").removesuffix(
                        "/variations"
                    )
                ).strip("/")
                self._read_json()
                result = self.service.propose_question_variations(question_id)
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/questions":
                request = self._read_json()
                document_ids = request.get("document_ids")
                result = self.service.ask(
                    question=request.get("question"),
                    document_id=request.get("document_id"),
                    document_ids=(
                        None if document_ids is None else tuple(document_ids)
                    ),
                    audience_level=request.get("audience_level"),
                    session_id=request.get("session_id"),
                    instruction_overrides=request.get("instruction_overrides"),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/interactions/") and parsed.path.endswith(
                "/review"
            ):
                interaction_id = unquote(
                    parsed.path.removeprefix("/api/interactions/").removesuffix(
                        "/review"
                    )
                ).strip("/")
                request = self._read_json()
                replacement = request.get("replacement_evidence_ids")
                result = self.service.review_interaction(
                    interaction_id=interaction_id,
                    review_label=request.get("review_label"),
                    corrected_answer=request.get("corrected_answer"),
                    required_facts=tuple(request.get("required_facts", [])),
                    prohibited_claims=tuple(request.get("prohibited_claims", [])),
                    replacement_evidence_ids=(
                        None if replacement is None else tuple(replacement)
                    ),
                    notes=request.get("notes", ""),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/corrections/") and parsed.path.endswith(
                "/approve"
            ):
                example_id = unquote(
                    parsed.path.removeprefix("/api/corrections/").removesuffix(
                        "/approve"
                    )
                ).strip("/")
                request = self._read_json()
                result = self.service.approve_correction(
                    example_id=example_id,
                    reviewer=request.get("reviewer"),
                    final_answer=request.get("final_answer"),
                )
                self._json(result)
                return
            if parsed.path.startswith("/api/corrections/") and parsed.path.endswith(
                "/edit"
            ):
                example_id = unquote(
                    parsed.path.removeprefix("/api/corrections/").removesuffix("/edit")
                ).strip("/")
                request = self._read_json()
                result = self.service.edit_correction(
                    example_id=example_id,
                    final_answer=request.get("final_answer"),
                )
                self._json(result)
                return
            if parsed.path.startswith("/api/corrections/") and parsed.path.endswith(
                "/audit"
            ):
                example_id = unquote(
                    parsed.path.removeprefix("/api/corrections/").removesuffix("/audit")
                ).strip("/")
                request = self._read_json()
                self._json(
                    self.service.audit_codex_approval(
                        example_id=example_id,
                        reviewer=request.get("reviewer"),
                        passed=request.get("passed"),
                    )
                )
                return
            if parsed.path.startswith("/api/corrections/") and parsed.path.endswith(
                "/reject"
            ):
                example_id = unquote(
                    parsed.path.removeprefix("/api/corrections/").removesuffix(
                        "/reject"
                    )
                ).strip("/")
                request = self._read_json()
                result = self.service.reject_correction(
                    example_id=example_id,
                    reviewer=request.get("reviewer"),
                    reason=request.get("reason", ""),
                )
                self._json(result)
                return
            if parsed.path == "/api/dataset/export":
                request = self._read_json()
                result = self.service.export_training_dataset(
                    output=request.get("output"),
                    seed=request.get("seed", 0),
                    manual_paper_splits=request.get("manual_paper_splits"),
                    trust_tier=request.get("trust_tier", "human-and-audited"),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/feedback":
                request = self._read_json()
                result = self.service.save_feedback(
                    interaction_id=request.get("interaction_id"),
                    verdict=request.get("verdict"),
                    issue_categories=request.get("issue_categories", []),
                    notes=request.get("notes", ""),
                    corrected_answer=request.get("corrected_answer", ""),
                    audience_level=request.get("audience_level"),
                )
                self._json(result, HTTPStatus.CREATED)
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found.")
        except (TypeError, ValueError, RuntimeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))

    def log_message(self, format: str, *args: object) -> None:
        """Log the numeric client address without a slow reverse-DNS lookup."""
        print(f"[review] {self.client_address[0]} {format % args}")


def create_server(
    service: ReviewService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a localhost-only review server without starting it."""
    if host != "127.0.0.1":
        raise ValueError("The review application binds only to 127.0.0.1.")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535].")

    class BoundHandler(ReviewRequestHandler):
        pass

    BoundHandler.service = service
    return ThreadingHTTPServer((host, port), BoundHandler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the localhost-only LocalML Scholar Paper Training Lab."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    """Run the review server until interrupted."""
    arguments = _parser().parse_args()
    service = ReviewService(arguments.repository)
    server = create_server(service, port=arguments.port)
    print(f"LocalML Scholar Paper Training Lab: http://127.0.0.1:{server.server_port}")
    print(f"Local workspace: {service.output_directory}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review lab.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
