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
            if parsed.path == "/api/questions":
                request = self._read_json()
                result = self.service.ask(
                    question=request.get("question"),
                    document_id=request.get("document_id"),
                    audience_level=request.get("audience_level", "undergraduate"),
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
        description="Run the localhost-only LocalML Scholar review lab."
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
    print(f"LocalML Scholar Review Lab: http://127.0.0.1:{server.server_port}")
    print(f"Feedback file: {service.feedback_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review lab.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
