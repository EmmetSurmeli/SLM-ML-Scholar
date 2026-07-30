"""HTTP boundary tests for the localhost-only review application."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

from localml_scholar.review_app.server import create_server
from localml_scholar.review_app.service import ReviewService


def _request(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    content_type = response.getheader("Content-Type")
    connection.close()
    return response.status, content_type, payload


def test_static_application_and_state_api(tmp_path):
    service = ReviewService(tmp_path)
    server = create_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, content_type, payload = _request(server, "GET", "/")
        assert status == 200
        assert content_type == "text/html; charset=utf-8"
        assert b"LocalML Scholar" in payload
        assert b"Save feedback for Codex" in payload

        status, content_type, payload = _request(server, "GET", "/api/state")
        assert status == 200
        assert content_type == "application/json; charset=utf-8"
        assert json.loads(payload)["papers"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_upload_question_and_feedback_http_workflow(tmp_path):
    service = ReviewService(tmp_path)
    server = create_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        paper_text = (
            b"# Optimizer Note\n\n## Method\n"
            b"Training uses Adam optimizer with learning rate 0.001.\n"
        )
        status, _, payload = _request(
            server,
            "POST",
            "/api/papers",
            paper_text,
            {
                "Content-Length": str(len(paper_text)),
                "X-Filename": "note.md",
            },
        )
        assert status == 201
        paper = json.loads(payload)

        question = json.dumps(
            {
                "question": "Which optimizer and learning rate are used?",
                "document_id": paper["document_id"],
                "audience_level": "researcher",
            }
        ).encode()
        status, _, payload = _request(
            server,
            "POST",
            "/api/questions",
            question,
            {
                "Content-Length": str(len(question)),
                "Content-Type": "application/json",
            },
        )
        assert status == 201
        interaction = json.loads(payload)
        assert interaction["audience_level"] == "researcher"

        feedback = json.dumps(
            {
                "interaction_id": interaction["interaction_id"],
                "audience_level": "beginner",
                "verdict": "correct",
                "issue_categories": [],
                "notes": "",
                "corrected_answer": "",
            }
        ).encode()
        status, _, payload = _request(
            server,
            "POST",
            "/api/feedback",
            feedback,
            {
                "Content-Length": str(len(feedback)),
                "Content-Type": "application/json",
            },
        )
        assert status == 201
        saved_feedback = json.loads(payload)
        assert saved_feedback["status"] == "pending_codex_review"
        assert saved_feedback["audience_level"] == "beginner"
        assert service.feedback_path.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_rejects_non_loopback_binding(tmp_path):
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(ReviewService(tmp_path), host="0.0.0.0")
