"""HTTP tests for the expanded local Paper Training Lab API."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

from localml_scholar.review_app.server import create_server
from localml_scholar.review_app.service import ReviewService
from localml_scholar.review_app.storage import atomic_write_json


def _request(server, method, path, value=None):
    body = None if value is None else json.dumps(value).encode()
    headers = (
        {}
        if body is None
        else {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
    )
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, json.loads(payload)


def test_training_lab_http_question_review_workflow(tmp_path):
    service = ReviewService(tmp_path)
    paper = service.add_paper(
        filename="paper.md",
        payload=(
            b"# Paper\n\n## Method\n"
            b"Training uses Adam optimizer with learning rate 0.001.\n"
        ),
    )
    server = create_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, session = _request(
            server,
            "POST",
            "/api/sessions",
            {"selected_paper_ids": [paper["document_id"]]},
        )
        assert status == 201

        status, questions = _request(
            server,
            "POST",
            f"/api/papers/{paper['document_id']}/questions/generate",
            {"count": 40},
        )
        assert status == 201
        assert len(questions) == 40

        status, manual = _request(
            server,
            "POST",
            "/api/questions/manual",
            {
                "question": "Which optimizer and learning rate are used?",
                "paper_ids": [paper["document_id"]],
            },
        )
        assert status == 201

        status, automatic = _request(
            server,
            "POST",
            "/api/automation/run",
            {
                "paper_ids": [paper["document_id"]],
                "question_ids": [manual["question_id"]],
                "generate_if_empty": False,
            },
        )
        assert status == 201
        assert automatic["status"] == "awaiting_user_review"
        assert automatic["semantic_judge_used"] is False
        saved_batches = json.loads(
            service.automatic_reviews_path.read_text(encoding="utf-8")
        )
        saved_batches[0]["status"] = "failed"
        saved_batches[0]["error"] = "legacy all-or-nothing failure"
        atomic_write_json(service.automatic_reviews_path, saved_batches)
        status, resumed = _request(
            server,
            "POST",
            f"/api/automation/batches/{automatic['batch_id']}/resume",
        )
        assert status == 200
        assert resumed["status"] == "awaiting_user_review"
        assert "error" not in resumed
        status, finalized = _request(
            server,
            "POST",
            f"/api/automation/batches/{automatic['batch_id']}/finalize",
            {
                "reviewer": "test-reviewer",
                "decisions": [
                    {
                        "review_id": automatic["reviews"][0]["review_id"],
                        "accepted": False,
                    }
                ],
            },
        )
        assert status == 200
        assert finalized["excluded_count"] == 1

        status, interaction = _request(
            server,
            "POST",
            f"/api/questions/{manual['question_id']}/run",
            {"session_id": session["session_id"]},
        )
        assert status == 201
        assert interaction["instruction_profile"]

        evidence_ids = [
            item["evidence_id"] for item in interaction["answer"]["evidence"]
        ]
        status, correction = _request(
            server,
            "POST",
            f"/api/interactions/{interaction['interaction_id']}/review",
            {
                "review_label": "correct",
                "corrected_answer": interaction["answer"]["answer_text"],
                "replacement_evidence_ids": evidence_ids,
            },
        )
        assert status == 201
        assert correction["review_status"] == "proposed"

        status, _approved = _request(
            server,
            "POST",
            f"/api/corrections/{correction['example_id']}/approve",
            {"reviewer": "test-reviewer"},
        )
        assert status == 200

        status, exported = _request(server, "POST", "/api/dataset/export", {"seed": 0})
        assert status == 201
        assert exported["report"]["paper_level_leakage"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
