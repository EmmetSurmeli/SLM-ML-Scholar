"""CLI tests for question generation, review runs, and dataset reports."""

from __future__ import annotations

import json

from localml_scholar.evaluation.cli import main
from localml_scholar.retrieval import RetrievalIndex, ingest_markdown
from localml_scholar.training_data import approve_correction, propose_correction


def _index(tmp_path):
    document = ingest_markdown(
        "# Tiny Paper\n\n## Method\nTraining uses Adam at learning rate 0.001.\n",
        source="tiny.md",
    )
    index = RetrievalIndex.build([document])
    path = tmp_path / "index.json"
    index.save(path)
    return document, path


def test_generate_and_run_paper_question_cli(tmp_path):
    document, index_path = _index(tmp_path)
    questions_path = tmp_path / "questions.json"
    assert (
        main(
            [
                "generate-paper-questions",
                "--index",
                str(index_path),
                "--paper",
                document.document_id,
                "--count",
                "40",
                "--output",
                str(questions_path),
            ]
        )
        == 0
    )
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    assert questions["candidate_only"] is True
    assert len(questions["questions"]) == 40
    assert all(item["review_status"] == "proposed" for item in questions["questions"])

    run_path = tmp_path / "review-results.json"
    assert (
        main(
            [
                "run-review-set",
                "--index",
                str(index_path),
                "--paper",
                document.document_id,
                "--questions",
                str(questions_path),
                "--output",
                str(run_path),
            ]
        )
        == 0
    )
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert len(run["results"]) == 40
    assert all(
        item["review_status"] == "pending_human_review" for item in run["results"]
    )


def test_export_and_report_cli_requires_explicit_approved_only(tmp_path, capsys):
    interaction = {
        "interaction_id": "i1",
        "paper_ids": ["paper-1"],
        "question": "What optimizer is used?",
        "answer": {
            "answer_text": "Adam. [C1]",
            "evidence": [
                {
                    "label": "C1",
                    "document_id": "paper-1",
                    "selected_text": "Adam is used.",
                }
            ],
        },
    }
    approved = approve_correction(
        propose_correction(interaction, review_label="correct"),
        reviewer="test",
    )
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps({"corrections": [approved.to_dict()]}), encoding="utf-8"
    )
    dataset = tmp_path / "dataset.json"
    assert (
        main(
            [
                "export-training-data",
                "--reviews",
                str(reviews),
                "--output",
                str(dataset),
            ]
        )
        == 2
    )
    assert "--approved-only" in capsys.readouterr().err
    assert (
        main(
            [
                "export-training-data",
                "--reviews",
                str(reviews),
                "--approved-only",
                "--output",
                str(dataset),
            ]
        )
        == 0
    )
    report_path = tmp_path / "report.json"
    assert (
        main(
            [
                "dataset-report",
                "--dataset",
                str(dataset),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["paper_level_leakage"] is False
    assert report["diversity"]["example_count"] == 1
