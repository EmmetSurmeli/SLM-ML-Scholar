from __future__ import annotations

import json
from pathlib import Path

from experiments.compare_answer_methods import main as comparison_main
from experiments.evaluate_extractive_answering import main as extractive_main
from experiments.evaluate_grounded_generation import main as generative_main
from experiments.grounded_qa_fixture import load_grounded_fixture_questions
from localml_scholar.answering import GroundedAnswerPipeline
from localml_scholar.answering.cli import main as answer_main
from localml_scholar.answering.evaluation import evaluate_grounded_answers
from localml_scholar.retrieval import RetrievalIndex


def test_authored_fixture_metrics_are_exact(
    grounded_index: RetrievalIndex,
) -> None:
    questions = load_grounded_fixture_questions()
    pipeline = GroundedAnswerPipeline(grounded_index)
    answers = {
        question.question_id: pipeline.answer(question.question)
        for question in questions
    }

    report = evaluate_grounded_answers(questions, answers)

    assert report.aggregate["answerability_correct"] == 1.0
    assert report.aggregate["citation_validity"] == 1.0
    assert report.aggregate["citation_coverage"] == 1.0
    assert report.aggregate["citation_recall"] == 1.0
    assert report.aggregate["abstention_precision"] == 1.0
    assert report.aggregate["abstention_recall"] == 1.0


def test_evaluation_zero_abstention_denominators_are_zero(
    grounded_index: RetrievalIndex,
) -> None:
    question = load_grounded_fixture_questions()[0]
    answer = GroundedAnswerPipeline(grounded_index).answer(question.question)

    report = evaluate_grounded_answers(
        (question,),
        {question.question_id: answer},
    )

    assert report.aggregate["abstention_precision"] == 0.0
    assert report.aggregate["abstention_recall"] == 0.0


def test_answer_cli_json_human_and_saved_artifact(
    grounded_index: RetrievalIndex,
    tmp_path: Path,
    capsys,
) -> None:
    index_path = grounded_index.save(tmp_path / "index.json")
    answer_path = tmp_path / "answer.json"

    assert (
        answer_main(
            [
                "--index",
                str(index_path),
                "--question",
                "How does a decoder prevent future token leakage?",
                "--json",
                "--save",
                str(answer_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "answer"
    assert payload["answer"]["validation"]["accepted"]
    assert answer_path.is_file()

    assert (
        answer_main(
            [
                "--index",
                str(index_path),
                "--question",
                "quantum hardware topology",
                "--verbose",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "status: abstained" in output
    assert "sources:" in output
    assert "validation:" in output


def test_extractive_experiment_writes_inspectable_summary(
    tmp_path: Path,
    capsys,
) -> None:
    destination = tmp_path / "extractive"

    assert extractive_main(["--output-directory", str(destination)]) == 0
    capsys.readouterr()
    state = json.loads(
        (destination / "extractive_evaluation.json").read_text(encoding="utf-8")
    )

    assert state["method"] == "extractive"
    assert state["answer_evaluation"]["aggregate"]["answerability_correct"] == 1.0
    assert len(state["answers"]) == 10


def test_generative_experiments_refuse_to_fabricate_without_checkpoint(
    tmp_path: Path,
    capsys,
) -> None:
    assert generative_main(["--output-directory", str(tmp_path / "gen")]) == 2
    assert "No checkpoint supplied" in capsys.readouterr().err
    assert comparison_main(["--output-directory", str(tmp_path / "cmp")]) == 2
    assert "requires --checkpoint" in capsys.readouterr().err
