"""Compare controlled extractive answering under four retrieval methods."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.grounded_qa_fixture import (  # noqa: E402
    build_grounded_fixture_index,
    load_grounded_fixture_questions,
)
from localml_scholar.answering import (  # noqa: E402
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
)
from localml_scholar.answering.evaluation import (  # noqa: E402
    evaluate_grounded_answers,
)
from localml_scholar.retrieval import SemanticRetrievalConfig  # noqa: E402
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def run_grounded_retriever_evaluation(output_directory: Path) -> dict:
    """Evaluate citation and support controls with each retrieval method."""
    output_directory.mkdir(parents=True, exist_ok=True)
    lexical = build_grounded_fixture_index()
    dimensions = min(8, len(lexical.chunks), len(lexical.vocabulary))
    index = lexical.enrich_semantic(SemanticRetrievalConfig(dimensions=dimensions))
    questions = load_grounded_fixture_questions()
    reports = {}
    for method in ("bm25", "semantic", "hybrid", "hybrid_reranked"):
        pipeline = GroundedAnswerPipeline(
            index,
            evidence_config=EvidenceSelectionConfig(retrieval_method=method),
        )
        answers = {
            question.question_id: pipeline.answer(
                question.question,
                method="extractive",
            )
            for question in questions
        }
        reports[method] = {
            "answer_evaluation": evaluate_grounded_answers(
                questions,
                answers,
            ).to_dict(),
            "answers": {
                question_id: answer.to_dict()
                for question_id, answer in sorted(answers.items())
            },
        }
    summary = {
        "experiment": "grounded_answer_retriever_regression",
        "answer_method": "extractive",
        "generative_evaluation": ("not_run_without_an_explicit_checkpoint"),
        "index_sha256": index.index_sha256,
        "semantic_sha256": index.semantic_index.semantic_sha256,
        "reports": reports,
    }
    atomic_write_text(
        output_directory / "grounded_retriever_evaluation.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/grounded_retriever_evaluation"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    summary = run_grounded_retriever_evaluation(parse_args(arguments).output_directory)
    print(
        json.dumps(
            {
                method: report["answer_evaluation"]["aggregate"]
                for method, report in summary["reports"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
