#!/usr/bin/env python3
"""Run one explicit human-approved local-paper benchmark."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from localml_scholar.evaluation.reports import (  # noqa: E402
    render_evaluation_summary,
    render_failure_report,
)
from localml_scholar.evaluation.runner import EvaluationRunner  # noqa: E402
from localml_scholar.evaluation.schemas import EvaluationConfig  # noqa: E402
from localml_scholar.evaluation.serialization import (  # noqa: E402
    load_benchmark,
    save_evaluation_run,
)
from localml_scholar.retrieval import RetrievalIndex  # noqa: E402
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument(
        "--method",
        choices=("retrieval_only", "top_passage", "extractive"),
        default="extractive",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf", "semantic", "hybrid", "hybrid_reranked"),
        default="bm25",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "real_paper_evaluation",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.benchmark is None or args.index is None:
        print(
            "No run performed: provide both --benchmark and --index. "
            "This experiment will not fabricate a paper or gold annotations.",
            file=sys.stderr,
        )
        return 2
    index = RetrievalIndex.load(args.index)
    benchmark = load_benchmark(args.benchmark, index=index)
    configuration = EvaluationConfig(
        mode=args.method,
        retrieval_method=args.retriever,
        top_k=args.top_k,
        retrieval_parameters={"top_k": args.top_k},
    )
    run = EvaluationRunner(benchmark, index, configuration).run()
    destination = args.output_directory
    save_evaluation_run(run, destination / "evaluation_run.json")
    atomic_write_text(
        destination / "summary.md",
        render_evaluation_summary(run, benchmark),
    )
    atomic_write_text(
        destination / "failures.md",
        render_failure_report(run, benchmark),
    )
    print(
        f"Evaluated {len(run.question_results)} approved questions; "
        f"results are in {destination}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
