#!/usr/bin/env python3
"""Compare local retrieval and answer configurations on one approved benchmark."""

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

from localml_scholar.answering import (  # noqa: E402
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
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
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation_config_comparison",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    index = RetrievalIndex.load(args.index)
    benchmark = load_benchmark(args.benchmark, index=index)
    answerer = (
        None
        if args.checkpoint is None
        else GroundedGenerativeAnswerer.from_checkpoint(
            args.checkpoint,
            config=GroundedGenerationConfig(greedy=True, seed=0),
        )
    )
    methods = ["top_passage", "extractive"]
    if answerer is not None:
        methods.extend(["generative", "generative_with_extractive_fallback"])
    rows = []
    for retriever in ("bm25", "semantic", "hybrid", "hybrid_reranked"):
        for method in methods:
            generative = method.startswith("generative")
            configuration = EvaluationConfig(
                mode=method,
                retrieval_method=retriever,
                top_k=args.top_k,
                model_checkpoint_sha256=(
                    answerer.checkpoint_sha256 if generative else None
                ),
                tokenizer_sha256=(
                    answerer.tokenizer.state_hash() if generative else None
                ),
                retrieval_parameters={"top_k": args.top_k},
            )
            pipeline = GroundedAnswerPipeline(
                index,
                evidence_config=EvidenceSelectionConfig(
                    retrieval_method=retriever,
                    retrieval_top_k=args.top_k,
                    evidence_top_k=min(4, args.top_k),
                ),
                generative_answerer=answerer,
            )
            run = EvaluationRunner(
                benchmark,
                index,
                configuration,
                answering_pipeline=pipeline,
            ).run()
            name = f"{retriever}__{method}"
            save_evaluation_run(
                run,
                args.output_directory / f"{name}.json",
            )
            rows.append(
                {
                    "name": name,
                    "run_id": run.run_id,
                    "configuration": configuration.to_dict(),
                    "metrics": run.aggregate_metrics,
                }
            )
    summary = {
        "control": (
            "Every row uses the same approved benchmark, index snapshot, "
            "question IDs, and top-k."
        ),
        "generative_checkpoint_supplied": answerer is not None,
        "runs": rows,
    }
    atomic_write_text(
        args.output_directory / "comparison.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(f"Saved {len(rows)} controlled runs to {args.output_directory}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
