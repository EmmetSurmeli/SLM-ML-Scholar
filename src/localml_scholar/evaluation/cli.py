"""Command-line entry point for local benchmark review and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from localml_scholar.answering import (
    EvidenceSelectionConfig,
    GroundedAnswerPipeline,
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
)
from localml_scholar.evaluation.attention_starter import (
    generate_attention_starter_benchmark,
)
from localml_scholar.evaluation.benchmark import (
    apply_review_decisions,
    generate_candidate_benchmark,
)
from localml_scholar.evaluation.comparison import compare_evaluation_runs
from localml_scholar.evaluation.human_review import build_review_queue
from localml_scholar.evaluation.reports import (
    render_benchmark_review,
    render_comparison_report,
    render_evaluation_summary,
    render_failure_report,
    render_review_queue,
)
from localml_scholar.evaluation.runner import EvaluationRunner
from localml_scholar.evaluation.schemas import EvaluationConfig
from localml_scholar.evaluation.serialization import (
    load_benchmark,
    load_evaluation_run,
    save_benchmark,
    save_comparison_report,
    save_evaluation_run,
    save_review_records,
)
from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.serialization import atomic_write_text


def _load_json_object(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON input does not exist: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Input is not valid UTF-8 JSON: {path}") from error
    if not isinstance(state, dict):
        raise ValueError("JSON input must contain one top-level object.")
    return state


def _write_markdown(path: Path, text: str) -> None:
    if path.suffix.casefold() not in {".md", ".markdown"}:
        raise ValueError("Markdown output path must end with .md or .markdown.")
    atomic_write_text(path, text)


def _generate_candidates(args: argparse.Namespace) -> None:
    index = RetrievalIndex.load(args.index)
    benchmark = generate_candidate_benchmark(
        index,
        args.document_id,
        name=args.name,
        benchmark_version=args.benchmark_version,
    )
    save_benchmark(benchmark, args.output)
    if args.review_report is not None:
        _write_markdown(args.review_report, render_benchmark_review(benchmark))
    print(
        f"Saved {len(benchmark.questions)} proposed questions to {args.output}. "
        "They are not approved gold data."
    )


def _review_benchmark(args: argparse.Namespace) -> None:
    candidates = load_benchmark(args.candidates)
    decisions = _load_json_object(args.decisions)
    reviewed = apply_review_decisions(candidates, decisions)
    save_benchmark(reviewed, args.output)
    if args.review_report is not None:
        _write_markdown(args.review_report, render_benchmark_review(reviewed))
    print(
        f"Saved reviewed benchmark with {len(reviewed.approved_questions)} "
        f"approved/edited questions to {args.output}."
    )


def _attention_starter(args: argparse.Namespace) -> None:
    index = RetrievalIndex.load(args.index)
    benchmark = generate_attention_starter_benchmark(
        index,
        args.document_id,
        benchmark_version=args.benchmark_version,
    )
    save_benchmark(benchmark, args.output)
    if args.review_report is not None:
        _write_markdown(args.review_report, render_benchmark_review(benchmark))
    print(
        f"Saved {len(benchmark.questions)} untrusted Attention-paper candidates "
        f"to {args.output}; human review is mandatory."
    )


def _run(args: argparse.Namespace) -> None:
    index = RetrievalIndex.load(args.index)
    benchmark = load_benchmark(args.benchmark, index=index)
    answerer = None
    checkpoint_sha256 = None
    tokenizer_sha256 = None
    if args.method in {"generative", "generative_with_extractive_fallback"}:
        if args.checkpoint is None:
            raise ValueError("Generative methods require --checkpoint.")
        answerer = GroundedGenerativeAnswerer.from_checkpoint(
            args.checkpoint,
            config=GroundedGenerationConfig(
                maximum_new_tokens=args.maximum_new_tokens,
                greedy=True,
                seed=args.seed,
            ),
        )
        checkpoint_sha256 = answerer.checkpoint_sha256
        tokenizer_sha256 = answerer.tokenizer.state_hash()
    evidence_config = EvidenceSelectionConfig(
        retrieval_method=args.retriever,
        retrieval_top_k=args.top_k,
        evidence_top_k=min(args.evidence_top_k, args.top_k),
    )
    pipeline = GroundedAnswerPipeline(
        index,
        evidence_config=evidence_config,
        generative_answerer=answerer,
    )
    configuration = EvaluationConfig(
        mode=args.method,
        retrieval_method=args.retriever,
        top_k=args.top_k,
        model_checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        random_seed=args.seed,
        retrieval_parameters={
            "top_k": args.top_k,
            "index_sha256": index.index_sha256,
        },
        evidence_selection_settings=evidence_config.to_dict(),
        acceptance_policy=pipeline.acceptance_config.to_dict(),
    )
    existing = None if args.resume is None else load_evaluation_run(args.resume)
    run = EvaluationRunner(
        benchmark,
        index,
        configuration,
        answering_pipeline=pipeline,
    ).run(existing_run=existing)
    save_evaluation_run(run, args.output)
    print(
        f"Saved {len(run.question_results)} question results to {args.output}; "
        f"automatic pass rate={run.aggregate_metrics['automatic_pass_rate']:.4f}."
    )


def _report(args: argparse.Namespace) -> None:
    run = load_evaluation_run(args.run)
    benchmark = load_benchmark(args.benchmark)
    if run.benchmark_sha256 != benchmark.benchmark_sha256:
        raise ValueError("Run and benchmark identities do not match.")
    if args.kind == "summary":
        rendered = render_evaluation_summary(run, benchmark)
    else:
        rendered = render_failure_report(run, benchmark)
    _write_markdown(args.output, rendered)
    print(f"Saved {args.kind} report to {args.output}.")


def _queue(args: argparse.Namespace) -> None:
    run = load_evaluation_run(args.run)
    benchmark = load_benchmark(args.benchmark)
    if run.benchmark_sha256 != benchmark.benchmark_sha256:
        raise ValueError("Run and benchmark identities do not match.")
    records = build_review_queue(
        run,
        benchmark,
        pass_sample_fraction=args.pass_sample_fraction,
        random_seed=args.seed,
    )
    save_review_records(records, args.output)
    if args.review_report is not None:
        _write_markdown(args.review_report, render_review_queue(records))
    print(f"Saved {len(records)} human-review records to {args.output}.")


def _compare(args: argparse.Namespace) -> None:
    baseline = load_evaluation_run(args.baseline)
    candidate = load_evaluation_run(args.candidate)
    report = compare_evaluation_runs(baseline, candidate)
    if args.output is not None:
        save_comparison_report(report, args.output)
    if args.markdown is not None:
        _write_markdown(args.markdown, render_comparison_report(report))
    if args.output is None and args.markdown is None:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print("Saved regression comparison.")


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic evaluation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate-candidates",
        help="Propose untrusted source-linked benchmark questions.",
    )
    generate.add_argument("--index", type=Path, required=True)
    generate.add_argument("--document-id", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--review-report", type=Path)
    generate.add_argument("--name")
    generate.add_argument("--benchmark-version", default="0.1-candidates")
    generate.set_defaults(handler=_generate_candidates)

    review = commands.add_parser(
        "review-benchmark",
        help="Apply explicit JSON approval/edit/rejection decisions.",
    )
    review.add_argument("--candidates", type=Path, required=True)
    review.add_argument("--decisions", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--review-report", type=Path)
    review.set_defaults(handler=_review_benchmark)

    starter = commands.add_parser(
        "attention-starter",
        help="Bind the untrusted 33-question Attention-paper starter to an index.",
    )
    starter.add_argument("--index", type=Path, required=True)
    starter.add_argument("--document-id", required=True)
    starter.add_argument("--output", type=Path, required=True)
    starter.add_argument("--review-report", type=Path)
    starter.add_argument("--benchmark-version", default="0.1-attention-candidates")
    starter.set_defaults(handler=_attention_starter)

    run = commands.add_parser("run", help="Run a human-approved benchmark.")
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--index", type=Path, required=True)
    run.add_argument(
        "--method",
        choices=(
            "retrieval_only",
            "top_passage",
            "extractive",
            "generative",
            "generative_with_extractive_fallback",
        ),
        default="extractive",
    )
    run.add_argument(
        "--retriever",
        choices=("bm25", "tfidf", "semantic", "hybrid", "hybrid_reranked"),
        default="bm25",
    )
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--evidence-top-k", type=int, default=4)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--maximum-new-tokens", type=int, default=64)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--resume", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(handler=_run)

    report = commands.add_parser("report", help="Render a Markdown run report.")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--benchmark", type=Path, required=True)
    report.add_argument("--kind", choices=("summary", "failures"), default="summary")
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(handler=_report)

    queue = commands.add_parser(
        "build-review-queue",
        help="Select failures, risky cases, and sampled automatic passes.",
    )
    queue.add_argument("--run", type=Path, required=True)
    queue.add_argument("--benchmark", type=Path, required=True)
    queue.add_argument("--pass-sample-fraction", type=float, default=0.1)
    queue.add_argument("--seed", type=int, default=0)
    queue.add_argument("--output", type=Path, required=True)
    queue.add_argument("--review-report", type=Path)
    queue.set_defaults(handler=_queue)

    compare = commands.add_parser(
        "compare",
        help="Compare exact question-level results across two runs.",
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--markdown", type=Path)
    compare.set_defaults(handler=_compare)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one evaluation command with concise failures and no silent recovery."""
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        args.handler(args)
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
