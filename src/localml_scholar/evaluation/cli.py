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
from localml_scholar.retrieval import RetrievalIndex, SearchFilters
from localml_scholar.review_app.service import ReviewService
from localml_scholar.serialization import atomic_write_text
from localml_scholar.training_data import (
    GroundedInstructionExample,
    build_dataset,
    dataset_report,
    generate_paper_questions,
    infer_instruction_profile,
    load_dataset,
    save_dataset,
)


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


def _write_json(path: Path, value: object) -> None:
    if path.suffix.casefold() != ".json":
        raise ValueError("JSON output path must end with .json.")
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
    )


def _resolve_paper(index: RetrievalIndex, value: str):
    matches = [
        document
        for document in index.documents
        if value
        in {
            document.document_id,
            document.source_name,
            document.title,
        }
    ]
    if len(matches) != 1:
        raise ValueError(
            "--paper must uniquely match a document ID, source name, or exact title."
        )
    return matches[0]


def _generate_paper_questions(args: argparse.Namespace) -> None:
    index = RetrievalIndex.load(args.index)
    paper = _resolve_paper(index, args.paper)
    candidates = generate_paper_questions(
        paper.document_id,
        paper.title or paper.source_name,
        count=args.count,
        section_titles=tuple(
            section.heading or "Untitled section" for section in paper.sections
        ),
    )
    _write_json(
        args.output,
        {
            "format_version": "1.0",
            "candidate_only": True,
            "paper_id": paper.document_id,
            "questions": [item.to_dict() for item in candidates],
        },
    )
    print(
        f"Saved {len(candidates)} proposed questions to {args.output}; "
        "none are human-approved."
    )


def _run_review_set(args: argparse.Namespace) -> None:
    index = RetrievalIndex.load(args.index)
    paper = _resolve_paper(index, args.paper)
    state = _load_json_object(args.questions)
    raw_questions = state.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("Questions artifact must contain a questions list.")
    pipeline = GroundedAnswerPipeline(index)
    results = []
    for item in raw_questions:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError("Every question must be a JSON object with question text.")
        question = item["question"]
        answer = pipeline.answer(
            question,
            method="extractive",
            top_k=8,
            filters=SearchFilters(document_id=paper.document_id),
        )
        results.append(
            {
                "question_id": item.get("question_id"),
                "paper_ids": [paper.document_id],
                "question": question,
                "question_type": item.get("question_type", "unknown"),
                "instruction_profile": infer_instruction_profile(question).to_dict(),
                "answer": answer.to_dict(),
                "review_status": "pending_human_review",
            }
        )
    _write_json(
        args.output,
        {
            "format_version": "1.0",
            "paper_id": paper.document_id,
            "results": results,
        },
    )
    print(f"Saved {len(results)} reviewable results to {args.output}.")


def _export_training_data(args: argparse.Namespace) -> None:
    if args.repository is not None:
        if args.reviews is not None:
            raise ValueError("Use either --repository or --reviews, not both.")
        result = ReviewService(args.repository).export_training_dataset(
            output=args.output,
            seed=args.seed,
            trust_tier=args.trust_tier,
        )
        print(
            f"Saved {len(result['dataset']['examples'])} {args.trust_tier} "
            f"examples to {args.output}."
        )
        return
    if args.reviews is None:
        raise ValueError("export-training-data requires --repository or --reviews.")
    try:
        state = json.loads(args.reviews.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reviews input does not exist: {args.reviews}"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Reviews input must be valid UTF-8 JSON.") from error
    raw = (
        state.get("examples", state.get("corrections"))
        if isinstance(state, dict)
        else state
    )
    if not isinstance(raw, list):
        raise ValueError("Reviews input must contain an examples/corrections list.")
    examples = tuple(GroundedInstructionExample.from_dict(item) for item in raw)
    if not args.approved_only:
        raise ValueError("Training exports require --approved-only.")
    dataset = build_dataset(
        examples,
        dataset_version=args.dataset_version,
        seed=args.seed,
        trust_tier=args.trust_tier,
    )
    save_dataset(dataset, args.output)
    print(f"Saved {len(dataset.examples)} {args.trust_tier} examples to {args.output}.")


def _dataset_report(args: argparse.Namespace) -> None:
    report = dataset_report(load_dataset(args.dataset))
    if args.output is None:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        _write_json(args.output, report)
        print(f"Saved dataset report to {args.output}.")


def _auto_review_workspace(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    paper_ids = tuple(args.paper_id or ())
    question_ids = None
    if args.all_pending:
        if paper_ids:
            raise ValueError(
                "Use either --paper/--paper-id or --all-pending, not both."
            )
        pending = [
            item
            for item in service.list_questions()
            if item["review_status"]
            in {
                "proposed",
                "needs_human_review",
                "ambiguous",
            }
        ]
        if not pending:
            raise ValueError("No pending questions are available for auto-review.")
        paper_ids = tuple(
            sorted({paper_id for item in pending for paper_id in item["paper_ids"]})
        )
        question_ids = tuple(item["question_id"] for item in pending)
    if not paper_ids:
        raise ValueError("auto-review requires --paper or --all-pending.")
    batch = service.run_automatic_review_batch(
        paper_ids=paper_ids,
        question_ids=question_ids,
        generate_if_empty=args.generate_if_empty,
        generated_question_count=args.generated_question_count,
    )
    if args.output is not None:
        _write_json(args.output, batch)
    print(
        f"Second-pass reviewed {len(batch['reviews'])} examples; "
        f"status={batch['status']}; calibration={batch['calibration_state']}."
    )


def _audit_workspace(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    result = service.create_audit_sample(
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    if args.output is not None:
        _write_json(args.output, result)
    print(
        f"Selected {result['selected_count']} of {result['population_count']} "
        "reviews for deterministic audit."
    )


def _calibration_workspace(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    result = service.state()["calibration"]
    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        _write_json(args.output, result)
        print(f"Saved calibration report to {args.output}.")


def _calibration_sample_workspace(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    result = service.create_calibration_sample(target_count=args.count, seed=args.seed)
    if args.output is not None:
        _write_json(args.output, result)
    print(
        f"Selected {result['selected_count']} of {result['population_count']} "
        f"reviews; coverage gaps={len(result['coverage_gaps'])}."
    )


def _rerun_historical_workspace(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    review_ids = None
    if args.sample_only:
        review_ids = tuple(service.state()["calibration_sample"].get("review_ids", []))
        if not review_ids:
            raise ValueError("The calibration sample is empty.")
    result = service.rerun_historical_reviews(review_ids=review_ids)
    if args.output is not None:
        _write_json(args.output, result)
    print(f"Appended {result['rerun_count']} non-destructive historical reruns.")


def _calibration_status_workspace(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    result = service.state()["calibration"]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


def _enable_auto_approval_workspace(args: argparse.Namespace) -> None:
    result = ReviewService(args.repository).set_auto_approval_enabled(enabled=True)
    print(f"Automatic approval state: {result['state']}.")


def _bulk_auto_review_workspace(args: argparse.Namespace) -> None:
    result = ReviewService(args.repository).bulk_auto_review(
        eligible_only=args.eligible_only
    )
    print(
        f"Bulk reviewed {len(result['batch']['reviews'])} eligible questions; "
        f"audit queue={result['audit']['selected_count']}."
    )


def _export_trust_tier(args: argparse.Namespace) -> None:
    service = ReviewService(args.repository)
    result = service.export_training_dataset(
        output=args.output,
        seed=args.seed,
        trust_tier=args.trust_tier,
    )
    print(
        f"Saved {len(result['dataset']['examples'])} {args.trust_tier} examples "
        f"to {result['output']}."
    )


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
        help="Bind the expanded untrusted Attention-paper starter to an index.",
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

    paper_questions = commands.add_parser(
        "generate-paper-questions",
        help="Generate 40-80 unapproved candidate questions for one local paper.",
    )
    paper_questions.add_argument(
        "--index", type=Path, default=Path("outputs/review_app/index.json")
    )
    paper_questions.add_argument("--paper", required=True)
    paper_questions.add_argument("--count", type=int, default=60)
    paper_questions.add_argument("--output", type=Path, required=True)
    paper_questions.set_defaults(handler=_generate_paper_questions)

    review_set = commands.add_parser(
        "run-review-set",
        help="Run proposed paper questions through the deterministic baseline.",
    )
    review_set.add_argument(
        "--index", type=Path, default=Path("outputs/review_app/index.json")
    )
    review_set.add_argument("--paper", required=True)
    review_set.add_argument("--questions", type=Path, required=True)
    review_set.add_argument("--output", type=Path, required=True)
    review_set.set_defaults(handler=_run_review_set)

    export = commands.add_parser(
        "export-training-data",
        help="Export an explicit grounded-instruction trust tier.",
    )
    export.add_argument("--reviews", type=Path)
    export.add_argument("--repository", type=Path)
    export.add_argument("--approved-only", action="store_true")
    export.add_argument("--dataset-version", default="1.2.2")
    export.add_argument(
        "--trust-tier",
        choices=("human-only", "human-and-audited", "include-codex-approved"),
        default="human-and-audited",
    )
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=_export_training_data)

    data_report = commands.add_parser(
        "dataset-report",
        help="Report paper splits and instruction-dataset diversity.",
    )
    data_report.add_argument("--dataset", type=Path, required=True)
    data_report.add_argument("--output", type=Path)
    data_report.set_defaults(handler=_dataset_report)

    auto_review = commands.add_parser(
        "auto-review",
        help="Run confidence-gated second-pass review in a local workspace.",
    )
    auto_review.add_argument("--repository", type=Path, default=Path.cwd())
    auto_review.add_argument("--paper-id", "--paper", action="append")
    auto_review.add_argument("--all-pending", action="store_true")
    auto_review.add_argument("--generate-if-empty", action="store_true")
    auto_review.add_argument("--generated-question-count", type=int, default=60)
    auto_review.add_argument("--output", type=Path)
    auto_review.set_defaults(handler=_auto_review_workspace)

    audit = commands.add_parser(
        "audit-sample",
        help="Create the deterministic 10%% plus mandatory-risk audit queue.",
    )
    audit.add_argument("--repository", type=Path, default=Path.cwd())
    audit.add_argument("--sample-fraction", "--rate", type=float, default=0.10)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(handler=_audit_workspace)

    calibration = commands.add_parser(
        "calibration-report",
        help="Report auto-review agreement, overrides, and activation state.",
    )
    calibration.add_argument("--repository", type=Path, default=Path.cwd())
    calibration.add_argument("--output", type=Path)
    calibration.set_defaults(handler=_calibration_workspace)

    calibration_sample = commands.add_parser(
        "calibration-sample",
        help="Create a deterministic, stratified calibration work queue.",
    )
    calibration_sample.add_argument("--repository", type=Path, default=Path.cwd())
    calibration_sample.add_argument("--count", type=int, default=50)
    calibration_sample.add_argument("--seed", type=int, default=42)
    calibration_sample.add_argument("--output", type=Path)
    calibration_sample.set_defaults(handler=_calibration_sample_workspace)

    rerun_historical = commands.add_parser(
        "rerun-historical-reviews",
        help="Append modern linked reruns without mutating original records.",
    )
    rerun_historical.add_argument("--repository", type=Path, default=Path.cwd())
    rerun_historical.add_argument("--sample-only", action="store_true")
    rerun_historical.add_argument("--output", type=Path)
    rerun_historical.set_defaults(handler=_rerun_historical_workspace)

    calibration_status = commands.add_parser(
        "calibration-status", help="Show readiness checks and exact blocking reasons."
    )
    calibration_status.add_argument("--repository", type=Path, default=Path.cwd())
    calibration_status.set_defaults(handler=_calibration_status_workspace)

    enable_auto = commands.add_parser(
        "enable-auto-approval",
        help="Explicitly enable automatic approval after every readiness gate passes.",
    )
    enable_auto.add_argument("--repository", type=Path, default=Path.cwd())
    enable_auto.set_defaults(handler=_enable_auto_approval_workspace)

    bulk_auto = commands.add_parser(
        "bulk-auto-review",
        help="Review eligible pending items after explicit calibration activation.",
    )
    bulk_auto.add_argument("--repository", type=Path, default=Path.cwd())
    bulk_auto.add_argument("--eligible-only", action="store_true", required=True)
    bulk_auto.set_defaults(handler=_bulk_auto_review_workspace)

    trust_export = commands.add_parser(
        "export-trust-tier",
        help="Export a deduplicated trust tier with paper-level splits.",
    )
    trust_export.add_argument("--repository", type=Path, default=Path.cwd())
    trust_export.add_argument(
        "--trust-tier",
        choices=("human-only", "human-and-audited", "include-codex-approved"),
        default="human-and-audited",
    )
    trust_export.add_argument("--seed", type=int, default=0)
    trust_export.add_argument("--output", type=Path, required=True)
    trust_export.set_defaults(handler=_export_trust_tier)
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
