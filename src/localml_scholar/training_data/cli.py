"""Command-line entry point for autonomous grounded-dataset curation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data.autonomous import AutonomousCurationConfig
from localml_scholar.training_data.trust import TRUST_TIERS


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions-per-paper", type=int, default=60)
    parser.add_argument("--maximum-examples-per-paper", type=int, default=40)
    parser.add_argument("--acceptance-threshold", type=float, default=0.97)
    parser.add_argument("--evidence-threshold", type=float, default=0.97)
    parser.add_argument("--maximum-repair-attempts", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-cross-paper", action="store_true")
    parser.add_argument("--no-multi-turn", action="store_true")
    parser.add_argument("--no-derivations", action="store_true")
    parser.add_argument("--no-abstentions", action="store_true")


def _config(arguments: argparse.Namespace) -> AutonomousCurationConfig:
    return AutonomousCurationConfig(
        questions_per_paper=arguments.questions_per_paper,
        maximum_examples_per_paper=arguments.maximum_examples_per_paper,
        acceptance_threshold=arguments.acceptance_threshold,
        evidence_threshold=arguments.evidence_threshold,
        maximum_repair_attempts=arguments.maximum_repair_attempts,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
        include_multi_turn=not arguments.no_multi_turn,
        include_derivations=not arguments.no_derivations,
        include_cross_paper=arguments.include_cross_paper,
        include_abstentions=not arguments.no_abstentions,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate a grounded local-paper dataset with Codex review passes."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="LocalML Scholar repository root (default: current directory).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser("curate-corpus")
    corpus.add_argument("--all-papers", action="store_true", required=True)
    _add_config_arguments(corpus)

    paper = commands.add_parser("curate-paper")
    paper.add_argument("--paper", required=True)
    _add_config_arguments(paper)

    new = commands.add_parser("process-new")
    _add_config_arguments(new)

    resume = commands.add_parser("resume-curation")
    resume.add_argument("--run", required=True)

    report = commands.add_parser("curation-report")
    report.add_argument("--run", required=True)

    export = commands.add_parser("export")
    export.add_argument("--run")
    export.add_argument(
        "--trust-tier",
        choices=sorted(TRUST_TIERS),
        default="codex-curated-only",
    )
    return parser


def _latest_run_id(service: ReviewService) -> str:
    runs = service._autonomous_curator().list_runs()
    if not runs:
        raise ValueError("No autonomous curation run exists yet; pass --run after one.")
    return runs[0]["run_id"]


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute one parsed CLI command and return its JSON result."""
    service = ReviewService(arguments.repository)
    if arguments.command == "curate-corpus":
        return service.start_autonomous_curation(config=_config(arguments))
    if arguments.command == "curate-paper":
        return service.start_autonomous_curation(
            paper_ids=(arguments.paper,), config=_config(arguments)
        )
    if arguments.command == "process-new":
        return service.process_new_papers_autonomously(config=_config(arguments))
    if arguments.command == "resume-curation":
        return service.resume_autonomous_curation(arguments.run)
    if arguments.command == "curation-report":
        return service.autonomous_curation_report(arguments.run)
    if arguments.command == "export":
        run_id = arguments.run or _latest_run_id(service)
        return service.export_autonomous_dataset(
            run_id, trust_tier=arguments.trust_tier
        )
    raise RuntimeError(f"Unhandled command: {arguments.command}")


def main() -> None:
    """Parse arguments, execute curation, and print strict JSON."""
    arguments = _parser().parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
