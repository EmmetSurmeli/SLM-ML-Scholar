"""Command-line entry point for autonomous grounded-dataset curation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from localml_scholar.review_app.service import ReviewService
from localml_scholar.training_data.autonomous import AutonomousCurationConfig
from localml_scholar.training_data.claim_alignment import diagnostic_claim_trace
from localml_scholar.training_data.preflight import (
    paper_ingestion_health,
    pipeline_self_test,
    rebuild_index_section_structure,
)
from localml_scholar.training_data.questions import generate_paper_questions
from localml_scholar.training_data.reviewer_reliability import (
    full_run_readiness,
    migrate_legacy_record,
    reliability_report,
)
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

    diagnose = commands.add_parser("diagnose-reviewers")
    diagnose.add_argument("--run", required=True)

    citation = commands.add_parser("citation-audit")
    citation.add_argument("--run", required=True)

    disagreement = commands.add_parser("disagreement-report")
    disagreement.add_argument("--run", required=True)

    claim_audit = commands.add_parser("claim-audit")
    claim_audit.add_argument("--run", required=True)

    repair = commands.add_parser("repair-report")
    repair.add_argument("--run", required=True)

    trace = commands.add_parser("claim-trace")
    trace.add_argument("--run", required=True)
    trace.add_argument("--candidate", required=True)

    diagnostic = commands.add_parser("diagnostic-curation")
    diagnostic.add_argument("--count", type=int, default=50)
    diagnostic.add_argument("--seed", type=int, default=42)

    freeze = commands.add_parser("freeze-diagnostic")
    freeze.add_argument("--run", required=True)
    freeze.add_argument(
        "--reason",
        default="invalid_for_readiness_due_to_upstream_pipeline_defects",
    )

    health = commands.add_parser("ingestion-health")
    health.add_argument(
        "--repair",
        action="store_true",
        help="Rebuild the local index with recovered section boundaries.",
    )

    commands.add_parser("pipeline-self-test")
    commands.add_parser("question-eligibility-report")

    pilot = commands.add_parser("pilot-curation")
    pilot.add_argument("--count", type=int, default=10)
    pilot.add_argument("--seed", type=int, default=42)

    usage = commands.add_parser("codex-usage-report")
    usage.add_argument("--run")

    commands.add_parser("full-run-readiness")

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


def _finished_controlled_diagnostics(
    service: ReviewService,
) -> list[dict[str, Any]]:
    """Return valid diagnostics that reached their declared sample size."""
    return [
        item
        for item in service._autonomous_curator().list_runs()
        if item.get("diagnostic", {}).get("controlled")
        and item.get("diagnostic", {}).get("valid_for_readiness", True)
        and isinstance(item.get("diagnostic", {}).get("count"), int)
        and len(item.get("records", [])) >= int(item["diagnostic"]["count"])
    ]


def _run_readiness(run_state: dict[str, Any]) -> dict[str, Any]:
    records = [migrate_legacy_record(item) for item in run_state["records"]]
    return full_run_readiness(reliability_report(records))


def _ingestion_health_report(service: ReviewService, *, repair: bool) -> dict[str, Any]:
    index = service._load_index()
    repair_report = {"documents_changed": 0, "index_changed": False}
    if repair:
        index, repair_report = rebuild_index_section_structure(index)
        if repair_report["index_changed"]:
            index.save(service.index_path)
    papers = []
    for document in index.documents:
        state = paper_ingestion_health(document).to_dict()
        state.update(
            {
                "title": document.title,
                "source_name": document.source_name,
                "section_titles": [
                    section.heading for section in document.sections if section.heading
                ],
            }
        )
        papers.append(state)
    return {
        "paper_count": len(papers),
        "healthy_papers": sum(
            item["healthy_for_question_generation"] for item in papers
        ),
        "unhealthy_papers": sum(
            not item["healthy_for_question_generation"] for item in papers
        ),
        "average_titled_section_fraction": (
            sum(item["titled_section_fraction"] for item in papers) / len(papers)
            if papers
            else 0.0
        ),
        "repair": repair_report,
        "papers": papers,
    }


def _question_eligibility_report(service: ReviewService) -> dict[str, Any]:
    index = service._load_index()
    papers = []
    type_counts: Counter[str] = Counter()
    suppressed = 0
    for document in index.documents:
        health = paper_ingestion_health(document)
        headings = tuple(
            section.heading for section in document.sections if section.heading
        )
        baseline = generate_paper_questions(
            document.document_id,
            document.title or document.source_name,
            count=80,
            section_titles=headings,
        )
        generated = (
            generate_paper_questions(
                document.document_id,
                document.title or document.source_name,
                count=80,
                section_titles=headings,
                paper_text=document.text,
            )
            if health.healthy_for_question_generation
            else ()
        )
        paper_suppressed = len(baseline) - len(generated)
        suppressed += paper_suppressed
        type_counts.update(item.question_type for item in generated)
        papers.append(
            {
                "paper_id": document.document_id,
                "title": document.title,
                "healthy": health.healthy_for_question_generation,
                "eligible_questions": len(generated),
                "templates_suppressed": paper_suppressed,
            }
        )
    return {
        "paper_count": len(papers),
        "question_templates_suppressed": suppressed,
        "eligible_question_type_counts": dict(sorted(type_counts.items())),
        "papers": papers,
    }


def _codex_usage_report(service: ReviewService, run_id: str | None) -> dict[str, Any]:
    runs = service._autonomous_curator().list_runs()
    if run_id is not None:
        runs = [item for item in runs if item["run_id"] == run_id]
        if not runs:
            raise ValueError(f"Unknown autonomous curation run: {run_id}")
    records = [record for run in runs for record in run.get("records", [])]
    roles: Counter[str] = Counter()
    for record in records:
        roles.update(
            str(item.get("pass_name", "unknown"))
            for item in record.get("codex_review_passes", [])
        )
    rejected = sum(bool(item.get("rejected_before_codex")) for item in records)
    return {
        "run_ids": [item["run_id"] for item in runs],
        "candidate_records": len(records),
        "candidates_sent_to_codex": sum(
            bool(item.get("codex_review_passes")) for item in records
        ),
        "codex_calls": sum(
            len(item.get("codex_review_passes", [])) for item in records
        ),
        "codex_calls_by_role": dict(sorted(roles.items())),
        "rejected_before_codex": rejected,
        "codex_calls_saved_estimate": rejected * 4,
        "deterministic_repair_successes": sum(
            item.get("answer", {}).get("deterministic_claim_repair", {}).get("outcome")
            == "fixed"
            for item in records
        ),
    }


def _pilot_readiness(run_state: dict[str, Any]) -> dict[str, Any]:
    """Apply the 1.2.6 gate before permitting a fresh 50-item diagnostic."""
    diagnostic = run_state.get("diagnostic", {})
    records = run_state.get("records", [])
    expected_count = int(diagnostic.get("count", 0))
    answerable = [
        item
        for item in records
        if item.get("expected_answerability", "answerable") == "answerable"
        and item.get("status") != "split_excluded"
    ]
    reviewed = [item for item in records if item.get("codex_review_passes")]
    retrieval_passes = sum(
        bool(item.get("answer", {}).get("sufficiency", {}).get("sufficient"))
        for item in answerable
    )
    retrieval_rate = retrieval_passes / len(answerable) if answerable else 0.0

    def reviewed_rate(field: str) -> float:
        return (
            sum(bool(item.get(field)) for item in reviewed) / len(reviewed)
            if reviewed
            else 0.0
        )

    structural_rate = reviewed_rate("citation_structural_valid")
    support_rate = reviewed_rate("citation_support_valid")
    relevance_rate = reviewed_rate("citation_relevance_valid")
    hard_rate = reviewed_rate("hard_reviewer_disagreement")
    failures = []
    if run_state.get("status") != "completed" or len(records) < expected_count:
        failures.append("pilot_incomplete")
    if any(str(item.get("status", "")).endswith("_failed") for item in records):
        failures.append("candidate_failure_present")
    if retrieval_rate < 0.95:
        failures.append("answerable_essential_retrieval_below_0_95")
    if structural_rate < 0.98:
        failures.append("citation_structure_below_0_98")
    if support_rate < 0.95:
        failures.append("citation_support_below_0_95")
    if relevance_rate < 0.95:
        failures.append("citation_relevance_below_0_95")
    if hard_rate > 0.15:
        failures.append("hard_reviewer_disagreement_above_0_15")
    if any(
        "leakage" in reason
        for item in records
        for reason in item.get("terminal_reasons", [])
    ):
        failures.append("leakage_detected")
    return {
        "ready": not failures,
        "failures": failures,
        "processed_count": len(records),
        "answerable_count": len(answerable),
        "reviewed_count": len(reviewed),
        "answerable_essential_retrieval_rate": retrieval_rate,
        "citation_structural_validity": structural_rate,
        "citation_support_rate": support_rate,
        "citation_relevance_rate": relevance_rate,
        "hard_reviewer_disagreement_rate": hard_rate,
    }


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
    if arguments.command == "freeze-diagnostic":
        return service._autonomous_curator().invalidate_for_readiness(
            arguments.run, arguments.reason
        )
    if arguments.command == "ingestion-health":
        return _ingestion_health_report(service, repair=arguments.repair)
    if arguments.command == "pipeline-self-test":
        return pipeline_self_test()
    if arguments.command == "question-eligibility-report":
        return _question_eligibility_report(service)
    if arguments.command == "codex-usage-report":
        return _codex_usage_report(service, arguments.run)
    if arguments.command == "pilot-curation":
        self_test = pipeline_self_test()
        if not self_test["passed"]:
            raise ValueError("The local pipeline self-test must pass before a pilot.")
        health = _ingestion_health_report(service, repair=True)
        if not health["healthy_papers"]:
            raise ValueError("No healthy papers are available for a pilot.")
        curator = service._autonomous_curator()
        created = curator.create_pilot(count=arguments.count, seed=arguments.seed)
        result = curator.resume(created["run_id"])
        result["pilot_preflight"] = {
            "self_test": self_test,
            "ingestion_health": {
                key: health[key]
                for key in (
                    "paper_count",
                    "healthy_papers",
                    "unhealthy_papers",
                    "average_titled_section_fraction",
                    "repair",
                )
            },
            "readiness": _pilot_readiness(result),
        }
        return result
    if arguments.command == "curation-report":
        return service.autonomous_curation_report(arguments.run)
    if arguments.command in {
        "diagnose-reviewers",
        "citation-audit",
        "disagreement-report",
        "claim-audit",
        "repair-report",
        "claim-trace",
    }:
        source = service._autonomous_curator().get_run(arguments.run)
        records = [migrate_legacy_record(item) for item in source["records"]]
        report = reliability_report(records)
        if arguments.command == "citation-audit":
            return {
                "run_id": arguments.run,
                "report": report,
                "records": [
                    {
                        "curation_record_id": item.get("curation_record_id"),
                        "question": item.get("question"),
                        "claim_citation_validation": item.get(
                            "claim_citation_validation"
                        ),
                    }
                    for item in records
                ],
            }
        if arguments.command == "disagreement-report":
            return {
                "run_id": arguments.run,
                "report": report,
                "records": [
                    {
                        "curation_record_id": item.get("curation_record_id"),
                        "question": item.get("question"),
                        "reviewer_disagreements": item.get(
                            "reviewer_disagreements", []
                        ),
                    }
                    for item in records
                ],
            }
        if arguments.command == "claim-audit":
            return {
                "run_id": arguments.run,
                "report": report,
                "records": [
                    {
                        "curation_record_id": item.get("curation_record_id"),
                        "question_id": item.get("question_id"),
                        "question": item.get("question"),
                        "supported_claim_graph": item.get("supported_claim_graph"),
                        "claim_alignment_metrics": item.get("claim_alignment_metrics"),
                    }
                    for item in records
                ],
            }
        if arguments.command == "repair-report":
            return {
                "run_id": arguments.run,
                "repair_success_rate": report.get("repair_success_rate"),
                "repair_success_by_type": report.get("repair_success_by_type", {}),
                "records": [
                    {
                        "curation_record_id": item.get("curation_record_id"),
                        "question_id": item.get("question_id"),
                        "repair_history": item.get("repair_history", []),
                    }
                    for item in records
                    if item.get("repair_history")
                ],
            }
        if arguments.command == "claim-trace":
            matches = [
                item
                for item in records
                if arguments.candidate
                in {item.get("question_id"), item.get("curation_record_id")}
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one candidate matching {arguments.candidate!r}; "
                    f"found {len(matches)}."
                )
            item = matches[0]
            answer = item.get("answer", {})
            return {
                "run_id": arguments.run,
                "candidate": arguments.candidate,
                "trace": diagnostic_claim_trace(
                    str(item.get("question", "")),
                    answer.get("evidence", []),
                    item.get("supported_claim_graph", {}),
                ),
                "supported_claim_graph": item.get("supported_claim_graph", {}),
            }
        return {"run_id": arguments.run, "report": report, "records": records}
    if arguments.command == "diagnostic-curation":
        curator = service._autonomous_curator()
        if arguments.count == 50:
            pilots = [
                item
                for item in curator.list_runs()
                if item.get("diagnostic", {}).get("kind")
                == "deterministic_preflight_pilot"
                and item.get("diagnostic", {}).get("valid_for_readiness", True)
            ]
            if not pilots or not _pilot_readiness(pilots[0])["ready"]:
                raise ValueError(
                    "A fresh 10-question deterministic-preflight pilot must pass "
                    "before a 50-candidate diagnostic."
                )
        if arguments.count != 50:
            if not 100 <= arguments.count <= 150:
                raise ValueError(
                    "Controlled diagnostics must use 50 candidates first or "
                    "100–150 candidates for the second run."
                )
            first_runs = [
                item
                for item in _finished_controlled_diagnostics(service)
                if item.get("diagnostic", {}).get("count") == 50
            ]
            if not first_runs or not _run_readiness(first_runs[0])["ready"]:
                raise ValueError(
                    "A valid 50-candidate diagnostic must pass every readiness "
                    "gate before the second diagnostic."
                )
            if arguments.seed == int(first_runs[0]["diagnostic"]["seed"]):
                raise ValueError(
                    "The second diagnostic must use a different deterministic seed."
                )
        created = curator.create_diagnostic(count=arguments.count, seed=arguments.seed)
        return curator.resume(created["run_id"])
    if arguments.command == "full-run-readiness":
        controlled = _finished_controlled_diagnostics(service)
        first = [
            item for item in controlled if item.get("diagnostic", {}).get("count") == 50
        ]
        second = [
            item
            for item in controlled
            if 100 <= int(item.get("diagnostic", {}).get("count", 0)) <= 150
        ]
        if not first:
            return {
                "ready": False,
                "failures": ["no_finished_valid_50_candidate_diagnostic"],
            }
        first_readiness = _run_readiness(first[0])
        if not first_readiness["ready"]:
            return {
                "ready": False,
                "failures": [f"first:{item}" for item in first_readiness["failures"]],
                "first_run_id": first[0]["run_id"],
            }
        if not second:
            return {
                "ready": False,
                "failures": ["no_finished_valid_second_diagnostic"],
                "first_run_id": first[0]["run_id"],
            }
        second_readiness = _run_readiness(second[0])
        return {
            **second_readiness,
            "first_run_id": first[0]["run_id"],
            "second_run_id": second[0]["run_id"],
            "failures": [f"second:{item}" for item in second_readiness["failures"]],
        }
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
