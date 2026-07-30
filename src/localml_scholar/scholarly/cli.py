"""Command-line interface for deterministic scholarly analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from localml_scholar.retrieval import RetrievalIndex
from localml_scholar.scholarly.artifacts import (
    render_checklist_markdown,
    render_comparison_markdown,
    render_notation_markdown,
)
from localml_scholar.scholarly.models import NotationEntry
from localml_scholar.scholarly.pipeline import ScholarlyAnalysisPipeline
from localml_scholar.scholarly.serialization import create_artifact, save_artifact

_COMMANDS = (
    "analyze",
    "glossary",
    "equations",
    "methods",
    "experiments",
    "summary",
    "reproduction-checklist",
    "compare",
    "research-gaps",
    "inspect",
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse scholarly CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create deterministic source-linked scholarly artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in _COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--index", type=Path, required=True)
        if command in {"compare", "research-gaps"}:
            subparser.add_argument("--document-id", action="append", required=True)
        else:
            subparser.add_argument("--document-id", required=True)
        subparser.add_argument(
            "--retrieval-method",
            choices=("bm25", "tfidf", "semantic", "hybrid", "hybrid_reranked"),
            default="bm25",
        )
        subparser.add_argument("--section-role")
        subparser.add_argument("--output", type=Path)
        subparser.add_argument("--json", action="store_true")
        subparser.add_argument("--verbose", action="store_true")
    return parser.parse_args(arguments)


def _select(
    args: argparse.Namespace,
    pipeline: ScholarlyAnalysisPipeline,
) -> tuple[object, str, tuple[str, ...]]:
    command = args.command
    document_ids = (
        tuple(args.document_id)
        if isinstance(args.document_id, list)
        else (args.document_id,)
    )
    if args.section_role and command in {
        "compare",
        "research-gaps",
        "summary",
        "reproduction-checklist",
        "inspect",
    }:
        raise ValueError(
            "--section-role is supported by analyze, glossary, equations, "
            "methods, and experiments."
        )
    if command == "compare":
        return pipeline.compare_papers(document_ids), "paper_comparison", document_ids
    if command == "research-gaps":
        return (
            pipeline.identify_research_gaps(document_ids),
            "research_gap_worksheet",
            document_ids,
        )
    document_id = document_ids[0]
    analysis = pipeline.analyze_paper(document_id)
    if args.section_role and not any(
        args.section_role in section.roles for section in analysis.paper.sections
    ):
        raise ValueError(
            f"Requested section role {args.section_role!r} is not present."
        )
    selected_section_ids = {
        section.section_id
        for section in analysis.paper.sections
        if args.section_role is None or args.section_role in section.roles
    }
    if command == "analyze":
        if args.section_role is not None:
            equation_ids = {
                item.equation_id
                for item in analysis.equations
                if item.section_id in selected_section_ids
            }
            evidence_names = (
                "assumptions",
                "claims",
                "methodology",
                "datasets",
                "metrics",
                "baselines",
                "hyperparameters",
                "results",
                "ablations",
                "limitations",
                "in_text_references",
            )
            return (
                {
                    "paper_id": analysis.paper.paper_id,
                    "section_role": args.section_role,
                    "sections": [
                        item.to_dict()
                        for item in analysis.paper.sections
                        if item.section_id in selected_section_ids
                    ],
                    "equations": [
                        item.to_dict()
                        for item in analysis.equations
                        if item.equation_id in equation_ids
                    ],
                    "equation_analyses": [
                        item.to_dict()
                        for item in analysis.equation_analyses
                        if item.equation_id in equation_ids
                    ],
                    "notation": _filtered_notation(
                        analysis.notation, selected_section_ids
                    ),
                    **{
                        name: [
                            item.to_dict()
                            for item in getattr(analysis, name)
                            if item.citation.section_id in selected_section_ids
                        ]
                        for name in evidence_names
                    },
                    "procedures": [
                        item.to_dict()
                        for item in analysis.procedures
                        if item.citation.section_id in selected_section_ids
                    ],
                    "experiments": [
                        item.to_dict()
                        for item in analysis.experiments
                        if item.citation.section_id in selected_section_ids
                    ],
                    "tables": [
                        item.to_dict()
                        for item in analysis.tables
                        if item.citation.section_id in selected_section_ids
                    ],
                },
                "paper_analysis_section",
                document_ids,
            )
        return analysis, "paper_analysis", document_ids
    if command == "glossary":
        return (
            {
                "paper_id": analysis.paper.paper_id,
                "entries": _filtered_notation(analysis.notation, selected_section_ids),
                "unresolved_symbols": list(analysis.unresolved_symbols),
            },
            "notation_glossary",
            document_ids,
        )
    if command == "equations":
        return (
            {
                "paper_id": analysis.paper.paper_id,
                "equations": [
                    item.to_dict()
                    for item in analysis.equations
                    if item.section_id in selected_section_ids
                ],
                "analyses": [
                    item.to_dict()
                    for item in analysis.equation_analyses
                    if any(
                        equation.equation_id == item.equation_id
                        and equation.section_id in selected_section_ids
                        for equation in analysis.equations
                    )
                ],
            },
            "equation_analysis",
            document_ids,
        )
    if command == "methods":
        return (
            tuple(
                item
                for item in analysis.methodology
                if item.citation.section_id in selected_section_ids
            ),
            "methodology",
            document_ids,
        )
    if command == "experiments":
        return (
            tuple(
                item
                for item in analysis.experiments
                if item.citation.section_id in selected_section_ids
            ),
            "experiments",
            document_ids,
        )
    if command == "summary":
        return pipeline.summarize_paper(document_id), "structured_summary", document_ids
    if command == "reproduction-checklist":
        return (
            pipeline.build_reproduction_checklist(document_id),
            "reproduction_checklist",
            document_ids,
        )
    if command == "inspect":
        return (
            {
                "analysis_id": analysis.analysis_id,
                "paper_id": analysis.paper.paper_id,
                "document_id": document_id,
                "index_sha256": pipeline.index.index_sha256,
                "counts": {
                    "sections": len(analysis.paper.sections),
                    "equations": len(analysis.equations),
                    "notation_entries": len(analysis.notation),
                    "assumptions": len(analysis.assumptions),
                    "methodology_fields": len(analysis.methodology),
                    "experiments": len(analysis.experiments),
                    "results": len(analysis.results),
                    "tables": len(analysis.tables),
                    "limitations": len(analysis.limitations),
                    "references": len(analysis.paper.references),
                },
                "warnings": list(analysis.warnings),
                "transformer_constructed": False,
            },
            "paper_analysis",
            document_ids,
        )
    raise RuntimeError("Unhandled scholarly command.")


def _filtered_notation(
    entries: tuple[NotationEntry, ...],
    section_ids: set[str],
) -> list[dict[str, Any]]:
    filtered = []
    for entry in entries:
        occurrences = tuple(
            item for item in entry.occurrences if item.section_id in section_ids
        )
        if not occurrences:
            continue
        state = entry.to_dict()
        state["occurrences"] = [item.to_dict() for item in occurrences]
        candidates = tuple(
            item
            for item in entry.definition_candidates
            if item.citation.section_id in section_ids
        )
        state["definition_candidates"] = [item.to_dict() for item in candidates]
        if entry.selected_definition not in candidates:
            state["selected_definition"] = None
        filtered.append(state)
    return filtered


def _state(value: object) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _state(item) for key, item in value.items()}
    return value


def _human(
    command: str,
    value: object,
    pipeline: ScholarlyAnalysisPipeline,
    document_ids: tuple[str, ...],
) -> str:
    if command == "glossary":
        return render_notation_markdown(pipeline.analyze_paper(document_ids[0]))
    if command == "reproduction-checklist":
        return render_checklist_markdown(value)
    if command == "compare":
        return render_comparison_markdown(value)
    return json.dumps(_state(value), ensure_ascii=False, indent=2, sort_keys=True)


def run(arguments: Sequence[str] | None = None) -> int:
    """Execute one scholarly CLI operation."""
    args = parse_args(arguments)
    index = RetrievalIndex.load(args.index)
    pipeline = ScholarlyAnalysisPipeline(
        index,
        retrieval_method=args.retrieval_method,
    )
    value, artifact_type, document_ids = _select(args, pipeline)
    if args.output is not None:
        artifact = create_artifact(
            value,
            artifact_type=artifact_type,
            index=index,
            document_ids=document_ids,
            config=pipeline.config,
        )
        save_artifact(artifact, args.output)
    if args.json:
        print(json.dumps(_state(value), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_human(args.command, value, pipeline, document_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
