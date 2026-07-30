#!/usr/bin/env python3
"""Summarize failures, likely causes, sections, and review priorities."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from localml_scholar.evaluation.serialization import (  # noqa: E402
    load_evaluation_run,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "failure_analysis.json",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    run = load_evaluation_run(args.run)
    failures = Counter()
    causes = Counter()
    question_types = Counter()
    sections = Counter()
    boilerplate = Counter()
    missing = Counter()
    for result in run.question_results:
        failures.update(result.failure_categories)
        causes[result.root_cause.primary_cause] += 1
        question_types[result.question_type] += 1
        sections.update(result.retrieval.wrong_section_chunk_ids)
        boilerplate.update(result.retrieval.boilerplate_chunk_ids)
        if result.concepts is not None:
            missing.update(result.concepts.missing_required)
    payload = {
        "run_id": run.run_id,
        "failure_frequencies": dict(sorted(failures.items())),
        "root_causes": dict(sorted(causes.items())),
        "question_types": dict(sorted(question_types.items())),
        "responsible_chunk_ids": dict(sorted(sections.items())),
        "boilerplate_chunk_ids": dict(sorted(boilerplate.items())),
        "missing_concepts": dict(sorted(missing.items())),
        "review_priority_question_ids": [
            item.question_id
            for item in run.question_results
            if not item.automatic_pass or item.automated_confidence == "low"
        ],
    }
    atomic_write_text(
        args.output,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(f"Saved failure analysis to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
