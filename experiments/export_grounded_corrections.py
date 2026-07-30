#!/usr/bin/env python3
"""Export only adjudicated, source-valid grounded correction examples."""

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

from localml_scholar.evaluation.human_review import (  # noqa: E402
    export_approved_corrections,
)
from localml_scholar.evaluation.reports import (  # noqa: E402
    render_correction_preview,
)
from localml_scholar.evaluation.serialization import (  # noqa: E402
    load_benchmark,
    load_review_records,
    save_corrections,
)
from localml_scholar.retrieval import RetrievalIndex  # noqa: E402
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    index = RetrievalIndex.load(args.index)
    benchmark = load_benchmark(args.benchmark, index=index)
    records = load_review_records(
        args.reviews,
        artifact_type="human_reviews",
    )
    corrections = export_approved_corrections(benchmark, index, records)
    save_corrections(corrections, args.output)
    if args.preview is not None:
        atomic_write_text(
            args.preview,
            render_correction_preview(corrections),
        )
    print(f"Exported {len(corrections)} human-approved corrections to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
