"""Evaluate deterministic scholarly extraction on authored fixtures."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scholarly_fixture import (  # noqa: E402
    build_pipeline,
    document_ids,
    load_scholarly_judgments,
    write_report,
)
from localml_scholar.scholarly import (  # noqa: E402
    citation_coverage,
    classification_accuracy,
    extraction_metrics,
)


def run(output: Path) -> dict:
    """Run field, role, equation, citation, and latency evaluation."""
    index, pipeline = build_pipeline()
    ids = document_ids(index)
    judgments = load_scholarly_judgments()
    per_paper = {}
    started = time.perf_counter()
    for source_name, expected in sorted(judgments.items()):
        paper_started = time.perf_counter()
        analysis = pipeline.analyze_paper(ids[source_name])
        datasets = [
            str(item.normalized_value)
            for item in analysis.datasets
            if item.category == "dataset"
        ]
        metrics = [str(item.normalized_value) for item in analysis.metrics]
        hyperparameters = [str(item.value["name"]) for item in analysis.hyperparameters]
        equations = [
            item.equation_number
            for item in analysis.equations
            if item.equation_number is not None
        ]
        symbols = [item.raw_symbol for item in analysis.notation]
        defined_symbols = [
            item.raw_symbol
            for item in analysis.notation
            if item.selected_definition is not None
        ]
        result_values = [str(item.value["raw_value"]) for item in analysis.results]
        method_fields = [item.category for item in analysis.methodology]
        reference_titles = [
            item.title for item in analysis.paper.references if item.title is not None
        ]
        experiment_names = [item.name for item in analysis.experiments]
        expected_roles = expected["section_roles"]
        actual_roles = {
            section.heading: section.roles[0]
            for section in analysis.paper.sections
            if section.heading in expected_roles
        }
        metadata = {
            name: (
                None
                if getattr(analysis.paper, name) is None
                else getattr(analysis.paper, name).normalized_value
            )
            for name in ("title", "authors", "year", "venue", "identifier")
        }
        serialized_records = [
            item.to_dict()
            for item in (
                analysis.paper.sections
                + analysis.equations
                + analysis.notation
                + analysis.assumptions
                + analysis.claims
                + analysis.methodology
                + analysis.procedures
                + analysis.datasets
                + analysis.metrics
                + analysis.baselines
                + analysis.hyperparameters
                + analysis.experiments
                + analysis.results
                + analysis.tables
                + analysis.ablations
                + analysis.limitations
                + analysis.paper.references
                + analysis.in_text_references
            )
        ]
        metric_results = {
            "datasets": extraction_metrics(datasets, expected["datasets"]).to_dict(),
            "metrics": extraction_metrics(metrics, expected["metrics"]).to_dict(),
            "hyperparameters": extraction_metrics(
                hyperparameters, expected["hyperparameters"]
            ).to_dict(),
            "equation_numbers": extraction_metrics(
                equations, expected["equation_numbers"]
            ).to_dict(),
            "symbols": extraction_metrics(
                symbols, expected.get("symbols", [])
            ).to_dict(),
            "symbol_definitions": extraction_metrics(
                defined_symbols, expected["defined_symbols"]
            ).to_dict(),
            "unresolved_symbols": extraction_metrics(
                analysis.unresolved_symbols, expected["unresolved_symbols"]
            ).to_dict(),
            "method_fields": extraction_metrics(
                method_fields, expected["method_fields"]
            ).to_dict(),
            "result_values": extraction_metrics(
                result_values, expected["result_values"]
            ).to_dict(),
            "experiment_grouping": extraction_metrics(
                experiment_names, expected["experiment_names"]
            ).to_dict(),
            "reference_titles": extraction_metrics(
                reference_titles, expected["reference_titles"]
            ).to_dict(),
        }
        count_accuracy = {
            "equations": float(len(analysis.equations) == expected["equation_count"]),
            "assumptions": float(
                len(analysis.assumptions) == expected["assumption_count"]
            ),
            "claims": float(len(analysis.claims) == expected["claim_count"]),
            "limitations": float(
                len(analysis.limitations) == expected["limitation_count"]
            ),
            "reference_links": float(
                sum(
                    item.validation == "validated"
                    for item in analysis.in_text_references
                )
                == expected["reference_link_count"]
            ),
        }
        metadata_accuracy = {
            name: float(metadata[name] == value)
            for name, value in expected["metadata"].items()
        }
        per_paper[source_name] = {
            **metric_results,
            "metadata": metadata,
            "metadata_field_accuracy": metadata_accuracy,
            "metadata_accuracy": sum(metadata_accuracy.values())
            / len(metadata_accuracy),
            "section_role_accuracy": classification_accuracy(
                actual_roles, expected_roles
            ),
            "count_accuracy": count_accuracy,
            "citation_coverage": citation_coverage(serialized_records),
            "counts": {
                "equations": len(analysis.equations),
                "notation": len(analysis.notation),
                "assumptions": len(analysis.assumptions),
                "methodology": len(analysis.methodology),
                "experiments": len(analysis.experiments),
                "results": len(analysis.results),
                "limitations": len(analysis.limitations),
                "references": len(analysis.paper.references),
            },
            "failures": {
                "metric_failures": [
                    name
                    for name, result in metric_results.items()
                    if result["exact_value_accuracy"] != 1.0
                ],
                "count_failures": [
                    name for name, value in count_accuracy.items() if value != 1.0
                ],
                "metadata_failures": [
                    name for name, value in metadata_accuracy.items() if value != 1.0
                ],
                "warnings": list(analysis.warnings),
            },
            "latency_seconds": time.perf_counter() - paper_started,
        }
    report = {
        "experiment": "authored_scholarly_extraction",
        "paper_count": len(per_paper),
        "index_sha256": index.index_sha256,
        "per_paper": per_paper,
        "latency_seconds_total": time.perf_counter() - started,
        "interpretation": (
            "Authored deterministic regression fixture only; heuristic extraction "
            "does not prove semantic correctness."
        ),
    }
    write_report(output, report)
    return report


if __name__ == "__main__":
    destination = PROJECT_ROOT / "outputs" / "scholarly_extraction" / "evaluation.json"
    result = run(destination)
    print(
        {
            name: {
                "dataset_f1": values["datasets"]["f1"],
                "metric_f1": values["metrics"]["f1"],
                "citation_coverage": values["citation_coverage"],
            }
            for name, values in result["per_paper"].items()
        }
    )
