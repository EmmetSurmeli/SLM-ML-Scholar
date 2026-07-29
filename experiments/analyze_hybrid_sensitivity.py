"""Descriptive one-factor sensitivity study for transparent hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.evaluate_retrieval_ablation import evaluate_method  # noqa: E402
from experiments.semantic_retrieval_fixture import (  # noqa: E402
    build_semantic_fixture_index,
    load_semantic_judgments,
)
from localml_scholar.retrieval import (  # noqa: E402
    HybridRetrievalConfig,
    RerankingConfig,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def _metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    return report["metrics"]["aggregate"]


def run_sensitivity(output_directory: Path) -> dict[str, Any]:
    """Evaluate a predefined descriptive grid without selecting on evaluation data."""
    output_directory.mkdir(parents=True, exist_ok=True)
    judgments = load_semantic_judgments()
    development = tuple(item for item in judgments if item["split"] == "development")
    evaluation = tuple(item for item in judgments if item["split"] == "evaluation")
    settings: list[dict[str, Any]] = []

    for dimension in (2, 4, 6, 8):
        index = build_semantic_fixture_index(dimensions=dimension)
        report = evaluate_method(
            index,
            judgments,
            label=f"latent_dimensions_{dimension}",
            method="hybrid",
            hybrid_config=HybridRetrievalConfig(fusion="rrf"),
            top_k=10,
        )
        settings.append(
            {
                "factor": "latent_dimensions",
                "value": dimension,
                "all_queries": _metric_summary(report),
            }
        )

    index = build_semantic_fixture_index(dimensions=6)
    controlled = (
        (
            "weighted_alpha",
            (0.25, 0.5, 0.75),
            lambda value: HybridRetrievalConfig(fusion="weighted", alpha=value),
            "hybrid",
        ),
        (
            "rrf_rank_constant",
            (10, 30, 60),
            lambda value: HybridRetrievalConfig(
                fusion="rrf",
                rrf_rank_constant=value,
            ),
            "hybrid",
        ),
        (
            "candidate_depth",
            (5, 10, 20),
            lambda value: HybridRetrievalConfig(
                fusion="rrf",
                lexical_candidate_count=value,
                semantic_candidate_count=value,
            ),
            "hybrid",
        ),
    )
    for factor, values, factory, method in controlled:
        for value in values:
            config = factory(value)
            development_report = evaluate_method(
                index,
                development,
                label=f"{factor}_{value}_development",
                method=method,
                hybrid_config=config,
                top_k=10,
            )
            evaluation_report = evaluate_method(
                index,
                evaluation,
                label=f"{factor}_{value}_evaluation",
                method=method,
                hybrid_config=config,
                top_k=10,
            )
            settings.append(
                {
                    "factor": factor,
                    "value": value,
                    "hybrid_configuration": config.to_dict(),
                    "development": _metric_summary(development_report),
                    "evaluation": _metric_summary(evaluation_report),
                }
            )

    redundancy_settings: list[dict[str, Any]] = []
    for threshold in (0.5, 0.8, 1.0):
        reranker = RerankingConfig(redundancy_threshold=threshold)
        rankings = {}
        relevance = {}
        for judgment in judgments:
            results = index.search(
                judgment["query"],
                method="hybrid_reranked",
                top_k=10,
                hybrid_config=HybridRetrievalConfig(fusion="rrf"),
                reranking_config=reranker,
            )
            rankings[judgment["query_id"]] = [result.chunk_id for result in results]
            relevance[judgment["query_id"]] = set(judgment["grades"])
        hit_rate = sum(
            bool(set(rankings[query_id][:3]) & relevance[query_id])
            for query_id in rankings
        ) / len(rankings)
        redundancy_settings.append(
            {
                "threshold": threshold,
                "reranking_configuration": reranker.to_dict(),
                "hit_rate_at_3": hit_rate,
            }
        )

    numeric_metrics = [
        entry["all_queries"]["reciprocal_rank"]
        for entry in settings
        if "all_queries" in entry
    ]
    summary = {
        "experiment": "hybrid_sensitivity",
        "study_type": "descriptive_not_tuned",
        "disclosure": (
            "Defaults were not selected by optimizing final evaluation queries."
        ),
        "development_query_count": len(development),
        "evaluation_query_count": len(evaluation),
        "settings": settings,
        "redundancy_settings": redundancy_settings,
        "latent_dimension_mrr_range": [
            min(numeric_metrics),
            max(numeric_metrics),
        ],
    }
    atomic_write_text(
        output_directory / "hybrid_sensitivity.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/hybrid_sensitivity"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    summary = run_sensitivity(parse_args(arguments).output_directory)
    print(
        json.dumps(
            {
                "study_type": summary["study_type"],
                "settings": len(summary["settings"]),
                "latent_dimension_mrr_range": summary["latent_dimension_mrr_range"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
