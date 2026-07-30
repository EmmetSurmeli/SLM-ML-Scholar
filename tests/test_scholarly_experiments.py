from __future__ import annotations

import json

from experiments.evaluate_paper_comparison import run as run_comparison
from experiments.evaluate_reproduction_checklists import run as run_checklists
from experiments.evaluate_research_gap_candidates import run as run_gaps
from experiments.evaluate_scholarly_extraction import run as run_extraction
from experiments.inspect_scholarly_analysis import run as run_inspection


def test_scholarly_extraction_experiment(tmp_path) -> None:
    path = tmp_path / "extraction.json"
    report = run_extraction(path)

    assert path.exists()
    assert report["paper_count"] == 3
    assert all(
        values["citation_coverage"] == 1.0 for values in report["per_paper"].values()
    )
    assert all(
        not values["failures"]["metric_failures"]
        and not values["failures"]["count_failures"]
        and not values["failures"]["metadata_failures"]
        for values in report["per_paper"].values()
    )
    assert json.loads(path.read_text())["index_sha256"] == report["index_sha256"]


def test_reproduction_comparison_and_gap_experiments(tmp_path) -> None:
    checklist = run_checklists(tmp_path / "checklist.json")
    comparison = run_comparison(tmp_path / "comparison.json")
    gaps = run_gaps(tmp_path / "gaps.json")

    assert all(
        values["risk_flag_metrics"]["f1"] == 1.0
        for values in checklist["papers"].values()
    )
    assert all(
        values[metric]["exact_value_accuracy"] == 1.0
        for values in checklist["papers"].values()
        for metric in (
            "found_field_metrics",
            "missing_field_detection",
            "ambiguity_detection",
            "conflict_detection",
        )
    )
    assert comparison["citation_coverage"] == 1.0
    assert comparison["false_superiority_claim_count"] == 0
    assert gaps["source_basis_validity"] == 1.0
    assert gaps["caution_label_presence"] == 1.0
    assert gaps["unsupported_novelty_claim_count"] == 0


def test_scholarly_inspection_reload_consistency(tmp_path) -> None:
    report = run_inspection(tmp_path / "inspection")

    assert report["artifact_reload_consistent"]
    assert not report["transformer_constructed"]
    assert report["equations"]
    assert report["notation_glossary"]
    assert report["reproduction_checklist"]["items"]
