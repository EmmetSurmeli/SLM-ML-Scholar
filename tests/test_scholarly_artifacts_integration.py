from __future__ import annotations

import json

import pytest

from localml_scholar.retrieval import RetrievalIndex, ingest_markdown
from localml_scholar.scholarly import (
    ScholarlyAnalysisPipeline,
    citation_coverage,
    create_artifact,
    extraction_metrics,
    load_analysis,
    load_artifact,
    render_checklist_markdown,
    render_comparison_markdown,
    render_notation_markdown,
    save_analysis,
    save_artifact,
)


def _ids(index):
    return {item.source_name: item.document_id for item in index.documents}


def test_summary_fields_are_cited_or_explicitly_missing(scholarly_index) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    summary = pipeline.summarize_paper(_ids(scholarly_index)["sparse_gate_network.md"])

    for field in summary.fields:
        assert field.status == "missing" or field.evidence
    assert summary.completeness["uncited_summary_fields"] == []
    assert "unresolved_symbols" in summary.completeness


def test_reproduction_checklist_order_status_conflicts_and_risks(
    scholarly_index,
) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    checklist = pipeline.build_reproduction_checklist(
        _ids(scholarly_index)["sparse_gate_network.md"]
    )
    labels = [item.item for item in checklist.items]

    assert labels[:4] == ["dataset", "split", "preprocessing", "sample sizes"]
    learning_rate = next(
        item for item in checklist.items if item.item == "learning rate"
    )
    assert learning_rate.status == "conflicting"
    assert any("conflicting" in item.reason for item in checklist.risk_flags)
    assert any(item.reason == "hardware missing" for item in checklist.risk_flags)


def test_comparison_dual_citations_and_false_superiority_prevention(
    scholarly_index,
) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    ids = _ids(scholarly_index)
    comparison = pipeline.compare_papers(
        [ids["sparse_gate_network.md"], ids["dense_gate_companion.md"]]
    )
    results = next(item for item in comparison.dimensions if item.name == "key_results")

    assert not results.comparable
    assert "no_superiority_ranking_permitted" in results.warnings
    assert all(results.values_by_paper.values())
    assert comparison.false_superiority_claim_count == 0


def test_research_gaps_distinguish_direct_and_inferred_and_carry_caution(
    scholarly_index,
) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    gaps = pipeline.identify_research_gaps(tuple(_ids(scholarly_index).values()))

    assert any(not item.system_inference for item in gaps)
    assert any(item.system_inference for item in gaps)
    assert all(item.citations for item in gaps)
    assert all(
        any("novel" in caution.casefold() for caution in item.cautions) for item in gaps
    )
    assert len({item.gap_id for item in gaps}) == len(gaps)


def test_analysis_artifact_round_trip_and_tamper_rejection(
    scholarly_index,
    tmp_path,
) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    analysis = pipeline.analyze_paper(_ids(scholarly_index)["sparse_gate_network.md"])
    path = save_analysis(
        analysis,
        tmp_path / "analysis.json",
        index=scholarly_index,
        config=pipeline.config,
    )
    loaded = load_analysis(path, index=scholarly_index)
    assert loaded.to_dict() == analysis.to_dict()

    state = json.loads(path.read_text())
    state["payload"]["paper"]["source_hash"] = "0" * 64
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="hash|identity|source"):
        load_analysis(path, index=scholarly_index)


def test_artifact_index_and_missing_source_rejection(scholarly_index, tmp_path) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    document_id = _ids(scholarly_index)["robust_mean_estimation.md"]
    summary = pipeline.summarize_paper(document_id)
    artifact = create_artifact(
        summary,
        artifact_type="structured_summary",
        index=scholarly_index,
        document_ids=(document_id,),
        config=pipeline.config,
    )
    path = save_artifact(artifact, tmp_path / "summary.json")
    assert load_artifact(path, index=scholarly_index).to_dict() == artifact.to_dict()

    other = RetrievalIndex.build(
        (ingest_markdown("# Other\n\nText.", source="other.md"),)
    )
    with pytest.raises(ValueError, match="index hash"):
        load_artifact(path, index=other)


def test_markdown_renderers_keep_citations(scholarly_index) -> None:
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    ids = _ids(scholarly_index)
    analysis = pipeline.analyze_paper(ids["sparse_gate_network.md"])
    checklist = pipeline.build_reproduction_checklist(ids["sparse_gate_network.md"])
    comparison = pipeline.compare_papers(
        [ids["sparse_gate_network.md"], ids["dense_gate_companion.md"]]
    )

    assert "| Symbol | Meaning |" in render_notation_markdown(analysis)
    assert "[Sparse Gate Networks" in render_notation_markdown(analysis)
    assert "| Item | Status |" in render_checklist_markdown(checklist)
    assert "| Dimension |" in render_comparison_markdown(comparison)


def test_metrics_have_explicit_empty_semantics_and_citation_coverage() -> None:
    exact = extraction_metrics(["a", "b"], ["a", "b"])
    partial = extraction_metrics(["a", "c"], ["a", "b"])
    assert exact.f1 == 1.0
    assert partial.precision == partial.recall == 0.5
    assert extraction_metrics([], []).f1 == 1.0
    assert citation_coverage([]) == 1.0


def test_pipeline_does_not_construct_transformer(monkeypatch, scholarly_index) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("transformer must not be constructed")

    monkeypatch.setattr(
        "localml_scholar.models.transformer_lm.TransformerLanguageModel.__init__",
        forbidden,
    )
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    analysis = pipeline.analyze_paper(next(iter(_ids(scholarly_index).values())))
    assert analysis.paper.paper_id
