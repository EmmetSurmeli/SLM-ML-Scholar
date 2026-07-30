from __future__ import annotations

from localml_scholar.scholarly import ScholarlyAnalysisPipeline


def _analysis(index, source):
    document = next(item for item in index.documents if item.source_name == source)
    return ScholarlyAnalysisPipeline(index).analyze_paper(document.document_id)


def test_assumptions_claims_and_qualifiers(scholarly_index) -> None:
    analysis = _analysis(scholarly_index, "robust_mean_estimation.md")

    assert any("finite variance" in item.source_text for item in analysis.assumptions)
    assert all(item.category == "assumption" for item in analysis.assumptions)
    assert any(
        item.metadata["claim_type"] == "theoretical_result" for item in analysis.claims
    )
    assert any(
        item.metadata["claim_type"] == "empirical_result" for item in analysis.claims
    )


def test_methodology_hyperparameters_conflicts_and_scope(scholarly_index) -> None:
    analysis = _analysis(scholarly_index, "sparse_gate_network.md")
    method_categories = {item.category for item in analysis.methodology}
    hyperparameters = {}
    for item in analysis.hyperparameters:
        hyperparameters.setdefault(item.value["name"], []).append(item)

    assert {"optimizer", "objective_function", "architecture", "preprocessing"} <= (
        method_categories
    )
    assert {item.value["raw_value"] for item in hyperparameters["learning_rate"]} == {
        "0.001",
        "0.0005",
    }
    assert all(
        item.validation == "conflicting" for item in hyperparameters["learning_rate"]
    )
    assert hyperparameters["batch_size"][0].value["parsed_value"] == 32


def test_procedure_dataset_metric_baseline_and_experiment_grouping(
    scholarly_index,
) -> None:
    analysis = _analysis(scholarly_index, "sparse_gate_network.md")

    assert len(analysis.procedures) == 1
    assert len(analysis.procedures[0].steps) == 4
    assert {
        item.normalized_value
        for item in analysis.datasets
        if item.category == "dataset"
    } == {"spiralbench"}
    assert {item.normalized_value for item in analysis.metrics} == {"accuracy", "f1"}
    assert any(item.normalized_value == "densenet" for item in analysis.baselines)
    experiment = next(
        item for item in analysis.experiments if item.name == "Experiments"
    )
    assert experiment.datasets
    assert experiment.metrics
    assert experiment.hyperparameters
    dense = _analysis(scholarly_index, "dense_gate_companion.md")
    assert {item.normalized_value for item in dense.baselines} == {"sgn"}


def test_result_percent_uncertainty_and_scope(scholarly_index) -> None:
    sparse = _analysis(scholarly_index, "sparse_gate_network.md")
    robust = _analysis(scholarly_index, "robust_mean_estimation.md")
    dense = _analysis(scholarly_index, "dense_gate_companion.md")

    assert any(item.value["unit"] == "%" for item in sparse.results)
    assert any(
        item.value["value"] == 0.25 and "confidence interval" in item.value["context"]
        for item in robust.results
    )
    assert all("scope_section_id" in item.metadata for item in sparse.results)
    sparse_raw = {item.value["raw_value"] for item in sparse.results}
    robust_raw = {item.value["raw_value"] for item in robust.results}
    assert "1" not in sparse_raw
    assert "90.8%" in {item.value["raw_value"] for item in dense.results}
    assert "10" not in robust_raw


def test_markdown_and_delimited_tables_preserve_raw_text(scholarly_index) -> None:
    sparse = _analysis(scholarly_index, "sparse_gate_network.md")
    robust = _analysis(scholarly_index, "robust_mean_estimation.md")

    markdown = sparse.tables[0]
    delimited = robust.tables[0]
    assert markdown.parsing_method == "markdown_pipe_table"
    assert markdown.headers == ("System", "Accuracy", "F1")
    assert markdown.caption == "Table 1: Test results"
    assert "| DenseNet |" in markdown.raw_text
    assert delimited.parsing_method == "explicit_delimiter_table"
    assert delimited.rows[0][0] == "SampleMean"


def test_ablations_limitations_and_references(scholarly_index) -> None:
    analysis = _analysis(scholarly_index, "sparse_gate_network.md")

    assert any("Removing" in item.source_text for item in analysis.ablations)
    assert any(
        item.metadata["limitation_type"] == "author_stated"
        for item in analysis.limitations
    )
    assert len(analysis.paper.references) == 2
    assert analysis.paper.references[0].year == 2021
    assert analysis.paper.references[0].authors == ("A. Author",)
    assert analysis.paper.references[0].title == "Dense Encoders"
    assert analysis.paper.references[1].authors == ("Doe", "Roe")
    assert analysis.paper.references[1].title == "Structured Pruning"
    assert analysis.in_text_references
    by_marker = {item.value: item for item in analysis.in_text_references}
    assert by_marker["[1]"].validation == "validated"
    assert by_marker["[1]"].metadata["resolved_reference_ids"] == [
        analysis.paper.references[0].reference_id
    ]
    assert by_marker["Doe and Roe (2022)"].validation == "validated"
    assert by_marker["Doe and Roe (2022)"].metadata["resolved_reference_ids"] == [
        analysis.paper.references[1].reference_id
    ]
