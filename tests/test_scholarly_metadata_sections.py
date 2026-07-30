from __future__ import annotations

from localml_scholar.retrieval import RetrievalIndex, ingest_markdown
from localml_scholar.scholarly import (
    ScholarlyAnalysisPipeline,
    citation_for_range,
    validate_source_citation,
)
from localml_scholar.scholarly.sections import classify_sections


def _analysis(index: RetrievalIndex, source_name: str):
    document = next(item for item in index.documents if item.source_name == source_name)
    return document, ScholarlyAnalysisPipeline(index).analyze_paper(
        document.document_id
    )


def test_explicit_metadata_and_abstract_are_source_linked(scholarly_index) -> None:
    document, analysis = _analysis(scholarly_index, "sparse_gate_network.md")

    assert (
        analysis.paper.title.value == "Sparse Gate Networks for Sequence Classification"
    )
    assert analysis.paper.authors.value == ["Mira Chen", "Pavel Ortiz"]
    assert analysis.paper.year.value == 2025
    assert analysis.paper.venue.value == "Open Methods Workshop"
    assert "SGN" in analysis.paper.abstract.value
    for field in (
        analysis.paper.title,
        analysis.paper.authors,
        analysis.paper.year,
        analysis.paper.venue,
        analysis.paper.abstract,
    ):
        validate_source_citation(document, field.citation, field.source_text)


def test_missing_metadata_remains_absent(scholarly_index) -> None:
    _, analysis = _analysis(scholarly_index, "robust_mean_estimation.md")
    assert analysis.paper.venue is None
    assert analysis.paper.identifier is None


def test_user_metadata_priority_requires_exact_source_match() -> None:
    text = "# Exact Title\nAuthors: Ada Example\n\n## Abstract\nA paper."
    document = ingest_markdown(
        text,
        source="paper.md",
        metadata={"title": "Exact Title", "venue": "Invented Venue"},
    )
    index = RetrievalIndex.build((document,))
    analysis = ScholarlyAnalysisPipeline(index).analyze_paper(document.document_id)

    assert analysis.paper.title.extraction_method.startswith("validated_user")
    assert analysis.paper.venue is None
    assert "user_metadata_not_present_in_source:venue" in analysis.warnings


def test_section_roles_preserve_ambiguity_unknown_appendix_and_references(
    scholarly_index,
) -> None:
    _, analysis = _analysis(scholarly_index, "robust_mean_estimation.md")
    roles = {section.heading: section.roles for section in analysis.paper.sections}

    assert roles["Assumptions and Theory"] == ("theory",)
    assert roles["Discussion and Limitations"] == ("discussion", "limitations")
    assert roles["References"] == ("references",)

    document = ingest_markdown("# Paper\n\n## Curious Material\nText.", source="x.md")
    classified = classify_sections(document)
    assert classified[1].roles == ("unknown",)


def test_source_citation_exact_ranges_and_deterministic_ids(scholarly_index) -> None:
    document, first = _analysis(scholarly_index, "sparse_gate_network.md")
    second = ScholarlyAnalysisPipeline(scholarly_index).analyze_paper(
        document.document_id
    )
    start = document.text.index("SpiralBench")
    citation = citation_for_range(document, start, start + len("SpiralBench"))

    assert first.analysis_id == second.analysis_id
    assert first.to_dict() == second.to_dict()
    assert (
        document.text[citation.start_character : citation.end_character]
        == "SpiralBench"
    )
    validate_source_citation(document, citation, "SpiralBench")
