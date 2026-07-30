from __future__ import annotations

import numpy as np
import pytest

from localml_scholar.retrieval import ingest_markdown
from localml_scholar.scholarly import (
    ScholarlyAnalysisPipeline,
    detect_equation_blocks,
    normalize_equation,
    symbol_spans,
)


def _paper(index, source):
    document = next(item for item in index.documents if item.source_name == source)
    return document, ScholarlyAnalysisPipeline(index).analyze_paper(
        document.document_id
    )


def test_display_inline_unicode_numbered_and_multiline_equations(
    scholarly_index,
) -> None:
    document, analysis = _paper(scholarly_index, "sparse_gate_network.md")

    assert {item.equation_number for item in analysis.equations} >= {"1", "2", "3"}
    assert any("\n" in item.raw_text for item in analysis.equations)
    assert any("\\[" in item.raw_text for item in analysis.equations)
    assert any("$L =" in item.raw_text for item in analysis.equations)
    for equation in analysis.equations:
        start = equation.citation.start_character
        end = equation.citation.end_character
        assert document.text[start:end] == equation.raw_text

    unicode_document = ingest_markdown(
        "# Paper\n\n## Theory\nα + β ≥ γ = 0 (4)\n",
        source="unicode.md",
    )
    equations = detect_equation_blocks(unicode_document)
    assert equations[0].equation_number == "4"


def test_malformed_delimiters_are_safe_and_overlaps_are_suppressed() -> None:
    document = ingest_markdown(
        "# Paper\n\n## Method\nUnclosed $x = y.\n\n$$x = y \\tag{1}$$\n",
        source="malformed.md",
    )
    equations = detect_equation_blocks(document)
    assert len(equations) == 2
    assert sum("\\tag{1}" in item.raw_text for item in equations) == 1


def test_conservative_normalization_preserves_case_indices_and_operators() -> None:
    assert normalize_equation("  W_Q   =  x_i − b  ") == "W_Q = x_i - b"
    with pytest.raises(ValueError, match="non-empty"):
        normalize_equation("")


def test_symbol_extraction_preserves_indices_greek_latex_and_case() -> None:
    symbols = {item[0] for item in symbol_spans(r"W_Q x_i + \theta + α + f(x)")}
    assert {"W_Q", "x_i", r"\theta", "α", "f", "x"} <= symbols
    assert "W_Q" != "w_q"


def test_glossary_definitions_conflicts_and_unresolved_symbols(scholarly_index) -> None:
    _, analysis = _paper(scholarly_index, "sparse_gate_network.md")
    entries = {entry.raw_symbol: entry for entry in analysis.notation}

    assert entries["W_g"].selected_definition is not None
    assert entries["q"].selected_definition is None
    assert entries["q"].ambiguity == "multiple conflicting definition candidates"
    assert "W_q" in analysis.unresolved_symbols
    assert all(entry.occurrences for entry in analysis.notation)


def test_equation_definition_links_mark_later_and_unresolved(scholarly_index) -> None:
    _, analysis = _paper(scholarly_index, "sparse_gate_network.md")
    by_id = {item.equation_id: item for item in analysis.equation_analyses}
    q_equation = next(
        item for item in analysis.equations if item.equation_number == "3"
    )
    linked = by_id[q_equation.equation_id]

    assert "q" in linked.symbols
    assert "q" in linked.defined_later_symbols
    assert "W_q" in linked.unresolved_symbols


def test_equation_aware_search_is_deterministic_and_citation_preserving(
    scholarly_index,
) -> None:
    document, _ = _paper(scholarly_index, "sparse_gate_network.md")
    pipeline = ScholarlyAnalysisPipeline(scholarly_index)
    first = pipeline.retrieve_equation_evidence(
        "What does W_g represent in Equation 1?",
        document_id=document.document_id,
    )
    second = pipeline.retrieve_equation_evidence(
        "What does W_g represent in Equation 1?",
        document_id=document.document_id,
    )

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert first[0].equation_signals["symbol_overlap"]
    assert not first[0].equation_signals["symbolic_equivalence_claimed"]
    assert np.isfinite(first[0].scholarly_score)
