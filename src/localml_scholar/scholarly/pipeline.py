"""Transformer-independent scholarly analysis pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from localml_scholar.retrieval import Document, RetrievalIndex
from localml_scholar.scholarly.artifacts import (
    build_reproduction_checklist,
    build_structured_summary,
    compare_analyses,
    identify_research_gaps,
)
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.equations import (
    analyze_equation_links,
    build_notation_glossary,
    detect_equation_blocks,
)
from localml_scholar.scholarly.extraction import (
    extract_ablations,
    extract_assumptions,
    extract_baselines,
    extract_claims,
    extract_datasets,
    extract_hyperparameters,
    extract_in_text_references,
    extract_limitations,
    extract_methodology,
    extract_metrics,
    extract_procedures,
    extract_results,
    extract_tables,
    group_experiments,
)
from localml_scholar.scholarly.models import (
    PaperAnalysis,
    PaperComparison,
    ReproductionChecklist,
    ResearchGapCandidate,
    StructuredSummary,
)
from localml_scholar.scholarly.paper import build_paper, extract_references
from localml_scholar.scholarly.retrieval import equation_aware_search
from localml_scholar.scholarly.sections import classify_sections


class ScholarlyAnalysisPipeline:
    """Run deterministic extraction without constructing a transformer."""

    def __init__(
        self,
        index: RetrievalIndex,
        *,
        retrieval_method: str = "bm25",
        scholarly_config: ScholarlyConfig | None = None,
    ) -> None:
        if not isinstance(index, RetrievalIndex):
            raise TypeError("index must be a RetrievalIndex.")
        if retrieval_method not in {
            "bm25",
            "tfidf",
            "semantic",
            "hybrid",
            "hybrid_reranked",
        }:
            raise ValueError("Unsupported retrieval method.")
        self.index = index
        self.retrieval_method = retrieval_method
        self.config = scholarly_config or ScholarlyConfig()

    def _document(self, document_id: str) -> Document:
        matches = tuple(
            document
            for document in self.index.documents
            if document.document_id == document_id
        )
        if len(matches) != 1:
            raise ValueError(f"Index does not contain document_id {document_id!r}.")
        return matches[0]

    def analyze_paper(self, document_id: str) -> PaperAnalysis:
        """Extract a complete immutable analysis from one indexed document."""
        document = self._document(document_id)
        sections = classify_sections(document)
        references = extract_references(document, sections)
        paper, metadata_warnings = build_paper(
            document,
            sections,
            references,
            self.config,
        )
        equations = detect_equation_blocks(document, self.config)
        notation, unresolved = build_notation_glossary(
            document,
            equations,
            sections,
            self.config,
        )
        equation_analyses = analyze_equation_links(document, equations, notation)
        assumptions = extract_assumptions(document, self.config)
        claims = extract_claims(document)
        methodology = extract_methodology(document)
        datasets = extract_datasets(document)
        metrics = extract_metrics(document)
        baselines = extract_baselines(document)
        hyperparameters = extract_hyperparameters(document)
        results = extract_results(document, sections)
        ablations = extract_ablations(document, sections)
        limitations = extract_limitations(document, sections)
        procedures = extract_procedures(document, sections)
        tables = extract_tables(document, self.config)
        experiments = group_experiments(
            document,
            sections,
            datasets,
            methodology,
            baselines,
            metrics,
            hyperparameters,
            results,
            ablations,
        )
        return PaperAnalysis(
            paper=paper,
            equations=equations,
            equation_analyses=equation_analyses,
            notation=notation,
            assumptions=assumptions,
            claims=claims,
            methodology=methodology,
            procedures=procedures,
            datasets=datasets,
            metrics=metrics,
            baselines=baselines,
            hyperparameters=hyperparameters,
            experiments=experiments,
            results=results,
            tables=tables,
            ablations=ablations,
            limitations=limitations,
            in_text_references=extract_in_text_references(document, references),
            unresolved_symbols=unresolved,
            warnings=metadata_warnings,
        )

    def build_notation_glossary(self, document_id: str):
        """Return complete notation entries and unresolved symbols."""
        analysis = self.analyze_paper(document_id)
        return analysis.notation, analysis.unresolved_symbols

    def extract_experiments(self, document_id: str):
        """Return source-scoped experiment records."""
        return self.analyze_paper(document_id).experiments

    def build_reproduction_checklist(
        self,
        document_id: str,
    ) -> ReproductionChecklist:
        """Build a cited checklist and document-completeness risk flags."""
        return build_reproduction_checklist(
            self.analyze_paper(document_id),
            self.config,
        )

    def summarize_paper(self, document_id: str) -> StructuredSummary:
        """Build a structured summary with no uncited factual fields."""
        return build_structured_summary(self.analyze_paper(document_id))

    def compare_papers(self, document_ids: Sequence[str]) -> PaperComparison:
        """Compare only extracted cited fields across distinct papers."""
        if isinstance(document_ids, (str, bytes)) or not isinstance(
            document_ids, Sequence
        ):
            raise TypeError("document_ids must be a sequence.")
        return compare_analyses(
            tuple(self.analyze_paper(document_id) for document_id in document_ids)
        )

    def identify_research_gaps(
        self,
        document_ids: Sequence[str],
    ) -> tuple[ResearchGapCandidate, ...]:
        """Return planning candidates with explicit no-novelty cautions."""
        if isinstance(document_ids, (str, bytes)) or not isinstance(
            document_ids, Sequence
        ):
            raise TypeError("document_ids must be a sequence.")
        return identify_research_gaps(
            tuple(self.analyze_paper(document_id) for document_id in document_ids),
            self.config,
        )

    def retrieve_equation_evidence(
        self,
        query: str,
        *,
        document_id: str,
        top_k: int = 5,
    ):
        """Apply opt-in equation signals over an unchanged base retriever."""
        return equation_aware_search(
            self.index,
            query,
            document_id=document_id,
            method=self.retrieval_method,
            top_k=top_k,
            config=self.config,
        )
