"""Summaries, checklists, comparisons, research gaps, and Markdown renderers."""

from __future__ import annotations

from collections import defaultdict

from localml_scholar.retrieval.documents import stable_identifier
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.models import (
    ChecklistItem,
    ComparisonDimension,
    PaperAnalysis,
    PaperComparison,
    ReproductionChecklist,
    ResearchGapCandidate,
    RiskFlag,
    ScholarlyEvidence,
    StructuredSummary,
    SummaryField,
)

_CHECKLIST_ORDER: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Data", "dataset", ("dataset",)),
    ("Data", "split", ("split",)),
    ("Data", "preprocessing", ("preprocessing",)),
    ("Data", "sample sizes", ("sample_count",)),
    ("Model", "architecture", ("architecture",)),
    ("Model", "dimensions", ("hidden_dimension",)),
    ("Model", "layers", ("layers",)),
    ("Model", "heads", ("heads",)),
    ("Model", "initialization", ("initialization",)),
    ("Training", "objective", ("objective_function",)),
    ("Training", "optimizer", ("optimizer",)),
    ("Training", "learning rate", ("learning_rate",)),
    ("Training", "schedule", ("learning_rate_schedule",)),
    ("Training", "batch size", ("batch_size",)),
    ("Training", "epochs or steps", ("epochs", "training_steps")),
    ("Training", "weight decay", ("weight_decay",)),
    ("Training", "dropout", ("dropout",)),
    ("Training", "random seeds", ("random_seed",)),
    ("Training", "stopping criteria", ("stopping_criteria",)),
    ("Evaluation", "metrics", ("metric",)),
    ("Evaluation", "baselines", ("baseline",)),
    ("Evaluation", "evaluation protocol", ("evaluation_protocol",)),
    ("Evaluation", "statistical tests", ("statistical_test",)),
    ("Evaluation", "number of runs", ("trials",)),
    ("Evaluation", "hardware", ("hardware",)),
    ("Evaluation", "software", ("software",)),
)


def _evidence_by_category(
    analysis: PaperAnalysis,
) -> dict[str, list[ScholarlyEvidence]]:
    grouped: dict[str, list[ScholarlyEvidence]] = defaultdict(list)
    for item in (
        analysis.methodology
        + analysis.datasets
        + analysis.metrics
        + analysis.baselines
        + analysis.hyperparameters
        + analysis.results
        + analysis.assumptions
        + analysis.claims
        + analysis.ablations
        + analysis.limitations
    ):
        category = (
            item.value.get("name")
            if item.category == "hyperparameter" and isinstance(item.value, dict)
            else item.category
        )
        grouped[str(category)].append(item)
    return grouped


def build_structured_summary(analysis: PaperAnalysis) -> StructuredSummary:
    """Create a citation-only summary without generated factual prose."""
    contribution_claims = tuple(
        item
        for item in analysis.claims
        if item.metadata.get("claim_type") == "contribution"
    )
    fields_source: tuple[tuple[str, tuple[ScholarlyEvidence, ...]], ...] = (
        (
            "paper_identity",
            tuple(
                item
                for item in (
                    analysis.paper.title,
                    analysis.paper.authors,
                    analysis.paper.year,
                    analysis.paper.venue,
                )
                if item is not None
            ),
        ),
        (
            "research_question",
            tuple(
                item
                for item in analysis.claims
                if item.metadata.get("claim_type") == "contribution"
            )[:1],
        ),
        ("main_contributions", contribution_claims),
        ("method", analysis.methodology),
        ("assumptions", analysis.assumptions),
        ("datasets", analysis.datasets),
        (
            "experiments",
            tuple(
                experiment.purpose
                for experiment in analysis.experiments
                if experiment.purpose is not None
            ),
        ),
        ("key_results", analysis.results),
        ("ablations", analysis.ablations),
        ("limitations", analysis.limitations),
        (
            "conclusions",
            tuple(
                item
                for item in analysis.claims
                if item.metadata.get("claim_type")
                in {"empirical_result", "theoretical_result"}
            ),
        ),
    )
    fields: list[SummaryField] = []
    for name, evidence in fields_source:
        cleaned = tuple(
            item for item in evidence if isinstance(item, ScholarlyEvidence)
        )
        validations = {item.validation for item in cleaned}
        status = (
            "missing"
            if not cleaned
            else "conflicting"
            if "conflicting" in validations
            else "ambiguous"
            if "ambiguous" in validations
            else "found"
        )
        fields.append(SummaryField(name=name, status=status, evidence=cleaned))
    role_counts: dict[str, int] = defaultdict(int)
    for section in analysis.paper.sections:
        for role in section.roles:
            role_counts[role] += 1
    completeness = {
        "fields_extracted": [field.name for field in fields if field.status == "found"],
        "fields_missing": [field.name for field in fields if field.status == "missing"],
        "fields_ambiguous": [
            field.name for field in fields if field.status == "ambiguous"
        ],
        "fields_conflicting": [
            field.name for field in fields if field.status == "conflicting"
        ],
        "unresolved_symbols": list(analysis.unresolved_symbols),
        "unparsed_table_warning_count": sum(
            len(table.warnings) for table in analysis.tables
        ),
        "uncited_summary_fields": [],
        "source_section_roles": dict(sorted(role_counts.items())),
    }
    return StructuredSummary(
        paper_id=analysis.paper.paper_id,
        fields=tuple(fields),
        completeness=completeness,
    )


def build_reproduction_checklist(
    analysis: PaperAnalysis,
    config: ScholarlyConfig | None = None,
) -> ReproductionChecklist:
    """Classify required reproduction details without filling absent values."""
    resolved = config or ScholarlyConfig()
    grouped = _evidence_by_category(analysis)
    items: list[ChecklistItem] = []
    for section, label, categories in _CHECKLIST_ORDER:
        values = tuple(
            item for category in categories for item in grouped.get(category, ())
        )
        normalized = {str(item.normalized_value) for item in values}
        status = (
            "not_found"
            if not values
            else "conflicting"
            if len(normalized) > 1
            and any(item.validation == "conflicting" for item in values)
            else "ambiguous"
            if any(item.validation == "ambiguous" for item in values)
            else "found"
        )
        items.append(
            ChecklistItem(
                section=section,
                item=label,
                status=status,
                values=values,
                notes=(
                    ("Multiple source values require scope review.",)
                    if status == "conflicting"
                    else ()
                ),
            )
        )
    by_label = {item.item: item for item in items}
    risk_specs = (
        ("dataset", "dataset unavailable in text", "Data", "high"),
        ("split", "data split unspecified", "Data", "high"),
        ("preprocessing", "preprocessing unspecified", "Data", "medium"),
        ("random seeds", "random seed missing", "Training", "medium"),
        (
            "epochs or steps",
            "number of training epochs or steps missing",
            "Training",
            "high",
        ),
        (
            "evaluation protocol",
            "evaluation protocol unspecified",
            "Evaluation",
            "high",
        ),
        ("hardware", "hardware missing", "Evaluation", "low"),
        ("number of runs", "number of runs missing", "Evaluation", "medium"),
    )
    risks = []
    for label, reason, section, severity in risk_specs:
        if by_label[label].status == "not_found":
            if label == "random seeds" and not resolved.risk_flag_missing_seed:
                continue
            if label == "hardware" and not resolved.risk_flag_missing_hardware:
                continue
            risks.append(
                RiskFlag(
                    risk_id=stable_identifier("risk", analysis.paper.paper_id, label),
                    reason=reason,
                    absence_scope="complete analyzed document",
                    checklist_section=section,
                    severity=severity,
                )
            )
    for item in items:
        if item.status == "conflicting":
            risks.append(
                RiskFlag(
                    risk_id=stable_identifier(
                        "risk", analysis.paper.paper_id, item.section, item.item
                    ),
                    reason=f"conflicting values for {item.item}",
                    absence_scope="retained source scopes",
                    checklist_section=item.section,
                    severity="high",
                    citations=tuple(value.citation for value in item.values),
                )
            )
    return ReproductionChecklist(
        paper_id=analysis.paper.paper_id,
        items=tuple(items),
        risk_flags=tuple(risks),
    )


def compare_analyses(analyses: tuple[PaperAnalysis, ...]) -> PaperComparison:
    """Compare cited normalized fields and mark invalid result comparisons."""
    if len(analyses) < 2:
        raise ValueError("At least two paper analyses are required.")
    if len({item.paper.paper_id for item in analyses}) != len(analyses):
        raise ValueError("Paper comparison requires distinct paper IDs.")
    dimensions = (
        ("task", "claim"),
        ("assumptions", "assumptions"),
        ("method", "methodology"),
        ("datasets", "datasets"),
        ("metrics", "metrics"),
        ("baselines", "baselines"),
        ("training_setup", "hyperparameters"),
        ("key_results", "results"),
        ("limitations", "limitations"),
    )
    output: list[ComparisonDimension] = []
    for label, attribute in dimensions:
        values_by_paper: dict[str, tuple[ScholarlyEvidence, ...]] = {}
        normalized_by_paper: list[set[str]] = []
        for analysis in analyses:
            values = (
                tuple(
                    item
                    for item in analysis.claims
                    if item.metadata.get("claim_type") == "contribution"
                )
                if attribute == "claim"
                else tuple(getattr(analysis, attribute))
            )
            values_by_paper[analysis.paper.paper_id] = values
            normalized_by_paper.append({str(item.normalized_value) for item in values})
        warnings: list[str] = []
        if any(not values for values in normalized_by_paper):
            relationship = "missing"
            comparable = False
            warnings.append("missing_information_is_not_a_difference")
        elif all(
            values == normalized_by_paper[0] for values in normalized_by_paper[1:]
        ):
            relationship = "shared"
            comparable = True
        else:
            relationship = "different"
            comparable = label not in {"key_results"}
        if label == "key_results":
            datasets = [
                {str(item.normalized_value) for item in analysis.datasets}
                for analysis in analyses
            ]
            metrics = [
                {str(item.normalized_value) for item in analysis.metrics}
                for analysis in analyses
            ]
            if any(value != datasets[0] for value in datasets[1:]):
                warnings.append("different_datasets")
                comparable = False
            if any(value != metrics[0] for value in metrics[1:]):
                warnings.append("different_metrics")
                comparable = False
            if not comparable:
                relationship = "incomparable"
                warnings.append("no_superiority_ranking_permitted")
        output.append(
            ComparisonDimension(
                name=label,
                values_by_paper=values_by_paper,
                relationship=relationship,
                comparable=comparable,
                warnings=tuple(warnings),
            )
        )
    return PaperComparison(
        paper_ids=tuple(item.paper.paper_id for item in analyses),
        dimensions=tuple(output),
        false_superiority_claim_count=0,
    )


def identify_research_gaps(
    analyses: tuple[PaperAnalysis, ...],
    config: ScholarlyConfig | None = None,
) -> tuple[ResearchGapCandidate, ...]:
    """Build a conservative worksheet; never make literature-wide novelty claims."""
    resolved = config or ScholarlyConfig()
    candidates: dict[tuple[str, str], ResearchGapCandidate] = {}
    caution = (
        "This candidate does not establish novelty; no external literature search "
        "was performed.",
    )
    for analysis in analyses:
        for limitation in analysis.limitations:
            item = ResearchGapCandidate(
                gap_id=stable_identifier(
                    "gap", analysis.paper.paper_id, limitation.evidence_id
                ),
                gap_type="explicit_limitation",
                statement=f"Investigate the stated constraint: {limitation.value}",
                source_basis="explicit author-stated limitation or constraint",
                citations=(limitation.citation,),
                system_inference=False,
                confidence="high",
                cautions=caution,
                question_template=(
                    "Does the method remain effective when the cited constraint "
                    "is relaxed?"
                ),
            )
            candidates[(item.gap_type, str(limitation.normalized_value))] = item
        for claim in analysis.claims:
            if claim.metadata.get("claim_type") != "future_work":
                continue
            item = ResearchGapCandidate(
                gap_id=stable_identifier(
                    "gap", analysis.paper.paper_id, claim.evidence_id
                ),
                gap_type="explicit_future_work",
                statement=str(claim.value),
                source_basis="explicit future-work statement",
                citations=(claim.citation,),
                system_inference=False,
                confidence="high",
                cautions=caution,
                question_template="Can the cited future-work direction be evaluated?",
            )
            candidates[(item.gap_type, str(claim.normalized_value))] = item
        if resolved.research_gap_from_missing_ablation and not analysis.ablations:
            basis = (
                analysis.paper.abstract
                or analysis.paper.title
                or next(iter(analysis.claims), None)
            )
            if basis is not None:
                item = ResearchGapCandidate(
                    gap_id=stable_identifier(
                        "gap", analysis.paper.paper_id, "missing_ablation"
                    ),
                    gap_type="missing_ablation",
                    statement=(
                        "A component-removal ablation was not found in the "
                        "analyzed text."
                    ),
                    source_basis=(
                        "system observation over the complete analyzed document; "
                        "citation identifies the paper, not proof of absence"
                    ),
                    citations=(basis.citation,),
                    system_inference=True,
                    confidence="medium",
                    cautions=caution,
                    question_template=(
                        "What is the effect of removing each cited method component?"
                    ),
                )
                candidates[(item.gap_type, analysis.paper.paper_id)] = item
    return tuple(candidates[key] for key in sorted(candidates))


def render_notation_markdown(analysis: PaperAnalysis) -> str:
    """Render a source-linked notation glossary."""
    lines = ["| Symbol | Meaning | Type | Source | Status |", "|---|---|---|---|---|"]
    for entry in analysis.notation:
        if entry.selected_definition is not None:
            meaning = entry.selected_definition.defining_text.replace("|", "\\|")
            source = entry.selected_definition.citation.format()
            status = "defined"
        elif entry.definition_candidates:
            meaning = "Multiple candidate definitions"
            source = "; ".join(
                item.citation.format() for item in entry.definition_candidates
            )
            status = "ambiguous"
        else:
            meaning = "Not resolved from source text"
            source = entry.occurrences[0].format()
            status = "unresolved"
        lines.append(
            f"| `{entry.raw_symbol}` | {meaning} | {entry.symbol_type} | "
            f"{source} | {status} |"
        )
    return "\n".join(lines)


def render_checklist_markdown(checklist: ReproductionChecklist) -> str:
    """Render checklist values without adding uncited factual content."""
    lines = ["| Item | Status | Value | Citation | Notes |", "|---|---|---|---|---|"]
    for item in checklist.items:
        values = "; ".join(str(value.value) for value in item.values) or "—"
        citations = "; ".join(value.citation.format() for value in item.values) or "—"
        lines.append(
            f"| {item.section}: {item.item} | {item.status} | {values} | "
            f"{citations} | {'; '.join(item.notes) or '—'} |"
        )
    return "\n".join(lines)


def render_comparison_markdown(comparison: PaperComparison) -> str:
    """Render structured multi-paper comparisons with comparability warnings."""
    paper_labels = list(comparison.paper_ids)
    lines = [
        "| Dimension | " + " | ".join(paper_labels) + " | Comparable? | Warnings |",
        "|---|" + "---|" * (len(paper_labels) + 2),
    ]
    for dimension in comparison.dimensions:
        values = []
        for paper_id in paper_labels:
            evidence = dimension.values_by_paper[paper_id]
            values.append(
                "; ".join(f"{item.value} {item.citation.format()}" for item in evidence)
                or "missing"
            )
        lines.append(
            f"| {dimension.name} | "
            + " | ".join(values)
            + f" | {'yes' if dimension.comparable else 'no'} | "
            + ("; ".join(dimension.warnings) or "—")
            + " |"
        )
    return "\n".join(lines)
