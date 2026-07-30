"""Deterministic scholarly section-role classification."""

from __future__ import annotations

import re
from collections.abc import Mapping

from localml_scholar.retrieval import Document, Section
from localml_scholar.scholarly.models import PaperSection
from localml_scholar.scholarly.source import citation_for_range

_ROLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("title", ("title",)),
    ("abstract", ("abstract", "summary")),
    ("introduction", ("introduction", "overview")),
    ("background", ("background", "preliminar")),
    ("related_work", ("related work", "prior work", "literature")),
    ("methodology", ("method", "approach", "model", "framework")),
    ("theory", ("theory", "theoretical analysis", "proof")),
    ("algorithm", ("algorithm", "procedure", "pseudocode")),
    ("data", ("data", "dataset", "corpus")),
    ("experiments", ("experiment", "evaluation", "empirical", "setup")),
    ("results", ("result", "finding")),
    ("ablation", ("ablation", "sensitivity", "component analysis")),
    ("discussion", ("discussion",)),
    ("limitations", ("limitation", "threats to validity")),
    ("conclusion", ("conclusion", "concluding")),
    ("references", ("references", "bibliography")),
    ("appendix", ("appendix", "supplement")),
)


def _normalize_heading(heading: str) -> str:
    value = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[A-Z])[\s.:_-]+", "", heading)
    return re.sub(r"\s+", " ", value.casefold()).strip(" #.:")


def classify_section(
    document: Document,
    section: Section,
    *,
    overrides: Mapping[str, tuple[str, ...]] | None = None,
) -> PaperSection:
    """Classify one section using inspectable heading and narrow content rules."""
    if overrides is not None and section.section_id in overrides:
        roles = tuple(overrides[section.section_id])
        reasons = ("explicit_user_override",)
        confidence = "high"
    else:
        heading = _normalize_heading(section.heading or "")
        found: list[str] = []
        reasons_list: list[str] = []
        for role, fragments in _ROLE_PATTERNS:
            matches = tuple(fragment for fragment in fragments if fragment in heading)
            if matches:
                found.append(role)
                reasons_list.append(f"heading_contains:{matches[0]}")
        if (
            not found
            and section.ordinal == 0
            and section.heading
            and section.level == 1
        ):
            found.append("title")
            reasons_list.append("first_level_one_heading")
        if not found:
            body_prefix = section.text[:500].casefold()
            if re.search(r"\babstract\b", body_prefix):
                found.append("abstract")
                reasons_list.append("content_contains_explicit_abstract_label")
            elif re.search(r"\breferences\b", body_prefix) and section.ordinal > 0:
                found.append("references")
                reasons_list.append("content_contains_explicit_references_label")
        roles = tuple(dict.fromkeys(found or ["unknown"]))
        reasons = tuple(reasons_list or ["no_recognized_heading_or_content_rule"])
        confidence = (
            "high"
            if any(
                role in {"abstract", "ablation", "limitations", "references"}
                for role in roles
            )
            else "medium"
            if roles != ("unknown",)
            else "low"
        )
    return PaperSection(
        section_id=section.section_id,
        heading=section.heading,
        roles=roles,
        reasons=reasons,
        confidence=confidence,
        citation=citation_for_range(
            document,
            section.start_character,
            section.end_character,
        ),
    )


def classify_sections(
    document: Document,
    *,
    overrides: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[PaperSection, ...]:
    """Classify every section without discarding unknown sections."""
    return tuple(
        classify_section(document, section, overrides=overrides)
        for section in document.sections
    )
