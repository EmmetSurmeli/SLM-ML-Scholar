"""Deterministic source-scoped scholarly field extraction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from localml_scholar.retrieval import Document
from localml_scholar.retrieval.documents import stable_identifier
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.models import (
    ExperimentRecord,
    ExtractedTable,
    PaperSection,
    Procedure,
    ReferenceEntry,
    ScholarlyEvidence,
)
from localml_scholar.scholarly.source import citation_for_range, section_for_range

_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"(?P<percent>%?)"
    r"(?:\s*(?:±|\+/-)\s*(?P<uncertainty>\d+(?:\.\d+)?))?"
    r"(?![A-Za-z0-9_])"
)
_ASSUMPTION = re.compile(
    r"\b(?:we assume|under the assumption|suppose that|provided that|subject to|"
    r"holds when|independent and identically distributed|finite variance|"
    r"stationary|convex|Gaussian)\b",
    re.IGNORECASE,
)
_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "contribution",
        re.compile(r"\b(?:we introduce|we propose|our contribution)\b", re.I),
    ),
    ("theoretical_result", re.compile(r"\b(?:we prove|we show|theorem)\b", re.I)),
    (
        "comparative_result",
        re.compile(r"\b(?:outperforms?|improves? over|better than)\b", re.I),
    ),
    (
        "empirical_result",
        re.compile(
            r"\b(?:we demonstrate|our results indicate|achieves?|reduces?)\b", re.I
        ),
    ),
    ("future_work", re.compile(r"\b(?:future work|we plan to|remains to)\b", re.I)),
)
_LIMITATION = re.compile(
    r"\b(?:limitation|does not|fails? when|restricted to|future work|"
    r"computationally expensive|sensitive to|requires)\b",
    re.IGNORECASE,
)
_RESULT_VERB = re.compile(
    r"\b(?:achieves?|obtains?|reports?|produces?|reduces?|improves?|"
    r"increases?|decreases?|reaches?|yields?)\b",
    re.IGNORECASE,
)
_NON_RESULT_UNIT = re.compile(
    r"^\s*(?:trials?|runs?|epochs?|steps?|samples?|examples?)\b",
    re.IGNORECASE,
)
_METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "perplexity",
    "bleu",
    "mean squared error",
    "mse",
    "sharpe ratio",
    "log loss",
    "calibration error",
)
_HYPERPARAMETERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "learning_rate",
        re.compile(
            r"\blearning\s+rate(?:\s+(?:of|is)|\s*=|:)?\s*"
            r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)",
            re.I,
        ),
    ),
    (
        "batch_size",
        re.compile(
            r"\bbatch\s+size(?:\s+(?:of|is)|\s*=|:)?\s*(?P<value>\d+)",
            re.I,
        ),
    ),
    ("epochs", re.compile(r"\b(?P<value>\d+)\s+epochs?\b", re.I)),
    (
        "training_steps",
        re.compile(r"\b(?P<value>\d+)\s+(?:training\s+)?steps?\b", re.I),
    ),
    (
        "weight_decay",
        re.compile(
            r"\bweight\s+decay(?:\s+(?:of|is)|\s*=|:)?\s*"
            r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)",
            re.I,
        ),
    ),
    (
        "dropout",
        re.compile(
            r"\bdropout(?:\s+(?:of|is)|\s*=|:)?\s*(?P<value>[0-9.]+)",
            re.I,
        ),
    ),
    (
        "random_seed",
        re.compile(
            r"\b(?:random\s+)?seed(?:\s+(?:of|is)|\s*=|:)?\s*"
            r"(?P<value>\d+)",
            re.I,
        ),
    ),
    (
        "hidden_dimension",
        re.compile(
            r"\bhidden\s+(?:size|dimension)(?:\s+(?:of|is)|\s*=|:)?\s*"
            r"(?P<value>\d+)",
            re.I,
        ),
    ),
    ("layers", re.compile(r"\b(?P<value>\d+)[ -]layers?\b", re.I)),
    ("heads", re.compile(r"\b(?P<value>\d+)[ -](?:attention )?heads?\b", re.I)),
    ("trials", re.compile(r"\b(?P<value>\d+)\s+(?:runs?|trials?)\b", re.I)),
)
_METHODOLOGY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "optimizer",
        re.compile(
            r"\b(?:optimizer|optimized using|we use)\s+"
            r"(?P<value>AdamW?|SGD|momentum)\b",
            re.I,
        ),
    ),
    (
        "objective_function",
        re.compile(r"\b(?:objective|loss function|minimize|maximize)\b[^.\n]*", re.I),
    ),
    (
        "architecture",
        re.compile(
            r"\b(?:architecture|encoder|decoder|network|regression model)\b[^.\n]*",
            re.I,
        ),
    ),
    (
        "preprocessing",
        re.compile(
            r"\b(?:preprocess|normalize|standardize|tokenize|filter)\w*\b[^.\n]*", re.I
        ),
    ),
    ("hardware", re.compile(r"\b(?:GPU|CPU|TPU|NVIDIA|AMD|Apple M\d)\b[^.\n]*", re.I)),
    (
        "software",
        re.compile(
            r"\b(?:PyTorch|TensorFlow|JAX|NumPy|scikit-learn|R version)\b[^.\n]*", re.I
        ),
    ),
)


def sentence_spans(text: str, start: int = 0, end: int | None = None):
    """Yield trimmed source sentence spans without rewriting their text."""
    limit = len(text) if end is None else end
    window = text[start:limit]
    for paragraph_match in re.finditer(
        r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)",
        window,
        re.DOTALL,
    ):
        paragraph_start = start + paragraph_match.start()
        paragraph = paragraph_match.group()
        if paragraph.lstrip().startswith("#"):
            newline = paragraph.find("\n")
            if newline < 0:
                continue
            paragraph_start += newline + 1
            paragraph = paragraph[newline + 1 :]
        for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\Z)", paragraph, re.DOTALL):
            left = paragraph_start + match.start()
            right = paragraph_start + match.end()
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                yield left, right


def _document_sentence_spans(document: Document) -> Iterator[tuple[int, int]]:
    """Yield sentences without allowing a span to cross a section boundary."""
    for section in document.sections:
        yield from sentence_spans(
            document.text,
            section.start_character,
            section.end_character,
        )


def make_evidence(
    document: Document,
    category: str,
    value: Any,
    start: int,
    end: int,
    method: str,
    confidence: str = "medium",
    *,
    normalized_value: Any | None = None,
    validation: str = "validated",
    metadata: dict[str, Any] | None = None,
) -> ScholarlyEvidence:
    """Construct one evidence item from an exact source slice."""
    source = document.text[start:end]
    normalized = (
        normalized_value
        if normalized_value is not None
        else value.casefold().strip()
        if isinstance(value, str)
        else value
    )
    return ScholarlyEvidence(
        evidence_id=stable_identifier(
            "ev", document.document_id, category, start, end, value
        ),
        category=category,
        value=value,
        normalized_value=normalized,
        citation=citation_for_range(document, start, end),
        source_text=source,
        extraction_method=method,
        confidence=confidence,
        validation=validation,
        metadata={} if metadata is None else metadata,
    )


def _sentence_matches(
    document: Document,
    category: str,
    predicate: Callable[[str], bool],
    method: str,
    *,
    value_transform: Callable[[str], Any] = lambda text: text,
    confidence: str = "high",
) -> tuple[ScholarlyEvidence, ...]:
    return tuple(
        make_evidence(
            document,
            category,
            value_transform(document.text[start:end]),
            start,
            end,
            method,
            confidence,
        )
        for start, end in _document_sentence_spans(document)
        if predicate(document.text[start:end])
    )


def extract_assumptions(
    document: Document,
    config: ScholarlyConfig,
) -> tuple[ScholarlyEvidence, ...]:
    """Extract explicit assumptions; inferred prerequisites remain opt-in."""
    explicit = list(
        _sentence_matches(
            document,
            "assumption",
            lambda text: bool(_ASSUMPTION.search(text)),
            "explicit_assumption_phrase",
        )
    )
    if config.include_inferred_assumptions:
        for start, end in _document_sentence_spans(document):
            text = document.text[start:end]
            if re.search(
                r"\b(?:requires?|necessary)\b", text, re.I
            ) and not _ASSUMPTION.search(text):
                explicit.append(
                    make_evidence(
                        document,
                        "inferred_prerequisite",
                        text,
                        start,
                        end,
                        "inferred_requirement_phrase",
                        "low",
                        validation="ambiguous",
                        metadata={"explicit": False},
                    )
                )
    return tuple(explicit)


def extract_claims(document: Document) -> tuple[ScholarlyEvidence, ...]:
    """Extract narrow explicit claim patterns while preserving qualifiers."""
    claims: list[ScholarlyEvidence] = []
    for start, end in _document_sentence_spans(document):
        text = document.text[start:end]
        for claim_type, pattern in _CLAIM_PATTERNS:
            if pattern.search(text):
                qualifiers = tuple(
                    value
                    for value in (
                        "on average",
                        "under these settings",
                        "statistically significant",
                        "in our experiments",
                        "may",
                    )
                    if value in text.casefold()
                )
                claims.append(
                    make_evidence(
                        document,
                        "claim",
                        text,
                        start,
                        end,
                        f"explicit_{claim_type}_phrase",
                        metadata={
                            "claim_type": claim_type,
                            "qualifiers": list(qualifiers),
                            "numbers": [
                                match.group() for match in _NUMBER.finditer(text)
                            ],
                        },
                    )
                )
                break
    return tuple(claims)


def extract_methodology(document: Document) -> tuple[ScholarlyEvidence, ...]:
    """Extract explicitly named methodology fields with source scope."""
    values: list[ScholarlyEvidence] = []
    for category, pattern in _METHODOLOGY_PATTERNS:
        for match in pattern.finditer(document.text):
            start, end = match.span()
            value = match.groupdict().get("value") or match.group()
            values.append(
                make_evidence(
                    document,
                    category,
                    value.strip(),
                    start,
                    end,
                    f"explicit_{category}_pattern",
                )
            )
    return _deduplicate(values)


def extract_hyperparameters(document: Document) -> tuple[ScholarlyEvidence, ...]:
    """Extract raw hyperparameter values without inventing common defaults."""
    values: list[ScholarlyEvidence] = []
    for name, pattern in _HYPERPARAMETERS:
        for match in pattern.finditer(document.text):
            start, end = match.span()
            raw_value = match.group("value")
            try:
                parsed: int | float = (
                    float(raw_value)
                    if any(marker in raw_value.casefold() for marker in (".", "e"))
                    else int(raw_value)
                )
            except ValueError:
                parsed = raw_value
            section = section_for_range(document, start, end)
            values.append(
                make_evidence(
                    document,
                    "hyperparameter",
                    {"name": name, "raw_value": raw_value, "parsed_value": parsed},
                    start,
                    end,
                    f"explicit_{name}_pattern",
                    normalized_value={"name": name, "value": parsed},
                    metadata={
                        "name": name,
                        "scope_section_id": section.section_id,
                    },
                )
            )
    by_name: dict[str, set[str]] = {}
    for item in values:
        name = item.value["name"]
        by_name.setdefault(name, set()).add(str(item.value["raw_value"]))
    return tuple(
        ScholarlyEvidence(
            **{
                **vars(item),
                "validation": (
                    "conflicting"
                    if len(by_name[item.value["name"]]) > 1
                    else item.validation
                ),
                "metadata": {
                    **item.metadata,
                    "conflicting_values": sorted(by_name[item.value["name"]]),
                },
            }
        )
        for item in values
    )


def extract_datasets(document: Document) -> tuple[ScholarlyEvidence, ...]:
    pattern = re.compile(
        r"\b(?P<name>[A-Z][A-Za-z0-9_-]*"
        r"(?:[ \t]+[A-Z][A-Za-z0-9_-]*){0,3})[ \t]+"
        r"(?:dataset|corpus)\b|\b(?:dataset|corpus)\s+(?:called|named)?\s*"
        r"(?P<after>[A-Z][A-Za-z0-9_-]+)",
    )
    values: list[ScholarlyEvidence] = []
    for match in pattern.finditer(document.text):
        name = match.group("name") or match.group("after")
        name = re.sub(r"^(?:The|A|An)[ \t]+", "", name)
        values.append(
            make_evidence(
                document,
                "dataset",
                name,
                *match.span(),
                "explicit_dataset_noun_pattern",
            )
        )
    sample_pattern = re.compile(
        r"\b(?P<count>\d[\d,]*)[ \t]+"
        r"(?:(?P<split>training|validation|test)[ \t]+)?"
        r"(?:examples|samples)\b",
        re.I,
    )
    for match in sample_pattern.finditer(document.text):
        count = int(match.group("count").replace(",", ""))
        split = (
            None if match.group("split") is None else match.group("split").casefold()
        )
        values.append(
            make_evidence(
                document,
                "sample_count",
                {"count": count, "split": split},
                *match.span(),
                "explicit_sample_count_pattern",
                normalized_value={"count": count, "split": split},
            )
        )
        if split is not None:
            values.append(
                make_evidence(
                    document,
                    "split",
                    split,
                    *match.span(),
                    "explicit_named_split_pattern",
                )
            )
    return _deduplicate(values)


def extract_metrics(document: Document) -> tuple[ScholarlyEvidence, ...]:
    values: list[ScholarlyEvidence] = []
    for metric in _METRICS:
        for match in re.finditer(rf"\b{re.escape(metric)}\b", document.text, re.I):
            values.append(
                make_evidence(
                    document,
                    "metric",
                    match.group(),
                    *match.span(),
                    "metric_lexicon_exact_phrase",
                )
            )
    return _deduplicate(values)


def extract_baselines(document: Document) -> tuple[ScholarlyEvidence, ...]:
    pattern = re.compile(
        r"\b(?P<name>[A-Z][A-Za-z0-9_+-]*(?:\s+[A-Z][A-Za-z0-9_+-]*){0,2})"
        r"\s+(?:is\s+)?(?:the\s+)?baseline\b"
        r"|\bbaseline\s+(?P<after>[A-Z][A-Za-z0-9_+-]+)",
    )
    return _deduplicate(
        [
            make_evidence(
                document,
                "baseline",
                match.group("name") or match.group("after"),
                *match.span(),
                "explicit_baseline_noun_pattern",
            )
            for match in pattern.finditer(document.text)
        ],
    )


def extract_results(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[ScholarlyEvidence, ...]:
    result_sections = {
        section.section_id
        for section in sections
        if set(section.roles) & {"results", "experiments", "ablation"}
    }
    values: list[ScholarlyEvidence] = []
    for start, end in _document_sentence_spans(document):
        section = section_for_range(document, start, end)
        text = document.text[start:end]
        numbers = list(_NUMBER.finditer(text))
        mentions_metric = any(metric in text.casefold() for metric in _METRICS)
        if section.section_id not in result_sections or not numbers:
            continue
        if not mentions_metric and _RESULT_VERB.search(text) is None:
            continue
        for match in numbers:
            prefix = text[: match.start()]
            suffix = text[match.end() :]
            if re.search(r"\b(?:table|figure|equation)\s*$", prefix, re.I):
                continue
            if _NON_RESULT_UNIT.match(suffix):
                continue
            raw = match.group()
            parsed = float(match.group("value"))
            values.append(
                make_evidence(
                    document,
                    "result",
                    {
                        "raw_value": raw,
                        "value": parsed,
                        "unit": "%" if match.group("percent") else None,
                        "uncertainty": match.group("uncertainty"),
                        "context": text,
                    },
                    start,
                    end,
                    "number_in_result_scope",
                    metadata={"scope_section_id": section.section_id},
                )
            )
    return tuple(values)


def extract_ablations(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[ScholarlyEvidence, ...]:
    ablation_sections = {
        section.section_id for section in sections if "ablation" in section.roles
    }
    pattern = re.compile(
        r"\b(?:without|removing|replacing|effect of|sensitivity to|"
        r"component analysis)\b",
        re.I,
    )
    return tuple(
        make_evidence(
            document,
            "ablation",
            document.text[start:end],
            start,
            end,
            "ablation_heading_or_phrase",
        )
        for start, end in _document_sentence_spans(document)
        if section_for_range(document, start, end).section_id in ablation_sections
        or pattern.search(document.text[start:end])
    )


def extract_limitations(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[ScholarlyEvidence, ...]:
    limitation_sections = {
        section.section_id for section in sections if "limitations" in section.roles
    }
    values = []
    for start, end in _document_sentence_spans(document):
        text = document.text[start:end]
        section_id = section_for_range(document, start, end).section_id
        if section_id in limitation_sections or _LIMITATION.search(text):
            limitation_type = (
                "author_stated"
                if section_id in limitation_sections or "limitation" in text.casefold()
                else "explicit_constraint"
            )
            values.append(
                make_evidence(
                    document,
                    "limitation",
                    text,
                    start,
                    end,
                    "limitation_section_or_phrase",
                    metadata={"limitation_type": limitation_type},
                )
            )
    return tuple(values)


def extract_in_text_references(
    document: Document,
    references: tuple[ReferenceEntry, ...],
) -> tuple[ScholarlyEvidence, ...]:
    """Extract citation markers and resolve only exact deterministic matches."""
    patterns = (
        re.compile(r"\[(?:\d+(?:\s*,\s*\d+)*)\]"),
        re.compile(r"\([A-Z][A-Za-z-]+ et al\.,\s*(?:19|20)\d{2}\)"),
        re.compile(r"\b[A-Z][A-Za-z-]+ and [A-Z][A-Za-z-]+ \((?:19|20)\d{2}\)"),
    )
    values = []
    for pattern in patterns:
        for match in pattern.finditer(document.text):
            if any(
                reference.citation.start_character
                <= match.start()
                < match.end()
                <= reference.citation.end_character
                for reference in references
            ):
                continue
            resolved = _resolve_reference_marker(match.group(), references)
            values.append(
                make_evidence(
                    document,
                    "in_text_reference",
                    match.group(),
                    *match.span(),
                    "explicit_reference_marker",
                    metadata={"resolved_reference_ids": list(resolved)},
                    confidence="high" if len(resolved) == 1 else "medium",
                    validation="validated" if resolved else "unresolved",
                )
            )
    return _deduplicate(values)


def _resolve_reference_marker(
    marker: str,
    references: tuple[ReferenceEntry, ...],
) -> tuple[str, ...]:
    number_match = re.fullmatch(r"\[([\d,\s]+)\]", marker)
    if number_match is not None:
        ordinals = tuple(
            int(value.strip()) for value in number_match.group(1).split(",")
        )
        if all(1 <= ordinal <= len(references) for ordinal in ordinals):
            return tuple(references[ordinal - 1].reference_id for ordinal in ordinals)
        return ()

    year_match = re.search(r"(?:19|20)\d{2}", marker)
    if year_match is None:
        return ()
    year = int(year_match.group())
    surnames = tuple(
        value.casefold()
        for value in re.findall(r"\b[A-Z][A-Za-z-]+\b", marker[: year_match.start()])
        if value.casefold() not in {"et", "al"}
    )
    candidates = []
    for reference in references:
        author_text = " ".join(reference.authors).casefold()
        if (
            reference.year == year
            and surnames
            and all(surname in author_text for surname in surnames)
        ):
            candidates.append(reference.reference_id)
    return tuple(candidates) if len(candidates) == 1 else ()


def extract_procedures(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[Procedure, ...]:
    algorithm_sections = {
        section.section_id
        for section in sections
        if set(section.roles) & {"algorithm", "methodology"}
    }
    procedures: list[Procedure] = []
    step_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)(?:\d+[.)]|[-*])\s+(?P<step>[^\n]+)$"
    )
    for section in document.sections:
        if section.section_id not in algorithm_sections:
            continue
        steps = []
        spans = []
        for match in step_pattern.finditer(section.text):
            start = section.start_character + match.start("step")
            end = section.start_character + match.end("step")
            spans.append((start, end))
            steps.append(
                make_evidence(
                    document,
                    "procedure_step",
                    match.group("step"),
                    start,
                    end,
                    "ordered_or_bulleted_step",
                )
            )
        if steps:
            start = min(item[0] for item in spans)
            end = max(item[1] for item in spans)
            name = section.heading or "Unnamed procedure"
            procedures.append(
                Procedure(
                    procedure_id=stable_identifier(
                        "proc", document.document_id, section.section_id, start, end
                    ),
                    name=name,
                    inputs=tuple(
                        match.group("value").strip()
                        for match in re.finditer(
                            r"(?im)^\s*Input\s*:\s*(?P<value>[^\n]+)",
                            section.text,
                        )
                    ),
                    outputs=tuple(
                        match.group("value").strip()
                        for match in re.finditer(
                            r"(?im)^\s*(?:Output|Return)\s*:\s*(?P<value>[^\n]+)",
                            section.text,
                        )
                    ),
                    steps=tuple(steps),
                    citation=citation_for_range(document, start, end),
                )
            )
    return tuple(procedures)


def extract_tables(
    document: Document,
    config: ScholarlyConfig,
) -> tuple[ExtractedTable, ...]:
    """Parse only explicit Markdown or consistently delimited text tables."""
    tables: list[ExtractedTable] = []
    if config.parse_markdown_tables:
        pattern = re.compile(
            r"(?m)^(?P<header>\s*\|[^\n]+\|\s*)\n"
            r"(?P<separator>\s*\|(?:\s*:?-+:?\s*\|)+\s*)\n"
            r"(?P<rows>(?:\s*\|[^\n]+\|\s*(?:\n|$))+)"
        )
        for match in pattern.finditer(document.text):
            headers = _pipe_cells(match.group("header"))
            row_lines = [
                line for line in match.group("rows").splitlines() if line.strip()
            ]
            rows = tuple(_pipe_cells(line) for line in row_lines)
            warnings = ()
            if any(len(row) != len(headers) for row in rows):
                continue
            start, end = match.span()
            caption = _preceding_caption(document.text, start)
            tables.append(
                ExtractedTable(
                    table_id=stable_identifier(
                        "table", document.document_id, start, end
                    ),
                    caption=caption,
                    headers=headers,
                    rows=rows,
                    citation=citation_for_range(document, start, end),
                    raw_text=document.text[start:end],
                    parsing_method="markdown_pipe_table",
                    warnings=warnings,
                )
            )
    if config.parse_delimited_tables:
        tables.extend(
            _delimited_tables(
                document, occupied=tuple(table.citation for table in tables)
            )
        )
    return tuple(sorted(tables, key=lambda item: item.citation.start_character))


def _pipe_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _preceding_caption(text: str, start: int) -> str | None:
    previous_end = max(0, start - 1)
    previous_start = text.rfind("\n", 0, previous_end)
    line = text[previous_start + 1 : previous_end].strip()
    return line if re.match(r"(?i)^(?:table|tab\.)\s+\d+", line) else None


def _delimited_tables(document: Document, occupied: tuple) -> list[ExtractedTable]:
    results: list[ExtractedTable] = []
    offset = 0
    run: list[tuple[int, str]] = []
    for line in document.text.splitlines(keepends=True) + [""]:
        stripped = line.rstrip("\r\n")
        delimiter = (
            "\t" if "\t" in stripped else "," if stripped.count(",") >= 2 else None
        )
        if delimiter is not None:
            run.append((offset, stripped))
        else:
            if len(run) >= 2:
                start = run[0][0]
                end = run[-1][0] + len(run[-1][1])
                if not any(
                    not (end <= item.start_character or item.end_character <= start)
                    for item in occupied
                ):
                    delimiter = "\t" if "\t" in run[0][1] else ","
                    cells = [
                        tuple(cell.strip() for cell in row.split(delimiter))
                        for _, row in run
                    ]
                    width = len(cells[0])
                    if width > 1 and all(len(row) == width for row in cells):
                        results.append(
                            ExtractedTable(
                                table_id=stable_identifier(
                                    "table", document.document_id, start, end
                                ),
                                caption=_preceding_caption(document.text, start),
                                headers=cells[0],
                                rows=tuple(cells[1:]),
                                citation=citation_for_range(document, start, end),
                                raw_text=document.text[start:end],
                                parsing_method="explicit_delimiter_table",
                            )
                        )
            run = []
        offset += len(line)
    return results


def group_experiments(
    document: Document,
    sections: tuple[PaperSection, ...],
    datasets: tuple[ScholarlyEvidence, ...],
    methods: tuple[ScholarlyEvidence, ...],
    baselines: tuple[ScholarlyEvidence, ...],
    metrics: tuple[ScholarlyEvidence, ...],
    hyperparameters: tuple[ScholarlyEvidence, ...],
    results: tuple[ScholarlyEvidence, ...],
    ablations: tuple[ScholarlyEvidence, ...],
) -> tuple[ExperimentRecord, ...]:
    """Group fields only within explicit experiment or ablation sections."""
    records = []
    for analyzed in sections:
        if not set(analyzed.roles) & {"experiments", "ablation"}:
            continue
        source = next(
            section
            for section in document.sections
            if section.section_id == analyzed.section_id
        )

        def within(
            items: Iterable[ScholarlyEvidence],
            scoped_section=source,
        ) -> tuple[ScholarlyEvidence, ...]:
            return tuple(
                item
                for item in items
                if scoped_section.start_character
                <= item.citation.start_character
                < item.citation.end_character
                <= scoped_section.end_character
            )

        purpose_span = next(
            sentence_spans(
                document.text,
                source.start_character,
                source.end_character,
            ),
            None,
        )
        purpose = (
            None
            if purpose_span is None
            else make_evidence(
                document,
                "experiment_purpose",
                document.text[purpose_span[0] : purpose_span[1]],
                *purpose_span,
                "first_experiment_section_sentence",
            )
        )
        records.append(
            ExperimentRecord(
                experiment_id=stable_identifier(
                    "exp", document.document_id, source.section_id
                ),
                name=source.heading or f"Experiment section {source.ordinal + 1}",
                purpose=purpose,
                datasets=within(datasets),
                methods=within(methods),
                baselines=within(baselines),
                metrics=within(metrics),
                hyperparameters=within(hyperparameters),
                results=within(results),
                ablations=within(ablations),
                citation=citation_for_range(
                    document, source.start_character, source.end_character
                ),
            )
        )
    return tuple(records)


def _deduplicate(
    values: list[ScholarlyEvidence],
    key: Callable[[ScholarlyEvidence], str] | None = None,
) -> tuple[ScholarlyEvidence, ...]:
    seen: set[object] = set()
    result = []
    for item in sorted(values, key=lambda entry: entry.citation.start_character):
        identity = (
            key(item)
            if key is not None
            else (
                item.category,
                item.citation.start_character,
                item.citation.end_character,
            )
        )
        if identity not in seen:
            result.append(item)
            seen.add(identity)
    return tuple(result)
