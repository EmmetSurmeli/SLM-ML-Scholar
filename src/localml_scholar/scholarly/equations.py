"""Text-only equation detection, notation extraction, and definition linking."""

from __future__ import annotations

import re
from collections import defaultdict

from localml_scholar.retrieval import Document
from localml_scholar.retrieval.documents import stable_identifier
from localml_scholar.scholarly.config import ScholarlyConfig
from localml_scholar.scholarly.models import (
    DefinitionCandidate,
    EquationAnalysis,
    EquationBlock,
    NotationEntry,
    PaperSection,
    ScholarlyEvidence,
)
from localml_scholar.scholarly.source import citation_for_range, section_for_range

_DISPLAY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("latex_dollars_display", re.compile(r"\$\$.+?\$\$", re.DOTALL)),
    ("latex_brackets_display", re.compile(r"\\\[.+?\\\]", re.DOTALL)),
    (
        "latex_equation_environment",
        re.compile(
            r"\\begin\{(?:equation\*?|align\*?|gather\*?)\}.+?"
            r"\\end\{(?:equation\*?|align\*?|gather\*?)\}",
            re.DOTALL,
        ),
    ),
    ("latex_parentheses_inline", re.compile(r"\\\(.+?\\\)", re.DOTALL)),
    ("latex_dollars_inline", re.compile(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)")),
)
_OPERATOR = re.compile(r"(?:<=|>=|≤|≥|=|∑|∫|∂|argmax|argmin|\blog\b)")
_MATH_SYMBOL = re.compile(r"(?:\\[A-Za-z]+|[α-ωΑ-Ω]|[A-Za-z](?:_[A-Za-z0-9{}]+)?)")
_TAG_NUMBER = re.compile(r"\\tag\{(?P<tag>[^}]+)\}")
_PAREN_NUMBER = re.compile(r"\((?P<paren>\d+[a-z]?)\)\s*(?:\$\$|\\\])?\s*$")
_LATEX_SYMBOL = re.compile(
    r"\\(?:hat|bar|tilde)\{\\?[A-Za-z]+\}"
    r"(?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9]+))?"
    r"|\\(?:mathcal|mathbf|boldsymbol)\{[A-Za-z]\}"
    r"(?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9]+))?"
    r"|\\[A-Za-z]+(?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9]+))?"
)
_UNICODE_GREEK = re.compile(r"[α-ωΑ-Ω](?:[₀-₉]+|_[A-Za-z0-9{}]+)?")
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z]"
    r"(?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9]+))?)(?![A-Za-z])"
)
_LATEX_OPERATORS = {
    r"\argmax",
    r"\argmin",
    r"\cdot",
    r"\frac",
    r"\geq",
    r"\leq",
    r"\log",
    r"\odot",
    r"\prod",
    r"\sum",
    r"\tag",
}
_DEFINITION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "where_symbol_definition",
        re.compile(
            r"(?i)\bwhere\s+(?P<symbol>\\?[A-Za-zα-ωΑ-Ω](?:_[\w{}]+)?)\s+"
            r"(?:is|denotes|represents)\s+(?P<meaning>[^.;\n]+)"
        ),
    ),
    (
        "let_symbol_definition",
        re.compile(
            r"(?i)\blet\s+(?P<symbol>\\?[A-Za-zα-ωΑ-Ω](?:_[\w{}]+)?)\s+"
            r"(?:be|denote|represent)\s+(?P<meaning>[^.;\n]+)"
        ),
    ),
    (
        "define_symbol_as",
        re.compile(
            r"(?i)\b(?:we\s+)?define\s+"
            r"(?P<symbol>\\?[A-Za-zα-ωΑ-Ω](?:_[\w{}]+)?)\s+as\s+"
            r"(?P<meaning>[^.;\n]+)"
        ),
    ),
    (
        "symbol_represents",
        re.compile(
            r"(?P<symbol>\\?[A-Za-zα-ωΑ-Ω](?:_[\w{}]+)?)\s+"
            r"(?i:represents|denotes)\s+(?P<meaning>[^.;\n]+)"
        ),
    ),
)


def normalize_equation(text: str) -> str:
    """Conservatively normalize whitespace and unambiguous minus characters."""
    if not isinstance(text, str) or not text:
        raise ValueError("Equation text must be non-empty.")
    value = text.replace("−", "-").replace("‐", "-")
    return re.sub(r"\s+", " ", value).strip()


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def detect_equation_blocks(
    document: Document,
    config: ScholarlyConfig | None = None,
) -> tuple[EquationBlock, ...]:
    """Detect non-overlapping equations from extracted text, never visual layout."""
    resolved = config or ScholarlyConfig()
    candidates: list[tuple[int, int, int, str]] = []
    for priority, (method, pattern) in enumerate(_DISPLAY_PATTERNS):
        for match in pattern.finditer(document.text):
            start, end = _trim_span(document.text, *match.span())
            if start < end:
                candidates.append((start, end, priority, method))
    offset = 0
    in_fence = False
    for line in document.text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        operator_count = len(_OPERATOR.findall(stripped))
        symbol_count = len(_MATH_SYMBOL.findall(stripped))
        equation_like = (
            operator_count >= resolved.equation_minimum_operator_count
            and symbol_count >= resolved.minimum_equation_line_symbol_count
            and len(stripped) <= 500
        )
        numbered = bool(
            (_TAG_NUMBER.search(stripped) or _PAREN_NUMBER.search(stripped))
            and _OPERATOR.search(stripped)
        )
        if stripped and not in_fence and (equation_like or numbered):
            line_start = offset + len(line) - len(line.lstrip())
            line_end = offset + len(line.rstrip("\r\n"))
            candidates.append(
                (
                    line_start,
                    line_end,
                    len(_DISPLAY_PATTERNS),
                    "numbered_or_operator_line",
                )
            )
        offset += len(line)
    selected: list[tuple[int, int, str]] = []
    for start, end, _priority, method in sorted(
        candidates,
        key=lambda item: (item[0], item[2], -(item[1] - item[0]), item[3]),
    ):
        if any(
            not (end <= kept_start or kept_end <= start)
            for kept_start, kept_end, _ in selected
        ):
            continue
        try:
            section_for_range(document, start, end)
        except ValueError:
            continue
        selected.append((start, end, method))
    blocks: list[EquationBlock] = []
    for start, end, method in sorted(selected):
        raw = document.text[start:end]
        tag_match = _TAG_NUMBER.search(raw)
        paren_match = _PAREN_NUMBER.search(raw)
        number = (
            tag_match.group("tag")
            if tag_match is not None
            else paren_match.group("paren")
            if paren_match is not None
            else None
        )
        section = section_for_range(document, start, end)
        blocks.append(
            EquationBlock(
                equation_id=stable_identifier(
                    "eq", document.document_id, start, end, raw
                ),
                document_id=document.document_id,
                section_id=section.section_id,
                raw_text=raw,
                normalized_text=normalize_equation(raw),
                equation_number=number,
                citation=citation_for_range(document, start, end),
                detection_method=method,
            )
        )
    return tuple(blocks)


def symbol_spans(text: str) -> tuple[tuple[str, int, int, str], ...]:
    """Return deterministic notation candidates and their local source spans."""
    candidates: list[tuple[str, int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, kind in (
        (_LATEX_SYMBOL, "latex"),
        (_UNICODE_GREEK, "greek"),
        (_IDENTIFIER, "indexed" if "_" in text else "variable"),
    ):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(
                left <= start < right or left < end <= right for left, right in occupied
            ):
                continue
            symbol = match.group(0)
            if symbol.split("_", 1)[0] in _LATEX_OPERATORS:
                occupied.append((start, end))
                continue
            if symbol in {"e", "i"} and kind == "variable":
                symbol_kind = "constant_or_variable"
            elif symbol.startswith("\\"):
                symbol_kind = "latex_symbol"
            elif "_" in symbol:
                symbol_kind = "indexed_variable"
            elif re.fullmatch(r"[α-ωΑ-Ω].*", symbol):
                symbol_kind = "greek_symbol"
            else:
                symbol_kind = kind
            candidates.append((symbol, start, end, symbol_kind))
            occupied.append((start, end))
    return tuple(sorted(candidates, key=lambda item: (item[1], item[2], item[0])))


def extract_definition_candidates(
    document: Document,
    sections: tuple[PaperSection, ...],
) -> tuple[DefinitionCandidate, ...]:
    """Extract cited definition candidates using explicit textual patterns."""
    role_by_section = {section.section_id: section.roles[0] for section in sections}
    candidates: list[DefinitionCandidate] = []
    for pattern_name, pattern in _DEFINITION_PATTERNS:
        for match in pattern.finditer(document.text):
            start, end = _trim_span(document.text, *match.span())
            section = section_for_range(document, start, end)
            candidates.append(
                DefinitionCandidate(
                    symbol=match.group("symbol"),
                    defining_text=document.text[start:end],
                    citation=citation_for_range(document, start, end),
                    distance_characters=0,
                    pattern=pattern_name,
                    confidence="high",
                    section_role=role_by_section.get(section.section_id, "unknown"),
                )
            )
    unique = {
        (
            item.symbol,
            item.citation.start_character,
            item.citation.end_character,
            item.pattern,
        ): item
        for item in candidates
    }
    return tuple(unique[key] for key in sorted(unique))


def build_notation_glossary(
    document: Document,
    equations: tuple[EquationBlock, ...],
    sections: tuple[PaperSection, ...],
    config: ScholarlyConfig | None = None,
) -> tuple[tuple[NotationEntry, ...], tuple[str, ...]]:
    """Build cited glossary entries while retaining conflicts and unresolved symbols."""
    resolved = config or ScholarlyConfig()
    definitions = extract_definition_candidates(document, sections)
    occurrences: dict[str, list] = defaultdict(list)
    types: dict[str, str] = {}
    for equation in equations:
        for symbol, start, end, kind in symbol_spans(equation.raw_text):
            absolute_start = equation.citation.start_character + start
            absolute_end = equation.citation.start_character + end
            occurrences[symbol].append(
                citation_for_range(document, absolute_start, absolute_end)
            )
            types.setdefault(symbol, kind)
    entries: list[NotationEntry] = []
    unresolved: list[str] = []
    for symbol in sorted(occurrences, key=lambda value: (value.casefold(), value)):
        matching = tuple(
            candidate
            for candidate in definitions
            if candidate.symbol == symbol
            and min(
                abs(candidate.citation.start_character - occurrence.start_character)
                for occurrence in occurrences[symbol]
            )
            <= resolved.definition_window_characters
        )
        meanings = {candidate.defining_text.casefold() for candidate in matching}
        selected = matching[0] if len(meanings) == 1 and matching else None
        ambiguity = (
            "multiple conflicting definition candidates" if len(meanings) > 1 else None
        )
        if not matching:
            unresolved.append(symbol)
        entries.append(
            NotationEntry(
                symbol_id=stable_identifier("sym", document.document_id, symbol),
                raw_symbol=symbol,
                normalized_symbol=symbol,
                symbol_type=types[symbol],
                occurrences=tuple(occurrences[symbol]),
                definition_candidates=matching,
                selected_definition=selected,
                ambiguity=ambiguity,
            )
        )
    return tuple(entries), tuple(unresolved)


def analyze_equation_links(
    document: Document,
    equations: tuple[EquationBlock, ...],
    notation: tuple[NotationEntry, ...],
) -> tuple[EquationAnalysis, ...]:
    """Link each equation to retained definitions without deriving new mathematics."""
    entry_by_symbol = {entry.raw_symbol: entry for entry in notation}
    analyses: list[EquationAnalysis] = []
    for equation in equations:
        symbols = tuple(
            dict.fromkeys(item[0] for item in symbol_spans(equation.raw_text))
        )
        definitions = tuple(
            candidate
            for symbol in symbols
            for candidate in entry_by_symbol[symbol].definition_candidates
        )
        unresolved = tuple(
            symbol
            for symbol in symbols
            if not entry_by_symbol[symbol].definition_candidates
        )
        defined_later = tuple(
            symbol
            for symbol in symbols
            if any(
                candidate.citation.start_character > equation.citation.end_character
                for candidate in entry_by_symbol[symbol].definition_candidates
            )
        )
        related: list[ScholarlyEvidence] = []
        line_end = document.text.find("\n", equation.citation.end_character)
        if line_end < 0:
            line_end = len(document.text)
        following_start = equation.citation.end_character
        while following_start < line_end and document.text[following_start].isspace():
            following_start += 1
        if following_start < line_end:
            text = document.text[following_start:line_end]
            related.append(
                ScholarlyEvidence(
                    evidence_id=stable_identifier(
                        "ev",
                        document.document_id,
                        "equation_related",
                        following_start,
                        line_end,
                    ),
                    category="equation_related_text",
                    value=text,
                    normalized_value=re.sub(r"\s+", " ", text).strip(),
                    citation=citation_for_range(document, following_start, line_end),
                    source_text=text,
                    extraction_method="following_line_text",
                    confidence="low",
                )
            )
        analyses.append(
            EquationAnalysis(
                equation_id=equation.equation_id,
                symbols=symbols,
                definitions=definitions,
                unresolved_symbols=unresolved,
                defined_later_symbols=defined_later,
                related_text=tuple(related),
            )
        )
    return tuple(analyses)
