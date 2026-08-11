# Claim-level citation validation

Whole-answer citation coverage is insufficient: a paragraph can contain one
valid citation while leaving other factual claims unsupported. Milestone 12A.4
therefore validates an answer as atomic `AnswerClaim` records.

## Canonical path

`normalize_citations` is the only parser used by answer repair, critics,
validation, migration, exports, and the dashboard. It converts `C1`, `[C1]`,
and bracketed current evidence/chunk IDs to `[C1]`, removes repeated labels in a
citation cluster, preserves first-use order, and reports unknown or malformed
markers. Unknown labels fail before adjudication.

`segment_answer_claims` then separates prose into stable factual, numeric,
equation, or qualification claims. Every substantive claim stores its labels,
support type, and support requirement. The parser is deliberately conservative
and is not a full natural-language proposition parser.

## Stable evidence identity

Display labels are answer-local and may change after repair. Stable evidence
identity instead hashes document ID, chunk ID, page range, line range, heading
path, and source hash when present. Invariants require:

- every label resolves to current evidence;
- each evidence item has document and chunk coordinates;
- cited documents belong to the selected paper set;
- repair cannot retain stale pre-repair identity;
- source-hash mismatches and split leakage are fatal.

## Deterministic gate

For every claim/citation pair, `validate_claim_citations` checks resolution,
non-empty passage text, selected source, section compatibility, numeric values,
and benchmark concept/alias presence when supplied. It returns separate
structural, support, and relevance results plus per-claim `CitationCritique`
records.

These checks catch obvious errors but do **not** establish entailment. A
semantic citation critic still evaluates whether a passage really supports a
claim. Conversely, the critic cannot waive a malformed label, wrong paper,
missing citation, stale identity, or source-hash mismatch.

## Acceptance

An accepted answer needs every substantive claim structurally cited and free
of deterministic support failures, every focused critic and final adjudicator
to accept, all configured 0.97 score/confidence floors, no unsupported claim or
uncertainty, deterministic grounding gates, and explicit derivation provenance
where applicable.

Tests cover canonical spellings, duplicates, unknown/malformed labels, multiple
and numeric claims, equations and qualifications, repeated/replaced evidence,
wrong paper/section, unsupported numbers, missing citations, legacy migration,
and repair outcomes.
## 1.2.5 pre-composition validation

Citation validation now occurs before and after composition. Before composition,
each `SupportedClaim` must resolve its current evidence labels, pass numeric and
entity alignment, and be direct or supporting for the question. After
composition, every substantive sentence must trace exactly to planned claim IDs.
The legacy sentence parser remains available for 1.2.3/1.2.4 migration and as an
independent final structural check.
