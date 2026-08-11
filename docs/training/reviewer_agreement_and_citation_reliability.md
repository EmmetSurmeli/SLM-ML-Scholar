# Reviewer agreement and citation reliability

Milestone 12A.4 replaces the old single disagreement bit with an auditable,
claim-aware reliability layer. It does not lower the 0.97 acceptance threshold,
train a model, or claim that deterministic checks prove semantic truth.

## Shared reviewer policy

Every review pass receives policy version `1.0` and the same definitions for
relevance, evidence sufficiency, factual support, completeness, acceptable
inference, citation support/relevance, abstention, derivation support, external
knowledge, and partial correctness. Roles remain distinct:

- the answerer selects evidence, constructs the target, and writes cited prose;
- the evidence critic sees no answer and asks whether the passages can support
  a correct answer;
- the answer critic judges the answer but not retrieval rank or score;
- the citation critic receives atomic claims and their cited passages and does
  not grade prose style;
- the final adjudicator receives deterministic conflict triggers and resolves
  only remaining semantic ambiguity.

Typed policy outcomes use `EvidenceDecision`, `ClaimSupport`, and
`CitationDecision`. Existing 1.2.3 `CodexReview` artifacts remain readable;
missing 1.2.4 fields are filled only in memory and the source artifact is never
rewritten.

## Disagreement policy

Hard disagreement covers support versus unsupported, sufficient versus
insufficient evidence, valid versus invalid citation, answer versus abstain,
and a final adjudicator overriding a factual majority. Soft disagreement covers
wording, verbosity, tone, optional detail, and technical style. A record can
carry several taxonomy labels.

Controlled runs stop for hard disagreement above 15%, structural citation
failure above 5%, unresolved support failure above 5%, malformed reviewer
output above 2%, any leakage, or any source-hash mismatch. Soft disagreement
alone never stops a run. Full-run readiness is stricter: hard disagreement must
be at most 10% and overall disagreement at most 15%.

## Repair diagnostics

Each repair records whether it changed evidence, answer text, citations,
structured target, or question interpretation. The next validation compares
pre/post gates and classifies the repair as `fixed`, `introduced_new_issue`, or
`unchanged`. This makes harmful repair strategies visible instead of counting
every repair attempt as progress.

## Temporary training exclusions

Until controlled diagnostics show stable behavior, complexity,
critical-reasoning, derivation, equation, figure-interpretation, historical-
impact, and cross-paper-synthesis examples are retained as evaluation/uncertain
artifacts but excluded from autonomous training materialization. They are not
deleted. The policy can be narrowed only from measured controlled-run results.

## Diagnostics

```bash
PYTHONPATH=src python3 -m localml_scholar.training_data.cli \
  diagnose-reviewers --run RUN_ID
PYTHONPATH=src python3 -m localml_scholar.training_data.cli \
  citation-audit --run RUN_ID
PYTHONPATH=src python3 -m localml_scholar.training_data.cli \
  disagreement-report --run RUN_ID
PYTHONPATH=src python3 -m localml_scholar.training_data.cli \
  diagnostic-curation --count 50 --seed 42
PYTHONPATH=src python3 -m localml_scholar.training_data.cli full-run-readiness
```

The dashboard exposes hard/soft disagreement, citation structure/support/
relevance, stale IDs, pairwise patterns, type metrics, and representative
failures. Generated diagnostic state remains under the ignored `outputs/`
directory.
## 1.2.5 shared claim graph

Evidence, answer, and citation critics now receive the same supported-claim graph
within their role-specific blind views. Hard disagreement is based on semantic
support/insufficiency/answer-versus-abstain conflicts, not wording differences.
Claim critiques are also compared by `claim_id`, and per-claim hard conflicts are
reported independently from example-level disagreement.
