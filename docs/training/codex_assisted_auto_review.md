# Codex-assisted auto-review

Milestone 12A.1 adds a conservative second pass after the existing deterministic
answer-and-grade pass. Its purpose is to reduce repetitive review work while
keeping the human/automated distinction permanent. A `codex_approved` record is
not human gold and is never serialized as `human_approved`.

## Decision model

Every resolved record uses one of these states:

- `human_approved`
- `codex_approved`
- `human_rejected`
- `codex_rejected`
- `needs_human_review`
- `ambiguous`
- `benchmark_problem`

`proposed` and the legacy `rejected` value remain readable for workspace
compatibility, but new decisions use the explicit states above.

The local site cannot launch genuinely independent Codex agents. It therefore
runs three deterministic configurations—evidence-strict, answer-strict, and
policy-strict—and records that they are correlated. Agreement is useful as a
consistency check, not independent replication.

## Approval gates

Automatic approval requires unanimous reviewer-profile approval, confidence at
or above the configured threshold (0.95 by default), enabled calibration, and
all 16 gates:

1. direct evidence;
2. answer relevance;
3. factual-claim support;
4. citations present;
5. citations resolve;
6. citations support attached claims;
7. citation relevance;
8. prohibited claims absent;
9. no unresolved contradiction;
10. correct answerability classification;
11. sufficient evidence;
12. required-concept coverage;
13. no external claim presented as paper-supported;
14. no inferred derivation presented as explicit paper content;
15. instruction following; and
16. confidence threshold.

An aggregate score cannot compensate for a failed gate. Confidence is a
diagnostic, not a correctness guarantee.

## Mandatory human routes

Historical impact, novelty, “first to” claims, cross-paper comparisons,
conflicting sources, inferred derivations or limitations, research gaps,
ambiguous benchmarks, external literature, extraction corruption, multiple
interpretations, numeric contradictions, unusual citation mappings, metadata
disagreement, uncertain figures/tables, and reviewer disagreement are routed to
`needs_human_review` by default.

## Correction lifecycle

The reviewer may propose corrected text, evidence, concepts, prohibited claims,
structured targets, or citations. Any correction is treated as a new candidate
and revalidated from scratch. It is never approved because the reviewer itself
generated it.

```text
answer → first pass → second pass → correction proposal
       → evidence/citation/answer revalidation → gates → decision
```

## Batch workflow

The Paper Training Lab supports paper-scoped batches, per-question failure
preservation, stop/resume, bulk human decisions, deterministic audit sampling,
and trust-tier export. Completed work is written atomically after every item.
No paper discovery, web request, or paper download is performed.

Recommended path:

```text
upload → index → generate questions → run answers → auto-review
→ inspect mandatory-human cases → audit sample → export a trust tier
```

Implementation: `training_data/auto_review.py`, `review_app/automation.py`, and
`review_app/service.py`. Validation is in
`tests/test_codex_assisted_auto_review.py` and the existing review-app tests.

Milestone 12A.2 keeps this reviewer unchanged and adds the human calibration
workflow around it. See [Calibration Lab](calibration_lab.md) and the
[bulk-approval policy](bulk_auto_approval_policy.md). Historical reruns are
append-only; the original review remains available with its content hash.
