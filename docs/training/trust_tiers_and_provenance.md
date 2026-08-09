# Trust tiers, provenance, and leakage controls

Trust is explicit metadata. Review origin is never collapsed into a single
“approved” boolean.

## Export tiers and weights

| Trust tier | Included records | Default weight |
|---|---|---:|
| `human-only` | `human_approved` | 1.0 |
| `human-and-audited` | human plus audited `codex_approved` | 1.0 / 0.9 |
| `include-codex-approved` | human plus all `codex_approved` | 1.0 / 0.9 / 0.6 |
| `codex-curated-only` | autonomous five-pass `codex_curated` only | 0.85 |
| `human-and-codex-curated` | human plus `codex_curated` | 1.0 / 0.85 |
| `all-trusted` | every eligible human, audited, Codex-approved, and Codex-curated record | tier-specific |

The legacy site/workspace export defaults to `human-and-audited`; autonomous
curation defaults to `codex-curated-only`. The lower-level
`build_dataset` API retains its conservative `human-only` default for backward
compatibility. Weights are written to example/dataset metadata; training code
does not apply them automatically, and future configuration may override them.

After a human confirms an audit, the browser exposes the effective status
`audited_codex_approved`. The serialized review status remains
`codex_approved` with `audit_status=human_confirmed`, preserving the original
producer identity and compatibility with existing datasets. It is never
relabeled as human gold.

`codex_curated` is separate from the older confidence-gated `codex_approved`
state. It requires the autonomous evidence, answer, citation, and adjudication
passes plus repair revalidation, source hashes, duplicate policy, and split
checks. Its metadata retains every pass and explicitly records
`human_approved=false`. Automated provenance is never upgraded merely because
the autonomous workflow completed.

Rejected, ambiguous, benchmark-problem, pending, unresolved-correction, and
calibration-routed records are excluded.

## Provenance

`ReviewProvenance` records producer and version, reviewer and version,
correction system, source hashes, answer hash, parent example IDs, benchmark
source, and independent validators. JSON is canonicalized before SHA-256 hashing.
Source hashes can be checked before reusing an old decision.

If the same system/version produced and reviewed an answer without an
independent validator, a circular-training warning is emitted. Codex-approved
records carrying circular warnings cannot enter trusted exports. Human review
creates distinct approval provenance rather than overwriting producer identity.

## Duplicate controls

Exports normalize punctuation/case, compare token-set similarity, and detect
repeated evidence/answer pairs. Stable union-find clusters receive SHA-derived
IDs. By default one highest-trust representative is selected per cluster, which
prevents many trivial paraphrases from dominating the dataset.

## Paper split protection

Splits are assigned by paper, and connected cross-paper examples are coalesced
to one split. Conflicting manual assignments raise an error. Records marked
`test_only` or listing `test_only_paper_ids` are rejected from training exports.
Auto-review never changes split metadata. Reviewed test-paper answers may remain
evaluation artifacts, but their corrections cannot become training examples.

## Boundaries

- Automated approval is not human gold.
- Confidence does not guarantee correctness.
- Human audit sampling is recommended but does not block autonomous curation.
- Provenance must survive every edit and export.
- Held-out papers must not leak into training.
- The project does not browse for or download papers automatically.

Implementation: `training_data/provenance.py`, `trust.py`, `duplicates.py`,
`dataset.py`, and `splits.py`.
