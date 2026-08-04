# Auto-review calibration and audit policy

Automatic approval starts disabled. New workspaces report
`calibration_required`; passing answers still route to human review until the
calibration policy qualifies and a human explicitly enables approval.

## States

- `calibration_required`: fewer than 50 paired human/automatic outcomes.
- `calibration_active`: enough examples meet every metric, awaiting explicit
  enablement.
- `auto_approval_enabled`: metrics qualify and a human enabled approval.
- `auto_approval_suspended`: agreement, override rate, or confidence calibration
  fails policy.

The default policy requires at least 50 examples, at least 95% agreement, no
more than 5% human overrides, and Brier score no greater than 0.08. These are
conservative operating thresholds, not evidence of semantic correctness.
Reports include confidence bins, agreement, overrides, Brier score, and reasons
for the current state. Threshold changes are recommended, never made silently.

## Required operating procedure

### Phase 1 — Calibration

Use roughly 50–100 examples. The automated reviewer and a human assess the same
items. Measure agreement and overrides.

### Phase 2 — Limited auto-approval

Explicitly enable the >=0.95 threshold only after calibration qualifies. Audit
at least 10% of automated approvals.

### Phase 3 — Scale the corpus

Move from the current pilot toward 10–15 papers only after audit quality is
acceptable. Do not jump straight to a large corpus.

### Phase 4 — Dataset targets

Use 100 trusted examples for training-pipeline smoke tests, 300+ diverse examples
for an initial instruction-tuning experiment, and 600+ for a more meaningful
first comparison. These workflow targets do not imply broad capability or
general intelligence.

## Audit sampling

`select_audit_sample` uses a stable SHA-256 rank and seed 42 by default. It
selects the configured rate (10% by default) plus 100% of decisions within 0.02
of the threshold, reviewer disagreements, and novel failure categories. Zero
and full-rate modes are supported. The returned precomputed reasons make every
selection explainable and reproducible.

If humans overturn too many automated approvals, the report enters
`auto_approval_suspended`. The operator should raise the threshold or tighten
gates, then recalibrate; the application does not change policy automatically.

Commands:

```bash
python3 -m localml_scholar.evaluation.cli audit-sample --rate 0.10 --seed 42
python3 -m localml_scholar.evaluation.cli calibration-report
```

`--sample-fraction` and `--rate` are aliases. All artifacts remain
repository-local.
