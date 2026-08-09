# Bulk automatic-approval policy

Bulk approval is disabled by default and requires a separate explicit enable.
Passing metrics only makes that enable action available; it does not change
policy state on its own.

Default readiness requirements are:

- at least 50 finalized human calibration pairs;
- at least 20 candidates the automatic policy would approve;
- automatic-approval precision at least 95%;
- false-approval rate no greater than 5%;
- overall agreement at least 95% and override rate no greater than 5%;
- near-threshold error rate no greater than 10%;
- every mandatory-human category kept out of automatic approval;
- zero source-hash, test-leakage, provenance, or duplicate errors.

Each check and its exact blocking reason is returned by `calibration-status`.
Thresholds are configurable in `CalibrationPolicy`, but lowering them changes a
safety policy and must be reviewed explicitly. A human override creates a
warning and updates the metrics; it does not silently mutate thresholds.

After activation, `bulk-auto-review --eligible-only` considers only pending,
non-test questions. Existing second-pass gates and mandatory-human routes still
apply. Eligible approvals retain the `codex_approved` serialized provenance
label until audit. At least 10% are sampled, plus every near-threshold,
disagreement, or novel-failure case. Human confirmation exposes the effective
trust status `audited_codex_approved`; it is not relabeled as human gold.

Bulk review never bypasses the audit queue or the duplicate/provenance checks in
trusted dataset export. Automatic review, calibration validation, audit, and
training approval are four distinct acts.

