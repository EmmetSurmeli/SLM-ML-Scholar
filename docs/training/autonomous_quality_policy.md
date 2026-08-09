# Autonomous quality policy

An example is `codex_curated` only when every focused critic accepts, the final
adjudicator accepts with confidence at or above 0.97 by default, all scored
dimensions meet the configured evidence threshold, deterministic grounding
gates pass, citations resolve, no unsupported claims remain, source hashes are
present, and split/duplicate policies pass.

Confidence is a reviewer score, not a calibrated probability. The initial
policy intentionally prefers discarding usable examples to admitting uncertain
ones. Derivations retain explicit structured provenance so paper-explicit steps
are distinguishable from mathematical inference and outside background.

## Terminal states

- `codex_curated`: eligible for the autonomous trust path.
- `rejected`: a critic/adjudicator rejected it or repairs were exhausted.
- `uncertain`: the review could not establish safe acceptance.
- `external_source_required`: local supplied sources cannot support the task.
- `insufficient_evidence`: no usable local evidence exists.
- `duplicate`: removed by normalized identity/cluster policy.
- `split_excluded`: withheld by leakage, balancing, or benchmark policy.

Only `codex_curated` is exportable through the autonomous-only tier.

## Safety stops

A run suspends on source-hash change, corrupt state, reviewer unavailability,
malformed reviewer output, repeated reviewer failure, or disagreement above the
configured rate. The persisted cursor allows a later restart. The quality
report records acceptance, repair, rejection, uncertainty, evidence/citation
validation, reviewer agreement, confidence, duplicate removal, split counts,
task diversity, high-risk categories, and representative accepted examples.

Machine-only audit metrics do not substitute for a human quality study. The
existing Calibration Lab, random audit samples, correction editor, and human
trust tiers remain available as optional oversight and do not block this mode.
