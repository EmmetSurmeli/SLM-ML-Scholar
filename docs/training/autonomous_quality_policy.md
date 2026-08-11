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

Before any critic call, package 1.2.6 requires a healthy paper structure, a
topic-eligible question, non-stopword essential concepts, usable evidence, and
deterministic claim answerability. Obvious failures terminate locally and are
reported as calls saved. Answerable and deliberate-abstention pools have
separate metrics.

## Terminal states

- `codex_curated`: eligible for the autonomous trust path.
- `rejected`: a critic/adjudicator rejected it or repairs were exhausted.
- `uncertain`: the review could not establish safe acceptance.
- `external_source_required`: local supplied sources cannot support the task.
- `insufficient_evidence`: no usable local evidence exists.
- `duplicate`: removed by normalized identity/cluster policy.
- `split_excluded`: withheld by leakage, balancing, or benchmark policy.
- `construction_failed`, `retrieval_failed`, `validation_failed`: isolated
  candidate defects that are persisted without discarding neighboring work.
- `reviewer_failed`, `repair_failed`: terminal reviewer/repair defects when the
  error is candidate-specific rather than service-wide.

Only `codex_curated` is exportable through the autonomous-only tier.

## Safety stops

A run suspends on source-hash change, corrupt state, reviewer unavailability,
malformed reviewer output, repeated reviewer failure, hard disagreement above
15%, structural citation failure above 5%, unresolved support failure above 5%,
any leakage, or any source-hash mismatch. Soft style disagreement alone does
not stop a run. The persisted cursor allows a later restart. The quality report
records acceptance, repair, rejection, uncertainty, citation structure/support/
relevance, hard/soft disagreement, confidence, duplicate removal, split counts,
task diversity, pairwise matrices, and representative failures.

Candidate isolation stops being safe when the same unexpected signature occurs
three times or more than 30% of processed work fails at one stage. Those
defaults are configurable but are not lowered during readiness evaluation.

A full run uses the stricter readiness policy: at least 98% structural citation
validity, 95% citation support/relevance and evidence validity, at most 10% hard
and 15% overall disagreement, positive repair success, and zero leakage, stale
evidence IDs, or source-hash mismatches. Controlled batches come first.

Machine-only audit metrics do not substitute for a human quality study. The
existing Calibration Lab, random audit samples, correction editor, and human
trust tiers remain available as optional oversight and do not block this mode.
