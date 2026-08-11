# Milestone 12A.6 pilot incident

Date: 2026-08-10
Package: 1.2.6
Run: `curation_fe778b0d917f4168b6fecb43bff4f5a4`

## Decision

The approved 10-item pilot was stopped and invalidated for readiness after
2/10 terminal records. It must not be resumed or used as evidence that the
curation pipeline is ready. No fresh 50-item diagnostic was started.

## Preserved outcome

- Cursor: `paper_index=2`, `question_index=0`.
- Record 1: deliberate insufficient-evidence item; zero Codex calls.
- Record 2: supported batch-size lookup; rejected after 15 Codex calls.
- Reviewer roles: three calls each to answerer, evidence critic, answer critic,
  citation critic, and final adjudicator.
- Source hashes, analysis hashes, paper splits, candidates, records, errors,
  and reviewer outputs remain unchanged in the ignored local run artifact.
- Final stage: `readiness_invalidated`.
- Invalidation marker:
  `failed_live_pilot_minibatch_alias_and_no_progress_repair_loop`.

## Verified causes

The evidence said that the minibatch size was 128 and the constructed claim
was numerically supported with a resolving citation. Two local policies used
the exact question token `batch`, however, and did not recognize `minibatch`:

1. the reproduction answer planner therefore labeled the claim only partially
   sufficient and inserted a "partial answer" prefix;
2. evidence sufficiency retained 0.5 query coverage, which caused deterministic
   answer-relevance, citation-relevance, and evidence-sufficiency gates to
   fail during revalidation.

All reviewer roles consistently requested removal of the unsupported partial
characterization. The second repair made no effective change, but the old loop
still launched a third complete review cycle.

## Fixes and regressions

- Reproduction planning recognizes both `batch` and `minibatch` wording.
- Evidence sufficiency uses the same narrow equivalence when checking retrieved
  passage text, without globally stemming or rewriting indexed terms.
- An identical no-progress correction stops after two review cycles instead of
  spending a third cycle.
- Authored tests cover the exact batch/minibatch answer, retrieval sufficiency,
  and the 10-call no-progress ceiling.

Verification:

```text
python3 -m localml_scholar.training_data.cli pipeline-self-test
  all 8 checks passed; Codex calls: 0

python3 -m ruff check .
python3 -m ruff format --check .
node --check src/localml_scholar/review_app/static/app.js
python3 -m pytest -q
  913 passed in 18.98s

python3 -m localml_scholar.training_data.cli ingestion-health
  14 healthy; 0 unhealthy; average titled-section fraction 1.0
```

## Next gate

The first replacement, `curation_78c29180704946dc9cc773a0568a8734`, confirmed
the batch-size repair by accepting that item in one five-role cycle. It then
exposed the same underlying design error on causal masking: a direct supported
claim was deemed partial because it did not repeat a generic `architecture`,
`model`, `layer`, `network`, or `component` marker. The run was interrupted
after 3/10 persisted records and frozen with invalidation marker
`failed_live_pilot_question_type_marker_answerability_gate`. It contains 20
persisted Codex calls; calls made by the interrupted fourth candidate were not
committed to the record, so they are not included in that count.

The answer planner now treats direct support plus required-concept coverage as
the answerability rule. Question-category markers remain part of relevance
classification but are no longer a second, hidden completeness requirement.
The exact causal-masking regression and complete suite pass:

```text
python3 -m pytest -q
  918 passed in 9.87s
```

The second replacement, `curation_2ffb967acc1d455ca4732c7340763ac8`, verified
that the partial-answer prefix was removed, but its causal-mask answer still
failed. Sentence-initial technical words (`Causal`, `Masking`, `Combined`, and
`Softmax`) were falsely treated as named entities absent from the evidence.
The sufficiency state also marked `causal` unmatched even though the passages
explicitly described subsequent-position blocking and the autoregressive
property. The run was stopped after 3/10 records and 20 persisted calls and
invalidated as
`failed_live_pilot_false_named_entities_and_causal_mask_coverage`.

The validators now use conservative entity forms and a narrow causal-mask
concept equivalence. Replaying all three stored answerer targets locally makes
their substantive masking claims direct, explicit, and sufficient.

## Completed bounded pilot

Third replacement pilot `curation_790d8f32e5a7499baac7da5b5f03009f`
completed all 10 approved items:

- 2 `codex_curated` records: batch size and causal masking;
- 2 deliberate insufficient-evidence outcomes with zero reviewer calls;
- 5 rejected answers;
- 1 uncertain answer;
- 80 persisted reviewer calls in total;
- 1.0 citation structural validity, support rate, and relevance rate;
- 0.25 hard reviewer disagreement, above the 0.15 readiness maximum.

The final dataset export exposed a separate local subset bug: manual split
assignments included non-test corpus papers that had no accepted examples. The
dataset builder rejected those unknown papers. Export now filters split
assignments to the selected examples, an authored multi-paper regression covers
the accepted-subset case, and the same run finalized without new reviewer
calls. Its dataset contains exactly the two `codex_curated` examples and no
human-approved label.

The pilot is complete but not ready. A fresh 50-item diagnostic, Milestone 12B,
full autonomous curation, and transformer training remain blocked.
