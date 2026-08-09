# Calibration Lab

Milestone 12A.2 turns human checking of automatic reviews into a dedicated,
local workflow. It does not train a model, call an external judge, or claim that
the heuristic confidence score is a probability.

## Workflow

1. Run automatic reviews on locally indexed papers.
2. Open **Calibration** and create the deterministic 50-item sample (or use the
   CLI with a larger count).
3. If an old record lacks current diagnostics, rerun it. The rerun is appended
   with the original snapshot and both content hashes; the original is never
   overwritten.
4. Inspect the question, paper, exact answer, evidence, reviewer gates, failure
   categories, and proposed correction.
5. Use one-click labels or keys: `A` approve, `C` correct, `P` partial, `I`
   incorrect, `S` should abstain, `B` benchmark problem, and `E` focus the edit.
6. Edit answer, retained evidence, required facts, or prohibited claims when
   needed. The pair stores before/after hashes.
7. Optionally perform the **separate** training-approval action. A calibration
   decision alone never enters the training dataset.

The sample seeks coverage across papers, question types, proposed labels,
abstentions, confidence buckets, gate failures, and near-threshold cases. The UI
reports any available stratum it could not include. Selection is deterministic
for the same reviews, count, and seed.

## Metrics

The report includes overall and approval/rejection agreement, automatic
approval precision and recall, false-approval and false-rejection rates,
override rate, mandatory-human routing accuracy, near-threshold behavior, and
breakdowns by confidence bucket, question type, failure category, paper, and
reviewer profile.

Confidence buckets are exactly `0.50–0.69`, `0.70–0.79`, `0.80–0.89`,
`0.90–0.94`, `0.95–0.97`, and `0.98–1.00`. Scores below 0.50 are reported as an
additional diagnostic group. The scores come from deterministic gates and are
not probabilistically calibrated.

## Local artifacts

The lab writes ignored JSON files under `outputs/review_app/`:

- `calibration_sample.json`
- `calibration_pairs.json`
- `historical_review_reruns.json`
- `paper_acquisition_queue.json`

The acquisition queue stores manually entered title, DOI, arXiv ID, citation,
reason, and category. It never fetches a paper. A local paper's references may
inform a suggestion, but adding the source remains a deliberate user action.

## Commands

```bash
python3 -m localml_scholar.evaluation.cli calibration-sample --count 50 --seed 42
python3 -m localml_scholar.evaluation.cli rerun-historical-reviews --sample-only
python3 -m localml_scholar.evaluation.cli calibration-report
python3 -m localml_scholar.evaluation.cli calibration-status
python3 -m localml_scholar.evaluation.cli enable-auto-approval
python3 -m localml_scholar.evaluation.cli bulk-auto-review --eligible-only
```

Fifty reviewed pairs is the minimum operating target. Seventy-five to one
hundred gives better coverage. The current six-paper corpus is useful for
workflow testing, not for claims of pretraining or broad scholarly ability.

