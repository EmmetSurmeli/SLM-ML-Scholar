# Autonomous corpus curation

> **1.2.6 deterministic preflight:** run local section, health, topic,
> retrieval, sufficiency, abstention, and claim-repair gates before Codex. The
> invalid partial 50-candidate run is frozen and must not be resumed. See
> [fast deterministic curation preflight](fast_deterministic_curation_preflight.md).

> **1.2.4 reliability gate:** a full corpus run now requires successful
> controlled diagnostics. Reviewer disagreement is split into hard and soft
> categories, and every substantive answer claim is citation-validated. See
> [reviewer agreement and citation reliability](reviewer_agreement_and_citation_reliability.md)
> and the [readiness policy](autonomous_curation_readiness_policy.md).

Milestone 12A.3 adds a no-human-gate path from locally supplied papers to a
versioned grounded-instruction dataset. It does not train a model and never
downloads papers. Paper extraction, retrieval, evidence, run state, and exports
remain in the repository workspace. The configured Codex CLI performs the five
reasoning review passes; that service may not itself be offline.

## Workflow

```text
local papers -> section recovery + ingestion health
             -> topic-eligible answerable/abstention question pools
             -> canonical concepts + deterministic grounded answer
             -> cheap claim repair or local terminal abstention
             -> Codex answerer
             -> evidence critic -> answer critic -> citation critic
             -> final adjudicator
             -> bounded evidence-first repair (default: two attempts)
             -> reject uncertainty -> deduplicate/balance
             -> paper-level split -> codex-curated export + report
```

The browser's **Autonomous Curation** page runs this flow for the selected
papers. **Process all newly uploaded papers** excludes papers that already
belong to a completed run. PDF, text, and Markdown inputs support multi-file
selection. The CLI equivalents are:

```bash
python3 -m localml_scholar.training_data.cli curate-corpus \
  --all-papers --questions-per-paper 60 --acceptance-threshold 0.97

python3 -m localml_scholar.training_data.cli curate-paper --paper PAPER_ID
python3 -m localml_scholar.training_data.cli process-new
python3 -m localml_scholar.training_data.cli resume-curation --run RUN_ID
python3 -m localml_scholar.training_data.cli curation-report --run RUN_ID
python3 -m localml_scholar.training_data.cli export \
  --run RUN_ID --trust-tier codex-curated-only
python3 -m localml_scholar.training_data.cli diagnose-reviewers --run RUN_ID
python3 -m localml_scholar.training_data.cli diagnostic-curation \
  --count 50 --seed 42
python3 -m localml_scholar.training_data.cli full-run-readiness
python3 -m localml_scholar.training_data.cli pipeline-self-test
python3 -m localml_scholar.training_data.cli ingestion-health
python3 -m localml_scholar.training_data.cli question-eligibility-report
python3 -m localml_scholar.training_data.cli codex-usage-report
python3 -m localml_scholar.training_data.cli pilot-curation --count 10 --seed 42
```

Run state is saved after every terminal question record in
`outputs/review_app/autonomous_curation_runs.json`. A cursor and stable stage ID
prevent completed questions from being rerun after an interruption. Source
document hashes are checked before resuming. A changed source suspends the run.
Routine per-candidate construction, retrieval, and validation failures are
saved and isolated. Repeated identical signatures or a high same-stage failure
rate suspends the run as a systemic defect.

## Splits and benchmark separation

Papers—not examples—are assigned to train, validation, or test. The default
fractions are 70/15/15; small corpora receive explicit sensible integer counts:
one paper is train-only, two papers are train/test, and three papers receive one
paper per split. Cross-paper examples are excluded if their papers have
different assignments.

Test-paper questions are retained as evaluation artifacts without answers or
corrections. They cannot enter `corrections.json` or a training export. This is
intended to test whether a future model can use evidence from unseen papers,
not memorize held-out papers.

## Trust and outputs

Only examples passing every configured gate become `codex_curated`. That label
means automated multi-pass review succeeded; it is not `human_approved` or
human gold. Uncertain, rejected, insufficient-evidence, duplicate, and
split-excluded records remain available for optional inspection but are not
exported for training.

Each completed run stores a dataset when at least one example qualified and a
manifest under `outputs/review_app/autonomous_curation/RUN_ID/`. Generated
outputs remain Git-ignored.

## Corpus-size recommendation

- The current six-paper corpus is appropriate for pipeline debugging.
- Use 10–15 diverse papers for the next autonomous quality run.
- Expand to 20–30 only after reviewing run-level error and disagreement trends.

Supplying 20–30 papers immediately is supported, but more inputs amplify any
systematic curation error. This corpus size is not enough to pretrain a general
language model.
