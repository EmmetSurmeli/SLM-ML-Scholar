# Paper Training Lab

## Fully Automated Curation (1.2.3)

The **Autonomous Curation** page is a parallel workflow for users who do not
want to label individual examples. Select papers, inspect the conservative
settings, and choose **Build Training Dataset**. The server creates a resumable
background run and performs local analysis, question generation, answering,
five Codex review passes, bounded repair, rejection, balancing, paper-level
splitting, export, and reporting. **Process all newly uploaded papers** scopes a
new run to papers absent from completed runs.

This mode does not bypass provenance: accepted records are `codex_curated`, not
human-approved. Uncertain and rejected records are retained only for optional
inspection. The existing Review, Calibration, and Corrections pages remain
available but do not block autonomous output. See
[autonomous corpus curation](autonomous_corpus_curation.md) and the
[quality policy](autonomous_quality_policy.md).

The Paper Training Lab is a loopback-only human-review application for building
grounded instruction data from locally indexed papers. It is a data-curation
tool, not an online-learning system. Saving or approving a record never changes
model parameters.

## Start the lab

From the repository root:

```bash
python3 -m pip install -e ".[app]"
localml-scholar-review
```

Open `http://127.0.0.1:8765`. The server refuses non-loopback bindings. Papers
are stored under `data/raw/review_app/`; indexes, interactions, questions,
corrections, opt-in session preferences, and dataset exports live under
`outputs/review_app/`. Both roots are Git-ignored.

## Workflow

1. Add PDF, UTF-8 text, or Markdown papers under **Papers**.
2. Inspect extracted metadata, sections, source text, and deterministic
   scholarly artifacts.
3. Select one, several, or all papers as the evidence scope.
4. Ask arbitrary questions naturally, including explanation-depth and format
   instructions.
5. Generate 40–80 proposed questions per paper or add manual questions.
6. Either run candidates individually, or use **Auto-review** to run every
   non-rejected question for one paper and prepare a conservative first pass.
7. Inspect each automatic draft. Edit its label, corrected answer, required
   facts, prohibited claims, and retained evidence, or uncheck it to exclude it.
8. Explicitly save the selected drafts as your reviews. No automatic draft is
   treated as a human judgment before this action.
9. Label individually reviewed answers as `correct`, `partial`, `incorrect`, `should_abstain`, or
   `benchmark_problem`.
10. Retain/reorder evidence, record required facts and prohibited claims, and
   edit a corrected answer.
11. Inspect the resulting correction proposal separately and approve it.
12. Create the deterministic audit queue and inspect mandatory-risk items.
13. Export an explicit trust tier with weights, duplicate control, paper-level
    splits, and a diversity report.
14. Use **Calibration** to validate a deterministic 50–100 item representative
    sample. Calibration approval is not training approval.

Automatic question generation, prompt variation, grading, and correction text
are suggestions. None are approved automatically. A second explicit approval
action is required before an example can enter an exported dataset.

## Automatic first-pass review

The **Auto-review** view reduces repetitive clicking but does not replace the
reviewer. For a selected paper, it uses every existing non-rejected question;
if the paper has no questions, it first creates the configured 40–80 proposed
candidates. It then runs the deterministic cited extractive answerer and saves
an editable draft for every response.

The first pass checks only behavior it can expose directly: expected
abstention, answer sufficiency, citation/claim validation, query-term coverage,
comparison source coverage, and whether a question type normally needs more
explanation than an extractive answer supplies. It proposes a five-way label,
confidence, rationale, retained evidence, facts, prohibited claims, and a
corrected-answer field. It is deliberately conservative: explanatory answers
are normally marked `partial` and flagged for editing because these rules
cannot judge semantic completeness or teaching quality.

One answer-construction error does not abort a new batch. The failed question
is retained as a visibly non-saveable review item while later questions keep
running. Batches created by the older all-or-nothing behavior expose **Resume
remaining questions**, which continues after their last completed question
without discarding the existing drafts.

At the final check, the reviewer may:

- keep a draft and save it unchanged;
- edit any proposed field before saving; or
- uncheck a draft to record it as excluded.

Saving creates a **proposed correction**, not an approved training record. The
reviewer must still open **Corrections**, inspect the saved proposal against the
paper, and approve it separately before dataset export. Batch state is stored
in `outputs/review_app/automatic_review_batches.json`. No external judge model,
web service, or semantic evaluator is called.

## Second pass, calibration, and audit

Each successful first-pass draft now includes evidence, answer, citation,
concept, completeness, instruction, style, and abstention scores from three
correlated deterministic configurations. Expanding a review card shows all 16
gates, mandatory-human categories, rationale, and provenance hashes.

Automatic approval is locked in `calibration_required` on a new workspace.
After 50–100 paired human outcomes qualify, the state becomes
`calibration_active`; a human must still explicitly enable it. Excess human
overrides move it to `auto_approval_suspended`. Confidence is not a guarantee.

The **Create 10% audit sample** action uses seed 42 and also includes every
near-threshold case, reviewer disagreement, and novel failure. Completed batch
items are persisted after each question. Batches support stop/resume without
discarding completed decisions.

The dataset page offers `human-only`, `human-and-audited` (default), and
`include-codex-approved` exports. Codex approvals remain labeled and weighted
as Codex approvals; they never become human gold.

## Multi-paper behavior

The Ask view accepts multiple selected papers. Retrieval is restricted to that
set and citations retain their source identity. The interaction records which
selected papers supplied evidence. If a comparison is requested but one source
does not supply evidence, the result is marked incomplete rather than filling
the gap from memory.

## Conversation privacy

Conversation turns and preferences are held in memory by default and expire
when the local server stops. Preference persistence is opt-in via the service
or API; only then is a redacted session record written to
`outputs/review_app/opt_in_sessions.json`. The lab does not silently persist
sensitive preferences.

## Command-line batch path

The same proposed-only and approved-only boundaries are available without the
browser:

```bash
python3 -m localml_scholar.evaluation.cli generate-paper-questions \
  --index outputs/review_app/index.json \
  --paper PAPER_ID \
  --count 60 \
  --output questions.json

python3 -m localml_scholar.evaluation.cli run-review-set \
  --index outputs/review_app/index.json \
  --paper PAPER_ID \
  --questions questions.json \
  --output review_results.json

python3 -m localml_scholar.evaluation.cli export-training-data \
  --reviews outputs/review_app/corrections.json \
  --approved-only \
  --output approved_dataset.json

python3 -m localml_scholar.evaluation.cli dataset-report \
  --dataset approved_dataset.json

python3 -m localml_scholar.evaluation.cli auto-review \
  --repository . --paper PAPER_ID

python3 -m localml_scholar.evaluation.cli audit-sample \
  --repository . --rate 0.10 --seed 42

python3 -m localml_scholar.evaluation.cli calibration-report --repository .

python3 -m localml_scholar.evaluation.cli calibration-sample \
  --repository . --count 50 --seed 42

python3 -m localml_scholar.evaluation.cli rerun-historical-reviews \
  --repository . --sample-only

python3 -m localml_scholar.evaluation.cli calibration-status --repository .

python3 -m localml_scholar.evaluation.cli enable-auto-approval --repository .

python3 -m localml_scholar.evaluation.cli bulk-auto-review \
  --repository . --eligible-only

python3 -m localml_scholar.evaluation.cli export-trust-tier \
  --repository . --trust-tier human-and-audited \
  --output outputs/review_app/trusted_dataset.json
```

## Progress targets

The dashboard shows 100, 300, and 600 approved-example workflow targets. They
measure review throughput only. They are not model-quality or capability
claims.
