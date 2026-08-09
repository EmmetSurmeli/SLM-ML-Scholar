# Local Paper Training Lab

Version 1.2.3 also exposes **Autonomous Curation**. This no-human-gate mode uses
the installed Codex CLI for structured review and therefore is distinct from
the fully local deterministic baseline. The website itself remains loopback
only, papers are supplied explicitly, and public web search/download is never
performed. See `docs/training/autonomous_corpus_curation.md` for the run and
privacy boundary.

## Purpose and boundary

This document describes the original Review Lab boundary, now expanded by
Milestone 12A into the Paper Training Lab. See
[`docs/training/paper_training_lab.md`](training/paper_training_lab.md) for the
complete current workflow. The application lets a reader:

1. add an extractable PDF, UTF-8 text file, or Markdown paper;
2. inspect deterministic scholarly artifacts;
3. select one, several, or all papers and ask an evidence-scoped question;
4. read the exact passages and citations used by the answer;
5. express depth, mathematical background, format, derivation, critique, or
   comparison needs naturally in the prompt;
6. generate proposed benchmark questions and save a five-way review label,
   evidence edits, factual targets, notes, and a corrected answer;
7. separately approve corrections and export a paper-split dataset.

The server binds only to `127.0.0.1`. It makes no cloud request and has no
external model API. The browser assets are packaged with the project and use no
CDN. The paper, index, interactions, and feedback stay under ignored repository
paths.

This interface does **not** turn feedback into training updates. A feedback
record is a review queue item. It captures enough context for a later Codex task
or manual data-curation step to diagnose and deliberately change extraction,
retrieval, prompts, evaluation fixtures, or a legally usable training corpus.
This separation prevents an unchecked answer correction from silently changing
the model.

## Install and run

From the repository root:

```bash
python3 -m pip install -e ".[app,dev]"
localml-scholar-review
```

Then open:

```text
http://127.0.0.1:8765
```

An equivalent source-tree command is:

```bash
PYTHONPATH=src python3 -m localml_scholar.review_app.server
```

Use `--port 9000` to select another loopback port. The application deliberately
rejects non-loopback binding.

## Local persistence

Default paths are:

| Content | Repository path | Git policy |
| --- | --- | --- |
| Uploaded paper bytes | `data/raw/review_app/` | ignored |
| Immutable retrieval snapshot | `outputs/review_app/index.json` | ignored |
| Complete question/answer snapshots | `outputs/review_app/interactions.json` | ignored |
| Legacy human feedback | `outputs/review_app/feedback.json` | ignored |
| Proposed questions | `outputs/review_app/question_candidates.json` | ignored |
| Proposed/approved corrections | `outputs/review_app/corrections.json` | ignored |
| Automatic review batches | `outputs/review_app/automatic_review_batches.json` | ignored |
| Approved dataset | `outputs/review_app/grounded_instruction_dataset.json` | ignored |
| Opt-in session preferences | `outputs/review_app/opt_in_sessions.json` | ignored |

The interface displays the resolved absolute paths for the current repository.
Writes use same-filesystem temporary files followed by atomic replacement.
Concurrent request mutations are serialized by the application service.

The **Auto-review** view can run all non-rejected questions for one paper and
prepare editable first-pass labels from deterministic evidence diagnostics.
These drafts are explicitly marked as non-semantic suggestions. A reviewer
must save or exclude each draft, and every saved review remains a proposed
correction until the separate Corrections approval step.

To have Codex inspect your reviews, use a prompt such as:

> Read `outputs/review_app/feedback.json`. Group the verified failures by cause,
> trace each one to retrieval, extraction, answering, or interface behavior, and
> propose fixes. Do not treat a reviewer correction as ground truth without
> checking its cited paper evidence.

The review file is ignored by Git because it may contain copyrighted paper
passages and personal notes. Export a deliberately curated, licensed fixture
separately if a correction should become a committed regression test.

## Ingestion behavior

- `.md` and `.markdown` use the existing exact-text ATX-heading ingestion.
- `.txt` uses exact UTF-8 plain-text ingestion.
- `.pdf` uses the optional local `pypdf` adapter to extract page text. Each
  extracted page is passed to the existing page-aware ingestion boundary, so
  citations retain page numbers.
- encrypted PDFs are rejected;
- scanned/image-only PDFs are rejected with an OCR instruction;
- the upload size limit is 30 MiB;
- adding a new file with the same sanitized filename replaces that logical
  source and rebuilds the immutable index.

PDF extraction is text-only. It does not perform OCR, recover visual equations,
reconstruct figure meaning, or guarantee reading order for complex layouts.
Always compare important answers with the rendered paper.

## Question and feedback semantics

Questions are scoped to the selected papers. One-paper queries use an explicit
`SearchFilters(document_id=...)`; subset comparisons build a temporary local
index over only those selected documents. The current trusted path is deterministic
extractive answering. It can select and quote relevant sentences with exact
citations or abstain when evidence is insufficient. It is not yet a trained
scholarly explanation model. The intended audience is recorded with the
interaction, but this extractive baseline does not rewrite cited passages at
different levels.

Each interaction record includes:

- question and selected document;
- intended audience level;
- exact answer text;
- evidence passages and source citations;
- sufficiency and validation state;
- index and evidence identities.

Legacy feedback records include:

- `correct`, `partially_correct`, or `incorrect`;
- the audience level against which that verdict was made;
- zero or more controlled issue categories;
- reviewer notes and optional corrected answer;
- a complete snapshot of the reviewed interaction;
- `pending_codex_review` status.

Non-correct verdicts require at least one issue category. This turns review into
structured diagnostic evidence without pretending that it is online learning.
Pedagogical categories include `too_advanced`, `too_basic`,
`missing_prerequisite`, and `unclear_explanation`. A reviewer may save separate
feedback records for the same answer at different audience levels. The current
training-data path instead uses `correct`, `partial`, `incorrect`,
`should_abstain`, or `benchmark_problem`, and requires a separate approval
before export. Canonical audience labels remain optional regression metadata;
the user-facing app infers an adaptive `InstructionProfile` without requiring a
static selector.

## Verification

The service and HTTP boundary tests cover:

- ingestion, replacement, listing, and immutable index reload;
- deterministic scholarly analysis;
- cited extractive questions;
- all three audience levels through interaction and feedback persistence;
- exact interaction snapshots in feedback;
- page preservation through the PDF adapter boundary;
- malformed filenames, encodings, state, IDs, and feedback;
- static application serving and the full upload/question/feedback HTTP flow;
- enforced loopback binding.

Run:

```bash
python3 -m pytest -q tests/test_review_app_service.py tests/test_review_app_server.py
python3 -m ruff check .
python3 -m ruff format --check .
```

## Confidence-gated second pass

Milestone 12A.1 adds a distinct second-pass review card showing confidence,
all mandatory gates, human-only risk routes, correction state, and provenance.
The dashboard exposes calibration and deterministic audit sampling. A
`codex_approved` item remains visibly Codex-approved after an audit; it is not
silently promoted to human gold. Batch stop/resume and per-item re-review retain
completed decisions through atomic local writes.
