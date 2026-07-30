# Local paper review application

## Purpose and boundary

The Review Lab is a narrow local interface for the human-in-the-loop stage of
LocalML Scholar. It lets a reader:

1. add an extractable PDF, UTF-8 text file, or Markdown paper;
2. inspect deterministic scholarly artifacts;
3. ask an evidence-scoped question;
4. read the exact passages and citations used by the answer;
5. choose whether the answer is being evaluated for a PhD researcher/professor,
   undergraduate, or high-school/beginner reader;
6. save a verdict, pedagogical issue labels, notes, and an optional corrected
   answer.

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
| Human review queue | `outputs/review_app/feedback.json` | ignored |

The interface displays the resolved absolute paths for the current repository.
Writes use same-filesystem temporary files followed by atomic replacement.
Concurrent request mutations are serialized by the application service.

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

Questions are scoped to the selected document with an explicit
`SearchFilters(document_id=...)`. The current trusted path is deterministic
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

Each feedback record includes:

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
feedback records for the same answer at different audience levels.

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
