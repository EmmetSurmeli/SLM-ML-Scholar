# Milestone 11 real-paper evaluation readiness audit

## Scope

This audit inspected the retrieval, grounded-answer, scholarly-analysis,
Review Lab, serialization, CLI, experiment, test, ignore, and documentation
surfaces before adding Milestone 11.5. It deliberately did not refactor the
transformer, tokenizer IDs, retrieval mathematics, or paper extraction rules.

## Components inspected

- `retrieval/`: immutable documents, sections, chunks, result metadata,
  citations, lexical/semantic/hybrid ranking, metrics, index hashes, and
  atomic JSON persistence
- `answering/`: evidence selection, sufficiency, answer artifacts, claim
  segmentation, citation parsing, support diagnostics, acceptance, fallback,
  checkpoint/tokenizer identity, and existing authored evaluation fixtures
- `scholarly/`: source-bound paper fields, equations, notation, experiments,
  summaries, checklists, artifact hashes, and CLI conventions
- `review_app/`: paper upload, exact interaction snapshots, audience labels,
  corrections, ignored local storage, and loopback-only service behavior
- package versioning, generated-output ignore rules, tests, experiments,
  README capability language, architecture, and roadmap

## Reusable infrastructure

The existing index already preserved exact chunk, document, section, source
range, page/line, and index identities. `GroundedAnswer` retained raw and
processed generation, evidence hashes, claims, validation results, fallback
state, and exact citations. Scholarly analysis supplied deterministic
candidate sources for metadata, equations, notation, methodology, experiments,
and limitations. Atomic UTF-8 JSON replacement and canonical hashing could be
reused without pickle or a new persistence dependency.

The answering code's older evaluation helper measured answerability, citation
coverage, and key-fact presence on authored fixtures. Retrieval metrics already
provided precision, recall, MRR, AP, and nDCG. These were sound building
blocks, but neither layer was a paper-level gold benchmark or a stage-wise
error-analysis system.

## Missing layers found

- no versioned, source-bound benchmark schema with approval status
- no hard boundary excluding proposed questions from official metrics
- no separate grades for retrieval, sufficiency, relevance, completeness,
  citation relevance, and audience level
- no question-type-aware section/boilerplate policy
- no required-concept aliases or prohibited-claim annotations
- no multi-label failure taxonomy or cautious root-cause attribution
- no deterministic selective human-review queue
- no human-approved correction export
- no per-question regression comparison
- no common batch runner or evaluation CLI
- no single evidence-backed target rendered at three audience levels

## Defects and fixes

1. Benchmark review edits initially could not replace `gold_evidence`, even
   though evidence selection is a required reviewer action. Review JSON now
   accepts and validates full `GoldEvidence` records.
2. Evaluation artifacts initially rejected any package version other than the
   currently installed patch version. That made cross-version regression
   comparison impossible. Format versions and hashes remain strict, while a
   recorded non-empty package version may be loaded and compared.
3. Scholarly artifacts had the same patch-version invalidation behavior.
   Their schema/hash validation remains strict, but older recorded package
   versions are no longer discarded solely because the package was patched.
4. The existing Review Lab used three UI-specific audience identifiers.
   Storage and UI now use the canonical `beginner`, `undergraduate`, and
   `researcher` values. The service maps the two legacy input values when
   reading older interactions.
5. Initial answer-relevance rules failed to recognize “blocks” and “prevents”
   as mechanism language. Both answer and audience graders now recognize
   those explicit causal/mechanistic verbs.
6. Required retrieval reporting omitted title-page success for metadata and
   abstract/introduction success for motivation questions. Both are explicit
   `RetrievalGrade` signals and are aggregated only for applicable question
   types.

## Benchmark schema decisions

- Question IDs are deterministic hashes of paper ID, normalized question,
  type, and audience.
- Paper and index hashes are immutable benchmark inputs.
- Answerability and paper sufficiency are separate compatible categorical
  labels.
- Gold evidence stores exact chunk and optional source ranges with graded
  relevance.
- Required concepts use manually reviewed aliases; prohibited claims remain
  explicit strings.
- Only `approved` and `edited` questions enter `EvaluationRunner`.
- Candidate retrieval is labeled untrusted and cannot become gold without an
  explicit review decision and notes.
- JSON artifacts use canonical hashes, atomic replacement, strict field
  validation, and stale-source/chunk rejection.

## Audience policy

All three audiences share one `StructuredAnswerTarget`. Deterministic renderers
may omit advanced detail but may not add a new factual claim or remove the core
claim/citation. Transparent statistics are diagnostic only. Historical,
synthesis, interpretation, ambiguous, and borderline style cases always
require human review.

## Human-review policy

The queue includes every automatic failure, every low-confidence case,
explicit method disagreements, historical/synthesis/interpretation questions,
ambiguous gold items, numerical contradictions, citation-valid but
low-relevance answers, and a seeded sample of passes. Pending labels are never
imputed. Duplicate adjudication is rejected. Only substantive, human-reviewed
corrections with approved gold chunks can enter the correction dataset;
`benchmark_problem` records are excluded.

## Failure and root-cause policy

Failures are multi-label and stable-ordered. Root cause is a likely diagnostic,
not a fact. When more than one stage is implicated, the artifact records a
primary cause, secondary causes, reasons, medium confidence, and a concrete
next inspection action. Correct abstention is informational and does not turn
an otherwise clean record into a failure.

## Verification commands

Run from the repository root:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
python3 -m pytest -q
python3 experiments/evaluate_real_paper_benchmark.py
python3 -m localml_scholar.evaluation.cli --help
```

The first experiment intentionally exits with status 2 when no user benchmark
and index are supplied. This is the verified non-fabrication behavior, not a
failed evaluation.

Final results on 2026-07-29:

- Ruff lint: clean
- Ruff format check: 191 files already formatted
- `git diff --check`: clean
- pytest: 686 passed in 7.73 seconds
- focused Milestone 11.5 evaluation tests: 55 collected, plus Review Lab
  canonical/legacy audience compatibility coverage
- authored benchmark: one approved question, Recall@1/3/5 `1.0`, answer
  relevance `0.825`, required-concept recall `0.5`, citation validity/support/
  relevance/coverage `1.0`
- detected smoke failures: `retrieval_wrong_section`,
  `required_concept_missing`
- Attention starter: 33 proposed, 0 approved
- review/correction: 1 queued, 1 explicitly adjudicated correction exported
- controlled comparison: 8 BM25/semantic/hybrid/hybrid-reranked ×
  top-passage/extractive runs
- real-paper experiment without explicit inputs: status 2, no fabricated run

The bigram fallback smoke, XOR experiment, scholarly extraction fixture, and
existing extractive-answer evaluation also completed under package 1.1.1.

## Unresolved limitations and deferred work

- Lexical heuristics do not prove semantic relevance, correctness, or
  entailment.
- Gold concepts, claims, answerability, and evidence require human judgment
  and may contain benchmark bias.
- Review agreement statistics need multiple independent reviewers.
- Source extraction errors can invalidate otherwise sound grading.
- Historical truth cannot be verified without a separately approved external
  evidence benchmark.
- The initial paper set is small; no population-level performance claim is
  justified.
- No external judge, neural entailment model, instruction training,
  transformer fine-tuning, OCR, or literature verification was added.
