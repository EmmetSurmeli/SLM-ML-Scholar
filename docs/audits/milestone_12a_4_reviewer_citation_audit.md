# Milestone 12A.4 reviewer and citation audit

Audit date: 2026-08-09
Frozen run: `curation_5d2d2687e0e74c61918e44894297303c`
Source package: 1.2.3
Audit implementation: 1.2.4

The original 1.9 MB run in
`outputs/review_app/autonomous_curation_runs.json` was loaded read-only and
migrated only in memory. Its ten records, hashes, pass outputs, repair history,
cursor, terminal state, and errors were not modified.

## Checks performed

- inspected all answerer, evidence-critic, answer-critic, citation-critic, and
  final-adjudicator inputs/outputs and confidence values;
- resolved answer citations against the evidence current at each terminal
  record;
- segmented terminal answers into atomic claims and reran structural/lexical
  gates;
- compared critic decisions pairwise and against the final adjudicator;
- separated factual conflicts from style-only differences;
- inspected every repair for evidence/answer/citation/target changes;
- verified source hashes, paper splits, held-out-paper isolation, and record
  lineage;
- audited CLI flags, reviewer blindness, prompt vocabulary, safety stops,
  dashboard metrics, Git ignore rules, and 1.2.3 artifact compatibility.

## Original-run aggregate findings

| Metric | Original run | 1.2.4 in-memory diagnosis |
|---|---:|---:|
| processed | 10 | 10 |
| evidence validation | 90% | 90% |
| legacy citation validation | 60% | — |
| claim-level citation structure | — | 30% |
| claim-level citation support | — | 30% |
| claim-level citation relevance | — | 30% |
| overall/hard disagreement | 70% | 70% |
| soft disagreement | not represented | 0% |
| stale stable evidence IDs | not represented | 0 |
| source hash mismatch | 0 | 0 |
| malformed reviewer output | 0% | 0% |
| leakage | 0 | 0 |

The 30% claim-level rate is expected for this frozen artifact: seven terminal
answers still contain raw pre-1.2.4 citation markers or leave an abstention/
limitation claim uncited. The previous whole-answer check counted any resolving
citation and therefore hid partially uncited answers.

## Candidate-by-candidate audit

All confidence values below are `pass=decision@confidence` in chronological
order. Evidence IDs name the terminal passages; the immutable JSON retains the
full passage text and structured targets.

### 1. Who wrote 1406.2661v1?

- Paper/type: `doc_027c3a57cbb2fab611372fab`, metadata.
- Evidence: `C1=chk_457fb86fb4dbf822d2d74d58` (page 1 footnotes and
  arXiv header); `C2=chk_ddaf1548a6930d8730f43ee0` (page 8
  acknowledgments).
- Terminal answer: abstains because neither passage establishes the complete
  author list; the second sentence cites raw chunk IDs.
- Repair: one attempt changed the answer/target after the citation critic asked
  to map the actual absence claim and distinguish inference from explicit fact.
- Passes: `answerer=reject@0.99`, `evidence=accept@0.99`,
  `answer=accept@0.99`, `citation=repair@0.98`,
  `adjudicator=repair@0.99`; then `answerer=repair@0.97`,
  `evidence=accept@0.99`, `answer=accept@0.99`,
  `citation=accept@0.99`, `adjudicator=accept@0.99`.
- Claim mapping: the abstention/complete-author-list claim has no local label;
  the footnote/acknowledgment claim resolves.
- Failure/disagreement: claim-citation structure failed; deterministic validator
  FAIL/citation critic ACCEPT; disagreement persisted after repair.
- Terminal: uncertain (`complete author list absent`, deterministic review
  failed).

### 2. What problem motivated the work?

- Paper/type: same paper, motivation.
- Evidence: `C1=chk_c21c227f4cb7f82ab5b1d840`, intractable partition
  function/gradient and MCMC mixing difficulty.
- Terminal answer: gives that motivation but retains a raw evidence-ID marker.
- Repair: one attempt removed “undirected models,” corrected “can only
  estimate,” and selected the single relevant passage.
- Passes: first cycle `repair/accept/accept/repair/repair` at
  `0.97/0.98/0.98/0.96/0.99`; second cycle
  `repair/accept/accept/accept/accept` at `0.99` throughout.
- Claim mapping: semantically relevant and lexically supported after in-memory
  label normalization; frozen terminal validation still rejected the raw marker.
- Failure/disagreement: no remaining critic conflict; frozen terminal reports
  citation and deterministic validation failure.
- Terminal: uncertain.

### 3. What is the central method?

- Paper/type: same paper, method.
- Evidence: `C1=chk_bbee965ac035b5a63d12a47a` and
  `C2=chk_ace25344b8fbdee21c1bb8c8`, adversarial framework and simultaneous
  generator/discriminator updates.
- Terminal answer: identifies the adversarial framework and training relation,
  but cites raw chunk IDs.
- Repairs: two. Retrieval/interpretation shifted from an unsupported abstention
  to stronger method evidence; structured targets remained inconsistent with
  the terminal answer.
- Passes by cycle: `repair/reject/accept/repair/repair`;
  `repair/accept/accept/accept/repair`; then
  `repair/accept/repair/repair/repair` (confidence 0.98–0.99).
- Claim mapping: terminal prose is structurally normalizable and supported, but
  the critic target included unrelated likelihood-estimation/update claims.
- Failure/disagreement: evidence-vs-answer, evidence-vs-citation, deterministic
  PASS/citation critic FAIL, and disagreement after repair.
- Terminal: rejected for reviewer disagreement, repair exhaustion, stale target
  semantics, and frozen citation/deterministic failures.

### 4. What are the main architectural components?

- Paper/type: same paper, architecture.
- Evidence: `C1=chk_fcac309de279c67cc1a09b2a`, generator/discriminator and
  gradient flow.
- Terminal answer: first sentence names generator and discriminator without a
  citation; second sentence cites the passage with a raw chunk ID.
- Repairs: two; removed exclusivity, unsupported `G`/`D` notation, stale IDs,
  and an unsupported auxiliary inference network.
- Passes: three repair cycles; final critics were
  `evidence=repair@0.97`, `answer=repair@0.97`,
  `citation=accept@0.97`, adjudicator `repair@0.98`.
- Claim mapping: first substantive claim missing citation, second supported.
- Failure/disagreement: answer/evidence critics versus citation critic,
  deterministic FAIL/citation ACCEPT, disagreement after repair.
- Terminal: rejected; repair limit exhausted.

### 5. What is the paper's key equation?

- Paper/type: same paper, equation.
- Evidence: `C1=chk_8d91e9911ab02256af54cb62` and
  `C2=chk_48a005cf1100f38fa3011c35`, Equation 1 description and gradient
  limitation.
- Terminal answer: abstains from reproducing the missing expression, but its
  first equation/absence claim is uncited; the later gradient claim uses raw
  chunk IDs.
- Repairs: two, chiefly provenance and stale evidence-ID corrections.
- Passes: `repair/accept/accept/accept/repair`;
  `repair/accept/accept/repair/repair`; then
  `repair/accept/repair/repair/repair` (0.94–0.99).
- Claim mapping: first claim missing citation and numeric support for “1”;
  second supported.
- Failure/disagreement: evidence critic versus answer/citation critics and
  disagreement after repair.
- Terminal: rejected for unresolved ambiguity and repair exhaustion.

### 6. How do the authors move from one equation to the next?

- Paper/type: same paper, derivation.
- Evidence: `C1=chk_40b8c74bd2438f00466c2104` and
  `C2=chk_48a005cf1100f38fa3011c35`.
- Terminal answer: correctly abstains from reconstructing an absent transition,
  then summarizes available Equation 1 context using raw evidence IDs.
- Repair: none; answerer requested repair but all critics/adjudicator accepted.
- Passes: `answerer=repair@0.98`; evidence/answer/citation/adjudicator all
  `accept@0.99`.
- Claim mapping: first two substantive claims lack normalized citations; final
  gradient claim resolves.
- Failure/disagreement: deterministic FAIL/citation critic ACCEPT.
- Terminal: uncertain; deterministic review and derivation provenance failed.

### 7. Explain the core idea intuitively.

- Paper/type: same paper, intuition.
- Evidence: none.
- Answer: deterministic insufficient-evidence abstention.
- Repair/review passes: none; Codex was correctly not called.
- Claim mapping: the generic abstention sentence has no citation, but this
  record is outside the reviewed-record safety denominator.
- Failure/disagreement: no reviewer disagreement.
- Terminal: insufficient evidence.

### 8. What is the computational complexity?

- Paper/type: same paper, complexity.
- Evidence: `C1=chk_ac022a4f0ba6bcacd6ad761d`, qualitative computational
  advantages only.
- Terminal answer: correctly says no asymptotic measure is supplied, then cites
  the qualitative advantages with a raw chunk ID.
- Repairs: two, removing unrelated evidence and making abstention explicit.
- Passes: first `repair/accept/repair/repair/repair`; second and third critics
  unanimously accept, while the adjudicator continues to request repair
  (0.97–0.99).
- Claim mapping: absence claim uncited; qualitative claim supported.
- Failure/disagreement: final adjudicator overrides critic majority,
  deterministic FAIL/citation ACCEPT, disagreement after repair.
- Terminal: rejected; repair limit exhausted.

### 9. What inductive bias does the method introduce?

- Paper/type: same paper, critical reasoning.
- Evidence: `C1=chk_597ff0461a7a938fd8e48534`, which does not state the
  requested inductive bias.
- Terminal answer: abstains, without a citation.
- Repair: none.
- Passes: `answerer=reject@0.99`; every critic and adjudicator
  `accept@0.99`.
- Claim mapping: the absence/abstention claim has no citation.
- Failure/disagreement: deterministic FAIL/citation critic ACCEPT.
- Terminal: uncertain; deterministic validation failed.

### 10. Which datasets were used?

- Paper/type: same paper, experiment.
- Evidence: `C1=chk_b232114ace3c2025c36f1269`, naming MNIST, TFD, and
  CIFAR-10.
- Terminal answer: qualified dataset list with a raw evidence-ID marker.
- Repairs: two; removed an exhaustiveness implication, unrelated claims, and
  stale evidence.
- Passes: `repair/accept/repair/accept/repair`;
  `repair/accept/accept/repair/repair`; then all five accept at 1.00.
- Claim mapping: normalized claim is supported and relevant.
- Failure/disagreement: no remaining critic conflict; frozen terminal still
  reports old citation/deterministic failures.
- Terminal: uncertain.

## Verified defects and fixes

1. **One scalar disagreement conflated policy and wording.** Added multi-label
   taxonomy and hard/soft severity; safety uses hard conflict.
2. **All reviewer roles shared one broad output vocabulary.** Added canonical
   typed policy outcomes and role-specific prompt contracts.
3. **Citation parsing differed across stages.** Added one canonical normalizer
   used before critics and acceptance.
4. **Whole-answer resolution hid uncited claims.** Added atomic claims and
   per-claim structural/support/relevance results.
5. **Display labels were mistaken for source identity.** Added stable identity
   from document/chunk/source coordinates and repair invariants.
6. **Final adjudication had no deterministic conflict policy.** Structural,
   evidence, unsupported-claim, ambiguity, and hard-conflict triggers are now
   passed before semantic adjudication.
7. **Repair attempts lacked outcome measurement.** Added changed-field and
   fixed/introduced/unchanged diagnostics.
8. **Insufficient-evidence records polluted safety denominators.** Only records
   with actual Codex passes count toward reviewer/citation stop rates.
9. **Original CLI citation labels survived into terminal prose.** The prior
   fix that maps raw IDs to current `[C#]` labels is now followed by canonical
   normalization and structural validation.

The previously verified 12A.3 fixes (Codex CLI flags, section-bounded
extraction, lexical repair retrieval, stale target removal, citation-label
conversion, and duplicate section-question removal) remain in the working tree
and are covered by regression tests.

## Tests added

Focused tests cover citation spellings/order/duplicates/malformed/unknown,
claim segmentation (multiple, numeric, equation, qualification), stable source
identity, wrong paper/section, missing and numerically unsupported citations,
hard/soft conflicts, validator/critic conflicts, deterministic adjudication,
repair outcomes, safety thresholds, leakage, deterministic stratified sampling,
readiness, report matrices, and non-mutating legacy migration.

## Unresolved limitations

- Sentence splitting is intentionally conservative and is not a semantic
  proposition parser.
- Lexical overlap and numeric/concept checks cannot prove entailment.
- The frozen run covers ten GAN-paper questions and cannot establish reliability
  across 14 papers or all question types.
- Repair success cannot be recovered perfectly from old artifacts because
  1.2.3 did not save every pre/post validation state.
- Figure interpretation, historical impact, cross-paper synthesis, derivation,
  equation, complexity, and critical-reasoning categories remain excluded from
  autonomous training pending controlled evidence.
- Milestone 12B remains blocked until controlled diagnostics pass and accepted
  examples have suitable licensing/provenance.

## Verification commands and results

The controlled run `diagnostic_a5a5d71ccba547a6af6261246da91f91`
selected 50 candidates with seed 42 from the verified frozen pool. It stopped
safely after 11 terminal candidates / 10 genuine reviewed candidates; the
remaining 39 were not run. Results:

| Controlled metric | Result | Target | Pass? |
|---|---:|---:|---|
| accepted | 0 | no quota | n/a |
| rejected / uncertain / insufficient | 9 / 1 / 1 | — | — |
| evidence validation | 90.9% | >=95% | no |
| citation structural validity | 20.0% | >=98% | no |
| citation support | 20.0% | >=95% | no |
| citation relevance | 20.0% | >=95% | no |
| hard disagreement | 60.0% | <=10% | no |
| overall disagreement | 60.0% | <=15% | no |
| soft disagreement | 0.0% | reported | — |
| repair success | 22.2% | positive | yes |
| stale IDs / source mismatches / leakage | 0 / 0 / 0 | all zero | yes |
| malformed reviewer output | 0.0% | <=2% | yes |

The first attempt stopped one record early under a superseded legacy
whole-answer citation rule. That rule was removed, reviewed-record denominators
were made explicit, and the same persisted run resumed for one additional
reviewed candidate. The final stop reason was correctly: **Hard reviewer
disagreement exceeded the configured safety threshold.** The second diagnostic
was not run, the full run is not ready, and Milestone 12B remains blocked.

Final verification commands:

```bash
PYTHONPATH=src python3 -m localml_scholar.training_data.cli \
  diagnose-reviewers --run curation_5d2d2687e0e74c61918e44894297303c
python3 -m pytest tests/test_reviewer_citation_reliability.py \
  tests/test_autonomous_curation.py -q
python3 -m pytest -q
python3 -m ruff check .
python3 -m ruff format --check .
node --check src/localml_scholar/review_app/static/app.js
git diff --check
```

The complete suite passed with 845 tests. Ruff lint/format, JavaScript syntax,
and `git diff --check` were clean. Generated run state remained Git-ignored.
