# Milestone 12A.5 claim-alignment audit

## Scope and artifact preservation

This audit reads, but does not modify, controlled Milestone 12A.4 run
`diagnostic_a5a5d71ccba547a6af6261246da91f91` in
`outputs/review_app/autonomous_curation_runs.json`. The artifact remains ignored
and byte-for-byte outside the 1.2.5 implementation. It contains 11 terminal
candidates: 10 genuine multi-pass reviews and one deterministic insufficient-
evidence terminal.

The reviewed denominator explains the reported 20% citation result: only
candidates 8 and 10 passed structural, support, and relevance validation. The
unreviewed insufficient-evidence candidate is part of evidence-validation counts
but not the citation-rate denominator.

## Terminal-candidate analysis

### 1. `question_3eb002c3f2c122f40157` — central method

- Question: “What is the central method?”
- Evidence: final evidence labels C1–C3 cover generative-model context,
  discriminator classification, and learning dynamics.
- Generated answer: two broad method claims cite C9 and C5/C6.
- Atomic claims/map: K1 (adversarial estimation and discrimination) → C9; K2
  (discriminator update and mutual improvement) → C5+C6.
- Unsupported/citation failures: every cited label is absent from the final
  evidence set; both claims are therefore unresolved and irrelevant to their
  cited final passages.
- Critics/adjudicator: evidence critic `accept/direct`; answer critic
  `repair/malformed`; citation critic `repair/malformed`; adjudicator
  `repair/malformed`.
- Repair history/outcome: two attempts, both unchanged. Retrieval changed the
  evidence ranking without rebuilding the claim/citation plan.
- Disagreement cause: evidence sufficiency was judged independently of the stale
  answer mapping, while answer/citation critics evaluated the stale labels.

### 2. `question_fc82de2e85657eb586ea` — architecture

- Evidence: final C1–C4 discuss the two MLPs, generated samples,
  discriminator rejection, and computational comparison.
- Generated answer: three architectural claims cite C9, C8+C10, and C2.
- Atomic claims/map: K1 (G and D are MLPs) → C9; K2 (generator samples and
  discriminator rejection) → C8+C10; K3 (gradient path) → C2.
- Failures: K1 and K2 use stale labels; K3 is structurally valid but its final
  passage does not establish the full gradient-path wording.
- Critics/adjudicator: `accept/direct`, `repair/unsupported`,
  `reject/unsupported`, `repair/unsupported`.
- Repairs: two unchanged attempts; the second broadened prose instead of
  narrowing it to the selected evidence.
- Disagreement: support versus unsupported and accept versus repair/reject.

### 3. `question_3331947374ea5a036c45` — intuition

- Evidence: none.
- Answer: deterministic abstention: insufficient indexed support.
- Claims/map: none; no ordinary answer was attempted.
- Critics/adjudicator/repairs: not run; this is the correct conservative
  terminal behavior.
- Outcome: `insufficient_evidence`; it lowers evidence validation to 10/11 but is
  excluded from the 10-record citation denominator.

### 4. `question_cfe3511731c6189df003` — inductive bias

- Evidence: C1 discusses a likelihood estimator's variance and poor behavior in
  high dimensions, not the requested inductive bias.
- Claims/map: K1 (evidence does not identify an inductive bias) → uncited; K2
  (variance/high-dimensional weakness) → C1; K3 (question cannot be answered) →
  C1.
- Failures: K1 is a substantive evidence-scope claim without traceability; K2
  is real but not directly responsive; K3 is an evidence-sufficiency decision
  expressed as if C1 stated it.
- Critics/adjudicator: `reject/irrelevant`, `repair/correct_abstention`,
  `repair/incomplete`, `repair/correct_abstention`.
- Repairs: two unchanged attempts.
- Disagreement: answer/abstention correctness versus irrelevant evidence.

### 5. `question_07efb81e1ec733360828` — results

- Evidence: final C1–C4 cover generator distribution, discriminator optimum,
  sample rejection, and Parzen evaluation.
- Claims/map: K1 (global optimum/Algorithm 1) → C2; K2 (equilibrium and D=1/2)
  → C8; K3 (non-saturating objective gradients) → C4; K4 (Table 1 values
  unavailable) → C5.
- Failures: K2 and K4 cite stale labels; numeric tokens 1 and 2 are not aligned
  to those final passages; K1 bundles theorem and algorithm claims under one
  citation; K3's citation does not establish the entire broad clause.
- Critics/adjudicator: evidence `repair/partial`; answer `repair/incomplete`;
  citation `repair/wrong_span`; adjudicator `repair/wrong_span`.
- Repairs: two unchanged attempts.
- Disagreement cause: all reviewers recognized partial evidence, but the generic
  rewrite never decomposed or remapped the claims.

### 6. `question_65db56c8bbfc99b79d0b` — false premise

- Evidence: final C1–C3 cover the objective proof, framework trade-offs, and
  synchronized optimization.
- Claims/map: K1 (qualified “No”) → uncited; K2 (global minimum −log 4) → C3;
  K3 (synchronization and Equation 1 gradient) → C4+C9; K4 (no practical
  guarantee) → C3+C4+C9.
- Failures: K1 has no claim trace; K2's number is not in cited C3; K3/K4 retain
  stale C4/C9; explicit theorem facts and inference are mixed.
- Critics/adjudicator: evidence `accept/direct`; answer `repair/supported`;
  citation `repair/incomplete`; adjudicator `repair/supported`.
- Repairs: two unchanged attempts.
- Disagreement: evidence can address the false premise, but the answer's
  inference/citation structure remains invalid.

### 7. `question_c2291865991e6f122439` — experiment extraction

- Evidence: final C1–C6 discuss discriminator behavior, TFD cross-validation,
  table results, optimization context, activations, and the theoretical loss.
- Claims/map: 11 claims covering datasets, architecture, activation details,
  loss, optimizer, metrics, and hyperparameters.
- Failures: C7/C8 are stale; four substantive qualification/absence claims are
  uncited; the theoretical loss contains unaligned numbers; several real claims
  are irrelevant to their cited passages; “backpropagation and dropout” is
  mislabeled as an optimizer answer.
- Critics/adjudicator: evidence `repair/partial`; answer `repair/incomplete`;
  citation `repair/unsupported`; adjudicator `repair/incomplete`.
- Repairs: two unchanged attempts.
- Disagreement cause: a broad checklist answer was generated before determining
  which requested fields the available passages could actually support.

### 8. `question_302c038f6115a856167f` — provenance

- Evidence: C1 explicitly discusses parameter copying and auxiliary inference;
  C2 explicitly contrasts Markov-chain sampling.
- Claims/map: K1 (three directly stated claims) → C1+C2; K2 (none requires
  inference) → C1+C2.
- Failures: final deterministic citation checks pass. The first repair fixed the
  mapping; the second was unchanged.
- Critics/adjudicator: all three critics accept; adjudicator still requests
  repair, creating a hard majority override under 12A.4 rules.
- Outcome: one of two citation-valid reviewed candidates, but rejected because
  generic adjudication did not share an atomic claim decision model.

### 9. `question_7d0b733d6244038243a9` — prerequisites

- Evidence: C1 covers inference networks/classifiers; C2 covers G/D training and
  Markov-chain terms.
- Claims/map: K1 (no explicit prerequisite list) → uncited; K2/K3 reading lists
  → C2/C1; K4 inference qualification → C1+C2.
- Failures: K1 is uncited; evidence terminology does not by itself validate all
  pedagogical recommendations; the response is only partially answerable from
  paper-only evidence.
- Critics/adjudicator: evidence `repair/partial`; answer `repair/incomplete`;
  citation `repair/partial`; adjudicator `repair/incomplete`.
- Repairs: two unchanged attempts.
- Disagreement cause: an inferred tutoring recommendation needs an explicit
  inference policy and atomic premises, not a generic paper citation.

### 10. `question_653feda324080241998e` — summary

- Evidence: C1 describes generator/discriminator competition; C2 describes
  limitations and synchronization.
- Claims/map: K1 contribution/mechanism → C1; K2 limitations → C2.
- Failures: none under the final deterministic checks; both broad claims were
  accepted by all critics and the adjudicator.
- Repairs: none. Outcome `uncertain` only because 12A.4 confidence/terminal policy
  did not reach autonomous acceptance.
- Result: the second of two citation-valid reviewed candidates.

### 11. `question_98d3a73820e377ed9908` — paper objective

- Evidence: C1 describes discriminator classification of model versus data
  samples.
- Claims/map: K1 (competitive generator/discriminator objective) → citation
  placed after the next sentence, so K1 is uncited; K2 (competition improves
  models to indistinguishability) → C1.
- Failures: sentence-level citation attachment leaves K1 uncited; C1 only
  partially supports K2's convergence-like wording.
- Critics/adjudicator: evidence `accept/direct`; answer `repair/incomplete`;
  citation `repair/incomplete`; adjudicator `repair/incomplete`.
- Repairs: attempt 1 fixed all citation gates; attempt 2 introduced a new failure
  by rewriting the answer and moving the citation again.
- Disagreement cause: repair operated on unrestricted prose and was not monotonic.

## Exact root cause of 20% citation support

The low rate was not caused by malformed reviewer output, stale source hashes, or
leakage. It arose from five interacting design defects:

1. Answer prose and citation labels were produced before the final evidence set
   stabilized. Five candidates retained at least one label absent from final
   evidence.
2. Sentence-level validation treated broad multi-clause sentences as one claim.
3. Evidence-scope and abstention explanations were emitted as substantive uncited
   claims.
4. Numeric checks found values/equation identifiers unsupported by the cited
   final passage or attached to the wrong context.
5. Generic repair rewrote the full answer; it could preserve a failure or undo a
   successful repair, as candidate 11 demonstrates.

## 1.2.5 corrective design

Version 1.2.5 makes `SupportedClaim` and `AnswerPlan` the authoritative answer
representation. Codex proposes atomic structured facts, deterministic validators
check evidence, relevance, numbers, entities, and inference labels, and the
composer renders only approved claim IDs. `corrected_answer` remains audit data
and is never copied into the training answer. Evidence replacement rebuilds the
claim graph. Repair outcomes are tracked by evidence, claim, citation,
completeness, and recomposition type.

## Verification commands

The final command results are recorded after implementation in the project README
and the final task report. The required set is:

```bash
python3 -m pytest -q tests/test_claim_alignment.py
python3 -m pytest -q
python3 -m ruff check .
python3 -m ruff format --check .
node --check src/localml_scholar/review_app/static/app.js
git diff --check
```

The frozen 12A.4 artifact was read only. A new 50-candidate diagnostic receives a
fresh run ID and uses the same frozen candidate pool and corpus hashes.

Local verification initially completed with 871 passing tests; additional
sampling, stdin-isolation, and readiness regressions were added after the
approved real run exposed those defects. The first attempt was invalidated after
12 records because it covered only two papers. The corrected sample covers all
12 eligible papers but is suspended at 1/50 because the configured Codex service
reached its usage limit. No completed 1.2.5 diagnostic metric exists, the second
run is ineligible, full-run readiness is false, and Milestone 12B remains blocked.
See `docs/runs/milestone_12a_5_controlled_diagnostic_report.md`.

Final local verification collected 873 tests. The 79 focused changed-path tests
and 869-test sandbox-safe suite passed. Ruff lint/format, JavaScript syntax, and
`git diff --check` passed. A final socket-inclusive run could not obtain local
port authorization after the account usage limit was reached; the most recent
authorized socket-inclusive run passed all 871 tests that existed before the two
new gating regressions were added.
