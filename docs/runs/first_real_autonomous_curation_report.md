# First real autonomous curation run

## Decision

**C - Fix curation defects first.** The run is not ready for Milestone 12B smoke
training and should not be resumed unchanged. It safely stopped after 10 terminal
candidates because reviewer disagreement was 70%, above the configured 10%
limit. No example passed every gate, so no training or validation dataset was
exported.

This report describes measured results from the frozen run. Fixes made after the
safety stop are listed separately and have not been credited as improvements to
the measured run.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `curation_5d2d2687e0e74c61918e44894297303c` |
| Package version | `1.2.3` |
| Run created | `2026-08-09T03:36:26.823+00:00` |
| Safety stop | `2026-08-09T04:21:03.539+00:00` |
| Report written | `2026-08-09T04:25:46Z` |
| Final status | `suspended` |
| Final stage | `safety_stop` |
| Cursor | paper 0, question 10 |
| Reviewer | `openai_codex_cli`, pipeline version `1.2.3` |

Run creation spent approximately 23 minutes performing deterministic scholarly
analysis over the expanded corpus before reviewer execution began. Review and
repair of the first 10 candidates took approximately 45 minutes.

## Corpus inventory and frozen paper split

The live corpus contained 14 papers, not the six anticipated by the original
task title. The deterministic seed-42 split assigned 10 train papers, two
validation papers, and two test papers. The run stores canonical document JSON
hashes for drift detection; raw PDF SHA-256 hashes are reported separately.

| Split | Paper | Canonical document SHA-256 | Raw file SHA-256 |
| --- | --- | --- | --- |
| train | Generative Adversarial Nets (`1406.2661v1.pdf`) | `ffdebecca089878cc459c2d2868d77a2d423f77e67271b975d4cc12019bc72e6` | `ff5819e3a7b713c3bd3107b7de3d51fe0a347aa5d8444f0efdcf2345ef0a8b63` |
| train | Adam (`1412.6980v9.pdf`) | `d8fc5d31518ea2ef0c419739f273b49888f814fd588b4db5d7b1ea91f3d5b6db` | `eab9c73ae2ceda884b94830bda99312254bac4806f6c9f045cbab90721ecda31` |
| train | Attention Is All You Need | `c534450d30f71544708d2ac47b4b8923c675f82630c93d3226df71374d5a31ec` | `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` |
| train | AlexNet/ImageNet classification | `7f9fb8e44ab274958e17bc22742f4f044492d565e53f38585d1be1779fe17872` | `90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771` |
| train | Vision Transformer (`2010.11929v2.pdf`) | `45502d4e116c05132652190b9605d1fd8307682565198fcb09f0b55993bce334` | `8ce7b83971a14508ca711a27c875c9b6914c4f6767cf3150fb1ca6c07aa056d6` |
| train | GPT-3/Few-Shot Learners (`2005.14165v4.pdf`) | `d023c0623e188f3b14fd9a9d04335cedc54fef1f823c02d8b90507a15498f78a` | `97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83` |
| train | Deep Residual Learning (`1512.03385v1.pdf`) | `34747a3bf72c1278f0d523c968b078cf2912c9b2738ba024eb93caf0197886d5` | `1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a` |
| train | Retrieval-Augmented Generation (`2005.11401v4.pdf`) | `4624feec12f22af904687e97094e2651a69617a0b0f004e84be360f64ae03a78` | `23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233` |
| train | FlashAttention (`2205.14135v2.pdf`) | `776a982525e8fb4ecdfed03b410530b3c37a69fd8fda882b69127c672c2af474` | `ca7f9fda10b90fc05dd291a3accc85e9c1a4a860b99b31928dab03ed3fcb14e4` |
| train | BERT (`1810.04805v2.pdf`) | `01d262d98df2c8f340ba4418c33f9ee47ae8101ee478caf82cf54781b7532074` | `5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e` |
| validation | InstructGPT (`2203.02155v1.pdf`) | `f691d464a81036f8f3fb07a8ed1bb1cf6377d2ff99c737ccb30b58a4627a38d8` | `c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98` |
| validation | Chinchilla (`2203.15556v1.pdf`) | `91438c7a442c52844ccfd07d77e4c88e8ca02fdeea465db98f5363c6c9c9d130` | `3fd3632a8ef48171bd25282990221d49535d75356192f068b3b2ebe08f2aedd4` |
| test | LoRA (`2106.09685v2.pdf`) | `21c715b5ec1595bf66167fe1bd84f2d454e99e0e00a80e935c3eaa31823de6b4` | `e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a` |
| test | Denoising Diffusion Probabilistic Models (`2006.11239v2.pdf`) | `9a0fd80edb92c1a86cecdbe8a0ba9856996bec1994c6eeb52aaed287a8bf10f4` | `aee5e07a802e8dfd2a386374c94fd61d1d056cb7e1e0fec4f28e8120ff5d8505` |

Every indexed source path existed at preflight, all 14 raw hashes were readable,
the 2,530-chunk index loaded successfully, and every source file had a matching
indexed document.

## Configuration

| Setting | Value |
| --- | --- |
| Questions per paper | 60 |
| Maximum accepted examples per paper | 40 |
| Acceptance threshold | 0.97 |
| Evidence threshold | 0.97 |
| Derivation/equation threshold | 0.99, enforced by code |
| Maximum repairs | 2 |
| Validation/test fractions | 0.15 / 0.15 |
| Seed | 42 |
| Multi-turn | enabled |
| Derivations | enabled |
| Abstentions | enabled |
| Cross-paper | disabled |
| Per-paper question-type cap | 12 |
| Maximum disagreement rate | 0.10 |

## Question generation and diversity

The frozen run generated 840 candidates, exactly 60 per paper.

| Question type | Count | Question type | Count |
| --- | ---: | --- | ---: |
| ablation | 42 | architecture | 26 |
| complexity | 29 | critical reasoning | 41 |
| derivation | 39 | equation | 42 |
| experiment | 44 | extension | 26 |
| extraction | 13 | false premise | 31 |
| hyperparameter | 6 | insufficient evidence | 28 |
| interpretation | 1 | intuition | 39 |
| limitation | 28 | metadata | 40 |
| method | 27 | motivation | 40 |
| multi-turn follow-up | 14 | prerequisites | 13 |
| provenance | 13 | reproduction | 42 |
| result | 27 | section comprehension | 124 |
| summary | 13 | teaching | 26 |
| user style | 26 |  |  |

Candidate-level aggregates were 81 equation/derivation prompts, 41 explicit
critical-reasoning prompts, 53 intuition or multi-turn natural prompts, 59
false-premise/insufficient-evidence prompts, 14 multi-turn prompts, and zero
cross-paper prompts. No dedicated `simplification` or `audience_adaptation`
candidate was generated, although `teaching` and `user_style` prompts were
present.

The frozen candidates contained 14 repeated within-paper prompt clusters and
110 duplicate excess records. The largest cluster had nine identical
`Untitled section` questions. This was a verified generator defect, not an
accepted-dataset deduplication result.

## Curation outcomes

| Metric | Measured value |
| --- | ---: |
| Candidates checkpointed | 10 |
| Candidates with Codex passes | 9 |
| Codex-curated | 0 |
| Rejected | 4 |
| Uncertain | 5 |
| Insufficient evidence | 1 |
| External source required | 0 |
| Duplicate terminal records | 0 |
| Split excluded | 0 |
| Entered repair | 7 |
| Accepted without repair | 0 |
| Accepted after one repair | 0 |
| Accepted after two repairs | 0 |
| Used two repair attempts | 5 |
| Explicit `repair_limit_exhausted` outcomes | 4 |
| Exact terminal question/answer duplicate clusters | 0 |

All 10 checkpointed candidates came from the first train paper, Generative
Adversarial Nets. The run stopped before it reached any other paper.

## Quality measurements

The repository's existing `autonomous_quality_report` measured:

| Metric | Value |
| --- | ---: |
| Structural evidence validation | 90.0% (9/10 had evidence hashes) |
| Citation validation | 60.0% |
| Reviewer agreement | 30.0% |
| Reviewer disagreement | 70.0% |
| Mean adjudicator confidence | 0.9889 across 9 reviewed records |
| Minimum adjudicator confidence | 0.98 |
| Acceptance rate | 0.0% |

Additional final-cycle diagnostics were:

| Diagnostic | Value |
| --- | ---: |
| Evidence critic accepted | 88.9% (8/9) |
| Evidence relevance score at least 0.97 | 33.3% (3/9) |
| Answer critic accepted | 66.7% (6/9) |
| Citation critic accepted | 77.8% (7/9) |
| Citation-support score at least 0.97 | 77.8% (7/9) |
| Citation-relevance score at least 0.97 | 66.7% (6/9) |
| Final adjudicator said `accept` | 55.6% (5/9) |
| Final pipeline accepted after all gates | 0.0% (0/10) |

There is no accepted-confidence mean or minimum because no record passed all
gates. High adjudicator confidence therefore did not imply dataset eligibility.

## Failure and repair analysis

The most common terminal categories were:

- deterministic validation failed: 9;
- citation validation failed: 4;
- reviewer disagreement explicitly recorded as a terminal reason: 4;
- repair limit exhausted: 4;
- insufficient local evidence: 1;
- derivation provenance incomplete: 1.

Seven of 10 records had reviewer disagreement according to the run-level flag.
Retrieval/evidence quality was also weak: only three final evidence-critic
scores reached 0.97. Repairs repeatedly requested stronger or narrower
evidence, removal of unsupported claims, qualification of incomplete lists,
and correction of claim-to-citation mappings.

The real run exposed four software defects before or during this sample:

1. the installed Codex CLI no longer supports `--ask-for-approval` on `exec`;
2. sentence extraction could cross a source-section boundary;
3. repair retrieval tried semantic/hybrid search on a lexical-only selected
   index;
4. repaired structured targets could retain stale evidence IDs.

The safety-stop sample then exposed two systemic data-path defects:

5. Codex corrected answers used raw evidence IDs (`[ev_…]`/`[chk_…]`) while
   deterministic validation requires display labels (`[C1]`);
6. repeated unnamed PDF sections generated 110 duplicate candidates.

All six defects received focused fixes and regression tests. The last two were
fixed after the frozen run stopped, so their effect is not included in the
metrics above. No threshold or correctness gate was weakened.

## Deduplication and balancing

No accepted record reached the exact/near-duplicate or per-paper balancing
stage. Consequently:

- accepted exact duplicates removed: 0;
- accepted near-duplicates removed: 0;
- per-paper caps applied: 0;
- accepted question-type balancing actions: 0;
- repeated terminal evidence/answer clusters: 0 exact clusters;
- candidate duplicates found before acceptance: 110 excess records.

Future run creation now removes repeated stable section-question IDs before
assembling each paper's 60 candidates.

## Dataset, provenance, and leakage

| Artifact | Count/status |
| --- | --- |
| Train examples | 0 |
| Validation examples | 0 |
| Held-out evaluation questions | 0; test papers were not reached |
| Autonomous corrections materialized | 0 |
| Dataset file | not created |
| Completed-run manifest | not created |
| Run state | preserved in ignored autonomous run storage |

Leakage checks found no accepted test records, no test answers, no test
corrections, and no cross-split accepted examples. The existing correction
store still contained one unrelated `proposed` record and no autonomous
records. Source hashes matched the frozen run throughout execution.

Generated PDFs, indexes, interactions, reviewer state, and curation artifacts
remain Git-ignored. The tracked working tree changed only because verified code
defects, tests, and this report were added.

## Safety-stop status

The final safety error was:

> Reviewer disagreement exceeded the configured safety threshold.

Two earlier resumable errors remain preserved in run history: lexical-only
semantic repair retrieval and an out-of-scope stale evidence ID. Both were
fixed before the successful 10-record sample proceeded. No malformed review
was silently approved and no fallback reviewer was used.

## Readiness assessment

### Milestone 12B smoke training

**Not ready.** The suggested minimum is at least 100 accepted train examples;
this run produced zero. Leakage and provenance controls were healthy, but the
retrieval/citation integration and disagreement rate were not.

### Meaningful first instruction-tuning comparison

**Not ready.** The preferred 300+ accepted examples were not approached, and
the run sampled only one paper before stopping.

### Recommendation

**C - Fix curation defects first.** Do not add more papers merely to increase
volume and do not start Milestone 12B. First create a fresh, small calibration
run after the citation-label and candidate-deduplication fixes, review at least
the first 10-20 terminal records, and require the disagreement and deterministic
validation failures to fall substantially before launching another 14-paper,
840-candidate run. The current 14-paper corpus is already broad enough for that
calibration; more papers are not the immediate need.

## Verification commands

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/test_autonomous_curation.py tests/test_scholarly_extraction.py
python3 -m ruff check .
python3 -m ruff format --check .
PYTHONPATH=src python3 -m pytest -q
git diff --check
```

Measured results:

- focused tests: 34 passed;
- Ruff lint: passed;
- Ruff formatting check: 222 files already formatted;
- complete test suite: 819 passed in 21.77 seconds when localhost socket
  binding was permitted;
- `git diff --check`: passed.
