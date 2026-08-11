# Milestone 12A.5 controlled diagnostic report

Date: 2026-08-10
Package: 1.2.5
Corpus: 14 locally indexed papers; 12 eligible and 2 held-out under the
deterministic paper split.

## Invalidated first attempt

Runs `diagnostic_d2ced0dcd64b4a768c3c4ab705179a23` and
`diagnostic_ad65c465aa964200b4ae771fb6a2ddfc` used the defective selection. The
first suspended after one record when a final-adjudicator call exceeded 300
seconds; the second was interrupted after 12 terminal records once sampling was
inspected. Each requested 50-item sample contained 25 questions from each of
only two papers. The selector balanced question-type subgroups but iterated
paper-major keys, so the count limit was reached before other papers entered the
sample.

Both runs are persisted as `invalid_diagnostic_sampling` with
`valid_for_readiness=false`. Their records and partial metrics must never be
used for readiness or capability claims.

The selector was corrected to round-robin first across papers and then across
question-type/failure subgroups within each paper. An authored regression proves
that a 50-item fixture across 14 papers covers all papers with a maximum count
difference of one.

## Corrected run

Run `diagnostic_1010993f09c949a480e5aa276a2ff2d1` has an exact 50-candidate
sample across all 12 eligible papers: two papers have five candidates and ten
papers have four. The other two papers remain held out.

The first terminal candidate was saved. Candidate two then suspended after the
Codex CLI returned nonzero. Its stderr contained a fatal model-cache decoding
error, `missing field base_instructions`, followed by plugin warnings. The
provider also printed `Reading additional input from stdin`; it now sets
`stdin=subprocess.DEVNULL`, and its regression test verifies that isolation.

On 2026-08-10 the installed CLI was `codex-cli 0.147.0-alpha.1.2`. A fresh
`models_cache.json` identified client version `0.147.0` and contained
`base_instructions` in every inspected model entry. A minimal raw
schema-constrained `codex exec` call then succeeded. The exact LocalML Scholar
provider and full output schema also succeeded independently for
`evidence_critic`, `answer_critic`, and `citation_critic`. This rules out the
current invocation flags, reviewer schema, authentication, and LocalML Scholar
adapter as the active cause. The smallest justified remediation was the normal
Codex cache refresh; the cache was not deleted or manually rewritten.

The successful raw call preserved the non-fatal warnings for inspection:

- plugin icon paths containing `..` were ignored because they did not resolve
  under plugin assets;
- the template-creator manifest exposed more than three default prompts, so
  excess prompts were ignored;
- rollout lookup reported a state-database discrepancy and used its documented
  fallback path.

The call returned valid schema-constrained JSON despite those warnings, so they
are not treated as fatal reviewer-availability failures.

The same run was resumed without resampling. Before resume it had cursor
`paper_index=0, question_index=1`, one persisted record, unchanged source and
analysis hashes, and the original paper-level split assignment. Source
verification passed and the artifact advanced to three persisted records while
continuing candidate four. No replacement diagnostic was created.

State verified immediately after recovery:

- resume process: active in `question_curation`
- persisted status: `suspended` until the run reaches its next terminal state;
  the retained availability error is historical
- minimum persisted candidates after recovery: 3/50
- measured readiness metrics: unavailable; the sample is incomplete
- second 100–150 run: ineligible
- full autonomous curation: blocked
- Milestone 12B: blocked

## Ablation-query retrieval finding

The completed record for “What does the ablation establish?” is a genuine
insufficient-evidence terminal, but its lexical ranking exposes a systemic query
preprocessing mismatch. `RetrievalIndex.search` sends all normalized query terms
to BM25, including `what`, `does`, and `the`. The later evidence-sufficiency
policy separately removes those stop terms and correctly evaluates only
`ablation` and `establish`. In this record BM25 therefore ranked chunks using
function-word matches while none matched the two meaningful terms.

This behavior is systemic because it follows the shared lexical query path; it
is not unique to this paper or record. No retrieval threshold, tokenizer, stop
list, index, or stored result was changed during this reviewer-availability
repair. Query preprocessing should be evaluated as a separate controlled
retrieval change after the 50-candidate diagnostic is stable.

The same run can be resumed without resampling:

```bash
python3 -m localml_scholar.training_data.cli --repository . \
  resume-curation \
  --run diagnostic_1010993f09c949a480e5aa276a2ff2d1
```

Readiness logic now ignores diagnostics marked invalid and diagnostics that have
not reached their declared count. It also requires a passing 50-candidate run
before creating a differently seeded 100–150 candidate run, and requires both
finished runs before reporting full-run readiness.
