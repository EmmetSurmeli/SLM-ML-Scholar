# Fast deterministic curation preflight

Package 1.2.6 moves cheap, local correctness checks in front of the configured
Codex reviewer. Codex review is now an expensive final quality layer, not the
mechanism used to discover basic ingestion, question-generation, retrieval, or
deterministic grounding defects.

## Execution order

```text
local source and index hashes
  -> inferred scholarly sections
  -> per-paper ingestion health
  -> topic-aware candidate eligibility
  -> canonical query concepts
  -> local retrieval, evidence sufficiency, and direct-answer preflight
  -> deterministic claim graph and repair
  -> Codex critics only for surviving candidates
```

Severe extraction defects block autonomous question generation. Ordinary
candidate construction, retrieval, or validation failures become persisted
terminal records and do not discard earlier work. Three repeated identical
errors, a stage failure rate above 30%, source corruption, index corruption,
leakage, or reviewer-wide unavailability remains a systemic stop.

Controlled pilots and diagnostics also precompute local answers before
selection. An answerable candidate must retrieve its primary technical terms,
bind a substantive passage from a compatible section to the resulting claim,
and satisfy type-specific direct-answer checks. Numerical reproduction prompts
must retrieve a value, while complexity prompts must retrieve an actual
complexity characterization. Generic result and dataset-extraction prompts are
currently excluded from the stable autonomous pilot pool; they remain useful
evaluation categories.

Pilot sampling rotates across stable question types and caps each paper before
repeating. A filesystem lease prevents two processes from resuming the same
cursor concurrently. Stale leases are reclaimed only after their owner process
is gone, and readiness-invalidated runs cannot be resumed.

## Section and health policy

`infer_scholarly_headings` recognizes numbered and nested headings, common
scholarly names, and conservative uppercase headings. Repeated headers are
suppressed. Once a References heading is reached, reference entries are not
treated as sections; an explicit Appendix can reopen heading inference.

`PaperIngestionHealth` reports section counts, titled/untitled and duplicate
fractions, extracted length, empty chunks, confidence, warnings, and the final
eligibility decision. The default block conditions include more than 25%
untitled structure with no recoverable headings, more than 50% duplicate
headings, no major scholarly section, or fewer than 20 non-whitespace source
characters.

The local index can be rebuilt with recovered boundaries:

```bash
python3 -m localml_scholar.training_data.cli ingestion-health --repair
```

This changes the ignored local index artifact, not the uploaded source files or
historical diagnostic records.

## Answerability and abstentions

Candidates carry `expected_answerability`: `answerable`, `partial`, `abstain`,
or `external_required`. Answerable retrieval metrics exclude deliberate
abstention tests. A local no-evidence result uses `GroundedAbstention`, which
records the reason and evidence attempt without turning generic refusal prose
into an uncited factual claim.

## Repair and reviewer use

The claim graph is built and narrowed locally before reviewer availability is
required. Unsupported claims are removed or narrowed, citations are remapped,
and the answer is recomposed from the repaired graph. Evidence-free candidates
terminate locally instead of consuming two repair cycles. Every record exposes
Codex call count, pass roles, pre-Codex rejection status, and deterministic
repair use.

## Cache and diagnostics

Paper health, inferred headings, and topic signals are cached under an exact
hash of source identity, parser identity, and section state. A changed source
or rebuilt index produces a new key; stale values are never reused.

The bounded real-paper pilot performs this preselection without Codex and
records how many candidates were removed. Only the final fixed set may reach
the reviewer. Deliberate abstention items are retained only when the same local
pipeline abstains as expected.

Useful commands:

```bash
python3 -m localml_scholar.training_data.cli pipeline-self-test
python3 -m localml_scholar.training_data.cli ingestion-health
python3 -m localml_scholar.training_data.cli question-eligibility-report
python3 -m localml_scholar.training_data.cli codex-usage-report
python3 -m localml_scholar.training_data.cli pilot-curation --count 10 --seed 42
```

The self-test uses authored local fixtures and makes zero Codex calls. A
reviewer-backed pilot is permitted only after that self-test and the ingestion
health gate pass. A fresh 50-question diagnostic is downstream of a successful
pilot; the historical broken run is not resumed.

## Verified 1.2.6 local gate

The completed local verification reported 14 healthy papers, no unhealthy
papers, an average titled-section fraction of 1.0, 367 template suppressions,
and all eight self-test checks passing with zero Codex calls. The bounded
pilot preselection removed 163 further retrieval/direct-answer failures and
fixed a set of 8 answerable plus 2 abstention questions across 10 papers. The
complete suite passed 918 tests; Ruff lint/format and JavaScript syntax checks
also passed.

After explicit approval, pilot `curation_fe778b0d917f4168b6fecb43bff4f5a4`
processed two items. The deliberate abstention terminated locally. The first
answerable item asked for batch size; its evidence explicitly stated a
minibatch size of 128, but exact-token completeness checks classified the
answer as partial. The unchanged repair then ran a third five-role cycle. The
pilot was stopped after 15 calls and frozen at its exact cursor.

The query/sufficiency and claim-planning paths now treat `batch` and
`minibatch` as a narrowly documented equivalence. A no-progress guard stops an
unchanged repeated correction after the second cycle, limiting this failure
mode to 10 calls. The failed pilot remains invalid for readiness.

The first replacement, `curation_78c29180704946dc9cc773a0568a8734`, accepted
the repaired batch-size item in one five-role cycle. Its next answerable item,
causal masking, was still classified partial because the answer planner
required an unrelated generic architecture marker. The run was stopped after
3/10 records and 20 persisted calls, then frozen. Question-type markers now
serve only relevance classification; direct supported claims determine
answerability together with explicit required concepts.

The second replacement, `curation_2ffb967acc1d455ca4732c7340763ac8`, verified
that the partial prefix was gone but found two further validators on the same
causal-mask item. `Causal`, `Masking`, `Combined`, and `Softmax` at sentence
starts were falsely treated as unsupported proper names, and the word `causal`
remained unmatched despite evidence stating subsequent-position blocking and
the autoregressive property. The run is frozen after 3/10 records and 20
persisted calls.

Named-entity extraction now keeps known entities, acronyms, identifiers,
multi-token names, and conservative named-subject predicates without treating
ordinary sentence-initial technical words as entities. Causal-mask sufficiency
uses a narrow compound-specific equivalence for subsequent/future and
autoregressive language. The exact stored answerer targets now produce direct,
explicit supported claims offline.

Third replacement `curation_790d8f32e5a7499baac7da5b5f03009f` completed all
10 items after explicit approval. It produced 2 `codex_curated`, 2 local
insufficient-evidence, 5 rejected, and 1 uncertain records with 80 reviewer
calls. The accepted batch-size and causal-masking answers each passed in one
five-role cycle with no repair. Citation structural validity, support, and
relevance were all 1.0; hard reviewer disagreement was 0.25, so the pilot fails
the 0.15 readiness gate.

Final dataset export initially supplied manual splits for every eligible corpus
paper even though only two papers had accepted examples. The split validator
correctly rejected those unknown assignments. Export now filters manual splits
to papers actually present in the selected examples, with an accepted-subset
multi-paper regression. The same run finalized without additional reviewer
calls and exported exactly two `codex_curated` examples. No fresh 50-question
diagnostic was created.
