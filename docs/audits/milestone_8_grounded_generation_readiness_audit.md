# Milestone 8 grounded-generation readiness audit

Audit date: 2026-07-23

## Scope

The complete repository was inspected before adding grounded answering, with
particular attention to:

- immutable retrieval documents, chunks, `SearchResult`, `Citation`, ranking,
  filters, index hashes, persistence, and CLIs;
- exact source character/line/page preservation through ingestion and
  chunking;
- character, byte, and BPE tokenizer contracts, state hashes, encode/decode
  policies, and checkpoint ownership;
- transformer inference mode, cache lifecycle, fixed context length,
  generation cropping behavior, checkpoint loading, and parameter identity;
- atomic JSON/NPZ helpers, configuration validation, output ignore rules, and
  existing experiment conventions.

Milestones 1–8 source files, tests, configurations, audits, derivations,
experiments, README, architecture, roadmap, `.gitignore`, and Git tracked-file
state were included.

## Verified foundations

1. `RetrievalIndex.search` is deterministic for a fixed immutable snapshot.
   Ties use stable identifiers, and both BM25 and TF-IDF retain full result
   metadata and exact source text.
2. Every chunk equals its document source slice. Citations retain chunk,
   document, heading, line, and known page identities.
3. Index JSON is atomic and transactional. Loading rebuilds and verifies
   statistics and hashes rather than trusting serialized derived arrays.
4. Byte and BPE tokenizers encode arbitrary Unicode through UTF-8 and have
   stable state hashes. Character tokenizers intentionally fail on unknown
   prompt characters.
5. Model-only checkpoints can contain complete tokenizer state and validate
   vocabulary compatibility.
6. `inference_mode()` restores nested model modes and rejects pending caches.
   Transformer generation leaves no backward cache.
7. The base generator crops ordinary prompts to the most recent model context.
   This is acceptable for free generation but not for grounded controls.
8. Generated outputs, checkpoints, indexes, caches, local corpora, and
   platform files are ignored. Tiny authored fixtures remain intentionally
   tracked.

## Verified defects and fixes

### Common-word evidence leakage

Initial answer-oriented selection accepted positive BM25 results whose only
matches were generic question words. This could pass irrelevant evidence into
the sufficiency gate. Selection now requires overlap with a documented
meaningful-query-term list; an authored quantum-hardware question retrieves no
eligible evidence and deterministically abstains.

### Verbatim negation false positive

The first support validator compared a claim's negation flag with the entire
cited passage. A verbatim source sentence could therefore be rejected because
another sentence in that passage used “not” or “never.” Exact source
substrings now bypass this passage-level warning while number, identifier,
symbol, structural citation, and source-range checks remain active.

### Weak extractive filler

Selecting one sentence from every evidence item caused weak passages to appear
only to fill the sentence budget. Sentence selection now requires new query
term coverage, a configurable score relative to the best sentence, and stops
at configured query coverage. This preserves multi-source answers while
reducing irrelevant cited prose. The authored fixture is reported honestly;
the heuristic still misses some key facts.

### Claim segmentation lifecycle

Wrapped extractive bullets initially merged following citation-only or
abstention lines. The deterministic segmenter now terminates bullets at those
boundaries and excludes Markdown headings. Multi-line exact source sentences
remain one claim.

### Effective configuration metadata

A per-call `top_k` override was used for retrieval but metadata recorded the
pipeline default. Metadata now records the effective `EvidenceSelectionConfig`
used by that call.

No retrieval mathematics, transformer mathematics, tokenizer IDs, or existing
checkpoint formats required modification.

## Evidence-context design

The evidence selector is a pure layer over search results. It preserves rank,
limits per-document chunks, suppresses heavily overlapping source ranges, and
creates exact `EvidenceItem` slices with deterministic `C#` labels. Evidence
characters are bounded before generation. Token-aware context construction
then removes lowest-ranked evidence before deterministically truncating one
remaining source prefix.

The complete prompt plus generation allowance must fit the model maximum.
Grounded generation never uses the base generator's context-cropping behavior,
so controls and evidence cannot silently fall off the left side.

Controls appear before and after quoted evidence. Evidence blocks explicitly
state that document text is source material rather than application
instructions. The prompt-injection fixture includes a fake `C99`; strict
answer-local citation validation rejects it if generated.

## Citation syntax

Only `[C1]`, `[C2]`, and comma-separated groups such as `[C1, C3]` are valid.
Labels are contiguous in selected-evidence order and local to one answer.
Every binding resolves to one exact `EvidenceItem` and structured `Citation`.
Unknown or malformed labels are rejected. A citation must occur after claim
content to attach to that claim.

## Abstention and acceptance

The sufficiency gate requires positive content evidence, minimum count/score,
meaningful term coverage, and matched-term count. Failure returns fixed
abstention without calling the transformer.

Generated answers default to complete citation coverage, known labels, valid
source linkage, lexical support threshold, matching numbers, and no simple
negation warning. The support score is diagnostic and is explicitly not
entailment. Invalid plain generative output remains visible and rejected.
Fallback mode preserves it, records reasons, and returns separately validated
extractive text.

## Extractive baseline

Source segmentation preserves exact character spans and fenced code. Candidate
ranking combines query coverage, compactness, and retrieval score.
Substantive lines are copied without paraphrase and receive one answer-local
citation each. This is the trusted baseline because the output source wording
can be mechanically verified.

## Generative limitations

- No useful or instruction-tuned checkpoint is assumed or bundled.
- Small existing checkpoints may be too short for the control prompt.
- Greedy output can be malformed, uncited, unsupported, or truncated.
- Byte/BPE output can require explicit replacement display.
- Citation and lexical validation do not establish semantic truth.
- There is no semantic retrieval or external fact checker.

## Explicitly deferred

Semantic embeddings, vector databases, hybrid retrieval, neural reranking,
external models/APIs, instruction tuning, web search, OCR, raw PDF parsing,
KV caching, optimized kernels, and large-scale training were not implemented.

## Verification commands

The final verification run from the repository root uses:

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
python3 -m pytest -q
python3 experiments/evaluate_extractive_answering.py
python3 experiments/evaluate_grounded_generation.py
python3 experiments/compare_answer_methods.py
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
python3 experiments/inspect_multi_head_attention.py
python3 experiments/inspect_bpe_tokenizer.py
python3 experiments/inspect_document_ingestion.py
python3 experiments/compare_retrievers.py
```

The generative and four-method commands without `--checkpoint` must exit with
an explicit message and no fabricated metrics. Final measured counts and
fixture results are recorded in the README after verification.

## Final results

- Ruff lint and format checks passed, and `git diff --check` was clean.
- The complete suite reported `532 passed in 3.20s`; the 51 dedicated
  answering tests reported `51 passed in 0.55s`.
- The extractive fixture made all 10 answerability decisions correctly.
  Citation validity, coverage, and recall were `1.0`; citation precision was
  `0.9`; key-fact recall was `0.85`; both abstention precision and recall were
  `1.0`.
- The unsupported quantum question and lexical synonym-mismatch question
  abstained without generation.
- A saved CLI answer reloaded and revalidated exactly against index
  `f130f4ee2babbdd6475e5010d31bf191ab79d62566b18ab00b7f9ca91ae2213e`.
- A deterministic random-checkpoint integration smoke attempted generation on
  eight answerable questions. All eight raw outputs failed validation,
  generative rejection was `1.0`, and explicit fallback preserved the raw
  output while returning validated extraction. The two insufficient questions
  never called the model.
- Bigram, XOR, single-head attention, decoder block, multi-head attention,
  controlled head comparison, BPE inspection, tokenizer comparison, document
  ingestion, retriever comparison, and tiny transformer overfit experiments
  all completed after the Milestone 9 changes.
- Generated indexes, answers, summaries, checkpoints, and caches appeared only
  under ignored paths. No corpus, checkpoint, cache, platform file, or
  environment-specific output was added to tracked state.
