# Milestone 6 tokenizer and corpus readiness audit

Audit date: 2026-07-23

This audit examined the 0.6.0 repository before Milestone 7 and records the
verified changes made for 0.7.0. It is an engineering and compatibility audit,
not a language-model quality evaluation.

## Files and behavior inspected

The audit covered:

- `src/localml_scholar/tokenizer.py`
- `src/localml_scholar/data.py`
- `src/localml_scholar/generation.py`
- `src/localml_scholar/serialization.py`
- `src/localml_scholar/models/transformer_lm.py`
- `src/localml_scholar/training/config.py`
- `src/localml_scholar/training/transformer.py`
- all model, loss, optimizer, and sequence-batching integration tests
- transformer training, overfit, generation, and comparison experiments
- current and legacy checkpoint tests
- README, architecture, roadmap, data policy, package configuration, and
  ignore rules

The 0.6.0 character tokenizer assigned sorted Unicode code points to contiguous
IDs. Whitespace and newlines were exact, fitting was deterministic, and unknown
characters raised. Raw text was chronologically split before vocabulary fitting
and each split was encoded independently. The sequence sampler consumed
arbitrary integer streams and did not require one-code-point token semantics.

## Leakage and boundary analysis

The original data path already fit the character vocabulary on the training
split only. Validation-only characters therefore failed explicitly. Train and
validation streams were distinct, so both the raw split-boundary pair and any
shifted sequence crossing that boundary were excluded.

Milestone 7 preserves that behavior and generalizes it:

- character vocabulary: fitted on raw training text only;
- byte tokenizer: fixed vocabulary, no fitting;
- BPE merge table: fitted on raw training text only;
- train and validation: encoded independently after the split.

Tests use validation-only code points and validation-only pair patterns to
verify isolation.

## Checkpoint compatibility analysis

Before this work, full training checkpoints stored a legacy character
vocabulary and token-stream SHA-256 hashes. Model-only checkpoints stored model
configuration and parameters but not a general tokenizer. Exact resumption
restored sampler and optimizer state, but tokenizer implementation identity was
implicitly character-only and raw corpus identity was not represented.

Version 0.7.0 now stores:

- complete versioned tokenizer state;
- tokenizer type, normalization, vocabulary size, and canonical state hash;
- raw-corpus UTF-8 hash and split metadata when the corpus path supplies them;
- isolated train and validation token-stream counts and hashes.

Recognized 0.5.1 and 0.6.0 training checkpoints migrate legacy character state
in memory. Recognized 0.5.0 and 0.6.0 model checkpoints remain loadable. The
source checkpoint is not modified, IDs are not remapped, and unknown identities
remain errors. Tests synthesize realistic 0.6.0 schemas and verify preserved
model logits and training state.

## Defects and limitations found

1. There was no tokenizer-wide interface; transformer APIs assumed
   `CharacterTokenizer`.
2. Tokenizer JSON did not use one schema across implementations and tokenizer
   JSON writes did not have the repository's NPZ-style atomic replacement.
3. There was no fixed byte tokenizer or arbitrary-Unicode tokenizer that could
   represent validation-only code points.
4. There was no independently trained byte-level BPE implementation.
5. Full checkpoints did not store a tokenizer state hash or raw corpus hash.
6. Model-only checkpoints could not carry tokenizer identity for standalone
   text generation.
7. Generation had no explicit policy for token IDs forming invalid UTF-8.
8. Training CLI resumption rebuilt data before restoring a general tokenizer,
   which would risk refitting learned BPE state.
9. Corpus metadata did not report raw character/byte counts, normalization,
   split policy, or tokenizer identity.
10. Token perplexity reporting did not address incomparable token units.

No defect was found in transformer forward/backward mathematics, attention,
optimizers, or shifted batching. Those implementations were not refactored.

## Fixes made

- Added a minimal `Tokenizer` ABC and unified version-2 JSON schema.
- Preserved exact character IDs and added in-memory legacy migration.
- Added raw-byte and deterministic byte-level BPE tokenizers.
- Added explicit no-normalization and strict/replacement UTF-8 policies.
- Added deterministic BPE tie-breaking, document-local pair counts,
  non-overlapping replacement, merge validation, and recursive byte expansion.
- Added atomic tokenizer JSON writes and canonical state hashing.
- Added `CorpusMetadata` and enforced split-before-fit for all policies.
- Generalized transformer training and generation without changing model math.
- Made resume restore the checkpoint tokenizer before corpus encoding; it never
  refits BPE.
- Added full tokenizer identity to training checkpoints and optional identity to
  model-only bundles.
- Added a byte-normalized comparison metric and an explicit token-perplexity
  comparability warning.
- Added targeted unit, malformed-state, Unicode, leakage, migration,
  checkpoint, generation, optimizer, and exact-resumption tests.

## Atomicity and error behavior

Tokenizer `load_state_dict` constructs a complete validated candidate before
assignment. Checkpoint loaders reconstruct fresh tokenizers, models, samplers,
and optimizers before returning; a mismatch cannot partially change a caller's
object. Current full checkpoints require the tokenizer hash. Invalid IDs,
wrong dtypes/ranks/shapes, unknown types/versions, malformed UTF-8, vocabulary
mismatches, content mismatches, and conflicting CLI selections produce
specific exceptions.

## Known limitations

- Raw files must be strict UTF-8 and non-empty for training.
- Character validation fails when a held-out code point is unseen.
- No normalization beyond the explicit `none` policy exists.
- The BPE implementation is a transparent reference algorithm, not a
  large-corpus-optimized trainer or encoder.
- Directory-corpus loading was intentionally deferred; BPE itself accepts
  multiple in-memory documents and preserves their boundaries.
- There are no BOS, EOS, PAD, UNK, or MASK tokens.
- Generated byte/BPE IDs can form invalid UTF-8; strict decoding raises and
  display generation opts into replacement.
- Checkpoints store corpus identity, not corpus contents.
- Tiny smoke runs do not establish tokenizer or language-model superiority.

## Verification

The final repository-wide verification used Python 3.13.5, NumPy 2.4.3,
pytest 9.1.1, and Ruff 0.15.18:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
python3 experiments/inspect_multi_head_attention.py
python3 experiments/compare_single_and_multi_head.py --steps 20
python3 experiments/overfit_tiny_transformer.py \
  --heads 1 --steps 40 \
  --output outputs/tiny_transformer_overfit_one_head
python3 experiments/overfit_tiny_transformer.py \
  --heads 2 --steps 40 \
  --output outputs/tiny_transformer_overfit_two_heads
python3 experiments/inspect_bpe_tokenizer.py
python3 experiments/compare_tokenizers.py --steps 12
```

Ruff lint and formatting checks passed, `git diff --check` produced no output,
and the complete suite reported `394 passed in 5.67s`. All Milestone 1–6
regression experiments completed successfully.

The transformer CLI was additionally run as an interrupted/resumed pair for
each tokenizer. Character continued from step 10 to 20; byte and BPE continued
from step 3 to 6. The byte/BPE resume commands omitted `--tokenizer`, so success
verified that the checkpoint restored tokenizer state before split encoding.
The BPE fallback tokenizer learned 16 rules and encoded 2,095 training code
points into 1,316 tokens.

The controlled 12-step comparison produced:

| tokenizer | vocabulary | merges | train tokens | bytes/token | validation BPB |
| --- | ---: | ---: | ---: | ---: | ---: |
| character | 30 | 0 | 1,248 | 1.0897 | 4.2317 |
| byte | 256 | 0 | 1,360 | 1.0000 | 6.2997 |
| BPE | 272 | 16 | 880 | 1.5455 | 3.8432 |

BPB used 48 fixed-seed sampled targets per tokenizer; represented byte counts
were 49, 48, and 81. These figures are integration observations, not a
tokenizer ranking. Every run round-tripped text exactly, resumed at step 6,
reached step 12, and reproduced logits exactly after checkpoint reload.

Artifact inspection parsed 15 new JSON files and 24 new NPZ files, loaded every
tokenizer JSON, loaded each current model/tokenizer bundle, and confirmed all
floating checkpoint arrays were finite. Git tracked only `.gitkeep` files
under raw data, processed data, and outputs. Real stored 0.5.0/0.6.0 model
checkpoints and 0.5.1/0.6.0 training checkpoints loaded successfully under
0.7.0 with their original character IDs and one-/two-head configurations.
