# Milestone 5 multi-head-readiness audit

Date: 2026-07-23

This audit inspected the verified Milestone 5 Part 2 code before and alongside
the Milestone 6 multi-head implementation.

## Scope inspected

The audit traced:

- deterministic recursive module and parameter registration;
- Q/K/V and output `Linear` modules on three-dimensional tensors;
- LayerNorm, residual, feed-forward, sequential, and top-level backward paths;
- one-forward/one-backward cache ownership and recursive inference mode;
- causal-mask broadcasting and N-dimensional masked softmax;
- model, optimizer, sampler, tokenizer, and full training checkpoints;
- float32 training and float64 finite-difference policy;
- generation context cropping and no-cache inference;
- command-line architecture configuration;
- Git ignore coverage for outputs, checkpoints, corpora, and caches.

## Readiness findings

### Foundations that were already suitable

`Linear` preserves arbitrary leading dimensions, so a fused projection can
produce `(B, T, H*d)` without changing its backward implementation.
`masked_softmax` and its explicit backward operate along the final axis and
already support broadcast masks over four-dimensional score tensors.
`Module` traverses multiple nested projection modules in deterministic
registration order and enforces one unmatched training forward. Decoder
residuals require exact `(B, T, D)` shapes, which provides a useful guard after
head concatenation and output projection.

Optimizer state is keyed by deterministic parameter position and validates
shape and dtype. Since a one-head fused module retains the legacy projection
names, order, and shapes, recognized single-head optimizer checkpoints can be
migrated without permuting state.

### Design decision: fused projections

Milestone 6 uses fused Q/K/V projections:

- query/key output width `H * key_dim`;
- value output width `H * value_dim`;
- explicit reshape `(B,T,H,d)` and transpose `(B,H,T,d)`;
- one shared causal mask of shape `(1,1,T,T)`;
- independent final-axis softmax for each batch/head/query row;
- inverse transpose/reshape before a mandatory output projection.

`key_dim` and `value_dim` are per-head dimensions. The model dimension need
not be divisible by the head count because the total projection widths are
explicit rather than inferred as `D/H`. Holding the per-head dimensions
constant while increasing `H` intentionally increases width and parameter
count.

## Verified defects and fixes

### Decoder checkpoint restoration could partially mutate

The decoder/feed-forward checkpoint helper loaded parameter arrays one at a
time. A malformed later tensor could fail after earlier tensors had already
been assigned. Restoration now validates every key, shape, dtype, and finite
value before mutating any parameter.

### Some component checkpoint writers were not atomic

The legacy attention and transformer-component writers used `np.savez`
directly while top-level model and training checkpoints already used atomic
replacement. They now use the existing `atomic_savez` helper.

### Architecture metadata had no head count

The transformer configuration, decoder configuration, training CLI, and tiny
overfit CLI had no `number_of_heads` field. Version 0.6.0 adds and validates
that field, includes it in serialized configuration, and passes it through
every decoder block. The new defaulted dataclass field is placed after the
pre-existing optional fields so legacy positional construction retains its
meaning; a regression test fixes that public compatibility contract.

### Existing decoder inspection assumed no head axis

The inspection details and regression tests expected single-head tensors
shaped `(B,T,...)`. Canonical multi-head inspection now reports
`(B,H,T,...)`; the `H=1` case retains a visible singleton axis so shape
semantics never depend on head count.

## Compatibility policy

The canonical decoder now always includes an output projection. This preserves
the architecture used by every Milestone 5 language-model checkpoint and makes
residual width independent of `H*d_v`.

Recognized compatibility paths are explicit:

- model-only checkpoint schema 1 / model version 0.5.0 migrates to
  `number_of_heads=1`;
- full training checkpoint schema 1 / package version 0.5.1 migrates to
  `number_of_heads=1`;
- decoder-block version 0.4.0 checkpoints migrate when they used the canonical
  output projection.

The migration occurs in memory; old files are not rewritten. Unknown versions,
missing arrays, changed names, incompatible shapes/dtypes, and an expected
head-count mismatch are rejected. Legacy standalone decoder checkpoints that
disabled the output projection are intentionally not migrated because that
noncanonical architecture has no output-projection state.

## Tests added

Coverage includes:

- one-, two-, three-, and four-head shape cases;
- a hand-computed two-head example with different head distributions;
- an independent per-head NumPy reference calculation;
- exact `H=1` legacy outputs, score gradients, input gradients, and parameter
  gradients;
- exhaustive float64 finite differences for `H=1`, `H=2`, and a complete
  two-head decoder block;
- forward and backward causality for multiple head counts;
- exact masked score-gradient zeros;
- lifecycle, optimizer, clipping, mode, dtype, and checkpoint behavior;
- full-model loss, backward, generation, and head-count validation;
- bitwise-exact interrupted/resumed two-head training;
- explicit legacy model and training checkpoint migration;
- deterministic inspection and controlled one-head/two-head experiments.

## Verification

Executed from the repository root with Python 3.13.5, NumPy 2.4.3, pytest
9.1.1, and Ruff 0.15.18:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
python3 experiments/inspect_multi_head_attention.py
python3 experiments/overfit_tiny_transformer.py --heads 1 --steps 40
python3 experiments/overfit_tiny_transformer.py --heads 2 --steps 40
python3 experiments/compare_single_and_multi_head.py --steps 20
python3 experiments/train_transformer_lm.py \
  --heads 2 --steps 20 --until-step 10 \
  --evaluation-interval 5 --checkpoint-interval 5 \
  --generation-length 10 \
  --output outputs/transformer_lm_multi_head_resume_smoke
python3 experiments/train_transformer_lm.py \
  --heads 2 --steps 20 --until-step 20 \
  --evaluation-interval 5 --checkpoint-interval 5 \
  --generation-length 10 \
  --output outputs/transformer_lm_multi_head_resume_smoke \
  --resume \
    outputs/transformer_lm_multi_head_resume_smoke/latest_training_checkpoint.npz
```

Ruff lint and format checks passed, `git diff --check` was clean, and the
complete suite reported `315 passed in 4.27s`.

The deterministic two-head inspection produced `(1,2,4,4)` probabilities,
exact future zeros, exact earlier-output independence, finite gradients, and
an exact `0.0` maximum difference between the fused one-head and legacy
outputs.

For the interrupted 40-step `abc` fixture:

- one head: 575 parameters, validation loss `2.0344989001750946` to
  `0.06828678771853447`, agreement `1.0`;
- two heads: 715 parameters, validation loss `1.5254340171813965` to
  `0.0011447585420683026`, agreement `1.0`.

Both reloaded logits and generation exactly. These separate initializations
are correctness fixtures, not a quality comparison.

The controlled 20-step `abcde` comparison held seed, corpus, schedule, model
dimension, and per-head dimensions constant. One head (539 parameters) ended
at validation loss `0.35880257189273834`; two heads (609 parameters) ended at
`0.3904109001159668`. Both generated the pattern and reloaded exactly. No
general head-count conclusion is drawn.

The two-head fallback-corpus run had 1,071 parameters. It resumed at step 10
and reached step 20, reducing validation loss from `3.4020278453826904` before
training to `2.7258946895599365` (perplexity `15.270069789089348`).

Artifact inspection parsed every new JSON file and opened all 22 NPZ
checkpoints created by the release commands with `allow_pickle=False`. The
repository's actual ignored 0.5.0 model checkpoint and 0.5.1 training
checkpoint also loaded successfully as `number_of_heads=1`; the legacy
training checkpoint preserved its recorded generation prefix.

## Remaining limitations

- full \(BHT^2\) score and probability materialization;
- no dropout, padding mask, KV cache, rotary embeddings, or optimized kernels;
- character tokenization and tiny local fixtures only;
- fixed context windows and full-context generation recomputation;
- NumPy CPU execution;
- no claim that more heads improve quality;
- no useful paper-assistance capability yet.
