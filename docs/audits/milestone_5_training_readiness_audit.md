# Milestone 5 Part 2 training-readiness audit

Date: 2026-07-23

This audit was completed before and alongside the Milestone 5 Part 2 training
and generation implementation. The release is package version `0.5.1`; it is
not named Milestone 6.

## Scope inspected

The audit traced the complete path needed for repeated transformer updates:

- chronological splitting and character vocabulary policy in `data.py`
- N-dimensional indexed cross-entropy in `losses.py`
- `Parameter` storage and deterministic `Module` traversal
- training/evaluation cache behavior in every embedding, linear, activation,
  normalization, attention, feed-forward, decoder-block, container, and
  top-level transformer module
- recursive mode propagation and pending-cache checks
- float32 execution and float64 validation policy
- global gradient measurement and clipping
- SGD, Momentum, and Adam parameter ordering and state
- model state dictionaries and model-only checkpoints
- tokenizer JSON persistence
- existing bigram generation and transformer validation tests
- Git ignore rules for corpora, NPZ checkpoints, caches, and outputs

## Verified foundations

### Repeated update cycles

The existing model already supported repeated
`zero_grad -> forward -> loss -> backward -> optimizer.step` cycles. Every
successful backward consumes the parent and child caches, and the existing
optimizer integration test covered all three optimizers.

### Evaluation and inference

`Module._store_forward_cache` was already a no-op in evaluation mode.
Consequently, a recursively evaluated transformer can run repeated forwards
without holding activations. Existing mode changes also reject a pending
training cache. Milestone 5 Part 2 adds a narrow `Module.inference_mode()`
context that:

1. rejects entry if any cache is pending;
2. records every nested module's mode;
3. temporarily disables caching recursively;
4. verifies that inference did not create a cache; and
5. restores every prior mode in `finally`.

Training lifecycle checks were not weakened. Backward after inference still
fails because inference creates no backward-capable cache.

### Shape and dtype support

The top-level model accepts integer `(B, T)` IDs and its validated components
preserve float32 or float64 `(B, T, D)` tensors. Its loss accepts `(B, T, V)`
logits with `(B, T)` integer targets. Sequence length is checked against the
configured context length.

### Data isolation

The existing chronological split fits the tokenizer only on training text and
encodes validation with that fixed vocabulary. Train and validation token
streams are formed independently. The pair crossing `split_index` is therefore
excluded intentionally. The new sequence sampler receives one isolated stream
and requires every target at `start + sequence_length` to remain in that same
stream.

### Reproducibility

Model initialization, batch sampling, evaluation sampling, and generation each
own a `numpy.random.Generator`. Evaluation reconstructs fixed samplers from
dedicated seeds and never advances the training sampler. The training sampler
serializes its complete PCG64 state and validates the stream hash, length,
shape settings, seed, and generator type before restoration.

## Issues found and fixes

### Model-only transformer saves were not atomic

The transformer checkpoint writer called `np.savez` directly even though the
documentation described checkpoint replacement as atomic. A crash could leave
a partial destination. The new dependency-free `atomic_savez` helper writes
and flushes a sibling temporary file, calls `fsync`, and replaces the
destination with `os.replace`. Transformer model-only checkpoints, optimizer
checkpoints, and full training checkpoints now use it.

### Optimizer state lacked an in-memory portable interface

Optimizers could save separate files but full training checkpoints could not
embed their state cleanly. `Optimizer.state_dict()` and
`Optimizer.load_state_dict()` now expose the same validated metadata and copied
arrays used by optimizer checkpoints.

### Malformed optimizer state could partially restore

Adam and Momentum validated and assigned state arrays in one loop. If a later
tensor was malformed, earlier tensors had already changed. Both loaders now
validate every array first and mutate state only after the complete candidate
passes.

### No resumable fixed-length sequence sampler existed

Milestone 1 batching sampled independent bigram examples and did not expose RNG
state. `SequenceBatchSampler` now samples uniformly with replacement from the
exact valid start range, creates shifted integer tensors, and supports exact
state restoration. This was missing functionality rather than a gradient
defect.

### Loss validation failure needed cache recovery

A malformed explicit target can fail after the model forward has created a
training cache. `TransformerTrainer.train_batch` clears the abandoned cache
before re-raising, so the model remains usable while the original error stays
visible.

## Checkpoint distinction

A model-only checkpoint contains architecture configuration and named model
parameters. It reproduces logits but cannot resume training.

A full training checkpoint additionally contains:

- format and package versions
- training configuration
- model mode
- optimizer type, hyperparameters, moments/velocities, and step counter
- completed step and best-validation metadata
- training sampler RNG state
- tokenizer vocabulary
- train and validation token-stream hashes and lengths
- accumulated history
- seed and clipping/weight-decay policy

The loader reconstructs fresh objects and rejects incompatible metadata or
arrays before returning them. Model-only checkpoints are explicitly rejected
by the full-state loader.

## Tests added

Focused coverage verifies:

- exact shifted sequences, valid start bounds, split isolation, and RNG restore
- configuration validation and all three optimizer paths
- repeated training, clipping, coupled L2 decay, and malformed-target recovery
- evaluation mode, parameters, gradients, optimizer state, and sampler
  preservation
- repeated no-cache inference and strict pending-cache failures
- greedy, seeded, temperature, and stable top-k generation
- context cropping and exact filtered probabilities
- full-state round trip, incompatible state rejection, and atomic save cleanup
- exact interrupted versus uninterrupted Adam trajectories
- deterministic tiny-pattern overfitting and output-preserving reload

## Verification commands and measured results

Executed from the repository root:

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
python3 experiments/overfit_tiny_transformer.py \
  --steps 120 \
  --output outputs/tiny_transformer_overfit_smoke
python3 experiments/train_transformer_lm.py \
  --steps 20 \
  --until-step 10 \
  --evaluation-interval 5 \
  --checkpoint-interval 5 \
  --generation-length 10 \
  --output outputs/transformer_lm_resume_smoke
python3 experiments/train_transformer_lm.py \
  --steps 20 \
  --until-step 20 \
  --evaluation-interval 5 \
  --checkpoint-interval 5 \
  --generation-length 10 \
  --output outputs/transformer_lm_resume_smoke \
  --resume outputs/transformer_lm_resume_smoke/latest_training_checkpoint.npz
```

At the time this audit was drafted, the complete suite reported `279 passed`.
The final formatting, regression experiments, artifact inspection, and exact
results are repeated at completion and reflected in the README.

The 575-parameter `abc` fixture reduced validation loss from
`2.0344989001750946` to `0.00043658849608618766` in 120 steps. It resumed from
step 60, reached greedy transition agreement `1.0`, and reloaded logits and
generation exactly.

The 931-parameter fallback-corpus smoke run resumed at step 10 and completed
step 20. Validation loss decreased from `3.3290658791859946` before training
to `2.7723965644836426`; the measured validation perplexity was
`15.996925771279688`. These numbers validate only the named deterministic
fixtures.

## Unresolved limitations

- one attention head
- character tokenization
- full quadratic attention matrices
- full-context recomputation during generation
- no KV cache, dropout, learning-rate schedule, padding mask, or mixed precision
- CPU NumPy execution without optimized kernels
- fixed-length random-with-replacement training batches
- exact continuation assumes compatible NumPy behavior and deterministic CPU
  operations
- no useful paper-assistance behavior or competitive language-model claim
