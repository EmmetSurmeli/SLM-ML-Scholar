# Milestone 3 decoder-block readiness audit

Date: 2026-07-23

Scope: inspect the complete Milestone 1–3 source, tests, configuration,
experiments, and documentation before composing a branched pre-normalized
decoder block. Existing code was changed only where Milestone 4 required a new
public composition; no verified foundation defect required correction.

## Abstractions inspected

### Nested heterogeneous modules

`Module.register_module` accepts independent child modules and recursively
enumerates their parameters. A decoder parent can therefore own two
LayerNorms, one attention head, and one feed-forward module whose own children
are two Linear layers and an activation.

Verified assumption: ready without modification.

### Three-dimensional tensor behavior

`Linear` treats only the final axis as features, flattens arbitrary leading
dimensions for parameter-gradient accumulation, and restores the original
input shape. LayerNorm normalizes only the final dimension. GELU is elementwise.
The attention head already requires and preserves \((B,T,D)\) when its output
projection is enabled.

Verified assumption: ready without modification. New tests cover batch size
one and multiple batches, sequence length one and longer sequences, smaller
and larger feed-forward widths, arbitrary key dimensions, and unequal value
dimensions with output projection.

### Residual branching

The project intentionally has no automatic graph. A residual parent must
therefore split the upstream gradient and accumulate branch results itself.
No foundation behavior prevents this: child backward calls return input
gradients without modifying the parent-owned upstream tensor.

Design decision: residual addition is implemented as explicit checked
functions rather than a `Module`. Addition has no parameter and needs no
standalone forward cache. `residual_add_backward` returns independent copies
for the identity and transformed branches, while
`PreNormDecoderBlock.backward` displays both accumulation points directly.

Verified assumption: ready without foundation modification.

### Independent caches and reverse consumption

Every registered layer instance owns a separate one-forward/one-backward
cache. The decoder uses distinct norm and affine instances. Its backward order
is:

1. feed-forward Linear 2, activation, Linear 1;
2. LayerNorm 2;
3. attention output/value/softmax/score/QKV paths;
4. LayerNorm 1.

The parent cache is consumed last. Repeated forward, malformed gradient, and
repeated backward tests verify that invalid calls do not silently overwrite
caches and that a complete backward leaves no pending cache.

Verified assumption: ready without modification.

### Recursive train/eval and parameter order

Mode propagation recursively reaches the block, both LayerNorms, attention
projections, feed-forward linears, and activation. Insertion-ordered explicit
registration produces stable, unique dotted names. Shared parameters remain
unsupported and are rejected by the existing traversal.

Verified assumption: ready without modification.

### Optimizers and clipping

SGD, Momentum, and Adam consume a deterministic tuple of `Parameter` objects
and do not depend on model type. Global clipping likewise operates on the
recursive parameter tuple. Two successive update cycles verify that all three
optimizers can reuse the block after caches are consumed.

Verified assumption: ready without modification.

### Precision and finite differences

Parameters and modules accept float32 or float64, reject integer and unsupported
floating dtypes, and require exact dtype agreement in backward. The generalized
gradient checker uses float64 by default, restores every perturbation, and
reports per-tensor coordinates and absolute/relative errors.

Verified assumption: ready without modification. New tests execute the full
block in float32 and check all coordinates of a tiny float64 block.

### Checkpoint conventions

Existing composed models use explicit model-specific NumPy checkpoints with
JSON configuration and deterministic parameter keys. Milestone 4 follows that
policy for both the feed-forward module and decoder block. The configuration
includes model version, all dimensions, biases, output-projection choice,
LayerNorm epsilon, activation, seed, and dtype. Loading reconstructs a
training-mode module, rejects incompatible type/version/key sets, validates
every parameter shape and dtype, and preserves float32 outputs exactly.

Verified assumption: suitable without changing previous checkpoints.

## Defects found and corrections made

No concrete Milestone 1–3 correctness defect was found. No working foundation
code was refactored for aesthetics.

Milestone 4 introduced:

- exact-shape residual forward and backward helpers;
- a checkpointable position-wise transformer feed-forward module;
- a checkpointable pre-normalized single-head decoder block;
- immutable inspection details for feed-forward and decoder intermediates.

## Tests added

- feed-forward hand calculations, shapes, dtypes, biases, lifecycle,
  deterministic initialization, parameter traversal, modes, checkpointing,
  optimizer/clipping integration, invalid inputs, and exhaustive gradients;
- residual exact-shape, dtype, finite-value, identity-gradient, and independent
  branch-copy behavior;
- decoder shapes, explicit residual equality, identity block, attention-only
  controlled fixture, absence of final LayerNorm, deterministic initialization,
  nested modes/names, and configuration validation;
- operational forward and backward causality;
- exhaustive input and parameter finite differences including both norms,
  Q/K/V and output projection, and both feed-forward affine layers;
- SGD, Momentum, Adam, clipping, repeated update cycles, checkpoint round trip,
  incompatible-version rejection, and deterministic inspection summary.

## Why the repository is ready for residual branching

Every transformed branch is already an independently differentiated `Module`
with its own cache and an input-gradient return value. A parent can therefore
make the graph topology explicit in ordinary Python: copy the upstream
gradient for a residual split, call child backward methods in reverse order,
and sum the returned transformed gradient with the identity gradient. The
general gradient checker validates the resulting full derivative without
requiring an automatic graph.

## Unresolved limitations

- one causal attention head only;
- one decoder block only, with no stack;
- no positional embeddings or vocabulary projection;
- no dropout, padding mask, KV cache, rotary representation, or optimized
  kernel;
- no transformer language-model objective, training, generation, or paper
  retrieval;
- attention still materializes \(T\times T\) scores and probabilities;
- checkpointing is explicit per composed model rather than a universal module
  serializer;
- one module instance still supports only one unmatched training forward.

## Verification commands and results

The following commands were run successfully from the repository root:

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
```

Measured results:

- Ruff formatted or confirmed 54 files, reported no lint violations, and
  confirmed every file was formatted.
- The final pytest pass reported `181 passed in 0.42s`.
- The unchanged 300-step bigram fallback smoke run reproduced 2,094 training
  examples, 232 validation examples, a 23-character vocabulary, 529
  parameters, and best sampled validation loss `1.5488034950125846`
  (perplexity `4.705836256201863`) at step 300.
- The unchanged 42-parameter XOR run reproduced final loss
  `7.506981534793376e-06`, predictions `[0, 1, 1, 0]`, and exact checkpoint
  output preservation.
- The unchanged single-head inspection reproduced synthetic loss
  `1.3372998086757004` and exactly zero future-token probabilities.
- The decoder inspection used 24 embedding parameters and 134 decoder
  parameters with \(D=4,d_k=2,d_v=3,D_{\mathrm{ff}}=7\). Synthetic loss was
  `5.861637709654549`.
- Decoder inspection gradient norms were `11.266325080145544` for embeddings,
  `6.900069083229121` for LayerNorm 1, `1.0274048134443496` for Q,
  `1.0013334591202478` for K, `9.351511542242077` for V,
  `6.086118806272695` for the attention output projection,
  `2.8571390777897805` for LayerNorm 2, `4.248104756012224` for feed-forward
  Linear 1, and `7.701071829944415` for feed-forward Linear 2.
- The inspection confirmed exact future-mask zeros, output-shape preservation,
  earlier-output independence after a future-token change, and at least one
  decoder parameter changed after Adam stepped.
- Generated summaries and checkpoints were inspected under `outputs/`. Git
  reports these paths as ignored, and no generated artifact is tracked.

All loss and timing values above describe deterministic correctness fixtures,
not model-capability or performance results.
