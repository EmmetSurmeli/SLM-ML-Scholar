# Milestone 2 attention-readiness audit

Date: 2026-07-23

Scope: the existing Milestone 1 and 2 source, tests, configurations,
experiments, and documentation were inspected before introducing Milestone 3
single-head causal self-attention. Changes were limited to the new attention
milestone because no blocking foundation defect was verified.

## Abstractions checked

### Multiple nested `Linear` modules

`Module.register_module` stores children explicitly in insertion-ordered
dictionaries. A parent can therefore own separate query, key, value, and
optional output projections. Duplicate child instances are rejected, avoiding
accidental cache sharing. The attention tests verify the expected dotted
parameter names and recursive mode propagation.

Finding: ready without modification.

### Deterministic parameter traversal

`named_parameters` visits direct parameters followed by child modules in
registration order and rejects shared parameters. Query, key, value, and
output parameters consequently have stable names and optimizer ordering.
Checkpoint tests verify that this ordering survives reconstruction.

Finding: ready without modification.

### Three-dimensional tensors and shape-preserving backward

`Linear` treats the final axis as features and flattens only the leading axes
for parameter-gradient calculation. It restores the original input shape for
the input gradient. Existing 3D linear tests and new attention tests cover
\((B,T,D)\) inputs.

Finding: ready without modification.

### Independent projection caches

Each `Linear` instance owns one cache. The attention head uses distinct module
instances for Q, K, V, and optional output projection, so each branch retains
its own input or context without overwriting another branch.

Finding: ready without modification.

### Forward/backward lifecycle

Training-mode modules permit one unmatched forward and consume the cache in
backward. Repeated forward, backward without a cache, and mode changes while a
cache is pending raise errors. `clear_cache` recursively recovers from an
abandoned computation. The attention module uses the same contract and clears
all child caches if its forward fails partway through.

Finding: ready without modification. Repeated attention forward/backward and
malformed-gradient tests were added.

### Parameter checkpointing

Milestone 2 uses model-specific versioned NumPy checkpoints with JSON
configuration plus deterministic dotted parameter keys. The attention head
follows the same explicit design and rejects missing or unexpected arrays.

Finding: the existing design is suitable; a versioned attention checkpoint and
exact float32 output round-trip test were added.

### Float32 training and float64 gradient checking

`Parameter`, `Module`, `Linear`, initialization utilities, optimizers, and the
general gradient checker preserve and validate float32/float64. Numerical
checks require float64 by default. No silent dtype conversion was identified.

Finding: ready without modification. The attention suite exercises float32
forward, backward, optimization, clipping, and checkpointing; exhaustive
finite differences use float64.

### Existing loss, optimizer, and training utilities

The N-dimensional cross-entropy interface already accepts sequence-shaped
logits and targets. Identity-keyed optimizers and global-norm clipping operate
on recursively enumerated parameters, independent of model type. These
interfaces require no attention-specific branching.

Finding: ready without modification. Adam and clipping integration tests were
added for the attention parameters.

### Repository and documentation boundary

Generated data, checkpoints, NumPy archives, caches, virtual environments, and
platform metadata remain ignored. The README accurately described Milestone 2
before this work and did not claim paper-assistant capability.

Finding: ready. Milestone 3 documentation continues to state that one
attention head is neither a full transformer nor a useful paper assistant.

## Defects found and fixes made

No concrete correctness defect was found in the Milestone 1 or 2
implementation. No existing foundation code was refactored merely for style.

Milestone 3 adds only the focused interfaces needed for attention:

- immutable, broadcastable boolean causal masks where `True` means allowed;
- independent stable masked-softmax forward and backward primitives;
- one explicitly differentiated causal self-attention head;
- read-only forward details for educational inspection;
- model-specific, versioned attention checkpoints.

## Attention integration decisions

- Q, K, V, and optional output projection reuse validated `Linear` modules.
- A single seeded generator is passed through the projections in deterministic
  construction order.
- The head accepts exactly \((B,T,D)\), while each `Linear` remains reusable
  for arbitrary leading dimensions.
- The mask has shape \((1,T,T)\) and broadcasts across the batch.
- Blocked probabilities and blocked score gradients are exactly zero.
- The default omits output projection and returns \(d_v\) features. Enabling
  it maps \(d_v\) back to the input dimension \(D\).
- Inspection tensors are read-only copies and do not become extra
  differentiation outputs.
- Dropout, multi-head attention, residual connections, and a transformer block
  are deliberately excluded.

## Tests added

- mask semantics for sequence lengths 1, 2, and larger;
- stable masked-softmax behavior, extreme logits, invalid masks, exact blocked
  zeros, dtype preservation, and exhaustive finite differences;
- forward shape and probability invariants across batches and dimensions;
- an independently calculated two-token example;
- forward and backward causality;
- exhaustive input and parameter finite differences, including optional output
  projection;
- cache misuse and malformed input/gradient handling;
- deterministic initialization and parameter traversal;
- float32 checkpoint, optimizer, clipping, and zero-gradient integration;
- deterministic attention-inspection summary generation.

## Unresolved limitations

- one head only; no multi-head composition;
- no dropout, attention bias other than projection biases, padding mask, or
  arbitrary user-supplied mask;
- no residual connection, normalization wrapper, feed-forward layer, or full
  decoder block;
- no attention-based language model, generation, or training experiment;
- no KV cache, rotary position representation, fused kernel, or GPU path;
- attention materializes the full \(T\times T\) score and probability tensors;
- model checkpoint logic remains model-specific rather than a universal module
  serialization framework;
- a module instance still supports one unmatched training forward and cannot
  be reused at multiple graph locations.

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
```

Results:

- Ruff formatted or confirmed 50 files, reported no lint violations, and
  confirmed every file was formatted.
- The final pytest pass reported `132 passed in 0.28s`.
- The unchanged 300-step bigram fallback smoke run used 2,094 training
  examples, 232 validation examples, a 23-character vocabulary, and 529
  parameters. Best sampled validation loss was `1.5488034950125846`
  (perplexity `4.705836256201863`) at step 300.
- The unchanged 1,000-step XOR run used 42 parameters, reduced
  cross-entropy from `0.6865651569496622` to
  `7.506981534793376e-06`, predicted `[0, 1, 1, 0]`, and preserved logits
  exactly through checkpoint reload.
- The attention inspection used one batch of four tokens with
  \(D=4,d_k=2,d_v=3\), 55 total embedding-plus-attention parameters, and
  synthetic loss `1.3372998086757004`. Future-token probabilities were
  exactly zero.
- Inspection gradient norms were `2.8693286294328257` for input embeddings,
  `0.2605448758200729` for the query projection,
  `0.5084771956554331` for the key projection, and
  `2.69169910965937` for the value projection.
- Generated summaries and checkpoints were inspected under `outputs/`. Git
  reports those paths as ignored; no generated artifact is tracked.

The timing and loss values describe deterministic correctness fixtures only.
They are not model-capability or performance claims.
