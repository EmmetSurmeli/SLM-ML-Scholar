# Milestone 4 transformer-readiness audit

Date: 2026-07-23

Scope: inspect the Milestone 1–4 source, tests, configurations, experiments,
and documentation before assembling the first decoder-only language-model
architecture. Existing components were modified only to export the new public
model; no verified foundation defect required correction.

## Readiness checks

### Stacked decoder blocks

`PreNormDecoderBlock` accepts and returns exactly \((B,T,D)\), consumes every
child cache in reverse order, and exposes all parameters recursively. Multiple
independent instances can therefore be applied sequentially without shape
conversion or shared state.

Finding: ready without modification.

### Container choice

A new `ModuleList` was considered but rejected as unnecessary.
`Sequential` already provides:

- deterministic child registration under names `0`, `1`, and so on;
- deterministic forward iteration;
- reverse-order backward propagation;
- recursive parameter traversal;
- recursive train/eval propagation;
- rejection of repeated module instances.

The transformer registers one `Sequential` decoder stack. Every block is
constructed independently from its own child seed.

Finding: the existing container is the smaller correct abstraction.

### Nested modules and recursive lifecycle

The top-level model owns two embeddings, the decoder-stack container, final
LayerNorm, and vocabulary Linear. Each decoder owns norms, attention, and
feed-forward children. `Module.has_pending_cache`, `clear_cache`, `train`,
`eval`, `parameters`, and `named_parameters` already recurse through this
entire hierarchy.

Finding: ready without modification. New tests verify repeated forward,
malformed backward, repeated backward, recursive modes, and complete cache
consumption.

### Three-dimensional activation path

Token and position embeddings produce \((B,T,D)\). Residual addition requires
exact shape equality. Decoder blocks preserve \((B,T,D)\); final LayerNorm
normalizes only \(D\); the vocabulary Linear maps the final axis to \(V\).

Finding: ready. Tests cover batch size one and multiple batches, sequence
length one through the configured maximum, and one or more decoder layers.

### Learned positions and batch accumulation

The existing `Embedding` uses indexed lookup in forward and `np.add.at` in
backward. Repeating position IDs \(0,\ldots,T-1\) across the batch therefore
accumulates every example into the same positional rows without new embedding
mathematics.

Finding: ready without behavioral changes. A two-batch test verifies token and
position table gradients equal exactly twice the corresponding one-batch
gradients for duplicated examples.

### Deterministic initialization

A top-level `SeedSequence` spawns independent child seeds for token embedding,
position embedding, every decoder block, and the language-model head. Each
block continues deriving independent internal attention and feed-forward
seeds.

Finding: ready. Tests verify equal top-level seeds reproduce every parameter,
different seeds change parameters, and separate blocks inside one model do not
start identically or share parameter identities.

### Deterministic parameter and checkpoint order

Explicit registration yields stable names beginning with token and position
tables, followed by `decoder_blocks.0...decoder_blocks.N...`, final norm, and
language-model head. A complete `state_dict` preserves that order and returns
copies. `load_state_dict` validates every key, shape, dtype, and finite value
before mutating any parameter.

Finding: ready. The versioned model checkpoint stores the exact validated
configuration and complete named state, rejects incompatible versions and
partial key sets, and reproduces float32 logits exactly.

### Optimizers and gradient clipping

SGD, Momentum, Adam, and global clipping operate on the recursive deterministic
parameter tuple and do not depend on model type.

Finding: ready. Two successive forward/backward/update cycles pass for all
three optimizers after clipping.

### Float32 and float64 policy

Every activation parameter is explicitly float32 or float64. Token IDs remain
integer. No token input is silently cast, clipped, or reshaped. The existing
finite-difference checker supports parameter-only checking when inputs are
discrete.

Finding: ready. Ordinary forward/backward, loss, optimizer, and checkpoint
tests cover float32; an exhaustive tiny model parameter check uses float64.

### Existing cross-entropy compatibility

The loss accepts arbitrary leading dimensions and treats the final axis as
classes. Transformer logits \((B,T,V)\) and targets \((B,T)\) therefore require
no wrapper loss or duplicated softmax.

Finding: ready without modification.

## Defects found and corrections made

No Milestone 1–4 correctness defect was found. No existing mathematical
primitive or cache behavior was changed.

Milestone 5 Part 1 adds:

- immutable validated `TransformerConfig`;
- public `TransformerLanguageModel`;
- learned token and position embedding composition;
- arbitrary positive stacks of independent single-head decoder blocks;
- final LayerNorm and untied vocabulary projection;
- explicit reverse-order backward orchestration;
- deterministic atomic `state_dict`/`load_state_dict`;
- versioned configuration-complete checkpoints.

## Tests added

- configuration validation and exact serialization;
- input validation for rank, dtype, range, empty dimensions, and context limit;
- output shape and dtype for several batch, sequence, and layer counts;
- independently controlled embedding, final-normalization, and vocabulary-head
  calculation;
- same-seed reproducibility, different-seed divergence, unique block
  initialization, and parameter non-sharing;
- deterministic names and explicit token/head weight untiedness;
- token and positional gradient accumulation across batch;
- multi-block forward and backward causality;
- direct N-dimensional cross-entropy compatibility;
- exhaustive finite differences for every parameter in a tiny float64 model;
- copied, ordered, atomic state dictionaries;
- exact float32 checkpoint round trip and incompatible-version rejection;
- SGD, Momentum, Adam, clipping, repeated cycles, recursive modes, zeroing, and
  cache misuse.

## Unresolved limitations

- each decoder block still contains one attention head;
- no dropout, rotary representation, KV cache, FlashAttention, or optimized
  kernel;
- token and output weights are intentionally untied;
- learned positions impose a fixed configured context limit;
- no tokenizer upgrade, training loop, sampling, evaluation, or transformer
  generation utility is introduced in this part;
- no stacked-model training has occurred;
- there is still no retrieval or paper-assistant behavior;
- checkpoints restore the existing policy of returning a training-mode model.

## Verification commands and results

The following commands were run successfully from the repository root:

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
PYTHONPATH=src python3 -c "import localml_scholar; print(localml_scholar.__version__)"
```

Results:

- Ruff formatted or confirmed 56 files, reported no lint violations, and
  confirmed every file was formatted.
- Pytest reported `219 passed in 0.97s`, including 38 new transformer
  architecture tests.
- The exhaustive tiny float64 fixture checked all 56 transformer parameters.
- The package import reported version `0.5.0`.
- A controlled architecture fixture independently reproduced token-plus-position
  composition, final LayerNorm, and vocabulary projection.
- Float32 logits integrated directly with the existing N-dimensional
  cross-entropy and completed manual backward.
- A two-layer float32 checkpoint reloaded configuration, deterministic
  parameter order, and logits exactly.
- No transformer training, evaluation, generation, sampling, or generated
  output artifact was introduced.

These results validate architecture and derivatives only. They are not
language-model quality, speed, or paper-assistant claims.
