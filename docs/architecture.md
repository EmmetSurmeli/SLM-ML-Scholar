# Architecture

## Intended local system

The eventual LocalML Scholar data flow is:

```text
Local PDF
    ↓
Document parser
    ↓
Structured sections, equations, captions, and references
    ↓
Local retrieval/index
    ↓
From-scratch decoder-only transformer
    ↓
Grounded explanation with page/section citations
```

All user documents and derived indexes are intended to remain on the user's
machine. A dedicated local PDF library is appropriate for document parsing;
the from-scratch constraint applies to the neural network and its training
machinery, not to decoding the PDF file format.

## Model and retrieval are separate

The language model learns general statistical and explanatory behavior during
training. The retrieval system indexes a newly loaded paper, selects relevant
passages and metadata at question time, and supplies that evidence as model
context. Loading a paper must not require retraining the model.

This separation has four practical benefits:

1. A paper can be added or removed quickly.
2. Answers can retain page and section provenance.
3. Retrieval quality and model quality can be evaluated independently.
4. Unsupported model claims can be checked against the supplied evidence.

The final answer layer should distinguish retrieved statements, model
inferences, and uncertainty. Page citations are a product requirement, not a
decorative feature.

## Implemented package boundaries

Milestones 1 through 6 establish interfaces that later components can
extend:

- `tokenizer.py` owns text/token conversion and vocabulary persistence.
- `data.py` owns local loading, chronological splits, examples, and seeded
  minibatches.
- `losses.py` owns explicit numerical forward/backward primitives.
- `nn/` owns explicit `Parameter`/`Module` foundations, initializers, layers,
  activations, normalization, sequential composition, causal masks, stable
  masked softmax, legacy single-head and canonical fused multi-head causal
  attention, residual primitives, a position-wise feed-forward module, and one
  pre-norm decoder block.
- `optim/` owns optimizers over identity-keyed parameters.
- `training/` owns configuration, explicit update cycles, deterministic
  evaluation, resumable training state, global clipping, and finite
  differences.
- `models/` composes primitives into checkpointable models.
- `optimizers.py` remains the Milestone 1 named-array SGD compatibility path.
- `generation.py` owns bigram and transformer autoregressive sampling
  independently of training.
- `serialization.py` owns atomic NPZ replacement without model dependencies.
- `utils.py` owns reproducibility, reporting math, and the bigram-specific
  compatibility gradient check.
- `experiments/` composes library pieces into reproducible runs but is not
  imported by the reusable model.

The bigram model retains its original one-matrix interface. New neural modules
register lightweight `Parameter` objects explicitly. Recursive enumeration
produces deterministic dotted names such as `network.0.weight`, while
optimizer state is keyed to parameter identity rather than a name.

## Manual backward and cache contract

There is no dynamic automatic-differentiation graph. Each trainable operation
implements its own backward formula.

In training mode, a primitive layer supports exactly one unmatched cached
forward. Backward consumes that cache. A second forward or mode transition
while the cache is pending raises an informative error. This makes accidental
cache overwrite and unsupported reuse of one layer instance explicit.
`Sequential` rejects duplicate child instances and runs backward in reverse
order.

Evaluation forwards do not cache and may be repeated, but cannot be followed
by backward. `clear_cache` exists only for explicit recovery after an abandoned
computation.

`inference_mode()` makes the existing behavior safe for an entire module tree.
It refuses to enter if a training cache is pending, snapshots every nested
mode, disables cache creation, checks for unexpected caches on exit, and
restores the snapshots. Generation and evaluation use this context. They do
not weaken the one-forward/one-backward training contract.

See `docs/numerical_precision.md` for the project-wide float32/float64 policy.

## Current attention architecture

`CausalSelfAttentionHead` remains as the validated legacy reference.
`MultiHeadCausalSelfAttention` is the canonical implementation used by decoder
blocks. It uses fused `Linear` projections with widths `H * key_dim`,
`H * key_dim`, and `H * value_dim`:

```text
X → fused Q/K/V projections
  → split (B,T,H*d) to (B,H,T,d)
  → per-head scaled QKᵀ
  → shared causal mask + independent per-head softmax
  → per-head A V
  → concatenate (B,T,H*d_v)
  → output Linear
  → Y (B,T,D)
```

The multi-head mask has shape `(1, 1, T, T)` and uses `True` for allowed
positions. It broadcasts across batches and heads. Softmax computes its
maximum and denominator independently for every batch/head/query row from
allowed entries only. Blocked probabilities and score gradients are exactly
zero.

Backward propagation first differentiates the output projection and inverts
the concatenation layout. Every head then produces probability, value, score,
query, and key gradients. Those per-head gradients are reassembled into the
fused layouts, passed through the three affine projections, and their shared
input branches are explicitly summed.

Per-head dimensions are explicit, so the model dimension need not be divisible
by the head count. Increasing `H` while holding per-head dimensions fixed
increases projection width and parameter count. With `H=1`, names, shapes,
initialization, output, and backward results match the legacy implementation
exactly.

Attention materializes complete `(B,H,T,T)` score and probability tensors. It
does not include dropout, a padding mask, KV caching, or an optimized kernel.

## Current decoder-block architecture

`PreNormDecoderBlock` composes the validated primitives in this exact order:

```text
X
├── LayerNorm 1 → multi-head causal attention ──┐
└────────────────────────────────────────────── + → R1
                                                │
                                                ├── LayerNorm 2
                                                │   → Linear
                                                │   → exact GELU
                                                │   → Linear ─┐
                                                └───────────── + → Y
```

The attention output projection is mandatory so arbitrary head counts and
per-head value dimensions map back to the model dimension before the first
residual.

Residual addition is an explicit checked numerical operation rather than a
module or automatic graph node. Both operands must have identical shape and
dtype. Backward returns independent copies of the upstream gradient to the
identity and transformed paths. `PreNormDecoderBlock.backward` then performs
both accumulation steps visibly:

1. add the second residual identity gradient to the gradient returned through
   feed-forward and LayerNorm 2;
2. add the first residual identity gradient to the gradient returned through
   attention and LayerNorm 1.

LayerNorm, the feed-forward network, concatenation, output projection, and
residual addition operate independently at each position. Multi-head causal
attention is the only cross-token operation, so the complete block preserves
its causal boundary.

The block has no final normalization, stack, position representation,
vocabulary projection, or training objective.

## Current decoder-only language-model architecture

`TransformerLanguageModel` is the permanent top-level model interface:

```text
integer token IDs (B, T)
    ├── learned token embedding ─────┐
    └── learned position embedding ─ + → hidden (B, T, D)
                                          ↓
                              decoder block 0
                                          ↓
                                      ...
                                          ↓
                              decoder block N - 1
                                          ↓
                                  final LayerNorm
                                          ↓
                         untied vocabulary Linear
                                          ↓
                                  logits (B, T, V)
```

The decoder stack uses the existing `Sequential` container. This provides
deterministic numbered registration, forward iteration, reverse-order
backward, recursive modes, and deterministic dotted parameter names without a
new `ModuleList` abstraction.

Learned positions are an independent `Embedding` table of shape
`(maximum_context_length, model_dimension)`. Position IDs `0..T-1` repeat
across the batch, so the existing indexed `np.add.at` backward naturally
accumulates positional gradients over all examples. Token and position outputs
are checked for exact shape equality before addition.

The vocabulary head is a separate `Linear(model_dimension, vocabulary_size)`.
It returns logits without applying softmax and does not share storage with the
token embedding. The existing N-dimensional cross-entropy consumes these
logits directly.

One top-level seed spawns independent child seed streams for both embeddings,
every decoder block, and the vocabulary head. `number_of_heads` is part of the
complete checkpointed configuration. The public state loader validates every
key, shape, dtype, and finite value before changing any parameter.

Version 0.6.0 recognizes 0.5.0 single-head model checkpoints and 0.5.1 full
training checkpoints. It adds `number_of_heads=1` in memory because the fused
one-head parameter names, order, and shapes are identical. Unknown versions
and incompatible state remain errors.

## Training, evaluation, generation, and checkpoint architecture

Milestone 5 Part 2 trains the existing architecture without introducing
another neural-network implementation:

```text
isolated train token stream
        ↓ private restorable RNG
shifted integer batches (B, T)
        ↓
TransformerLanguageModel.forward
        ↓ logits (B, T, V)
N-D softmax cross-entropy
        ↓ explicit logit gradient
TransformerLanguageModel.backward
        ↓
optional coupled L2 decay → global norm/clip → optimizer step
```

`SequenceBatchSampler` samples valid starts uniformly with replacement. A
start is valid only when its last shifted target remains in the supplied token
stream. Training and validation samplers never share a stream or RNG.

Evaluation constructs new fixed-seed samplers on every call and combines batch
losses by predicted-token count. It therefore does not change the next
training batch. Parameter values, optimizer state, and gradient buffers remain
unchanged.

Generation accepts a batched integer prompt, crops only the model input to the
most recent configured context, and recomputes the retained context at every
new token. Greedy decoding, positive temperature, and stable top-\(k\)
sampling are supported. There is no KV cache.

Model-only checkpoints preserve architecture and parameters for inference.
Full training checkpoints additionally preserve optimizer tensors and step,
training configuration, sampler RNG state, tokenizer vocabulary, corpus
identity hashes, best-validation state, history, mode, clipping/decay policy,
and seed. Full checkpoint replacement is atomic, and loading reconstructs
fresh objects before returning a usable trainer.

## Current models

The Milestone 1 baseline predicts the next character from the immediately
preceding character:

\[
P(x_{t+1}\mid x_t).
\]

It has no representation of words, equations, passages, papers, or questions.
Its value is as a minimal correctness baseline for tokenization, dataset
isolation, loss math, manual gradients, optimization, generation, persistence,
and experiment bookkeeping.

Milestone 2 adds reusable mathematical components and a two-layer MLP. The XOR
experiment demonstrates composition and manual backward correctness on four
synthetic examples; it is not a language model and does not add paper
understanding.

Milestone 3 adds a single attention head and an inspection experiment. It
validates causal information flow and the local derivatives needed by a future
decoder, but it is not itself a sequence language model or transformer.

Milestone 4 adds one mathematically complete decoder block. Its deterministic
inspection composes token embeddings with the block, validates future-token
independence, runs every manual backward path, and applies one optimizer step.
One block without position information or a vocabulary output is still not a
transformer language model.

Milestone 5 Part 1 assembles the complete decoder-only transformer
architecture. Part 2 adds a deterministic training and generation path and
proves exact interrupted resumption on a tiny fixture. The model is now a
trained character-level transformer in the literal engineering sense, but its
small fixture results do not establish general language ability or paper
understanding.

Milestone 6 replaces the decoder's canonical attention path with fused
multi-head causal attention while preserving `H=1` exactly. Training,
evaluation, autoregressive generation, model checkpoints, and exact full-state
resumption support multiple heads. The tokenizer and experimental corpora
remain deliberately small and character-level.

## Future module constraints

The core neural-network path will continue to use Python, NumPy array storage
and matrix multiplication, and standard-library code. It will not depend on
automatic differentiation, framework layers, framework losses, framework
optimizers, pretrained-model APIs, or external tokenizers. PDF parsing,
plotting, tests, and the application will stay outside that core and may use
focused third-party libraries when their value is clear.
