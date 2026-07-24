# LocalML Scholar

LocalML Scholar is a portfolio project for building a fully local small
language model and research-paper assistant from first principles. The eventual
goal is to help students inspect difficult machine-learning papers, ask
grounded questions about equations and methods, and create reproduction plans
without sending documents to an internet service.

## Why local, and why from scratch?

Research papers can contain unpublished ideas, class work, annotations, or
other material a reader does not want to upload. A local retrieval and
generation pipeline offers a clear privacy boundary and can work without a
network connection.

The language model is being built from first principles to make its numerical
behavior inspectable. The core implementation uses Python and NumPy for arrays
and matrix multiplication. Forward propagation, backward propagation, loss,
optimization, training, and generation are explicit code—not calls to an ML
framework or automatic-differentiation system. This is an educational and
engineering constraint, not a claim that a from-scratch model is automatically
faster or more capable.

## Current status: Milestone 5 Part 1

The package version is `0.5.0`.

Milestone 1 is complete and independently audited. Its character-level bigram
learns a \(V\times V\) table of next-character logits and conditions only on
the previous character. The audit verifies:

- deterministic character vocabulary construction and JSON persistence
- chronological train/validation splitting with no boundary example leakage
- reproducible minibatches
- stable softmax and indexed cross-entropy
- manually derived logit and weight gradients
- SGD with optional weight decay
- exhaustive or sampled finite-difference gradient checking
- seeded temperature sampling and greedy generation
- checkpoint, history, and run-summary artifacts
- versioned model configuration and a mathematical implementation map

Milestone 2 adds a reusable manually differentiated neural-network foundation:

- lightweight `Parameter` objects and deterministic nested `Module` traversal
- explicit one-forward/one-backward cache lifecycle
- `Linear`, `Embedding`, ReLU, exact GELU, and LayerNorm
- `Sequential` composition and a two-layer MLP
- coupled-decay SGD, classical momentum, and bias-corrected Adam
- global gradient norm calculation and clipping
- generalized input and parameter finite-difference checking
- float32/float64 precision policy with no silent parameter downcasting
- versioned MLP and optimizer state checkpoints
- a deterministic XOR correctness demonstration

No automatic differentiation or external neural-network implementation is
used. Every trainable operation contains an explicit backward formula connected
to a derivation and numerical test.

Milestone 3 implements the smallest causal-attention unit needed for the next
stage:

- an immutable boolean causal mask where `True` means allowed
- stable masked softmax over valid entries only
- an explicit masked-softmax backward pass
- query, key, and value projections
- scaled dot-product scores and weighted value aggregation
- an optional output projection
- manual gradients through every attention operation
- exhaustive finite differences on tiny float64 inputs and parameters
- forward and backward causality tests
- checkpoint, optimizer, clipping, and float32 integration
- a deterministic embedding-to-attention inspection experiment

This is **one educational, unoptimized attention head only**. It is not
multi-head attention, a transformer block, or a language model. It materializes
the complete score matrix and has no dropout, padding mask, residual path, or
KV cache.

Milestone 4 composes the validated components into one mathematically complete
pre-normalized decoder block:

- `LayerNorm -> single-head causal attention -> residual addition`
- `LayerNorm -> Linear -> exact GELU -> Linear -> residual addition`
- exact-shape residual checks with no silent broadcasting
- explicit identity and transformed gradient branches
- manual accumulation at both residual joins
- configurable key, value, and feed-forward dimensions
- optional attention and feed-forward biases
- an attention output projection enabled by default
- versioned feed-forward and decoder-block checkpoints
- exhaustive full-block input and parameter finite differences
- operational forward and backward causality validation
- deterministic embedding-to-decoder inspection and optimizer step

All gradients remain manually implemented. The block is educational and
unoptimized, and attention still uses one head.

Milestone 5 Part 1 assembles the first complete decoder-only transformer
language-model architecture:

- validated immutable `TransformerConfig`
- learned token embeddings
- learned absolute position embeddings
- exact-shape embedding addition
- arbitrary positive stacks of independent decoder blocks
- final LayerNorm
- separate, untied vocabulary projection
- unnormalized logits shaped `(batch, sequence, vocabulary)`
- direct compatibility with the existing N-dimensional cross-entropy
- explicit reverse-order manual backward through the complete model
- deterministic independent child initialization from one top-level seed
- deterministic atomic state dictionaries and versioned checkpoints
- multi-block causality and exhaustive tiny-model parameter checks

The public `TransformerLanguageModel` now exposes `forward`, `backward`,
recursive parameters, state loading, checkpoints, and train/eval modes through
the existing manual module conventions.

The project is **still not an SLM and not yet a research-paper assistant**. The
bigram cannot understand a paper, explain an equation, retrieve evidence, or
maintain context beyond one character. The MLP is a synthetic integration
fixture; the attention head and decoder block are numerical and causality
fixtures. The transformer architecture has not been trained or evaluated and
has no transformer generation or sampling interface. It therefore makes no
language-understanding or paper-assistance claim.

## Installation

Python 3.10 or newer is required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

The runtime dependency is NumPy. Pytest and Ruff are development dependencies.
The model package does not use PyTorch, TensorFlow, JAX, Keras, Hugging Face,
autograd libraries, pretrained APIs, or external tokenizers.

## Testing

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m ruff format --check .
```

Tests cover serialization, deterministic data and generation, numerical
stability, hand-calculated loss, finite-difference gradients, repeated-row
gradient accumulation, 2D/3D layer behavior, LayerNorm, hand-computed optimizer
trajectories, global clipping, nested module behavior, checkpoint identity,
XOR learning, attention-mask semantics, extreme-logit masked softmax,
hand-computed attention, forward/backward causality, every attention parameter
gradient, residual branch accumulation, position-wise feed-forward behavior,
controlled decoder fixtures, full decoder-block gradients, three optimizer
integrations, learned positional accumulation, independent decoder stacking,
top-level state atomicity, cross-entropy compatibility, multi-block causality,
exhaustive transformer parameter gradients, dtype policy, cache misuse, and
malformed inputs.

### Verified implementation

The following commands were executed successfully from the repository root on
2026-07-23 with Python 3.13.5, NumPy 2.4.3, pytest 9.1.1, and Ruff 0.15.18:

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
python3 experiments/inspect_single_head_attention.py
python3 experiments/inspect_pre_norm_decoder_block.py
PYTHONPATH=src python3 -c \
  "import localml_scholar; print(localml_scholar.__version__)"
```

Ruff reported no lint or formatting errors, and pytest reported `219 passed`.
The 300-step fallback-corpus smoke run used 2,094 training examples, 232
validation examples, a 23-character vocabulary, and 529 parameters. Its best
sampled validation loss was `1.5488034950125846` (perplexity
`4.705836256201863`) at step 300. These figures verify this one fixture and
configuration only; they are not language-model capability claims.

The 42-parameter XOR MLP reduced mean cross-entropy from
`0.6865651569496622` to `7.506981534793376e-06` in 1,000 full-batch Adam
steps and predicted `[0, 1, 1, 0]`. Its reloaded checkpoint reproduced logits
bit-for-bit. This demonstrates correctness on four deterministic synthetic
examples only.

The 55-parameter deterministic attention inspection reported exact tensor
shapes, scaled scores, the causal mask, probabilities, synthetic loss, and
input/Q/K/V gradient norms. Its synthetic loss was `1.3372998086757004`, and
all future-token probabilities were exactly zero. Its measured values are
recorded in the
[Milestone 2 attention-readiness audit](docs/audits/milestone_2_attention_readiness_audit.md);
this is a mathematical inspection, not a quality or performance benchmark.

The deterministic decoder-block inspection reports both normalized paths,
Q/K/V and score shapes, the causal mask and probabilities, both residual
outputs, the feed-forward hidden shape, every parameter-group gradient norm,
and whether one Adam step changed a block parameter. Its exact measured values
are recorded in the
[Milestone 3 decoder-block readiness audit](docs/audits/milestone_3_decoder_block_readiness_audit.md).
The 134-parameter decoder fixture produced synthetic loss
`5.861637709654549`, preserved earlier outputs after changing a future token,
and updated at least one parameter.

The transformer architecture adds no training or performance result. Its
verification consists of controlled forward calculations, direct
cross-entropy integration, multi-block causality, exact float32 checkpoint
reload, and exhaustive finite differences across all 56 parameters of a tiny
one-layer float64 configuration.

## Training

Put a legally usable UTF-8 corpus at `data/raw/corpus.txt`; data is not
downloaded automatically.

```bash
python experiments/train_bigram.py \
  --input data/raw/corpus.txt \
  --config configs/bigram_small.json
```

For a fast plumbing check, omit `--input` to use the tiny built-in fallback
corpus:

```bash
python experiments/train_bigram.py --config configs/bigram_small.json
```

The configured output directory receives:

- `best_model.npz` — lowest sampled validation-loss checkpoint
- `final_model.npz` — weights after the final update
- `tokenizer.json` — exact character-to-ID vocabulary
- `history.json` — evaluation losses, perplexities, and generated samples
- `run_summary.json` — configuration, split sizes, parameter count, seed,
  best result, and artifact paths

The same configuration, corpus bytes, NumPy behavior, and environment produce
the same random draws and update sequence. Evaluation is a reproducible
minibatch estimate, not an exhaustive validation loss.

This training script applies only to the Milestone 1 bigram. No transformer
training loop is included in Milestone 5 Part 1.

## Transformer architecture interface

The architecture accepts integer token IDs and returns logits. It does not
apply softmax:

```python
import numpy as np

from localml_scholar import TransformerConfig, TransformerLanguageModel
from localml_scholar.losses import softmax_cross_entropy_loss_and_gradient

config = TransformerConfig(
    vocabulary_size=32,
    maximum_context_length=16,
    model_dimension=8,
    number_of_layers=2,
    key_dimension=4,
    value_dimension=4,
    feed_forward_dimension=16,
    dtype=np.float64,
    seed=7,
)
model = TransformerLanguageModel(config)
token_ids = np.array([[1, 4, 2, 7]], dtype=np.int64)
targets = np.array([[4, 2, 7, 3]], dtype=np.int64)

logits = model.forward(token_ids)
loss, grad_logits = softmax_cross_entropy_loss_and_gradient(logits, targets)
model.backward(grad_logits)
```

This demonstrates interface compatibility only; it is not a training loop or
a capability example.

## XOR foundation demonstration

Run the MLP, exact GELU, generalized cross-entropy, manual backward passes, and
Adam together:

```bash
python3 experiments/train_mlp_xor.py
```

The default run writes:

- `outputs/mlp_xor/model.npz` — versioned MLP configuration and parameters
- `outputs/mlp_xor/optimizer.npz` — Adam moments and step counter
- `outputs/mlp_xor/run_summary.json` — configuration, loss, probabilities,
  predictions, history, and round-trip result

The script supports `--seed`, `--steps`, `--learning-rate`, `--hidden-dim`,
`--report-interval`, and `--output-directory`. It is a correctness
demonstration, not a capability benchmark.

## Single-head attention inspection

Run a deterministic token-ID → embedding → causal-attention forward/backward
calculation:

```bash
python3 experiments/inspect_single_head_attention.py
```

The script prints the input, Q/K/V, score, probability, and output shapes along
with raw scaled scores, the causal allowed mask, attention probabilities, a
synthetic scalar loss, and gradient norms. It asserts that every future-token
probability is exactly zero and writes:

```text
outputs/attention_inspection/run_summary.json
```

The output directory is ignored by Git. This experiment does not train a model.

## Pre-norm decoder-block inspection

Run a deterministic token-ID → embedding → one-decoder-block forward/backward
calculation and optimizer step:

```bash
python3 experiments/inspect_pre_norm_decoder_block.py
```

The script prints the embedding, normalization, Q/K/V, attention,
feed-forward, residual, and final output shapes. It verifies exact future-mask
zeros, output-shape preservation, earlier-output independence under a future
token change, finite gradients, and a real decoder-parameter update. It writes:

```text
outputs/decoder_block_inspection/run_summary.json
```

This is an educational correctness inspection, not language-model training or
a performance benchmark.

## Example generation

```python
from localml_scholar.generation import generate_text
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer.load("outputs/bigram_small/tokenizer.json")
model = BigramLanguageModel.load_checkpoint(
    "outputs/bigram_small/best_model.npz"
)

sample = generate_text(
    model,
    tokenizer,
    seed_text="l",
    max_new_tokens=120,
    temperature=0.9,
    seed=7,
)
print(sample)
```

Set `greedy=True` for argmax decoding. A fixed sampling seed makes stochastic
generation reproducible.

## Repository structure

```text
configs/                 Reproducible experiment configurations
data/                    Ignored local raw and processed data
docs/
  architecture.md        Intended retrieval-plus-model system
  roadmap.md             Explicit incremental project milestones
  numerical_precision.md Float32/float64 policy
  audits/                 Evidence-backed milestone audits
  derivations/           Math connected to source functions
experiments/             Reproducible training and inspection scripts
src/localml_scholar/
  data.py                 Local loading, splits, examples, minibatches
  tokenizer.py            Character tokenizer
  losses.py               N-D stable softmax and indexed cross-entropy
  nn/                     Layers, attention, FFN, and one decoder block
  optim/                  SGD, momentum, and Adam
  training/               Gradient clipping and finite differences
  optimizers.py           Milestone 1 compatibility SGD
  generation.py           Autoregressive sampling
  utils.py                Bigram checks and reporting utilities
  models/bigram.py        Trainable bigram model
  models/mlp.py           Two-layer foundation model
  models/transformer_lm.py Decoder-only transformer architecture
tests/                    Unit and numerical correctness tests
outputs/                  Ignored generated run artifacts
```

Mathematical details are in:

- [linear layer](docs/derivations/linear_layer.md)
- [embedding layer](docs/derivations/embedding_layer.md)
- [activations](docs/derivations/activations.md)
- [LayerNorm](docs/derivations/layer_normalization.md)
- [optimizers and clipping](docs/derivations/optimizers.md)
- [generalized gradient checking](docs/derivations/gradient_checking.md)
- [single-head causal attention](docs/derivations/single_head_causal_attention.md)
- [pre-norm decoder block](docs/derivations/pre_norm_decoder_block.md)
- [transformer language model](docs/derivations/transformer_language_model.md)
- [Milestone 1 audit](docs/audits/milestone_1_audit.md)
- [Milestone 2 attention-readiness audit](docs/audits/milestone_2_attention_readiness_audit.md)
- [Milestone 3 decoder-block readiness audit](docs/audits/milestone_3_decoder_block_readiness_audit.md)
- [Milestone 4 transformer-readiness audit](docs/audits/milestone_4_transformer_readiness_audit.md)

## Limitations

- One character of context cannot represent syntax, semantics, paper
  structure, notation, or factual evidence.
- Character tokenization is inefficient for a later language model and has no
  unknown-token policy.
- A validation-only character causes an explicit error because the tokenizer
  is intentionally fit on training text only.
- Minibatches are sampled with replacement; the trainer does not yet implement
  epochs, schedules, resume, or early stopping.
- The bigram experiment does not save optimizer or sampler resume state.
- The module system supports one unmatched training forward per layer and does
  not support shared/reused module instances in one graph.
- Exact GELU uses standard-library scalar `erf`; it prioritizes transparent
  mathematics over throughput.
- LayerNorm is implemented, but RMSNorm is not.
- The transformer has a decoder stack, learned positions, and vocabulary
  logits, but every block still uses one attention head.
- There is no dropout, rotary representation, KV cache, FlashAttention,
  mixed-precision path, weight tying, or optimized kernel.
- There is no transformer training loop, trained transformer checkpoint,
  evaluation, sampling, or generation path.
- Learned positions impose a fixed maximum context length.
- Attention uses \(O(T^2)\) score/probability storage and has no KV cache or
  optimized kernel.
- The fallback corpus exists only for tests and smoke runs. It is not useful
  training data.
- Generated bigram text should not be interpreted as meaningful language.

## Roadmap

The next architectural milestone is multi-head causal self-attention: split
the model dimension across independently validated heads, concatenate their
outputs, apply an output projection, and validate the complete multi-head
backward pass before scaling transformer training. Training, generation, local
PDF parsing, retrieval, licensed ML-paper specialization, evidence-based
evaluation, an application, and measured performance work remain later
milestones.

See [the full roadmap](docs/roadmap.md) and
[the architecture](docs/architecture.md).

## Evidence rule

No performance or capability claim belongs in this project without evidence.
Accuracy claims require a defined dataset and evaluation procedure. Speed
claims require hardware, shapes, precision, batch/context sizes, warm-up,
compilation settings, latency distribution, and numerical tolerance. Claims
about paper-grounded answers require exact source-page evidence. Until those
measurements exist, the project documentation states only what the code is
designed to do and what verified tests demonstrate.
