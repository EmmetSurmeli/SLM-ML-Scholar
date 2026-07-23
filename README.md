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

## Current status: Milestone 2

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

The project is **still not an SLM and not yet a research-paper assistant**. The
bigram cannot understand a paper, explain an equation, retrieve evidence, or
maintain context beyond one character. The MLP is a synthetic integration
fixture, not a language model.

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
XOR learning, dtype policy, cache misuse, and malformed inputs.

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
```

Ruff reported no lint or formatting errors, and pytest reported `90 passed`.
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
  roadmap.md             Ten explicit project milestones
  numerical_precision.md Float32/float64 policy
  audits/                 Evidence-backed milestone audits
  derivations/           Math connected to source functions
experiments/             CLI training composition
src/localml_scholar/
  data.py                 Local loading, splits, examples, minibatches
  tokenizer.py            Character tokenizer
  losses.py               N-D stable softmax and indexed cross-entropy
  nn/                     Parameters, modules, layers, and composition
  optim/                  SGD, momentum, and Adam
  training/               Gradient clipping and finite differences
  optimizers.py           Milestone 1 compatibility SGD
  generation.py           Autoregressive sampling
  utils.py                Bigram checks and reporting utilities
  models/bigram.py        Trainable bigram model
  models/mlp.py           Two-layer foundation model
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
- [Milestone 1 audit](docs/audits/milestone_1_audit.md)

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
- Milestone 2 has no attention, causal masking, residual block, sequence model,
  or language-model training path.
- The fallback corpus exists only for tests and smoke runs. It is not useful
  training data.
- Generated bigram text should not be interpreted as meaningful language.

## Roadmap

The next milestone is the smallest possible decoder-only transformer forward
pass built from the validated components, beginning with a single attention
head and exhaustive forward/backward testing. It will be expanded only after
that minimal path is numerically correct. Local PDF parsing, retrieval,
licensed ML-paper specialization, evidence-based evaluation, an application,
and measured performance work remain later milestones.

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
