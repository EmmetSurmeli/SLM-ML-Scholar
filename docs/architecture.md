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

Milestones 1 and 2 establish interfaces that later components can extend:

- `tokenizer.py` owns text/token conversion and vocabulary persistence.
- `data.py` owns local loading, chronological splits, examples, and seeded
  minibatches.
- `losses.py` owns explicit numerical forward/backward primitives.
- `nn/` owns explicit `Parameter`/`Module` foundations, initializers, layers,
  activations, normalization, and sequential composition.
- `optim/` owns optimizers over identity-keyed parameters.
- `training/` owns global gradient clipping and generalized finite differences.
- `models/` composes primitives into checkpointable models.
- `optimizers.py` remains the Milestone 1 named-array SGD compatibility path.
- `generation.py` owns autoregressive sampling independently of training.
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

See `docs/numerical_precision.md` for the project-wide float32/float64 policy.

## Current model

The only language model still predicts the next character from the immediately
preceding character:

\[
P(x_{t+1}\mid x_t).
\]

It has no representation of words, equations, passages, papers, or questions.
Its value is as an end-to-end correctness baseline for tokenization, dataset
isolation, loss math, manual gradients, optimization, generation, persistence,
and experiment bookkeeping.

Milestone 2 adds reusable mathematical components and a two-layer MLP. The XOR
experiment demonstrates composition and manual backward correctness on four
synthetic examples; it is not a language model and does not add paper
understanding.

## Future module constraints

The core neural-network path will continue to use Python, NumPy array storage
and matrix multiplication, and standard-library code. It will not depend on
automatic differentiation, framework layers, framework losses, framework
optimizers, pretrained-model APIs, or external tokenizers. PDF parsing,
plotting, tests, and the application will stay outside that core and may use
focused third-party libraries when their value is clear.
