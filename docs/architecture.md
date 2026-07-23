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

## Milestone 1 boundaries

The current package establishes small interfaces that later components can
extend:

- `tokenizer.py` owns text/token conversion and vocabulary persistence.
- `data.py` owns local loading, chronological splits, examples, and seeded
  minibatches.
- `losses.py` owns explicit numerical forward/backward primitives.
- `models/` owns parameters, forward computation, gradient accumulation,
  modes, and checkpoints.
- `optimizers.py` updates named parameter arrays and validates gradients.
- `generation.py` owns autoregressive sampling independently of training.
- `utils.py` owns reproducibility, reporting math, and gradient checking.
- `experiments/` composes library pieces into reproducible runs but is not
  imported by the reusable model.

For the bigram model, the parameter and gradient mappings each contain one
matrix named `weights`. A future transformer will expose many named arrays
through the same conceptual boundary. Optimizer state can then grow without
putting update rules inside layers.

## Current model

Milestone 1 predicts the next character from only the immediately preceding
character:

\[
P(x_{t+1}\mid x_t).
\]

It has no representation of words, equations, passages, papers, or questions.
Its value is as an end-to-end correctness baseline for tokenization, dataset
isolation, loss math, manual gradients, optimization, generation, persistence,
and experiment bookkeeping.

## Future module constraints

The core neural-network path will continue to use Python, NumPy array storage
and matrix multiplication, and standard-library code. It will not depend on
automatic differentiation, framework layers, framework losses, framework
optimizers, pretrained-model APIs, or external tokenizers. PDF parsing,
plotting, tests, and the application will stay outside that core and may use
focused third-party libraries when their value is clear.

