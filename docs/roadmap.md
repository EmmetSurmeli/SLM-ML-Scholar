# Roadmap

Every milestone should have tests, numerical checks where applicable, a
reproducible small run, documented limitations, and evidence for any
performance or capability claim. A milestone is not complete merely because
its forward pass runs.

## Milestone 1 — Bigram baseline

Status: complete and audited.

- Deterministic character tokenizer and local chronological data pipeline
- Stable softmax, indexed cross-entropy, and manual combined gradient
- Trainable \(V\times V\) bigram matrix and selected-row gradient accumulation
- Named-array SGD with optional weight decay
- Full and deterministic-subset finite-difference checking
- Seeded temperature and greedy generation
- Checkpoints, training history, summary, tests, and mathematical documentation

This is an educational systems baseline, not a paper assistant or an SLM.

## Milestone 2 — Neural-network foundations

Status: complete and verified.

- explicit `Parameter` and deterministic recursive `Module` registration
- dense layers over arbitrary leading dimensions
- token/position-ready embeddings with repeated-index accumulation
- ReLU and exact \(x\Phi(x)\) GELU
- population-variance LayerNorm
- Sequential composition and a two-layer MLP
- coupled-decay SGD, classical momentum, and bias-corrected Adam
- scaled global gradient norm and clipping
- generalized input/parameter finite-difference validation
- explicit float32/float64 policy
- versioned model and optimizer checkpoints
- deterministic XOR integration demonstration

Every primitive has a manual backward pass, hand-computed fixtures, and
finite-difference coverage. RMSNorm was intentionally deferred because
LayerNorm satisfies the transformer normalization prerequisite.

## Milestone 3 — Decoder-only transformer

Next planned step: build the smallest possible decoder-only transformer forward
pass from the validated components, beginning with a single attention head and
exhaustive forward/backward testing.

Then incrementally implement:

- token embeddings
- positional embeddings or rotary embeddings
- causal masking
- scaled dot-product attention
- multi-head attention
- residual connections
- normalization
- feed-forward network
- vocabulary projection
- cross-entropy training
- autoregressive generation
- manual backward passes for all operations

Start with tiny tensors that permit exhaustive finite-difference checks.

## Milestone 4 — Tokenization and training

Add:

- a byte-level or BPE tokenizer implemented from scratch
- checkpointing and resume support
- learning-rate schedules
- batching by context windows
- training and validation curves
- a versioned model configuration system
- TinyStories-scale experimentation where computationally feasible

Report compute, data, wall time, and validation methodology with every run.

## Milestone 5 — Correctness reference

Only after the independent implementation works:

- create an equivalent PyTorch model for validation
- compare forward outputs
- compare gradients
- compare short training trajectories
- report numerical tolerances
- keep PyTorch outside the main implementation and dependency path

The reference must not become the source of the manual implementation.

## Milestone 6 — Local paper retrieval

Add:

- local PDF parsing with a dedicated local library
- section-aware chunking
- page metadata
- equation and figure-caption extraction where feasible
- local vector or lexical retrieval
- citation-preserving context assembly
- no cloud dependencies

Do not implement a PDF parser from raw bytes. Evaluate extraction failures
explicitly, especially multi-column ordering, scanned documents, and equations.

## Milestone 7 — ML-paper specialization

Build a legally usable, versioned training and evaluation corpus from:

- permissively licensed educational material
- open course notes
- author-provided or openly licensed papers
- synthetic instruction examples
- manually created equation explanations
- paper reproduction checklists

Track source URL or local provenance, creator, license, allowed uses, acquisition
date, transformations, and split assignment for every item. Do not
indiscriminately scrape copyrighted textbooks or papers.

Target capabilities:

- notation glossary generation
- equation explanation
- prerequisite identification
- method summaries
- implementation checklists
- experiment extraction
- limitation analysis
- reproduction planning

## Milestone 8 — Evaluation

Create a manually reviewed benchmark with:

- symbol definition
- equation explanation
- derivation correctness
- section comprehension
- experimental-detail extraction
- paper-grounded question answering
- unsupported-claim detection
- reproduction-plan quality

Compare:

- bigram baseline
- transformer without retrieval
- transformer with retrieval
- alternative model sizes
- ablated retrieval strategies
- optional larger external baselines when available

Factuality evaluation must point to exact page-level evidence in the supplied
paper. Record annotator instructions and disagreement.

## Milestone 9 — Local application

Build a lightweight local interface for:

- opening a paper
- viewing extracted sections
- asking questions
- displaying retrieved evidence
- showing page citations
- creating a notation glossary
- producing a reproduction checklist
- saving notes locally

Define the local privacy boundary and storage paths in the interface.

## Milestone 10 — Performance optimization

Profile before optimizing. Potential targets include:

- KV caching
- fused RMSNorm and residual operations
- specialized matrix multiplication
- quantized inference
- CPU cache-aware layouts
- batch-size-one inference latency

Any claim of outperforming PyTorch must report:

- hardware
- model configuration
- tensor shapes
- precision
- batch size
- context length
- warm-up procedure
- compilation settings
- latency distribution
- numerical tolerance
- whether PyTorch eager and `torch.compile` were tested
