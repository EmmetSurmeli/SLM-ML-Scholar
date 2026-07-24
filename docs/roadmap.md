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

## Milestone 3 — Single-head causal self-attention

Status: complete and verified.

- immutable broadcastable causal masks with explicit allowed/blocked semantics
- stable masked softmax and an explicit vector-Jacobian product
- token-wise query, key, and value projections
- scaled dot-product scores and causal value aggregation
- optional output projection
- manual gradients for every matrix operation and all shared-input paths
- exhaustive float64 finite differences for inputs and parameters
- hand-calculated forward fixture
- operational forward and backward causality checks
- float32 checkpoint, optimizer, and clipping integration
- deterministic embedding-to-attention inspection experiment

This is one educational attention head, not a transformer block or language
model.

## Milestone 4 — Minimal decoder block

Status: complete and verified.

- exact-shape residual forward and branch-gradient primitives
- reusable position-wise `Linear -> GELU -> Linear` feed-forward module
- pre-normalized attention residual
- pre-normalized feed-forward residual
- optional attention output projection with explicit compatibility validation
- manual backward flow through both residual branch splits and accumulations
- stable deterministic nested parameter names and versioned checkpoints
- controlled identity and attention-only fixtures
- forward and backward causality tests
- exhaustive float64 input and parameter finite differences
- float32 execution, three optimizer integrations, and gradient clipping
- deterministic embedding-to-decoder inspection and one optimizer step

This is one educational single-head decoder block, not a stacked transformer
or language model.

## Milestone 5 — Decoder-only transformer language model

### Part 1: architecture

Status: complete and verified.

- validated immutable transformer configuration
- learned token embeddings
- learned absolute position embeddings with batch gradient accumulation
- exact-shape embedding composition
- arbitrary positive stack of independent decoder blocks
- final LayerNorm
- untied vocabulary projection producing \((B,T,V)\) logits
- direct N-dimensional cross-entropy compatibility
- explicit reverse-order manual backward
- deterministic top-level child seeding
- deterministic atomic state dictionaries and versioned checkpoints
- multi-block forward/backward causality
- exhaustive parameter finite differences for a tiny model
- float32 execution and optimizer/clipping compatibility

Part 1 deliberately adds no training loop, transformer generation, sampling,
evaluation, multi-head attention, or performance optimization.

### Part 2: deterministic training and generation

Status: complete and verified.

- validated immutable transformer training configuration
- restorable random-with-replacement sequence batching
- explicit full-model training steps with N-D cross-entropy
- coupled L2 weight decay and global norm clipping
- fixed-seed train/validation evaluation isolated from training RNG
- recursive no-cache inference lifecycle with strict training guards
- greedy, temperature, and stable top-\(k\) autoregressive generation
- atomic model, optimizer, and full training checkpoints
- exact interrupted-versus-uninterrupted Adam resumption
- deterministic tiny-pattern overfitting
- configurable character-corpus training and resume CLI

At its 0.5.1 verification point, the model remained a tiny, character-level,
single-head educational transformer. Part 2 added no retrieval or paper
assistant.

## Milestone 6 — Multi-head causal self-attention

Status: complete and verified.

- fused Q/K/V projections with explicit per-head dimensions
- explicit `(B,T,H*d) -> (B,H,T,d)` layouts and inverse concatenation
- shared causal mask with independent per-head stable softmax
- mandatory output projection back to the model dimension
- complete manual backward through all heads and fused projections
- hand-computed and independent per-head reference fixtures
- exhaustive float64 input, parameter, and decoder-block finite differences
- exact one-head equivalence to the original implementation
- forward/backward causality for multiple head counts
- decoder, language-model, training, evaluation, and generation integration
- explicit 0.5.0 model and 0.5.1 training checkpoint migration
- exact interrupted/resumed multi-head training
- deterministic inspection and controlled head-count comparison experiments

The implementation is educational and unoptimized. It materializes quadratic
attention tensors and makes no claim that more heads improve quality.

## Milestone 7 — Tokenization and scaled training

Improve the language-modeling foundation through tokenizer and corpus
infrastructure, beginning with a byte-level or independently implemented BPE
tokenizer, padding-aware batching if needed, and controlled training
experiments before adding paper retrieval.

The tokenizer must remain independently implemented and versioned. Corpus
provenance, licensing, encoding policy, train/validation isolation, unknown or
byte handling, and checkpoint compatibility must be explicit before scaling
data or context lengths. Do not add retrieval in this milestone.

Report compute, data, wall time, and validation methodology with every run.

## Milestone 8 — Correctness reference

Only after the independent implementation works:

- create an equivalent PyTorch model for validation
- compare forward outputs
- compare gradients
- compare short training trajectories
- report numerical tolerances
- keep PyTorch outside the main implementation and dependency path

The reference must not become the source of the manual implementation.

## Milestone 9 — Local paper retrieval

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

## Milestone 10 — ML-paper specialization

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

## Milestone 11 — Evaluation

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

## Milestone 12 — Local application

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

## Milestone 13 — Performance optimization

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
