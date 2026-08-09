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

## Milestone 7 — Byte and BPE tokenization infrastructure

Status: complete and verified.

- minimal explicit tokenizer contract with deterministic state hashes
- preserved, versioned character tokenizer and legacy schema migration
- fixed 256-symbol UTF-8 byte tokenizer
- independently trained byte-level BPE with no unknown token
- deterministic frequency/lexicographic pair selection
- document-local counting and left-to-right non-overlapping replacement
- ranked BPE encoding and corpus-independent recursive decoding
- explicit `normalization="none"` and strict/replacement UTF-8 policies
- raw chronological split before character/BPE fitting
- corpus content, split, tokenizer, and token-stream identity metadata
- complete tokenizer state in full resumable checkpoints
- tokenizer-aware model bundles, generation, CLI selection, and resume
- exact interrupted/resumed character, byte, and BPE training
- transparent BPE inspection and controlled tokenizer comparison
- bytes-per-token and sampled bits-per-byte reporting

The BPE path is a correctness-oriented reference implementation, not a
large-corpus-optimized tokenizer. Token-level perplexity is not compared across
different token units without byte normalization.

## Milestone 8 — Document ingestion and transparent retrieval

Status: complete and verified.

- exact immutable documents, ordered sections, deterministic chunks, and
  structured citations
- strict UTF-8 plain-text and narrow ATX-Markdown ingestion
- externally supplied page-text PDF adapter without OCR or PDF parsing
- independent retrieval-only lexical terms with original source spans
- exact reconstruction and overlap validation
- independently implemented sparse TF-IDF cosine and BM25
- deterministic ranking, metadata filters, and term-level explanations
- versioned atomic immutable index snapshots with transactional reconstruction
- corpus/configuration change explanations and full-rebuild policy
- project-authored fixture corpus with exact chunk relevance judgments
- Precision@k, Recall@k, MRR, and Hit Rate@k
- build, inspect, and search CLI with no answer generation
- controlled retriever comparison and ingestion inspection experiments

The five-query fixture validates implementation behavior only. It does not
establish general retrieval quality. Direct PDF extraction, semantic search,
reranking, and answer generation were absent at this stage.

## Milestone 9 — Controlled grounded answer generation

Status: complete and verified.

- deterministic answer-oriented BM25/TF-IDF evidence selection
- meaningful-term filtering, range-overlap suppression, and source diversity
- transparent evidence-sufficiency gate with deterministic abstention
- exact tokenizer-aware grounded prompts with quoted-document isolation
- trusted top-passage and sentence-level extractive baselines
- explicit local-checkpoint transformer generation with raw output retention
- strict inline `[C#]` parsing and exact evidence/citation bindings
- claim segmentation, complete citation coverage, and source-range validation
- transparent term, number, identifier, equation-symbol, negation, and exact
  quote diagnostics
- strict generated-answer acceptance and optional extractive fallback
- versioned atomic grounded-answer artifacts
- authored answerability, citation, support, content, causality, code,
  numerical, synonym, and prompt-injection fixtures
- extractive, generative, and controlled method-comparison experiments
- human/JSON answer CLI with evidence inspection and artifact saving

The deterministic extractive path is the trusted baseline. Citation linkage
and lexical support heuristics do not prove semantic truth or entailment. No
useful generative checkpoint is assumed.

## Milestone 10 — Semantic retrieval extension

Status: complete and verified.

- deterministic TF-IDF term-chunk matrices and NumPy truncated SVD
- explicit effective-rank validation and SVD sign canonicalization
- normalized chunk embeddings and exact full-corpus semantic cosine
- query projection with known/OOV term explanations
- maximum-positive-score weighted fusion and reciprocal rank fusion
- bounded lexical/semantic candidate union with component ranks and scores
- transparent feature-based reranking and source-range redundancy penalties
- version 2 semantic index state, hashes, atomic persistence, and reload
- explicit enrichment of recognized 0.8.0/0.9.0 lexical-only indexes
- MAP, graded nDCG, mean relevant rank, no-result, and category metrics
- project-authored synonym, paraphrase, notation, numerical, negation, code,
  acronym, rare-term, broad, multi-source, and misleading-overlap fixtures
- retrieval ablation, descriptive sensitivity, semantic inspection, and
  grounded-answer regression experiments

LSA is a linear corpus-dependent baseline, not a modern neural embedding
model. Similarity does not prove relevance or factual support. BM25 and TF-IDF
remain independently available.

## Milestone 11 — Paper-specific scholarly analysis

Status: complete and verified.

- canonical paper identity and source-verifiable metadata priority
- multi-label section roles with retained ambiguity and unknown sections
- text-only equation blocks, exact offsets, numbering, and conservative
  normalization
- source-linked notation glossaries with conflicts and unresolved symbols
- cited assumptions, claims, methods, procedures, datasets, metrics,
  hyperparameters, experiments, results, ablations, limitations, and references
- deterministic local numbered and unique author/year reference linking
- regular Markdown/delimiter table parsing without visual-layout invention
- equation-aware retrieval reranking with unchanged source identities
- citation-only structured summaries and completeness reporting
- reproduction checklists with missing, ambiguous, conflicting, and risk states
- cross-paper comparison with incompatible-result warnings
- research-gap worksheets that do not claim novelty
- versioned atomic artifacts, section-filtered CLI views, authored fixtures,
  extraction metrics, and deterministic demonstration experiments

There is no OCR, visual PDF recovery, symbolic equation solver, external
metadata lookup, learned scholarly extractor, or literature-wide novelty
verification. Citation provenance does not prove semantic correctness.

## Milestone 11.5 — Real-paper evaluation and failure analysis

Status: complete.

- source/index-bound human-approved benchmark schemas
- deterministic scholarly candidate generation and explicit review decisions
- original 33-question untrusted *Attention Is All You Need* starter (expanded
  with all 80 specified prompts in Milestone 12A)
- retrieval, section, boilerplate, sufficiency, relevance, concept,
  prohibited-claim, citation, completeness, abstention, and audience grades
- one cited structured target with three deterministic audience renderers
- multi-label failure taxonomy and uncertainty-aware likely root causes
- deterministic selective review queues and adjudication records
- per-question, paper, system, failure, review, and regression reports
- atomic evaluation artifacts and source-validated correction export
- extraction-only evaluation without a transformer; explicit checkpoint
  identity for generative evaluation

The automated signals remain heuristics. They do not prove semantic
correctness, and proposed questions or unreviewed corrections are never
treated as gold/training data.

## Deferred — Correctness reference

Only after the independent implementation works:

- create an equivalent PyTorch model for validation
- compare forward outputs
- compare gradients
- compare short training trajectories
- report numerical tolerances
- keep PyTorch outside the main implementation and dependency path

The reference must not become the source of the manual implementation.

## Milestone 12A — Paper Training Lab and grounded dataset curation

Status: complete and verified.

- loopback-only paper corpus, Ask, Benchmarks, Review, Corrections, Dataset,
  and Evaluation workflows
- deterministic 40–80 question generation per paper and an expanded untrusted
  *Attention Is All You Need* pool containing all 80 specified prompts
- arbitrary user-authored and multi-paper questions
- adaptive instruction profiles inferred from prompts, recent turns, opt-in
  preferences, and explicit overrides
- separate evidence scope and presentation interpretation
- provenance-labelled facts and derivations (`paper_explicit`,
  `mathematical_inference`, `external_knowledge`, `uncertain`)
- proposed-only automatic questions, variations, and correction suggestions
- explicit human approval before dataset inclusion
- paper-level train/validation/test splits, cross-paper coalescing, and prompt-
  variation leakage checks
- dataset diversity diagnostics and 100/300/600 review-throughput targets
- CLI batch generation, baseline runs, approved-only export, and reports

This milestone curates data but performs no model training or fine-tuning. It
does not call an external LLM, download papers, or automatically approve gold
facts.

The next recommended milestone is:

> Milestone 12B: grounded instruction tuning of the custom local transformer using Codex-curated and/or human-verified grounded examples, evaluated on completely held-out papers with deterministic extractive answering retained as the trusted baseline.

## Milestone 12A.1 — Confidence-gated auto-review and audit sampling

Status: complete in package 1.2.1, pending real-corpus calibration.

The Paper Training Lab now performs a second pass with three explicitly
correlated deterministic reviewer configurations, 16 non-compensating gates,
mandatory human routes, correction revalidation, provenance/circularity
checks, deterministic audit sampling, calibration state, duplicate clusters,
and provenance-weighted trust-tier exports. Automatic approval is disabled by
default and cannot be enabled until at least 50 paired human outcomes meet the
policy and a human explicitly enables it.

Before Milestone 12B, calibrate on 50–100 examples, inspect every mandatory-risk
route, audit at least 10%, confirm held-out paper isolation, and build a legally
usable corpus with preserved source licenses. Codex approval remains distinct
from human gold.

## Milestone 12A.2 — Calibration Lab and bulk-readiness policy

Status: complete in package 1.2.2; real-corpus human calibration remains user
work.

The local site now selects representative calibration samples, preserves
historical reviews through linked append-only reruns, presents one-click and
keyboard validation cards, reports false-approval-first metrics, and explains
every readiness failure. Bulk approval remains disabled until at least 50
human pairs and 20 positive candidates meet all metric and integrity gates and
a human explicitly enables it. Calibration approval is not training approval.
Audited automated examples retain their producer provenance.

The optional acquisition queue records local paper suggestions but performs no
network request or download. Expand from the initial workflow corpus to 10–15,
then 20–30 legally usable papers only after calibration behavior is understood.

## Milestone 12A.3 — Fully automated Codex-reviewed corpus curation

Status: complete in package 1.2.3; a real autonomous corpus run remains an
explicit user action.

The Paper Training Lab can now ingest a selected local corpus, generate 40–80
questions per paper, answer from indexed evidence, run five schema-constrained
Codex passes, repair evidence/answers at most twice, reject uncertainty,
deduplicate and balance accepted records, enforce paper-level splits, and
export `codex_curated` data without individual human approval. Run state and
stage IDs are persisted for restart. Test-paper questions are evaluation-only;
their answers and corrections never enter training data.

This automated trust class is not human gold. Separate passes reduce but do not
eliminate correlated or systematic reviewer error. The deterministic extractive
answer remains the trusted baseline, optional human audit remains available,
and no model is trained in this milestone. Test the workflow on 10–15 diverse
papers before expanding to 20–30.

## Milestone 12B — Grounded scholarly explanation training

Status: planned.

Exact next milestone:

> Milestone 12B: grounded instruction tuning of the custom local transformer
> using Codex-curated and/or human-verified grounded examples, evaluated on
> completely held-out papers with deterministic extractive answering retained
> as the trusted baseline.

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

Derive instruction examples from the immutable structured artifacts, retain
exact evidence and licensing metadata, train only the project transformer, and
evaluate generated explanations against deterministic extraction and citation
validation. The deterministic path remains the trusted baseline and every
generated claim must retain or recover exact evidence.

## Milestone 13 — Evaluation

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

## Milestone 14 — Local application

Status: partially implemented early as the narrow localhost-only Review Lab.

The implemented slice supports local paper upload, exact source inspection,
deterministic scholarly artifacts, cited extractive questions, and structured
feedback snapshots for later Codex/manual review. It deliberately does not
perform automatic training from feedback.

Complete the broader lightweight local interface with:

- opening a paper
- viewing extracted sections
- asking questions
- displaying retrieved evidence
- showing page citations
- creating a notation glossary
- producing a reproduction checklist
- saving notes locally

Define the local privacy boundary and storage paths in the interface.
Add rendered-PDF navigation, citation-to-page linking, paper removal/export,
review adjudication, and explicitly licensed dataset curation before marking
the milestone complete.

## Milestone 15 — Performance optimization

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
