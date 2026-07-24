# Tokenization and byte-pair encoding

This document defines the token units used by LocalML Scholar 0.7.0. The
implementation is independent: it uses Python, NumPy arrays for returned token
IDs, and the standard library. It does not use an external tokenizer package.

## 1. Why tokenization exists

A language model consumes vocabulary indices, not Python strings. A tokenizer
defines a deterministic mapping

\[
\operatorname{encode}: \text{text}\rightarrow(t_1,\ldots,t_N),
\qquad 0\leq t_i<V,
\]

and a corresponding decoder. The token-to-ID mapping is part of model identity:
row \(i\) of the input embedding and column \(i\) of the vocabulary projection
refer to tokenizer ID \(i\). Changing that mapping while keeping model weights
would silently change the model.

`Tokenizer` in `src/localml_scholar/tokenizer.py` is the minimal contract. It
exposes `encode`, `decode`, `token_bytes`, `vocabulary_size`, complete versioned
state, and a canonical SHA-256 state hash.

## 2. Character tokenization

`CharacterTokenizer` treats each Python string element as one Unicode code
point. Fitting sorts the distinct training code points, so IDs are reproducible:

\[
\mathcal V=\operatorname{sort}(\operatorname{set}(x_{\text{train}})).
\]

It preserves newlines, tabs, spaces, and repeated whitespace exactly. A
validation character outside the training vocabulary is an error; there is no
unknown token. Unicode-equivalent visual strings may differ. For example, a
precomposed `é` and the two-code-point sequence `e` plus a combining accent
receive different encodings when both exist. No Unicode normalization occurs.

Character tokenization is transparent but typically produces long sequences,
and a training-only vocabulary cannot represent unseen validation code points.

## 3. Byte tokenization

`ByteTokenizer` first applies Python's strict UTF-8 encoder:

\[
x \xrightarrow{\text{UTF-8}} (b_1,\ldots,b_N),
\qquad b_i\in\{0,\ldots,255\}.
\]

Each byte value is its token ID, so \(V=256\), fitting is unnecessary, and every
valid Python string is encodable without an unknown token. ASCII code points
usually use one byte; many mathematical symbols and non-Latin code points use
multiple bytes.

`encode` returns one-dimensional `np.int64` IDs. `decode` reconstructs bytes and
uses strict UTF-8 by default. Arbitrary generated IDs may be invalid UTF-8, so
user-facing generation explicitly requests `errors="replace"`; this display
policy is never silent. `encode_bytes` and `decode_to_bytes` provide exact
byte-oriented inspection.

## 4. BPE training

`BytePairTokenizer` starts each document as raw byte IDs. For separate token
sequences \(s^{(1)},\ldots,s^{(M)}\), adjacent-pair frequency is

\[
C(a,b)=\sum_{m=1}^{M}\sum_{i=1}^{|s^{(m)}|-1}
\mathbf 1[(s_i^{(m)},s_{i+1}^{(m)})=(a,b)].
\]

The sum is document-local: no pair spans two documents. At every rank, the
eligible pair has frequency at least `minimum_pair_frequency`. Selection uses:

1. greatest frequency;
2. lexicographically smallest integer pair \((a,b)\) on a tie.

Thus training uses no random state and is deterministic. A selected pair
\((a,b)\) receives ID \(256+r\) at rank \(r\). Replacement scans left to right
and is non-overlapping. With \((a,a)\mapsto c\),

```text
[a, a, a] -> [c, a]
```

not an overlapping two-merge result.

Training stops at the target vocabulary size, the optional maximum merge count,
or when no pair reaches the minimum frequency. With \(R\) rules,

\[
V=256+R.
\]

`count_adjacent_pairs`, `replace_pair_non_overlapping`, and
`BytePairTokenizer.train_with_trace` implement these stages directly.

## 5. BPE encoding

New text starts as UTF-8 bytes. For the current sequence, encoding finds all
adjacent pairs present in the learned merge table, selects the available rule
with the smallest rank, replaces every non-overlapping occurrence, and repeats
until no rule is available.

The repeated availability check matters because an earlier merge can create a
pair consumed by a later merge. For rules

```text
rank 0: (a, b) -> 256
rank 1: (256, c) -> 257
rank 2: (257, 256) -> 258
```

`abcab` becomes `[258]`. Applying rules without respecting the current sequence
and merge priority can produce a different tokenization.

The reference implementation favors clarity over speed. It does not mutate the
merge table and never merges across separate `encode` calls.

## 6. BPE decoding

Every learned token is an acyclic binary tree whose children have smaller IDs:

\[
\operatorname{expand}(t)=
\begin{cases}
[t], & t<256,\\
\operatorname{expand}(l_t)\Vert\operatorname{expand}(r_t), & t\geq256.
\end{cases}
\]

The constructor validates child ordering, contiguous IDs and ranks, and
duplicate pairs, then precomputes byte expansions. Decoding concatenates those
expansions and applies the same explicit UTF-8 error policy as byte
tokenization. It needs no training corpus. For every valid string \(x\),

\[
\operatorname{decode}(\operatorname{encode}(x))=x.
\]

## 7. Vocabulary size and model size

For model dimension \(D\) and untied input/output vocabulary matrices, changing
the vocabulary by \(\Delta V\) changes the two weight matrices by approximately

\[
2D\Delta V
\]

parameters, plus \(\Delta V\) output biases when enabled. Larger tokens may
reduce sequence length, which indirectly reduces the quadratic attention work
for a fixed raw byte span. A larger vocabulary also makes the vocabulary
projection more expensive. Neither tradeoff is universally superior.

## 8. Compression measures

For a text with \(N_{\text{bytes}}\) UTF-8 bytes and \(N_{\text{tokens}}\)
tokens, the comparison experiment reports

\[
\text{tokens per byte} =
\frac{N_{\text{tokens}}}{N_{\text{bytes}}},
\qquad
\text{bytes per token} =
\frac{N_{\text{bytes}}}{N_{\text{tokens}}}.
\]

These describe token sequence length only; they are not file-compression
algorithms and do not include the size of the merge table.

## 9. Perplexity comparability

Mean token cross-entropy and token perplexity depend on the token unit. A byte
tokenizer predicts more units than a BPE tokenizer for the same text, so raw
token perplexities are not directly comparable.

For sampled target tokens representing exactly \(N_{\text{bytes}}\) bytes and
total negative log-likelihood

\[
\operatorname{NLL}=-\sum_i \log p(t_i\mid t_{<i}),
\]

the experiment reports

\[
\operatorname{BPB}=
\frac{\operatorname{NLL}}{N_{\text{bytes}}\log 2}.
\]

Each token's NLL is assigned to the exact byte expansion returned by
`token_bytes`. The experiment also records the evaluated token and byte counts.
Its BPB is a fixed-seed sampled estimate, not a full-corpus benchmark.

## 10. Split isolation and leakage

The controlled language-model policy is:

1. split raw text chronologically by Python code-point index;
2. fit a character vocabulary or BPE merges on training text only;
3. encode the two raw splits independently.

Byte tokenization has no fitting step. Because train and validation streams are
separate, neither a token nor a shifted target crosses the boundary.

Using validation characters to construct a vocabulary or validation pairs to
learn BPE rules leaks held-out distribution information even though it does not
use target probabilities. Production tokenizers may instead be trained on a
separately governed broad corpus; that is a different experimental protocol.

## 11. Normalization

The only supported policy is `normalization = "none"`. The implementation does
not lowercase, strip, normalize NFC/NFD, collapse whitespace, rewrite quotes,
or change newlines. Exact preservation makes identity and provenance easier to
audit, but visually equivalent Unicode representations may tokenize
differently.

## 12. Serialization and checkpoint compatibility

Tokenizer JSON has these top-level fields:

```text
tokenizer_format_version
tokenizer_type
normalization
vocabulary_size
state
metadata
```

Canonical JSON produces a stable state hash. Tokenizer files use a flushed,
`fsync`-completed sibling temporary file followed by `os.replace`; loading
constructs and validates a candidate before changing an existing object.
Pickle is never used.

Full training checkpoints store complete tokenizer state and hash. Current
model-only checkpoints can store the same state for generation. Loading
validates the tokenizer vocabulary against the model. Recognized 0.5/0.6
legacy character metadata migrates in memory without rewriting the source
checkpoint or remapping an ID.

## 13. Complexity

The transparent BPE trainer recounts pairs and rebuilds sequences after each
merge. For total current sequence length \(N_r\) at rank \(r\), its work is
roughly

\[
O\left(\sum_{r=0}^{R-1}N_r\right).
\]

Encoding likewise scans the current sequence to discover the next ranked rule
and scans again to replace it. This is intentionally not an optimized
linked-list/heap tokenizer and can be slow for large corpora or merge tables.

## 14. Source and test mapping

- Interface, character/byte/BPE implementations, training, merge validation,
  encoding, decoding, hashing, and migration:
  `src/localml_scholar/tokenizer.py`
- Atomic JSON replacement: `src/localml_scholar/serialization.py`
- Raw split-before-fit, UTF-8 loading, and corpus metadata:
  `src/localml_scholar/data.py`
- Tokenizer-aware training state:
  `src/localml_scholar/training/transformer.py`
- Model/tokenizer bundles: `src/localml_scholar/models/transformer_lm.py`
- User-facing decode policy: `src/localml_scholar/generation.py`
- Hand-checked byte/BPE behavior:
  `tests/test_byte_tokenizer.py`, `tests/test_bpe_tokenizer.py`
- Split and leakage behavior: `tests/test_tokenizer_corpus.py`
- Training, generation, checkpoint, and exact-resume integration:
  `tests/test_tokenizer_transformer_integration.py`
- Controlled metrics and inspection:
  `experiments/compare_tokenizers.py`,
  `experiments/inspect_bpe_tokenizer.py`
