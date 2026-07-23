# Embedding layer

## Notation and forward

For vocabulary size \(V\) and embedding dimension \(D\), the trainable table is

\[
E\in\mathbb{R}^{V\times D}.
\]

Integer IDs may have any shape

\[
X\in\{0,\ldots,V-1\}^{s_1\times\cdots\times s_k}.
\]

The lookup is

\[
Y_{i_1,\ldots,i_k,:}=E_{X_{i_1,\ldots,i_k},:},
\]

with output shape \((s_1,\ldots,s_k,D)\). `Embedding.forward` checks that IDs
are integers, non-empty, non-negative, and below \(V\), then caches a copy of
the IDs in training mode.

## Backward

There is no derivative with respect to the discrete integer IDs. Let

\[
G=\frac{\partial L}{\partial Y}.
\]

For table row \(v\),

\[
\boxed{
\frac{\partial L}{\partial E_{v,:}}
=
\sum_{i_1,\ldots,i_k:
X_{i_1,\ldots,i_k}=v}
G_{i_1,\ldots,i_k,:}
}.
\]

Repeated IDs must add into the same table row. Buffered advanced-index `+=`
does not guarantee that behavior, so `Embedding.backward` uses
`np.add.at(weight.grad, indices, grad_output)` and returns `None` for the
integer input gradient.

## Initialization and numerical considerations

Rows are initialized from a zero-mean normal distribution with standard
deviation \(1/\sqrt D\), using an explicit seeded RNG. The table and upstream
gradient must share an explicitly selected `float32` or `float64` dtype.
Indices are copied into the cache so later caller mutation cannot change where
gradients accumulate.

## Source and tests

- Source: `nn/embedding.py`
- Lookup values, arbitrary leading shape, invalid IDs, and repeated
  accumulation: `tests/test_embedding.py`
- Exhaustive table finite differences with repeated IDs:
  `test_embedding_weight_passes_gradient_check_with_repeated_ids`
