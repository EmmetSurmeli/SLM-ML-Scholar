# Multi-head causal self-attention

This document derives the fused multi-head attention implementation in
`localml_scholar.nn.attention.MultiHeadCausalSelfAttention`. NumPy supplies
array storage and matrix multiplication; every backward operation below is
implemented explicitly.

## 1. Why multiple heads?

A single attention head creates one query-key similarity geometry and one
value representation. Multiple heads create \(H\) independently parameterized
subspaces. A receiving token can therefore assign different attention
distributions in different heads—for example, one head may weight a nearby
token while another weights an earlier delimiter. This mechanism provides
representational capacity; it does not guarantee that trained heads will learn
distinct or human-interpretable roles.

## 2. Dimensions and fused layout

Let \(B\) be batch size, \(T\) sequence length, \(D\) model dimension, \(H\)
head count, \(d_k\) per-head query/key dimension, and \(d_v\) per-head value
dimension.

| Tensor | Shape |
| --- | --- |
| \(X\) | \(B\times T\times D\) |
| \(W_Q,W_K\) | \(D\times(Hd_k)\) |
| \(W_V\) | \(D\times(Hd_v)\) |
| \(Q_{\mathrm{flat}},K_{\mathrm{flat}}\) | \(B\times T\times(Hd_k)\) |
| \(V_{\mathrm{flat}}\) | \(B\times T\times(Hd_v)\) |
| \(Q,K\) | \(B\times H\times T\times d_k\) |
| \(V\) | \(B\times H\times T\times d_v\) |
| \(S,A\) | \(B\times H\times T\times T\) |
| \(O_{\mathrm{heads}}\) | \(B\times H\times T\times d_v\) |
| \(O_{\mathrm{cat}}\) | \(B\times T\times(Hd_v)\) |
| \(W_O\) | \((Hd_v)\times D\) |
| \(Y\) | \(B\times T\times D\) |

The implementation uses fused Q/K/V projections. A flat query is reshaped

\[
(B,T,Hd_k)\rightarrow(B,T,H,d_k)
\]

and transposed to

\[
(B,H,T,d_k).
\]

The key and value tensors use the same transformation. Head outputs reverse
this exact operation:

\[
(B,H,T,d_v)\rightarrow(B,T,H,d_v)\rightarrow(B,T,Hd_v).
\]

The order is head-major within the final concatenated feature axis: all
features for head 0, then all features for head 1, and so on. Backward uses
the exact inverse reshape and transpose, so no coordinate is silently
permuted.

This project treats \(d_k\) and \(d_v\) as explicit per-head dimensions.
Consequently, \(D\) need not be divisible by \(H\). Total projection widths
are \(Hd_k\) and \(Hd_v\), and the mandatory output projection maps the
concatenated value width back to \(D\).

## 3. Forward pass

### Fused affine projections

\[
Q_{\mathrm{flat}}=XW_Q+b_Q,\qquad
K_{\mathrm{flat}}=XW_K+b_K,\qquad
V_{\mathrm{flat}}=XW_V+b_V.
\]

After splitting into heads, each head \(h\) has its own slices
\(Q^{(h)},K^{(h)},V^{(h)}\).

### Per-head scaled scores

\[
S^{(h)}
=
\frac{Q^{(h)}(K^{(h)})^\top}{\sqrt{d_k}}.
\]

The product is batched over both \(B\) and \(H\). Dividing by
\(\sqrt{d_k}\) keeps score variance from growing proportionally to \(d_k\)
under common independence assumptions, which reduces avoidable softmax
saturation.

### Shared causal mask and independent softmax

The boolean mask has shape \(1\times1\times T\times T\):

\[
M_{ij}=[j\le i].
\]

`True` means allowed. Broadcasting reuses this relation across every batch and
head without allocating \(BH\) copies. For each \((b,h,i)\) row,

\[
A_{bhij} =
\begin{cases}
\displaystyle
\frac{\exp(S_{bhij}-m_{bhi})}
{\sum_{k:M_{ik}}\exp(S_{bhik}-m_{bhi})},
&M_{ij},\\[8pt]
0,&\neg M_{ij},
\end{cases}
\]

where \(m_{bhi}\) is the maximum over allowed entries only. Each head
normalizes independently along its own final key-position axis. Blocked
probabilities are exactly zero.

### Value aggregation, concatenation, and output projection

\[
O^{(h)}=A^{(h)}V^{(h)}.
\]

After concatenation,

\[
O_{\mathrm{cat}}
=
\operatorname{Concat}\left(
O^{(1)},\ldots,O^{(H)}
\right),
\]

and the canonical output is

\[
Y=O_{\mathrm{cat}}W_O+b_O.
\]

The output projection restores the model dimension, allowing exact-shape
residual addition in a decoder block.

## 4. Backward pass

Let \(G_Y=\partial L/\partial Y\). The existing manual `Linear.backward`
for the output projection computes

\[
G_{O_{\mathrm{cat}}}=G_YW_O^\top,
\quad
G_{W_O}=O_{\mathrm{cat,flat}}^\top G_{Y,\mathrm{flat}},
\quad
G_{b_O}=\sum G_Y.
\]

The gradient is split back to \(G_{O^{(h)}}\) using the inverse concatenation
layout.

For each head,

\[
O^{(h)}=A^{(h)}V^{(h)}
\]

gives

\[
G_A^{(h)}=G_O^{(h)}(V^{(h)})^\top,
\qquad
G_V^{(h)}=(A^{(h)})^\top G_O^{(h)}.
\]

For one softmax row, the vector-Jacobian product is

\[
G_S
=
A\odot
\left(
G_A-\sum_j G_{A,j}A_j
\right).
\]

The sum is independent for every \((B,H,T)\) row. Since blocked coordinates
have \(A=0\), their score gradients are exactly zero; the implementation also
applies the mask explicitly.

With \(c=1/\sqrt{d_k}\) and \(S=cQK^\top\),

\[
G_Q=cG_SK,
\qquad
G_K=cG_S^\top Q.
\]

The per-head gradients are transposed and reshaped back to their fused flat
layouts. The three fused affine backward passes accumulate their parameter
gradients and produce three branches to the shared input:

\[
G_X
=
G_X^{(Q)}+G_X^{(K)}+G_X^{(V)}.
\]

No automatic-differentiation graph performs this sum.

## 5. Causality

For every head and query position \(i\),

\[
O_i^{(h)}
=
\sum_{j=0}^{i}A_{ij}^{(h)}V_j^{(h)}.
\]

The probability row also depends only on keys \(0,\ldots,i\) and the
token-wise query at \(i\). Therefore changing \(X_j\) for \(j>i\) cannot alter
head output \(i\), the concatenation at \(i\), or the token-wise output
projection at \(i\). Concatenating heads does not mix sequence positions, so
it cannot break causality.

Tests verify this operationally for one, two, and four heads and verify that a
loss on an earlier output produces zero input gradient at prohibited future
positions.

## 6. One-head compatibility

When \(H=1\), fused projection shapes reduce to the legacy single-head shapes:

\[
D\times(Hd_k)=D\times d_k,\qquad
D\times(Hd_v)=D\times d_v.
\]

The parameter names, initialization order, score scale, mask, softmax,
aggregation, and output projection are also identical. The only internal
difference is a singleton head axis. Tests copy every legacy parameter into
the fused module and require exact equality for outputs, input gradients,
score gradients, and every parameter gradient.

Version 0.6.0 loaders migrate recognized 0.5.0 model-only and 0.5.1 full
training checkpoints by adding `number_of_heads=1` in memory. Unknown versions
or incompatible shapes remain hard errors.

## 7. Numerical and memory considerations

- The valid-entry maximum stabilizes softmax even when logits have very large
  magnitude.
- Masked probabilities and masked score gradients are explicit zeros.
- Float64 is used for centered finite differences; ordinary training may use
  float32 without a silent downcast.
- `transpose` can produce a noncontiguous NumPy view. Subsequent `reshape`
  operations are allowed to return a view or allocate a copy; the
  implementation depends only on documented axis order, never on contiguity or
  shared storage.
- Score computation costs
  \[
  O(BHT^2d_k),
  \]
  while value aggregation costs \(O(BHT^2d_v)\). The implementation
  materializes \(BHT^2\) scores and probabilities, so attention storage is
  quadratic in sequence length and linear in head count.
- Q/K/V projection widths and parameter count grow with \(H\) when per-head
  dimensions are held constant. With all biases enabled, attention has
  \(H(2Dd_k+2Dd_v)+H(2d_k+d_v)+D\) scalar parameters.
- The implementation is optimized for auditability, not kernel count or
  throughput. It has no dropout, padding mask, KV cache, or fused low-level
  kernel.

## 8. Source and validation map

| Stage | Source | Principal tests |
| --- | --- | --- |
| fused projections and head split | `MultiHeadCausalSelfAttention.forward` | shape, deterministic initialization, and independent NumPy reference tests |
| shared causal mask | `causal_attention_mask` plus broadcast in `forward` | exact-zero and causality tests |
| per-head stable softmax | `masked_softmax` | extreme-logit and row-sum tests |
| value aggregation and concatenation | `_join_heads` and `forward` | hand-computed two-head fixture |
| all backward equations | `MultiHeadCausalSelfAttention.backward` | exhaustive input and parameter finite differences |
| singleton-head equivalence | fused module with \(H=1\) | exact legacy forward/backward comparison |
| decoder/model integration | `PreNormDecoderBlock`, `TransformerLanguageModel` | full-block finite differences, generation, checkpoints, and resume |
| educational inspection | `experiments/inspect_multi_head_attention.py` | experiment summary integration test |

The tiny exhaustive checks cover input coordinates, all fused Q/K/V weights
and biases, output-projection weight and bias, both \(H=1\) and \(H=2\), and
the full two-head decoder block.
