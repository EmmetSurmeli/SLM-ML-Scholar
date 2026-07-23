# Single-head causal self-attention

This document derives the forward and backward computations implemented in
`localml_scholar.nn.attention`. The implementation is one educational,
unoptimized attention head. It does not implement a transformer block or
multi-head attention.

## 1. Motivation

Self-attention lets each token select information from tokens in the same
sequence:

- a **query** describes what the receiving token is looking for;
- a **key** describes what each source token offers for matching;
- a **value** contains the information contributed by a source token;
- query-key similarity determines the weighting of the values.

In a decoder, position \(i\) must not use positions \(j>i\). A causal mask
blocks those query-key pairs before softmax, preventing future-token leakage
through the attention mixing operation.

## 2. Notation and shapes

Let:

- \(B\): batch size;
- \(T\): sequence length;
- \(D\): input embedding dimension;
- \(d_k\): query and key dimension;
- \(d_v\): value dimension.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| \(X\) | \(B\times T\times D\) | input sequence representations |
| \(W_Q,W_K\) | \(D\times d_k\) | query and key weights |
| \(b_Q,b_K\) | \(d_k\) | optional query and key biases |
| \(W_V\) | \(D\times d_v\) | value weights |
| \(b_V\) | \(d_v\) | optional value bias |
| \(Q,K\) | \(B\times T\times d_k\) | projected queries and keys |
| \(V\) | \(B\times T\times d_v\) | projected values |
| \(S,A\) | \(B\times T\times T\) | scaled scores and probabilities |
| \(O\) | \(B\times T\times d_v\) | weighted value aggregate |
| \(W_O\) | \(d_v\times D\) | optional output-projection weight |
| \(b_O\) | \(D\) | optional output-projection bias |
| \(Y\) | \(B\times T\times D\) | output when projection is enabled |

All matrix products below are independently batched over \(B\).

## 3. Forward derivation

### Projections

The same input supplies three independently learned views:

\[
Q=XW_Q+b_Q,\qquad
K=XW_K+b_K,\qquad
V=XW_V+b_V.
\]

The existing `Linear.forward` applies each affine transformation over the last
dimension and preserves the leading batch and sequence dimensions.

### Scaled dot-product scores

For batch element \(b\), receiving position \(i\), and source position \(j\),

\[
S_{bij}
=
\frac{1}{\sqrt{d_k}}
\sum_{r=1}^{d_k}Q_{bir}K_{bjr}.
\]

Equivalently,

\[
S=\frac{QK^\top}{\sqrt{d_k}},
\]

where the transpose exchanges the final two axes of \(K\). If query and key
components have roughly unit variance and are sufficiently independent, an
unscaled dot product has variance proportional to \(d_k\). Multiplication by
\(1/\sqrt{d_k}\) keeps the score scale more nearly constant as \(d_k\)
changes, reducing premature softmax saturation.

### Causal masking and stable softmax

Define a boolean allowed mask:

\[
M_{ij} =
\begin{cases}
\mathrm{True}, & j\le i,\\
\mathrm{False}, & j>i.
\end{cases}
\]

`causal_attention_mask` returns shape \(1\times T\times T\), so the same mask
broadcasts across the batch. In this project, `True` always means **allowed**.

For each attention row, only allowed entries contribute to the maximum and
normalizer:

\[
m_{bi}=\max_{j:M_{ij}} S_{bij},
\]

\[
A_{bij} =
\begin{cases}
\displaystyle
\frac{\exp(S_{bij}-m_{bi})}
{\sum_{k:M_{ik}}\exp(S_{bik}-m_{bi})},
& M_{ij},\\[10pt]
0, & \neg M_{ij}.
\end{cases}
\]

Subtracting the valid-entry maximum is algebraically neutral but prevents
large positive scores from overflowing the exponential. Blocked entries are
assigned exactly zero after normalization. Every causal row is valid because
the diagonal is allowed. The general `masked_softmax` primitive nevertheless
rejects any all-masked row.

### Weighted values

Each receiving position takes a probability-weighted combination of the
source values:

\[
O=AV,
\qquad
O_{bif}=\sum_{j=1}^{T}A_{bij}V_{bjf}.
\]

If output projection is enabled:

\[
Y=OW_O+b_O.
\]

Without it, the module returns \(O\) with final dimension \(d_v\). With it,
the module returns \(Y\) with final dimension \(D\).

## 4. Masked-softmax backward derivation

Consider one row and write \(p=\operatorname{softmax}(z)\) over its allowed
coordinates. The softmax Jacobian is

\[
\frac{\partial p_i}{\partial z_j}
=p_i(\mathbb{1}[i=j]-p_j).
\]

Given upstream gradient \(g_i=\partial L/\partial p_i\),

\[
\frac{\partial L}{\partial z_j}
=\sum_i g_i\frac{\partial p_i}{\partial z_j}
=\sum_i g_i p_i(\mathbb{1}[i=j]-p_j).
\]

Separating the \(i=j\) term gives

\[
\frac{\partial L}{\partial z_j}
=p_jg_j-p_j\sum_i g_ip_i
=p_j\left(g_j-\sum_i g_ip_i\right).
\]

Applied to every batch and query row:

\[
G_S
=
A\odot
\left(
G_A-\sum_jG_{A,j}A_j
\right),
\]

where the sum retains a length-one final axis for broadcasting.
`masked_softmax_backward` implements precisely this vector-Jacobian product.
At a blocked coordinate \(A_{bij}=0\), multiplication by \(A_{bij}\) makes
the score gradient exactly zero. The implementation applies the boolean mask
again explicitly so this invariant is direct and testable.

## 5. Attention matrix backward derivation

Suppose the upstream gradient with respect to \(O\) is \(G_O\). If output
projection is enabled, `Linear.backward` first converts the gradient with
respect to \(Y\) into \(G_O\) and accumulates \(G_{W_O}\) and \(G_{b_O}\).

### Weighted aggregation

For

\[
O=AV,
\]

the differential is

\[
dO=(dA)V+A(dV).
\]

Using the Frobenius inner product to identify the coefficients of \(dA\) and
\(dV\):

\[
G_A=G_OV^\top,
\qquad
G_V=A^\top G_O.
\]

### Scaled scores

For

\[
S=cQK^\top,\qquad c=\frac{1}{\sqrt{d_k}},
\]

the differential is

\[
dS=c(dQ)K^\top+cQ(dK)^\top.
\]

The resulting gradients are

\[
G_Q=cG_SK,
\qquad
G_K=cG_S^\top Q.
\]

### Projection and shared-input accumulation

Each existing `Linear.backward` computes its own affine gradients. For example,
the query projection gives

\[
G_{W_Q}=X_{\mathrm{flat}}^\top (G_Q)_{\mathrm{flat}},
\qquad
G_{b_Q}=\sum_{\text{leading axes}}G_Q,
\qquad
G_X^{(Q)}=G_QW_Q^\top.
\]

The key and value projections give corresponding terms. Since \(X\) is an
input to all three branches, the total input gradient is a sum:

\[
G_X=G_X^{(Q)}+G_X^{(K)}+G_X^{(V)}.
\]

This accumulation is explicit in `CausalSelfAttentionHead.backward`; it is not
performed by an automatic-differentiation graph.

## 6. Causality

For output position \(i\),

\[
O_i=\sum_{j=0}^{i}A_{ij}V_j,
\]

because \(A_{ij}=0\) when \(j>i\). Also, the nonzero probabilities in row
\(i\) are normalized only from scores \(S_{i0},\ldots,S_{ii}\). Those scores
depend on \(Q_i\) and keys \(K_0,\ldots,K_i\), all produced by token-wise
linear projections. Therefore \(O_i\) is independent of input tokens
\(X_j\) for \(j>i\).

This statement is specific to the attention path implemented here. A future
model must preserve causality in every other sequence-mixing operation too.
Tests change a future input token and verify earlier outputs remain unchanged.
A complementary backward test places loss only on position zero and verifies
that later input positions receive exactly zero gradient.

## 7. Numerical considerations

- **Stable masking:** the row maximum is computed over valid coordinates only.
  A large blocked logit cannot influence the normalization.
- **No all-masked rows:** rejecting these rows avoids the undefined
  \(-\infty-(-\infty)\) form and a zero normalization denominator.
- **Exact masked zeros:** the forward probabilities and backward score
  gradients are explicitly zeroed with the boolean mask.
- **Precision:** ordinary model construction supports float32 or float64 and
  does not silently downcast. Tiny finite-difference fixtures use float64.
- **Finite differences:** centered differences use a small perturbation but
  remain sensitive to rounding and near-zero gradients. Tests combine absolute
  and relative criteria rather than relying on relative error alone.
- **Score scaling:** \(1/\sqrt{d_k}\) is stored in the module dtype and used in
  both forward and backward.
- **Non-finite values:** inputs, intermediate score tensors, gradients, and
  primitive softmax arguments are validated explicitly.

## 8. Source and test mapping

| Mathematics | Source | Principal validation |
| --- | --- | --- |
| causal relation \(j\le i\) | `nn/masks.py::causal_attention_mask` | `tests/test_attention_masks.py` |
| stable valid-entry softmax | `nn/attention.py::masked_softmax` | `tests/test_masked_softmax.py` |
| softmax vector-Jacobian product | `nn/attention.py::masked_softmax_backward` | exhaustive centered differences in `tests/test_masked_softmax.py` |
| Q/K/V projections and scaled scores | `nn/attention.py::CausalSelfAttentionHead.forward` | hand-computed and shape tests in `tests/test_attention.py` |
| aggregation and all backward paths | `nn/attention.py::CausalSelfAttentionHead.backward` | exhaustive module finite differences in `tests/test_attention.py` |
| future-token independence | causal mask plus token-wise projections | forward and backward causality tests in `tests/test_attention.py` |
| checkpoint reconstruction | attention `save_checkpoint`/`load_checkpoint` | checkpoint round-trip test in `tests/test_attention.py` |
| inspectable end-to-end computation | `experiments/inspect_single_head_attention.py` | experiment-summary integration test |

The gradient tests check every coordinate of tiny float64 inputs and all
parameters, including all projection biases and the optional output
projection. The implementation and tests deliberately stop at one head.
