# Pre-normalized single-head decoder block

This document derives the forward and backward computations in
`localml_scholar.nn.transformer`. The implementation contains one causal
attention head and one position-wise feed-forward network. It is not a stacked
transformer or a language model.

## 1. Motivation

A decoder block combines four ideas:

- causal self-attention mixes information from the current and earlier
  positions;
- a position-wise feed-forward network applies a nonlinear transformation
  independently to every token;
- residual paths retain the incoming representation and provide direct
  gradient routes;
- LayerNorm controls the feature scale independently at every token position.

The causal mask remains the mechanism that prevents future-token leakage. The
normalizations, feed-forward transformation, and residual additions do not mix
sequence positions.

## 2. Pre-norm versus post-norm

This milestone uses pre-normalization:

\[
R_1
=
X+\operatorname{Attention}(\operatorname{LN}_1(X)),
\]

\[
Y
=
R_1+\operatorname{FFN}(\operatorname{LN}_2(R_1)).
\]

Each transformed branch receives a normalized input, while each residual
identity path bypasses normalization.

A post-normalized arrangement instead applies normalization after a residual
addition, such as

\[
\operatorname{LN}(X+\operatorname{Attention}(X)).
\]

Pre-norm is convenient for this explicit implementation because the identity
gradient route remains visually and mathematically separate from the
normalization derivative. Pre-norm is common in modern decoder designs, but
this project does not claim it is universally superior; optimization behavior
depends on architecture, scale, initialization, and training conditions.

## 3. Tensor shapes

Let:

- \(B\): batch size;
- \(T\): sequence length;
- \(D\): model dimension;
- \(d_k\): query/key dimension;
- \(d_v\): value dimension;
- \(D_{\mathrm{ff}}\): feed-forward hidden dimension.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| \(X\) | \(B\times T\times D\) | block input |
| \(U\) | \(B\times T\times D\) | first normalized tensor |
| \(Q,K\) | \(B\times T\times d_k\) | attention queries and keys |
| \(V_{\mathrm{attn}}\) | \(B\times T\times d_v\) | attention values |
| \(S,A\) | \(B\times T\times T\) | attention scores and probabilities |
| \(H\) | \(B\times T\times D\) | attention output after projection if needed |
| \(R_1\) | \(B\times T\times D\) | first residual output |
| \(V\) | \(B\times T\times D\) | second normalized tensor |
| \(Z_1,A_{\mathrm{ff}}\) | \(B\times T\times D_{\mathrm{ff}}\) | FFN pre-activation and activation |
| \(F\) | \(B\times T\times D\) | feed-forward output |
| \(Y\) | \(B\times T\times D\) | block output |

An attention output projection is enabled by default, mapping \(d_v\) to
\(D\). If it is disabled, construction requires \(d_v=D\). Residual addition
never broadcasts or changes shape.

## 4. Forward derivation

### First normalization

LayerNorm operates on the final feature dimension:

\[
U=\operatorname{LN}_1(X).
\]

For each batch and token position, the mean and population variance use the
\(D\) features only:

\[
\mu=\frac{1}{D}\sum_{r=1}^{D}X_r,
\qquad
\sigma^2=\frac{1}{D}\sum_{r=1}^{D}(X_r-\mu)^2.
\]

The existing LayerNorm applies learned scale and shift:

\[
U
=
\gamma_1\odot
\frac{X-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta_1.
\]

### Causal attention

The existing single-head attention module computes

\[
Q=UW_Q+b_Q,\qquad
K=UW_K+b_K,\qquad
V_{\mathrm{attn}}=UW_V+b_V,
\]

\[
S=\frac{QK^\top}{\sqrt{d_k}},
\qquad
A=\operatorname{MaskedSoftmax}(S),
\]

\[
O=AV_{\mathrm{attn}}.
\]

When output projection is enabled,

\[
H=OW_O+b_O.
\]

Otherwise \(H=O\), and construction requires \(d_v=D\).

### First residual

\[
R_1=X+H.
\]

`residual_add` checks that both tensors have exactly the same shape and dtype
before this addition. NumPy broadcasting is not accepted.

### Second normalization

\[
V=\operatorname{LN}_2(R_1).
\]

This normalization again operates independently over the last dimension.

### Position-wise feed-forward transformation

For every batch element and sequence position:

\[
Z_1=VW_1+b_1,
\]

\[
A_{\mathrm{ff}}=\operatorname{GELU}(Z_1),
\]

\[
F=A_{\mathrm{ff}}W_2+b_2.
\]

The same weights are used at each position, but no values are mixed between
positions. Exact GELU,

\[
\operatorname{GELU}(z)=z\Phi(z),
\]

is the default. ReLU is supported only because it is already a validated
project primitive.

### Second residual

\[
Y=R_1+F.
\]

There is deliberately no final LayerNorm in this milestone.

## 5. Feed-forward backward derivation

Let \(G_F=\partial L/\partial F\). Flattening only the leading batch and
sequence dimensions for the weight sums, the second affine layer gives

\[
G_{W_2}
=
A_{\mathrm{ff,flat}}^\top G_{F,\mathrm{flat}},
\]

\[
G_{b_2}
=
\sum_{\text{batch,position}}G_F,
\]

\[
G_{A_{\mathrm{ff}}}
=
G_FW_2^\top.
\]

For exact GELU,

\[
\frac{d}{dz}\left(z\Phi(z)\right)
=
\Phi(z)+z\phi(z),
\]

so

\[
G_{Z_1}
=
G_{A_{\mathrm{ff}}}
\odot
\left(\Phi(Z_1)+Z_1\phi(Z_1)\right).
\]

The first affine layer gives

\[
G_{W_1}
=
V_{\mathrm{flat}}^\top G_{Z_1,\mathrm{flat}},
\]

\[
G_{b_1}
=
\sum_{\text{batch,position}}G_{Z_1},
\]

\[
G_V
=
G_{Z_1}W_1^\top.
\]

`TransformerFeedForward.backward` composes the already validated
`Linear.backward` and `GELU.backward` operations in exactly this reverse order.
See `activations.md` and `linear_layer.md` for their complete primitive
derivations.

## 6. Residual backward derivation

For a general residual expression

\[
Y=X+F(X),
\]

the differential is

\[
dY=dX+dF.
\]

Given \(G_Y\), the addition sends the same upstream gradient to both operands:

\[
G_X^{(\mathrm{identity})}=G_Y,
\qquad
G_F=G_Y.
\]

After differentiating the transformed branch,

\[
G_X^{(\mathrm{transformed})}
=
G_Y\frac{\partial F}{\partial X}.
\]

The total is

\[
\frac{\partial L}{\partial X}
=
G_X^{(\mathrm{identity})}
+G_X^{(\mathrm{transformed})}
=
G_Y
+G_Y\frac{\partial F}{\partial X}.
\]

The identity branch receives an exact copy because the derivative of addition
with respect to either same-shaped operand is the identity map.
`residual_add_backward` returns two independent copies so one branch cannot
accidentally overwrite the other.

## 7. Complete decoder-block backward flow

Start with

\[
G_Y=\frac{\partial L}{\partial Y}.
\]

### Reverse the second residual

For \(Y=R_1+F\):

\[
G_{R_1}^{(\mathrm{identity,2})}=G_Y,
\qquad
G_F=G_Y.
\]

Backpropagate through the feed-forward module:

\[
G_V
=
G_F\frac{\partial F}{\partial V}.
\]

Backpropagate through the second LayerNorm:

\[
G_{R_1}^{(\mathrm{ff})}
=
G_V\frac{\partial V}{\partial R_1}.
\]

Accumulate at the first residual output:

\[
G_{R_1}
=
G_{R_1}^{(\mathrm{identity,2})}
+G_{R_1}^{(\mathrm{ff})}.
\]

### Reverse the first residual

For \(R_1=X+H\):

\[
G_X^{(\mathrm{identity,1})}=G_{R_1},
\qquad
G_H=G_{R_1}.
\]

Backpropagate through attention:

\[
G_U
=
G_H\frac{\partial H}{\partial U}.
\]

The attention module itself accumulates its query, key, and value input
gradients:

\[
G_U
=
G_U^{(Q)}+G_U^{(K)}+G_U^{(V)}.
\]

Backpropagate through the first LayerNorm:

\[
G_X^{(\mathrm{attention})}
=
G_U\frac{\partial U}{\partial X}.
\]

Finally:

\[
G_X
=
G_X^{(\mathrm{identity,1})}
+G_X^{(\mathrm{attention})}.
\]

`PreNormDecoderBlock.backward` spells out both residual splits and both sums.
No automatic graph discovers or accumulates these branches.

## 8. Causality

Consider output position \(i\):

- LayerNorm uses only the \(D\) features at position \(i\);
- each feed-forward affine and GELU operation is position-wise;
- residual addition combines tensors at the same position;
- causal attention is the only operation that mixes sequence positions;
- its mask sets attention probabilities to zero when source position \(j>i\).

Therefore output position \(i\) cannot depend on input position \(j>i\).
Forward tests change one future token and verify earlier outputs remain
bit-for-bit unchanged. A backward test places loss only at position zero and
verifies all later input-position gradients are exactly zero.

## 9. Numerical considerations

- **Float32 execution:** layers and parameters preserve explicitly requested
  float32 without silently downcasting.
- **Float64 validation:** exhaustive finite differences use float64 to reduce
  perturbation rounding error.
- **LayerNorm epsilon:** a finite positive epsilon is part of the checkpointed
  block configuration.
- **Stable masked softmax:** attention subtracts the maximum over allowed
  entries only.
- **Exact masked zeros:** future probabilities and their score gradients are
  exactly zero.
- **Residual accumulation:** each upstream residual gradient is copied before
  branch-specific backward operations and then summed explicitly.
- **No broadcasting:** residual operand shapes and dtypes must match exactly.
- **Finite differences:** the tests use centered differences and combined
  absolute/relative acceptance. Near-zero coordinates rely on the absolute
  criterion instead of a misleading large relative error.
- **Non-finite values:** inputs, residual outputs, backward outputs, parameter
  buffers, and optimizer inputs receive explicit finite-value validation.

## 10. Source and test mapping

| Equation or invariant | Source | Validation |
| --- | --- | --- |
| \(Y=X+F\), exact-shape addition | `nn/transformer.py::residual_add` | residual tests in `tests/test_pre_norm_decoder_block.py` |
| residual branch gradient copies | `nn/transformer.py::residual_add_backward` | direct copy/independence and identity-block tests |
| position-wise FFN forward/backward | `nn/transformer.py::TransformerFeedForward` | hand calculation and exhaustive finite differences in `tests/test_transformer_feed_forward.py` |
| first pre-norm attention residual | `nn/transformer.py::PreNormDecoderBlock.forward` | controlled attention-only and residual-inspection tests |
| second pre-norm FFN residual | `nn/transformer.py::PreNormDecoderBlock.forward` | identity and exact residual tests |
| full reverse branch accumulation | `nn/transformer.py::PreNormDecoderBlock.backward` | exhaustive all-coordinate block gradient check |
| causal independence | existing attention mask plus position-wise operations | forward and backward causality tests |
| deterministic persistence | decoder `save_checkpoint`/`load_checkpoint` | exact float32 checkpoint round trip and incompatible-version rejection |
| end-to-end inspection | `experiments/inspect_pre_norm_decoder_block.py` | experiment-summary integration test |

The block gradient fixture checks every coordinate of the input and every
trainable parameter: both LayerNorm scales and shifts, Q/K/V projections,
attention output projection, and both feed-forward affine layers including all
enabled biases.
