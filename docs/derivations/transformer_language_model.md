# Decoder-only transformer language-model architecture

This document maps the Milestone 5 Part 1 architecture to
`TransformerLanguageModel`. It assembles already validated manual operations;
it does not introduce a training loop, generation algorithm, or new attention
mathematics.

## Configuration and shapes

Let:

- \(B\): batch size;
- \(T\): sequence length;
- \(V\): vocabulary size;
- \(D\): model dimension;
- \(N\): number of decoder blocks;
- \(T_{\max}\): maximum context length.

The input is an integer tensor

\[
X_{\mathrm{id}}\in\mathbb{Z}^{B\times T},
\qquad
0<T\le T_{\max}.
\]

The output is a floating logit tensor

\[
L\in\mathbb{R}^{B\times T\times V}.
\]

`TransformerConfig` validates every dimension, bias option, LayerNorm epsilon,
floating dtype, and deterministic seed before any parameter is created.

## Token and learned position embeddings

Let

\[
E_{\mathrm{tok}}\in\mathbb{R}^{V\times D}
\]

be the token table and

\[
E_{\mathrm{pos}}\in\mathbb{R}^{T_{\max}\times D}
\]

the learned position table. For position IDs

\[
P_{bi}=i,
\]

the initial hidden representation is

\[
H^{(0)}_{bi}
=
E_{\mathrm{tok}}[X_{\mathrm{id},bi}]
+
E_{\mathrm{pos}}[P_{bi}].
\]

The two embedding outputs have exact shape \((B,T,D)\). `residual_add`
validates equality before addition, so NumPy broadcasting cannot hide a
composition error.

Token IDs are discrete and have no gradient. During backward, the hidden
gradient is copied to both embedding branches. Each table uses the existing
indexed `Embedding.backward`:

\[
\frac{\partial L}{\partial E[r]}
=
\sum_{(b,i):\,\mathrm{index}_{bi}=r}
\frac{\partial L}{\partial H^{(0)}_{bi}}.
\]

For positions, every batch contains the same IDs \(0,\ldots,T-1\), so
`np.add.at` accumulates positional gradients across every batch element.

## Decoder stack

For \(n=1,\ldots,N\):

\[
H^{(n)}
=
\operatorname{DecoderBlock}_n(H^{(n-1)}).
\]

Every block is a separate `PreNormDecoderBlock` instance with independent
parameters and an independent child seed. `Sequential.forward` applies them in
increasing order. Backward reverses that order:

\[
G^{(n-1)}
=
G^{(n)}
\frac{\partial H^{(n)}}{\partial H^{(n-1)}}.
\]

The notation above represents the complete block Jacobian; each block already
implements its attention, feed-forward, normalization, and residual gradients
explicitly.

## Final normalization and vocabulary projection

After the stack:

\[
\widetilde H=\operatorname{LayerNorm}_{\mathrm{final}}(H^{(N)}).
\]

The untied language-model head is

\[
L=\widetilde H W_{\mathrm{vocab}}+b_{\mathrm{vocab}},
\]

where

\[
W_{\mathrm{vocab}}\in\mathbb{R}^{D\times V}.
\]

No softmax is applied. `softmax_cross_entropy_loss_and_gradient` accepts
logits shaped \((B,T,V)\) and indexed targets shaped \((B,T)\).

## Complete backward order

Given

\[
G_L=\frac{\partial \mathcal{L}}{\partial L},
\]

`TransformerLanguageModel.backward` executes:

1. language-model `Linear.backward`;
2. final `LayerNorm.backward`;
3. decoder blocks \(N,N-1,\ldots,1\);
4. exact-copy split of the embedding-addition gradient;
5. position `Embedding.backward`;
6. token `Embedding.backward`.

The method returns `None` because integer token IDs are not differentiable.
Every trainable parameter gradient remains available through the recursive
parameter interface.

## Causality

Learned position lookup, embedding addition, final LayerNorm, and vocabulary
projection operate independently at each sequence position. Each decoder block
is already causal. Composing any positive number of causal blocks therefore
preserves:

\[
L_i\ \text{is independent of}\ X_{\mathrm{id},j}
\quad\text{for }j>i.
\]

Tests change a future token and verify earlier logits remain bit-for-bit
unchanged. A reverse test places loss only on an earlier logit row and verifies
future token-table and position-table rows receive exactly zero gradients.

## Initialization and persistence

One top-level seed creates independent child seed streams for:

- token embedding;
- position embedding;
- every decoder block;
- language-model head.

The final LayerNorm has deterministic constant initialization. Equal
configurations and seeds reproduce every parameter; separate blocks do not
share identities or begin with identical projected weights.

`state_dict` follows deterministic named-parameter order and returns copies.
`load_state_dict` validates the complete state before mutation. Checkpoints
store model/checkpoint versions, the exact serialized configuration, and every
named array. Token embeddings and vocabulary projection remain separate
parameters; weight tying is not used.

## Source and test mapping

| Architecture stage | Source | Validation |
| --- | --- | --- |
| configuration | `models/transformer_lm.py::TransformerConfig` | configuration tests |
| token and position lookup | `TransformerLanguageModel.forward` plus `Embedding` | controlled forward and accumulation tests |
| embedding addition | `residual_add` | exact-shape architecture fixture |
| decoder stack | existing `Sequential` and `PreNormDecoderBlock` | multi-layer shape, determinism, gradient, and causality tests |
| final normalization | existing `LayerNorm` | controlled forward fixture and finite differences |
| vocabulary logits | existing `Linear` | hand calculation and cross-entropy integration |
| complete reverse flow | `TransformerLanguageModel.backward` | exhaustive parameter finite differences |
| state and checkpoint | transformer state/checkpoint methods | atomic state and exact float32 reload tests |

The exhaustive fixture checks all 56 scalar parameters of a tiny one-layer
float64 model. No gradient is requested with respect to integer token IDs.
