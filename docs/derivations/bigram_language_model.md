# Bigram language model: derivation and implementation map

This note derives exactly the mathematics implemented in the Milestone 1 NumPy
model. It uses a vocabulary of \(V\) characters and a minibatch of \(B\)
next-character examples. Token IDs are integers in \(\{0,\ldots,V-1\}\).

## 1. Bigram parameterization

The complete model is one trainable matrix

\[
W \in \mathbb{R}^{V \times V}.
\]

For current token \(x_b=i\), row \(W_i\) contains the logits for the next token:

\[
z_b = W_{x_b}, \qquad z_b \in \mathbb{R}^{V}.
\]

Thus \(W_{ij}\) is the unnormalized score assigned to next token \(j\) when the
current token is \(i\). There is no hidden state or context beyond the current
character. `BigramLanguageModel.forward` in
`src/localml_scholar/models/bigram.py` performs this row selection.

Given a token sequence \(x_1,\ldots,x_T\), `next_token_pairs` in
`src/localml_scholar/data.py` creates

\[
(x_1,x_2),(x_2,x_3),\ldots,(x_{T-1},x_T).
\]

The chronological train/validation boundary pair is intentionally discarded:
examples are made independently inside each split, so no training target comes
from validation.

## 2. Stable softmax

For one logit vector \(z\), ordinary softmax is

\[
p_j = \frac{\exp(z_j)}{\sum_{k=1}^{V}\exp(z_k)}.
\]

Exponentiating a large positive logit can overflow. Let

\[
m = \max_k z_k.
\]

Subtracting the same constant from every logit does not change the ratio:

\[
p_j =
\frac{\exp(z_j-m)}
{\sum_{k=1}^{V}\exp(z_k-m)}.
\]

Every shifted logit is at most zero, so its exponential is at most one.
`stable_softmax` in `src/localml_scholar/losses.py` implements this operation
along the final array axis.

## 3. Indexed cross-entropy

For integer target \(y_b\), the negative log-likelihood of one example is

\[
\ell_b = -\log p_{b,y_b}.
\]

The mean minibatch loss is

\[
L = \frac{1}{B}\sum_{b=1}^{B}\ell_b.
\]

There is no reason to allocate a \(B\times V\) target matrix solely to select
one probability per row. `multiclass_cross_entropy` uses NumPy indexed access.

For additional stability, the combined
`softmax_cross_entropy_forward` does not take the logarithm of a rounded
probability. With shifted logits \(\tilde z_{bj}=z_{bj}-m_b\),

\[
\ell_b =
\log\left(\sum_{k=1}^{V}\exp(\tilde z_{bk})\right)
-\tilde z_{b,y_b}.
\]

## 4. Softmax plus cross-entropy gradient

For one example, define the one-hot indicator
\(\mathbf{1}[j=y]\). Differentiating the log-softmax gives

\[
\frac{\partial \ell}{\partial z_j}
= p_j-\mathbf{1}[j=y].
\]

Because \(L\) is the mean of \(B\) losses,

\[
\boxed{
\frac{\partial L}{\partial z_{bj}}
= \frac{p_{bj}-\mathbf{1}[j=y_b]}{B}
}
\]

or, in matrix notation,

\[
\frac{\partial L}{\partial z}=\frac{p-y}{B}.
\]

`softmax_cross_entropy_backward` starts with a copy of \(p\), subtracts one
only at each indexed target, then divides the whole array by \(B\). The
implementation avoids constructing the full one-hot matrix.

## 5. Accumulation into \(W\)

Since \(z_b=W_{x_b}\), example \(b\) affects only the row selected by \(x_b\):

\[
\frac{\partial L}{\partial W_{ij}}
=
\sum_{b:x_b=i}
\frac{\partial L}{\partial z_{bj}}.
\]

Repeated current tokens must sum into the same row. In
`BigramLanguageModel.backward`, `np.add.at(grad_weights, input_ids,
grad_logits)` performs this accumulation. Ordinary advanced-index `+=` is not
correct when an index repeats because its buffered writes need not accumulate
each occurrence.

`BigramLanguageModel.loss_and_backward` connects the forward loss, the
logit-gradient calculation, and this selected-row accumulation. Gradients are
explicit arrays and are never produced by automatic differentiation.

## 6. Stochastic gradient descent

For learning rate \(\eta>0\), plain SGD updates every parameter by

\[
W \leftarrow W-\eta \frac{\partial L}{\partial W}.
\]

With the optional weight-decay coefficient \(\lambda\), this implementation
uses coupled L2 decay:

\[
W \leftarrow
W-\eta\left(\frac{\partial L}{\partial W}+\lambda W\right).
\]

`SGD.step` in `src/localml_scholar/optimizers.py` implements this update after
checking that every named gradient matches its parameter's shape.
`SGD.zero_grad` clears the persistent gradient buffers before the next
minibatch. The named-array interface leaves room for momentum and Adam state in
later milestones.

## 7. Perplexity

For mean token cross-entropy \(L\), perplexity is

\[
\operatorname{PPL}=\exp(L).
\]

It can be read as an effective number of equally likely next-token choices,
but only when models use compatible tokenization and evaluation data.
`safe_perplexity` in `src/localml_scholar/utils.py` caps the exponential input
at the largest representable float threshold to avoid overflow. It does not
make a poor loss better; it only makes reporting numerically defined.

## 8. Centered finite-difference checking

For one scalar parameter \(\theta_i\), the centered numerical derivative is

\[
g_{\text{numerical},i}
\approx
\frac{
L(\theta_i+\epsilon)-L(\theta_i-\epsilon)
}{
2\epsilon
}.
\]

The analytical and numerical values are compared with

\[
r_i =
\frac{
\left|g_{\text{analytical},i}-g_{\text{numerical},i}\right|
}{
\left|g_{\text{analytical},i}\right|
+
\left|g_{\text{numerical},i}\right|
+\delta
}.
\]

The small \(\delta>0\) prevents division by zero when both gradients are near
zero. `check_bigram_gradients` in `src/localml_scholar/utils.py` can inspect
every entry for a tiny vocabulary or a seeded, deterministic subset for a
larger matrix. It restores each perturbed weight in a `finally` block and
reports the worst matrix index, both gradient values, relative error, and
tolerance on failure.

Finite differences are a correctness diagnostic, not a training method. Their
cost scales with the number of checked parameters, and excessively small
\(\epsilon\) can amplify floating-point cancellation.

