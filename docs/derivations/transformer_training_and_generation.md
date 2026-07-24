# Transformer training and autoregressive generation

This document describes Milestone 5 Part 2 (`0.5.1`). All neural-network
derivatives remain explicit NumPy code; no automatic-differentiation framework
is involved.

## 1. Next-token objective and teacher forcing

For tokens \(x_1,\ldots,x_n\), a causal language model estimates

\[
P(x_{t+1}\mid x_1,\ldots,x_t).
\]

During training, the input window contains the true preceding tokens:

\[
X=[x_s,\ldots,x_{s+T-1}],\qquad
Y=[x_{s+1},\ldots,x_{s+T}].
\]

Supplying the true history at every position is teacher forcing. The
`SequenceBatchSampler` in `data.py` constructs these integer tensors without
crossing the chronological split.

## 2. Sequence cross-entropy and perplexity

The transformer returns logits
\(Z\in\mathbb{R}^{B\times T\times V}\). For target \(y_{bt}\), the mean
negative log-likelihood is

\[
\mathcal{L}
=-\frac{1}{BT}
\sum_{b=1}^{B}\sum_{t=1}^{T}
\log
\frac{\exp Z_{bt,y_{bt}}}
{\sum_{v=1}^{V}\exp Z_{btv}}.
\]

The stable implementation subtracts the maximum logit in each vocabulary row.
Its logit gradient is

\[
\frac{\partial\mathcal{L}}{\partial Z_{btv}}
=\frac{P_{btv}-\mathbf{1}[v=y_{bt}]}{BT}.
\]

The averaging factor appears exactly once in
`softmax_cross_entropy_backward`. Layer backward methods then sum the already
scaled contributions; they do not divide again.

Perplexity is

\[
\operatorname{PPL}=\exp(\mathcal{L}).
\]

`safe_perplexity` caps the exponent at the largest representable float64 value
instead of allowing an overflow.

## 3. Complete manual backward path

One `TransformerTrainer.train_batch` call follows:

\[
\text{targets}
\rightarrow \mathcal{L}
\rightarrow G_Z
\rightarrow \text{LM head}
\rightarrow \text{final LayerNorm}
\rightarrow \text{decoder blocks in reverse}
\rightarrow \text{token and position embeddings}.
\]

The language-model head applies the `Linear.backward` equations. Final
LayerNorm uses the population-variance derivative in
`layer_normalization.md`. Each decoder block uses the residual, attention, and
feed-forward derivatives derived in `pre_norm_decoder_block.md` and
`single_head_causal_attention.md`. At the embedding sum, the upstream gradient
is copied to both branches. `Embedding.backward` uses `np.add.at`, so repeated
token IDs and positions accumulate correctly.

Integer token IDs are discrete. The model therefore returns no gradient with
respect to IDs; it accumulates gradients only in trainable parameters.

## 4. Global norm clipping and coupled weight decay

For parameter gradients \(g^{(p)}\), the global norm is

\[
\lVert g\rVert_2
=\sqrt{\sum_p\sum_i(g_i^{(p)})^2}.
\]

If the configured maximum is \(c\) and \(\lVert g\rVert_2>c\), every gradient
is scaled by

\[
g^{(p)}\leftarrow
g^{(p)}
\frac{c}{\lVert g\rVert_2+\epsilon}.
\]

The relative direction of the complete gradient vector is preserved. The
trainer reports norms before and after clipping.

Optional weight decay is documented as coupled L2 regularization:

\[
g^{(p)}\leftarrow g^{(p)}+\lambda\theta^{(p)}
\]

before norm measurement and clipping. This keeps the convention identical
across SGD, Momentum, and Adam while leaving the validated optimizer algorithms
unchanged when \(\lambda=0\).

## 5. Adam state and exact resumption

Adam maintains first and second moments plus a completed update count. The
equations are derived in `optimizers.md`. Bias correction depends on the exact
step \(t\), so restoring only parameters is insufficient. Full checkpoints
restore both moments, the step counter, hyperparameters, and deterministic
parameter order.

## 6. Evaluation

Evaluation uses independent fixed-seed samplers. For batch \(k\) with mean loss
\(\ell_k\) over \(n_k\) predicted tokens, the combined result is

\[
\ell_{\mathrm{eval}}
=\frac{\sum_k n_k\ell_k}{\sum_k n_k}.
\]

This remains correct if a future evaluator supports unequal final batches.
Evaluation:

- runs in recursive no-cache mode;
- does not call backward or an optimizer;
- leaves parameter gradients unchanged;
- does not consume the training sampler RNG; and
- restores every caller mode.

## 7. Autoregressive generation

The chain rule factorizes a sequence probability:

\[
P(x_{1:n})
=\prod_{t=1}^{n}P(x_t\mid x_{<t}).
\]

At each step, `generate_transformer_ids` evaluates the current context, selects
the last-position logits, chooses one next token, appends it, and repeats.

Greedy decoding uses

\[
x_{\text{next}}=\arg\max_v Z_v.
\]

Sampling first applies positive temperature \(\tau\):

\[
Z'_v=Z_v/\tau.
\]

Smaller temperature sharpens the distribution; larger temperature flattens it.
Greedy mode validates but ignores temperature because positive scaling cannot
change the argmax.

For top-\(k\), only the \(k\) greatest logits participate in stable softmax.
All other probabilities are exactly zero. Stable sorting makes equal-logit
ties prefer lower vocabulary indices.

## 8. Context-window truncation

If the prompt plus generated text is longer than the configured context, only
the most recent tokens are passed to the model:

\[
\text{context}
=x_{\max(1,t-C+1):t},
\]

where \(C\) is the maximum context length. The returned sequence still contains
the full prompt and all generated tokens. This is explicit truncation, not a KV
cache: the complete retained context is recomputed at every step.

## 9. Training and inference cache lifecycle

Training modules support exactly one cached forward followed by one backward.
A second forward or a mode change with an unmatched cache raises.

`Module.inference_mode()` is the narrow no-gradient mechanism:

- it cannot discard a pending training graph;
- evaluation forwards do not cache;
- backward during or after inference has no matching cache and fails;
- it checks for unexpected caches before exit; and
- it restores nested modes even when inference raises.

This preserves educationally explicit training state while enabling repeated
generation forwards.

## 10. Deterministic resumption

Reproducing the uninterrupted trajectory requires:

1. exact model parameters and architecture;
2. optimizer type, hyperparameters, moments/velocity, and step;
3. training batch-sampler bit-generator state;
4. completed training step;
5. tokenizer vocabulary;
6. train/validation stream identity;
7. clipping and coupled-decay configuration;
8. best-validation metadata and history; and
9. the same deterministic numerical environment.

The full checkpoint stores each item. Its test compares uninterrupted and
interrupted runs exactly, including the next sampled batch.

## 11. Limitations

The trained fixture remains character-level, tiny, single-head, CPU-oriented,
and unoptimized. Attention is quadratic in context length. There is no dropout,
schedule, padding mask, KV cache, multi-head attention, mixed precision, or
large-corpus training. Generated samples are correctness demonstrations, not
evidence of broad language ability or paper understanding.

## 12. Source and test mapping

| Concept | Source | Primary tests |
|---|---|---|
| chronological streams and shifted batches | `data.py` | `test_sequence_batch_sampler.py` |
| N-D loss and logit gradient | `losses.py` | `test_softmax_cross_entropy.py`, `test_transformer_training.py` |
| no-cache lifecycle | `nn/module.py` | `test_nn_module.py`, `test_transformer_training.py` |
| training/evaluation | `training/transformer.py` | `test_transformer_training.py` |
| clipping | `training/clipping.py` | `test_gradient_clipping.py`, `test_transformer_training.py` |
| optimizer state | `optim/` | `test_optimizers.py`, `test_transformer_training_checkpoint.py` |
| generation | `generation.py` | `test_transformer_generation.py` |
| exact resumption | `training/transformer.py` | `test_transformer_training_checkpoint.py` |
| tiny overfit | `experiments/overfit_tiny_transformer.py` | `test_transformer_training.py` |
| corpus CLI | `experiments/train_transformer_lm.py` | smoke command in the audit |
