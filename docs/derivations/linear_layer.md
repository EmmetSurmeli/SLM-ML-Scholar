# Linear layer

## Notation and shapes

Let an input have arbitrary leading dimensions and final feature size
\(D_{\text{in}}\):

\[
X \in \mathbb{R}^{s_1\times\cdots\times s_k\times D_{\text{in}}}.
\]

Flatten the leading dimensions into \(N=\prod_i s_i\):

\[
\bar X\in\mathbb{R}^{N\times D_{\text{in}}}.
\]

Parameters are

\[
W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}},
\qquad
b\in\mathbb{R}^{D_{\text{out}}}.
\]

Bias is optional.

## Forward

\[
\bar Y=\bar XW+b.
\]

The result is restored to
\((s_1,\ldots,s_k,D_{\text{out}})\). NumPy broadcasting adds \(b\) to every
flattened row.

`Linear.forward` validates dtype and final dimension, then caches the input
reference in training mode. Mutating that input before backward is unsupported.
Evaluation mode stores no cache.

## Backward

Let

\[
G=\frac{\partial L}{\partial \bar Y}
\in\mathbb{R}^{N\times D_{\text{out}}}.
\]

Using differentials,

\[
d\bar Y=d\bar X\,W+\bar X\,dW+db.
\]

The scalar loss differential is

\[
dL=\operatorname{tr}(G^\top d\bar Y).
\]

For the input term,

\[
\operatorname{tr}(G^\top d\bar XW)
=\operatorname{tr}((GW^\top)^\top d\bar X),
\]

so

\[
\boxed{\frac{\partial L}{\partial \bar X}=GW^\top}.
\]

For the weight term,

\[
\operatorname{tr}(G^\top\bar X\,dW)
=\operatorname{tr}((\bar X^\top G)^\top dW),
\]

so

\[
\boxed{\frac{\partial L}{\partial W}=\bar X^\top G}.
\]

Every output row shares the bias, hence

\[
\boxed{\frac{\partial L}{\partial b_j}=\sum_{n=1}^{N}G_{nj}}.
\]

`Linear.backward` computes these expressions, accumulates parameter gradients,
restores the input-gradient shape, and consumes the one pending cache.

## Initialization and numerical considerations

The default Xavier uniform initialization samples

\[
W_{ij}\sim U\left[
-\sqrt{\frac{6}{D_{\text{in}}+D_{\text{out}}}},
\sqrt{\frac{6}{D_{\text{in}}+D_{\text{out}}}}
\right].
\]

He-normal initialization is also available with standard deviation
\(\sqrt{2/D_{\text{in}}}\). Both use an explicit `numpy.random.Generator` and
return the requested `float32` or `float64` dtype.

## Source and tests

- Source: `nn/linear.py`, `nn/initialization.py`
- Hand calculations, 2D/3D shapes, no-bias mode, dtype behavior:
  `tests/test_linear.py`
- Exhaustive input/weight/bias finite differences:
  `test_linear_passes_input_weight_and_bias_gradient_check`
