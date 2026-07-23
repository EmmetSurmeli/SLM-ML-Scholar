# Layer normalization

## Notation and shapes

LayerNorm operates independently on the final feature dimension. Flatten all
leading dimensions into \(N\) rows:

\[
X\in\mathbb{R}^{N\times D}.
\]

For each row, this implementation uses **population variance**, with denominator
\(D\), not \(D-1\):

\[
\mu=\frac{1}{D}\sum_{j=1}^{D}x_j,
\qquad
c_j=x_j-\mu,
\]

\[
v=\frac{1}{D}\sum_{j=1}^{D}c_j^2,
\qquad
r=(v+\epsilon)^{-1/2},
\qquad
\hat x_j=c_jr.
\]

With affine parameters,

\[
y_j=\gamma_j\hat x_j+\beta_j.
\]

Gamma starts at one and beta at zero. `affine=False` omits both.

## Affine gradients

Let \(g_j=\partial L/\partial y_j\), and let sums below include every flattened
row when accumulating shared feature parameters. Direct differentiation gives

\[
\boxed{
\frac{\partial L}{\partial \gamma_j}
=\sum_n g_{nj}\hat x_{nj}
},
\qquad
\boxed{
\frac{\partial L}{\partial \beta_j}
=\sum_n g_{nj}
}.
\]

For the input derivation define

\[
u_j=\frac{\partial L}{\partial \hat x_j}=g_j\gamma_j
\]

or \(u_j=g_j\) without affine parameters.

## Input-gradient derivation

Start from

\[
\hat x_j=c_jr.
\]

Its differential is

\[
d\hat x_j=r\,dc_j+c_j\,dr.
\]

The reciprocal-standard-deviation differential is

\[
dr=-\frac{1}{2}(v+\epsilon)^{-3/2}dv
=-\frac{1}{2}r^3dv.
\]

Since

\[
v=\frac{1}{D}\sum_k c_k^2,
\]

\[
dv=\frac{2}{D}\sum_k c_k\,dc_k.
\]

Substitute these into

\[
dL=\sum_j u_j\,d\hat x_j:
\]

\[
dL
=r\sum_j u_j\,dc_j
-\frac{r^3}{D}
\left(\sum_j u_jc_j\right)
\left(\sum_k c_k\,dc_k\right).
\]

Therefore the gradient with respect to centered values is

\[
\frac{\partial L}{\partial c_j}
=r u_j
-\frac{r^3c_j}{D}\sum_k u_kc_k.
\]

But \(c_j=x_j-\mu\), and

\[
d c_j=d x_j-\frac{1}{D}\sum_k dx_k.
\]

This subtracts the mean of the centered-value gradient. Using
\(\sum_j c_j=0\) and \(\hat x_j=rc_j\), the expression simplifies to

\[
\boxed{
\frac{\partial L}{\partial x_j}
=
r\left[
u_j
-\frac{1}{D}\sum_k u_k
-\hat x_j\frac{1}{D}\sum_k u_k\hat x_k
\right]
}.
\]

Equivalently,

\[
\frac{\partial L}{\partial x}
=\frac{r}{D}
\left[
D u-\sum_k u_k-\hat x\sum_k(u_k\hat x_k)
\right].
\]

`LayerNorm.backward` implements this final form without constructing a
Jacobian.

## Cache and numerical considerations

Training forward caches only \(\hat x\) and \(r\); those are sufficient for
backward. Epsilon is positive and cast to the configured input dtype. Nearly
constant rows remain finite because \(v+\epsilon>0\). `float64` is used for
finite-difference validation; ordinary training may explicitly use `float32`.

The output variance is

\[
\operatorname{Var}(\hat x)=\frac{v}{v+\epsilon},
\]

which approaches one but is not asserted to equal one exactly.

## Source and tests

- Source: `nn/normalization.py`
- Population mean/variance, affine values, 2D/3D shapes, and nearly constant
  rows: `tests/test_layer_norm.py`
- Exhaustive input/gamma/beta finite differences:
  `test_layer_norm_input_gamma_and_beta_pass_finite_differences`
