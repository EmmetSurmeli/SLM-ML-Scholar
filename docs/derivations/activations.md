# ReLU and exact GELU

Both activations are elementwise, so their input and output shapes are
identical. Given upstream gradient \(g=\partial L/\partial y\), backward
multiplies \(g\) elementwise by the scalar derivative.

## ReLU

### Forward

\[
\operatorname{ReLU}(x)=\max(0,x).
\]

### Backward

\[
\frac{d}{dx}\operatorname{ReLU}(x)=
\begin{cases}
1 & x>0,\\
0 & x<0.
\end{cases}
\]

The derivative is undefined at zero. This project explicitly chooses zero:

\[
\left.\frac{d}{dx}\operatorname{ReLU}(x)\right|_{x=0}=0.
\]

Therefore

\[
\boxed{\frac{\partial L}{\partial x}
=g\,\mathbf{1}[x>0]}.
\]

`ReLU.forward` caches the boolean positive mask and input dtype. Backward
requires a matching shape/dtype and consumes the cache.

## Exact GELU

This implementation uses the exact Gaussian Error Linear Unit rather than the
common tanh approximation.

Let the standard normal cumulative distribution and density be

\[
\Phi(x)=\frac{1}{2}
\left(1+\operatorname{erf}\left(\frac{x}{\sqrt 2}\right)\right),
\qquad
\phi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}.
\]

### Forward

\[
\operatorname{GELU}(x)=x\Phi(x).
\]

### Backward

Because \(\Phi'(x)=\phi(x)\), the product rule gives

\[
\frac{d}{dx}\operatorname{GELU}(x)
=\Phi(x)+x\phi(x).
\]

Thus

\[
\boxed{
\frac{\partial L}{\partial x}
=g\left(\Phi(x)+x\phi(x)\right)
}.
\]

`GELU.forward` caches the input; `GELU.backward` evaluates this exact formula.

## Numerical considerations

NumPy does not provide the required scalar `erf` in the allowed core API, so
the CDF uses standard-library `math.erf` over the flattened array and restores
the explicitly selected dtype. This is mathematically transparent but is not
yet performance-optimized. In large tails, `erf` and the exponential
naturally saturate to their floating-point limits: GELU approaches zero for
large negative input and the identity for large positive input.

ReLU finite differences are not evaluated at exactly zero because a centered
difference there selects neither one-sided convention. GELU is smooth and is
checked at zero, near zero, and in both tails.

## Source and tests

- Source: `nn/activations.py`
- Hand forward/derivative values and zero convention:
  `tests/test_activations.py`
- Finite differences:
  `test_relu_matches_finite_differences_away_from_kink` and
  `test_gelu_matches_finite_differences_near_zero_and_in_tails`
