# Generalized finite-difference checking

For a scalar objective \(L\) and one coordinate \(\theta_i\), the centered
finite difference is

\[
g_{\text{numerical},i}
=
\frac{
L(\theta_i+\epsilon)-L(\theta_i-\epsilon)
}{
2\epsilon
}.
\]

The caller supplies an objective that returns both \(L\) and the analytical
gradient with respect to module output. The checker performs one ordinary
training forward/backward to obtain input and parameter gradients, then uses
evaluation-mode forwards for numerical perturbations.

For each coordinate it reports absolute error

\[
e_{\text{abs}}=|g_{\text{analytical}}-g_{\text{numerical}}|
\]

and a diagnostic relative error

\[
e_{\text{rel}}=
\frac{
e_{\text{abs}}
}{
|g_{\text{analytical}}|+|g_{\text{numerical}}|+\delta
}.
\]

Pass/fail uses the robust mixed criterion

\[
e_{\text{abs}}
\le
\text{atol}
+
\text{rtol}\max(
|g_{\text{analytical}}|,
|g_{\text{numerical}}|
).
\]

This prevents a large relative ratio between two harmless near-zero values
from dominating the decision.

All coordinates are checked by default. `max_checks_per_tensor` selects a
sorted, seeded subset independently for each larger tensor. Diagnostics remain
separate for input and every deterministic dotted parameter name.

Before checking, the implementation copies the input and every parameter.
Each scalar perturbation has a local `finally` restoration, and an outer
`finally` restores complete tensors and the original module mode. Tests compare
restored parameters bit-for-bit, verify deterministic subsets, exercise
float64-by-default policy, and confirm failure messages contain tensor names,
indices, analytical values, and numerical values.

Source: `training/gradient_check.py`.

Tests: `tests/test_gradient_checking.py` plus each layer's focused test module.
