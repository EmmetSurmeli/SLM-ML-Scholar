# Optimizers and global gradient clipping

Let \(\theta_t\) be a parameter at step \(t\), \(g_t\) its already-computed
minibatch gradient, and \(\eta>0\) the learning rate. Optimizers never invoke
backward; they only validate and consume explicit `Parameter.grad` arrays.

## SGD

Plain stochastic gradient descent is

\[
\boxed{\theta_{t+1}=\theta_t-\eta g_t}.
\]

Optional weight decay uses **coupled L2 regularization**:

\[
\boxed{
\theta_{t+1}
=\theta_t-\eta(g_t+\lambda\theta_t)
}.
\]

This is not decoupled AdamW-style decay. The pre-update parameter contributes
to the same update vector as the gradient.

## Classical momentum

Velocity is initialized to a zero array for every parameter. This project uses
the unscaled-gradient convention requested by the roadmap:

\[
\boxed{v_t=\beta v_{t-1}+g_t},
\]

\[
\boxed{\theta_{t+1}=\theta_t-\eta v_t},
\qquad 0\le\beta<1.
\]

Some libraries instead multiply the new gradient by \(1-\beta\); that is not
the convention here.

## Adam

First and second moments start at zero, and the global step counter starts at
zero. After incrementing to \(t\):

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

The zero initialization biases both moments toward zero. Bias correction is

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

The update is

\[
\boxed{
\theta_{t+1}
=
\theta_t-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}.
\]

Epsilon is outside the square root. State dictionaries are keyed by
`Parameter` object identity, so equal names or equal values cannot alias
moments. Checkpoints serialize moments in deterministic parameter order and
validate optimizer type, hyperparameters, shapes, and dtypes on load.

## Global gradient clipping

Across parameters \(p\), the global Euclidean norm is

\[
\lVert g\rVert_2
=
\sqrt{\sum_p\sum_i(g_i^{(p)})^2}.
\]

If this exceeds maximum \(M>0\), every gradient receives one common scale:

\[
g_i^{(p)}
\leftarrow
g_i^{(p)}
\frac{M}{\lVert g\rVert_2+\epsilon}.
\]

Using one factor preserves all directions and relative magnitudes. The
function returns the pre-clipping norm. A zero norm remains zero. Non-finite
gradient elements are rejected.

The implementation uses a scaled sum-of-squares calculation instead of
directly squaring huge values. This avoids avoidable overflow for finite
components such as \(10^{300}\).

## Gradient lifecycle

Layer backward calls accumulate into persistent buffers. A training step is:

1. `optimizer.zero_grad()`
2. module forward
3. loss and explicit logits gradient
4. module backward
5. optional global clipping
6. `optimizer.step()`

Skipping step 1 intentionally accumulates gradients across graphs.

## Source and tests

- Source: `optim/base.py`, `optim/sgd.py`, `optim/momentum.py`,
  `optim/adam.py`, `training/clipping.py`
- Multi-step hand calculations, state isolation, and checkpoint continuation:
  `tests/test_optimizers.py`
- Norm, uniform scaling, zero/non-finite behavior, and huge finite values:
  `tests/test_gradient_clipping.py`
