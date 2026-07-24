# Optimization Notes

## Gradient Descent

Gradient descent updates parameters in the negative gradient direction. A
learning_rate controls the step size, and an optimizer may accumulate momentum.

## Adaptive Updates

Adam tracks first and second moment estimates. Bias correction compensates for
moments initialized at zero.

## Reproducibility

Deterministic experiments record the random seed, data split, optimizer
configuration, and evaluation procedure.
