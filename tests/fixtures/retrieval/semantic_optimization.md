# Optimization

## Update Magnitude

The learning rate controls how far gradient descent moves the parameters on
each update. A value of 0.01 produces a smaller parameter change than 0.1 when
the gradient is fixed.

## Vocabulary Bridge

Optimization notes sometimes use step size as another name for learning rate.
Both phrases refer to the scalar multiplier applied to the descent direction.

## Bias Correction

Adam divides its moving averages by beta-dependent factors during early
updates. This bias correction compensates for initializing the moment
estimates at zero.
