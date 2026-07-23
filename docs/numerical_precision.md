# Numerical precision policy

LocalML Scholar uses an explicit precision policy because silent dtype changes
make manual-gradient bugs difficult to diagnose.

- Trainable parameters must be NumPy `float32` or `float64`; integer,
  `float16`, extended, object, and complex parameters are rejected.
- A parameter's gradient buffer has exactly the same shape and dtype as its
  data.
- Layer inputs and upstream gradients must match the layer's parameter dtype.
  Modules without parameters preserve their input dtype and require matching
  upstream gradients.
- Ordinary training may explicitly select `float32` or `float64`. Constructors
  default to `float64` for the small educational fixtures in this repository.
- Centered finite-difference checks require `float64` by default. A caller may
  opt into a lower-precision check only by disabling that requirement and
  choosing appropriate, documented epsilon/tolerances.
- Loss functions preserve `float32` or `float64` logits. Targets and embedding
  IDs must use an integer dtype.
- Initializers generate with a caller-owned seeded RNG and explicitly cast to
  the requested supported dtype.
- Public numerical boundaries reject NaN and infinity unless a function
  explicitly documents a reporting behavior. `safe_perplexity` is one such
  boundary: it maps excessive positive loss to the largest finite `float64`
  value and `-inf` to zero.
- No function silently downcasts a supplied floating tensor. Internal
  `float64` scalar evaluation may be used for diagnostics or standard-library
  `erf`; results return in the explicitly selected tensor dtype.

The policy is enforced by `Parameter`, `Module._validate_float_array`, the loss
validators, optimizers, gradient clipping, checkpoint loaders, and tests.
