# Representation Normalization

## LayerNorm

LayerNorm standardizes the features of each token using that token's feature
mean and population variance. Learned scale and shift parameters then restore
an affine degree of freedom.

## Phrase Bridge

When a question asks how to normalize activations inside a transformer, it may
be referring to LayerNorm. The operation acts across the final feature
dimension rather than across the batch.

## Equation

For feature vector x, the normalized coordinate is
x_hat_j = (x_j - mu) / sqrt(variance + epsilon). The epsilon term prevents an
unstable division when coordinates are nearly constant.
