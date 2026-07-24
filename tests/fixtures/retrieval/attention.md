# Causal Attention

## Masking Future Positions

A decoder applies a causal mask before softmax. Position i may attend to
positions j less than or equal to i, while every future position j greater than
i receives zero probability. This prevents future-token leakage during
autoregressive prediction.

```python
allowed = key_position <= query_position
scores[~allowed] = blocked
```

## Scaled Scores

Query and key vectors form dot-product scores. Dividing by the square root of
the key dimension controls the scale before softmax.
