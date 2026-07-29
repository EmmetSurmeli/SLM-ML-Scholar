# Decoder Attention

## Triangular Restriction

At position i, the decoder may use keys at positions j less than or equal to i.
It is forbidden from attending to later positions. Therefore an earlier
prediction is independent of tokens appended to the suffix.

## Terminology Bridge

Sequence-model authors call accidental access to unavailable suffix tokens
future information leakage. A causal mask is the usual safeguard against that
training-time shortcut.

## Score Scaling

Dot products grow with the key dimension. Dividing attention logits by the
square root of the key width keeps their typical magnitude controlled before
softmax.
