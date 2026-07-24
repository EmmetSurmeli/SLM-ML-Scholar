"""Manually differentiated neural-network building blocks."""

from localml_scholar.nn.activations import GELU, ReLU
from localml_scholar.nn.attention import (
    AttentionDetails,
    CausalSelfAttentionHead,
    masked_softmax,
    masked_softmax_backward,
)
from localml_scholar.nn.containers import Sequential
from localml_scholar.nn.embedding import Embedding
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.masks import causal_attention_mask
from localml_scholar.nn.module import Module
from localml_scholar.nn.normalization import LayerNorm
from localml_scholar.nn.parameter import Parameter
from localml_scholar.nn.transformer import (
    DecoderBlockDetails,
    FeedForwardDetails,
    PreNormDecoderBlock,
    TransformerFeedForward,
    residual_add,
    residual_add_backward,
)

__all__ = [
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "AttentionDetails",
    "CausalSelfAttentionHead",
    "causal_attention_mask",
    "masked_softmax",
    "masked_softmax_backward",
    "DecoderBlockDetails",
    "FeedForwardDetails",
    "PreNormDecoderBlock",
    "TransformerFeedForward",
    "residual_add",
    "residual_add_backward",
]
