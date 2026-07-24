"""From-scratch language-model components for LocalML Scholar."""

from localml_scholar._version import __version__
from localml_scholar.generation import generate_transformer_ids
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.models.mlp import MLP
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.nn.attention import (
    CausalSelfAttentionHead,
    MultiHeadCausalSelfAttention,
)
from localml_scholar.nn.parameter import Parameter
from localml_scholar.nn.transformer import PreNormDecoderBlock
from localml_scholar.tokenizer import CharacterTokenizer
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.training.transformer import TransformerTrainer

__all__ = [
    "BigramLanguageModel",
    "CausalSelfAttentionHead",
    "CharacterTokenizer",
    "generate_transformer_ids",
    "MLP",
    "MultiHeadCausalSelfAttention",
    "Parameter",
    "PreNormDecoderBlock",
    "TransformerConfig",
    "TransformerLanguageModel",
    "TransformerTrainer",
    "TransformerTrainingConfig",
    "__version__",
]
