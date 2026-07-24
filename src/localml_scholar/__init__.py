"""From-scratch language-model components for LocalML Scholar."""

from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.models.mlp import MLP
from localml_scholar.nn.attention import CausalSelfAttentionHead
from localml_scholar.nn.parameter import Parameter
from localml_scholar.nn.transformer import PreNormDecoderBlock
from localml_scholar.tokenizer import CharacterTokenizer

__all__ = [
    "BigramLanguageModel",
    "CausalSelfAttentionHead",
    "CharacterTokenizer",
    "MLP",
    "Parameter",
    "PreNormDecoderBlock",
]
__version__ = "0.4.0"
