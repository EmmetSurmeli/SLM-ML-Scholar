"""From-scratch language-model components for LocalML Scholar."""

from localml_scholar._version import __version__
from localml_scholar.data import CorpusMetadata, TokenStreamDataset
from localml_scholar.generation import (
    generate_transformer_ids,
    generate_transformer_text,
)
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
from localml_scholar.tokenizer import (
    BPETrainingConfig,
    BytePairTokenizer,
    ByteTokenizer,
    CharacterTokenizer,
    MergeRule,
    Tokenizer,
    load_tokenizer,
    save_tokenizer,
)
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.training.transformer import TransformerTrainer

__all__ = [
    "BigramLanguageModel",
    "BPETrainingConfig",
    "BytePairTokenizer",
    "ByteTokenizer",
    "CausalSelfAttentionHead",
    "CharacterTokenizer",
    "CorpusMetadata",
    "generate_transformer_ids",
    "generate_transformer_text",
    "load_tokenizer",
    "MergeRule",
    "MLP",
    "MultiHeadCausalSelfAttention",
    "Parameter",
    "PreNormDecoderBlock",
    "save_tokenizer",
    "Tokenizer",
    "TokenStreamDataset",
    "TransformerConfig",
    "TransformerLanguageModel",
    "TransformerTrainer",
    "TransformerTrainingConfig",
    "__version__",
]
