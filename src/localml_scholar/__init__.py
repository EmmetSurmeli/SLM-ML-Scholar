"""LocalML Scholar's manual language-model and deterministic retrieval components."""

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
from localml_scholar.retrieval import (
    BM25Config,
    Chunk,
    ChunkingConfig,
    Citation,
    Document,
    IndexConfig,
    LexicalTokenizerConfig,
    PageText,
    RetrievalIndex,
    SearchFilters,
    SearchQuery,
    SearchResult,
    Section,
    ingest_file,
    ingest_files,
    ingest_markdown,
    ingest_pdf_text,
    ingest_plain_text,
)
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
    "BM25Config",
    "BPETrainingConfig",
    "BytePairTokenizer",
    "ByteTokenizer",
    "CausalSelfAttentionHead",
    "CharacterTokenizer",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "CorpusMetadata",
    "Document",
    "generate_transformer_ids",
    "generate_transformer_text",
    "IndexConfig",
    "ingest_file",
    "ingest_files",
    "ingest_markdown",
    "ingest_pdf_text",
    "ingest_plain_text",
    "LexicalTokenizerConfig",
    "load_tokenizer",
    "MergeRule",
    "MLP",
    "MultiHeadCausalSelfAttention",
    "PageText",
    "Parameter",
    "PreNormDecoderBlock",
    "RetrievalIndex",
    "save_tokenizer",
    "SearchFilters",
    "SearchQuery",
    "SearchResult",
    "Section",
    "Tokenizer",
    "TokenStreamDataset",
    "TransformerConfig",
    "TransformerLanguageModel",
    "TransformerTrainer",
    "TransformerTrainingConfig",
    "__version__",
]
