"""Manual local language-model, retrieval, and grounded-answer components."""

from localml_scholar._version import __version__
from localml_scholar.answering import (
    AnswerAcceptanceConfig,
    AnswerValidation,
    EvidenceItem,
    EvidenceSelectionConfig,
    EvidenceSufficiency,
    ExtractiveAnswerer,
    GroundedAnswer,
    GroundedAnswerPipeline,
    GroundedGenerativeAnswerer,
    load_grounded_answer,
    save_grounded_answer,
)
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
    "AnswerAcceptanceConfig",
    "AnswerValidation",
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
    "EvidenceItem",
    "EvidenceSelectionConfig",
    "EvidenceSufficiency",
    "ExtractiveAnswerer",
    "generate_transformer_ids",
    "generate_transformer_text",
    "GroundedAnswer",
    "GroundedAnswerPipeline",
    "GroundedGenerativeAnswerer",
    "IndexConfig",
    "ingest_file",
    "ingest_files",
    "ingest_markdown",
    "ingest_pdf_text",
    "ingest_plain_text",
    "LexicalTokenizerConfig",
    "load_grounded_answer",
    "load_tokenizer",
    "MergeRule",
    "MLP",
    "MultiHeadCausalSelfAttention",
    "PageText",
    "Parameter",
    "PreNormDecoderBlock",
    "RetrievalIndex",
    "save_tokenizer",
    "save_grounded_answer",
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
