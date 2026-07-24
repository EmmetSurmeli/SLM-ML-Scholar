"""Evidence-controlled local answering with explicit citations and validation."""

from localml_scholar.answering.citations import (
    CitationOccurrence,
    CitationSyntaxError,
    citation_labels,
    format_inline_citation,
    parse_inline_citations,
)
from localml_scholar.answering.context import (
    GroundedContext,
    build_grounded_context,
    render_grounded_prompt,
)
from localml_scholar.answering.evaluation import (
    AnswerEvaluation,
    GroundedQuestion,
    QuestionAnswerMetrics,
    evaluate_grounded_answers,
    load_grounded_questions,
)
from localml_scholar.answering.evidence import (
    EvidenceSelection,
    EvidenceSelectionConfig,
    assess_evidence_sufficiency,
    select_evidence,
)
from localml_scholar.answering.extractive import (
    ExtractiveAnswerConfig,
    ExtractiveAnswerer,
    ExtractiveResult,
)
from localml_scholar.answering.generative import (
    GroundedGeneration,
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
)
from localml_scholar.answering.models import (
    AnswerValidation,
    CitationBinding,
    ClaimSupport,
    EvidenceItem,
    EvidenceSufficiency,
    GroundedAnswer,
    GroundedClaim,
)
from localml_scholar.answering.pipeline import (
    ABSTENTION_TEXT,
    GroundedAnswerPipeline,
)
from localml_scholar.answering.serialization import (
    ANSWER_FORMAT_VERSION,
    load_grounded_answer,
    save_grounded_answer,
)
from localml_scholar.answering.validation import (
    AnswerAcceptanceConfig,
    assess_claim_support,
    validate_answer_text,
)

__all__ = [
    "ABSTENTION_TEXT",
    "ANSWER_FORMAT_VERSION",
    "AnswerAcceptanceConfig",
    "AnswerEvaluation",
    "AnswerValidation",
    "assess_claim_support",
    "assess_evidence_sufficiency",
    "build_grounded_context",
    "CitationBinding",
    "citation_labels",
    "CitationOccurrence",
    "CitationSyntaxError",
    "ClaimSupport",
    "EvidenceItem",
    "EvidenceSelection",
    "EvidenceSelectionConfig",
    "EvidenceSufficiency",
    "ExtractiveAnswerConfig",
    "ExtractiveAnswerer",
    "ExtractiveResult",
    "format_inline_citation",
    "GroundedAnswer",
    "GroundedAnswerPipeline",
    "GroundedClaim",
    "GroundedContext",
    "GroundedGeneration",
    "GroundedGenerationConfig",
    "GroundedGenerativeAnswerer",
    "GroundedQuestion",
    "load_grounded_answer",
    "parse_inline_citations",
    "QuestionAnswerMetrics",
    "render_grounded_prompt",
    "save_grounded_answer",
    "select_evidence",
    "evaluate_grounded_answers",
    "load_grounded_questions",
    "validate_answer_text",
]
