"""Untrusted starter questions for reviewing *Attention Is All You Need*."""

from __future__ import annotations

from typing import Any

from localml_scholar.evaluation.schemas import (
    Benchmark,
    BenchmarkQuestion,
    ConceptGroup,
    GoldEvidence,
)
from localml_scholar.retrieval import RetrievalIndex, SearchFilters

_STARTER_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "question": "Who wrote the paper?",
        "type": "metadata",
        "sections": ("title",),
        "concepts": ("authors",),
    },
    {
        "question": "What is the exact title of the paper?",
        "type": "metadata",
        "sections": ("title",),
        "concepts": ("Attention Is All You Need",),
    },
    {
        "question": "When was the paper published?",
        "type": "metadata",
        "sections": ("title",),
        "concepts": ("publication year",),
    },
    {
        "question": "What problem was the Transformer designed to solve?",
        "type": "motivation",
        "sections": ("abstract", "introduction"),
        "concepts": ("sequence transduction",),
    },
    {
        "question": "Why did the authors remove recurrence?",
        "type": "motivation",
        "sections": ("introduction",),
        "concepts": ("parallelization", "sequential computation"),
    },
    {
        "question": "What is scaled dot-product attention?",
        "type": "architecture",
        "sections": ("attention", "model architecture"),
        "concepts": ("queries", "keys", "values"),
    },
    {
        "question": "Why are attention logits divided by the square root of d_k?",
        "type": "equation",
        "sections": ("attention",),
        "concepts": ("large dot products", "softmax gradients"),
    },
    {
        "question": "What is multi-head attention?",
        "type": "architecture",
        "sections": ("multi-head attention",),
        "concepts": ("multiple representation subspaces",),
    },
    {
        "question": "Why is positional encoding needed?",
        "type": "architecture",
        "sections": ("positional encoding",),
        "concepts": ("sequence order",),
    },
    {
        "question": "How do the encoder and decoder differ?",
        "type": "architecture",
        "sections": ("encoder", "decoder"),
        "concepts": ("masked attention", "cross-attention"),
    },
    {
        "question": "How is masking used in decoder self-attention?",
        "type": "architecture",
        "sections": ("decoder", "attention"),
        "concepts": ("future positions",),
    },
    {
        "question": "What role do residual connections play?",
        "type": "architecture",
        "sections": ("encoder", "decoder"),
        "concepts": ("residual connection", "layer normalization"),
    },
    {
        "question": "What is the position-wise feed-forward network?",
        "type": "architecture",
        "sections": ("feed-forward",),
        "concepts": ("two linear transformations",),
    },
    {
        "question": "Which optimizer was used?",
        "type": "hyperparameter",
        "sections": ("training",),
        "concepts": ("Adam",),
    },
    {
        "question": "What Adam beta values were used?",
        "type": "hyperparameter",
        "sections": ("training",),
        "concepts": ("beta_1", "beta_2"),
    },
    {
        "question": "How many warmup steps were used?",
        "type": "hyperparameter",
        "sections": ("training",),
        "concepts": ("warmup steps",),
    },
    {
        "question": "What dropout rate was used?",
        "type": "hyperparameter",
        "sections": ("regularization", "training"),
        "concepts": ("dropout",),
    },
    {
        "question": "What label smoothing value was used?",
        "type": "hyperparameter",
        "sections": ("regularization", "training"),
        "concepts": ("label smoothing",),
    },
    {
        "question": "Which datasets were used?",
        "type": "experiment",
        "sections": ("experiments",),
        "concepts": ("WMT",),
    },
    {
        "question": "What BLEU scores were reported?",
        "type": "result",
        "sections": ("results", "experiments"),
        "concepts": ("BLEU",),
    },
    {
        "question": "What hardware was used for training?",
        "type": "experiment",
        "sections": ("training", "experiments"),
        "concepts": ("GPU",),
    },
    {
        "question": "How long did training take?",
        "type": "experiment",
        "sections": ("training", "experiments"),
        "concepts": ("training time",),
    },
    {
        "question": "What ablation studies were performed?",
        "type": "ablation",
        "sections": ("model variations", "experiments"),
        "concepts": ("attention heads", "model size"),
    },
    {
        "question": "How does self-attention complexity compare with recurrence?",
        "type": "comparison",
        "sections": ("why self-attention",),
        "concepts": ("computational complexity", "sequential operations"),
    },
    {
        "question": "What limitations does the paper state explicitly?",
        "type": "limitation",
        "sections": ("conclusion", "discussion"),
        "concepts": ("limitations",),
    },
    {
        "question": "What limitations can only be inferred from its experiments?",
        "type": "interpretation",
        "sections": ("experiments", "conclusion"),
        "concepts": ("experimental scope",),
        "sufficiency": "partially_sufficient",
        "answerability": "ambiguous",
    },
    {
        "question": "Does the paper prove Transformers are always better than RNNs?",
        "type": "false_premise",
        "sections": ("results", "conclusion"),
        "prohibited": ("Transformers are always better than RNNs",),
        "sufficiency": "partially_sufficient",
        "answerability": "ambiguous",
    },
    {
        "question": "Did the paper invent attention?",
        "type": "false_premise",
        "sections": ("introduction", "references"),
        "prohibited": ("The paper invented attention",),
        "sufficiency": "partially_sufficient",
        "answerability": "ambiguous",
    },
    {
        "question": "Did the paper train a large language model?",
        "type": "false_premise",
        "sections": ("abstract", "experiments"),
        "prohibited": ("The paper trained an LLM",),
        "sufficiency": "partially_sufficient",
        "answerability": "ambiguous",
    },
    {
        "question": "Does the paper discuss instruction tuning?",
        "type": "insufficient_evidence",
        "sections": ("abstract", "conclusion"),
        "concepts": ("instruction tuning",),
        "sufficiency": "insufficient",
        "answerability": "unanswerable",
    },
    {
        "question": "How did this paper contribute to later large language models?",
        "type": "historical_impact",
        "sections": ("introduction", "conclusion"),
        "prohibited": ("The paper alone caused the LLM revolution",),
        "sufficiency": "external_required",
        "answerability": "external_sources_required",
    },
    {
        "question": "What parts of the later LLM revolution are not covered?",
        "type": "external_context_required",
        "sections": ("conclusion",),
        "concepts": ("later developments",),
        "sufficiency": "external_required",
        "answerability": "external_sources_required",
    },
    {
        "question": "What steps are needed to reproduce the base model training?",
        "type": "reproduction",
        "sections": ("training", "experiments"),
        "concepts": ("optimizer", "dataset", "model configuration"),
    },
)


def generate_attention_starter_benchmark(
    index: RetrievalIndex,
    document_id: str,
    *,
    benchmark_version: str = "0.1-attention-candidates",
) -> Benchmark:
    """Bind 33 review prompts to candidate evidence from one local paper index."""
    if not isinstance(index, RetrievalIndex):
        raise TypeError("index must be RetrievalIndex.")
    documents = {item.document_id: item for item in index.documents}
    if document_id not in documents:
        raise ValueError("document_id is absent from the retrieval index.")
    questions: list[BenchmarkQuestion] = []
    audiences = ("beginner", "undergraduate", "researcher")
    for position, template in enumerate(_STARTER_QUESTIONS):
        results = index.search(
            template["question"],
            method="bm25",
            top_k=3,
            filters=SearchFilters(document_id=document_id),
        )
        evidence = tuple(
            GoldEvidence(
                chunk_id=item.chunk_id,
                section_id=next(
                    chunk.section_id
                    for chunk in index.chunks
                    if chunk.chunk_id == item.chunk_id
                ),
                relevance_grade=max(1, 3 - rank),
            )
            for rank, item in enumerate(results)
            if item.score > 0.0
        )
        questions.append(
            BenchmarkQuestion.create(
                paper_id=document_id,
                question=template["question"],
                question_type=template["type"],
                audience_level=audiences[position % len(audiences)],
                answerability=template.get("answerability", "paper_answerable"),
                paper_sufficiency=template.get("sufficiency", "sufficient"),
                expected_sections=tuple(template.get("sections", ())),
                gold_evidence=evidence,
                required_concepts=tuple(
                    ConceptGroup(item) for item in template.get("concepts", ())
                ),
                prohibited_claims=tuple(template.get("prohibited", ())),
                review_status="proposed",
                metadata={
                    "candidate_generator": "attention_starter_v1",
                    "candidate_evidence_only": True,
                    "trusted_gold": False,
                    "review_instruction": (
                        "Verify the question, labels, concepts, and every evidence "
                        "chunk against the paper before approval."
                    ),
                },
            )
        )
    document = documents[document_id]
    return Benchmark(
        name="Attention Is All You Need starter candidates",
        benchmark_version=benchmark_version,
        index_sha256=index.index_sha256,
        document_hashes={document_id: document.content_sha256},
        questions=tuple(sorted(questions, key=lambda item: item.question_id)),
    )


def attention_starter_question_count() -> int:
    """Return the fixed number of starter prompts."""
    return len(_STARTER_QUESTIONS)
