"""Deterministic candidate-question and prompt-variation generation."""

from __future__ import annotations

from collections import Counter

from localml_scholar.training_data.schemas import QuestionCandidate

# This exact catalog is the Milestone 12A Attention-paper starter. Every entry is
# proposed data: its concepts, evidence, and wording require human review.
ATTENTION_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Who wrote Attention Is All You Need?", "metadata"),
    ("What problem does the paper address?", "motivation"),
    ("What is the Transformer?", "method"),
    ("What are the main components of the architecture?", "architecture"),
    ("What tasks are evaluated?", "experiment"),
    ("Which datasets are used?", "experiment"),
    ("What BLEU scores are reported?", "result"),
    ("How long did training take?", "reproduction"),
    ("What hardware was used?", "reproduction"),
    ("What optimizer was used?", "reproduction"),
    ("What Adam beta values were used?", "hyperparameter"),
    ("How many warmup steps were used?", "hyperparameter"),
    ("What dropout rate was used?", "hyperparameter"),
    ("What label smoothing value was used?", "hyperparameter"),
    ("How many encoder layers are used?", "architecture"),
    ("How many decoder layers are used?", "architecture"),
    ("What is the model dimension?", "hyperparameter"),
    ("How many attention heads are used?", "hyperparameter"),
    ("What are the differences between the base and big models?", "comparison"),
    ("What is self-attention?", "architecture"),
    ("What is scaled dot-product attention?", "equation"),
    ("Why divide the attention logits by sqrt(d_k)?", "equation"),
    ("What are queries, keys, and values?", "architecture"),
    ("What is multi-head attention?", "architecture"),
    ("Why use multiple heads?", "architecture"),
    ("Why is positional encoding required?", "architecture"),
    ("How are positional encodings constructed?", "equation"),
    ("Why use sinusoidal positional encodings?", "architecture"),
    (
        "How does encoder self-attention differ from decoder self-attention?",
        "comparison",
    ),
    ("What does causal masking do?", "architecture"),
    ("What role do residual connections play?", "architecture"),
    ("What role does LayerNorm play?", "architecture"),
    ("What does the feed-forward network do?", "architecture"),
    ("Why is the Transformer easier to parallelize than an RNN?", "comparison"),
    (
        "How does path length differ between self-attention and recurrence?",
        "comparison",
    ),
    ("What is the computational complexity of self-attention?", "complexity"),
    ("When does self-attention become expensive?", "complexity"),
    ("What tradeoff exists between sequence length and model width?", "complexity"),
    ("Why might self-attention help long-range dependencies?", "interpretation"),
    (
        "Does the paper empirically prove that attention always handles "
        "long-range dependencies better?",
        "false_premise",
    ),
    ("What are the main experimental claims?", "experiment"),
    ("Which baselines are compared?", "experiment"),
    ("What ablation studies were run?", "ablation"),
    ("What effect did changing the number of heads have?", "ablation"),
    ("What effect did model dimensionality have?", "ablation"),
    ("What evidence supports the chosen positional encoding?", "experiment"),
    (
        "Which result most strongly supports the paper's main claim?",
        "critical_reasoning",
    ),
    ("How convincing are the experimental comparisons?", "critical_reasoning"),
    ("Did the paper invent attention?", "false_premise"),
    ("Did the paper invent self-attention?", "false_premise"),
    ("Did the paper train an LLM?", "false_premise"),
    ("Does the paper discuss instruction tuning?", "insufficient_evidence"),
    ("Does the paper discuss RLHF?", "insufficient_evidence"),
    ("Does the paper prove Transformers are always better than RNNs?", "false_premise"),
    (
        "What claims are architectural arguments rather than empirical findings?",
        "critical_reasoning",
    ),
    ("What does the paper not establish?", "limitation"),
    ("What limitations are stated explicitly?", "limitation"),
    ("What limitations can be reasonably inferred?", "limitation"),
    ("What details are needed to reproduce the base model?", "reproduction"),
    ("What reproduction details are ambiguous or missing?", "reproduction"),
    ("What would make exact replication difficult today?", "reproduction"),
    ("Why was this paper important for later language models?", "external_context"),
    (
        "Which parts of modern LLMs come directly from the Transformer architecture?",
        "external_context",
    ),
    (
        "Which important modern LLM techniques are absent from this paper?",
        "external_context",
    ),
    ("Can this paper alone explain the later LLM revolution?", "false_premise"),
    (
        "What additional papers would be needed to trace the development from "
        "the Transformer to modern LLMs?",
        "external_context",
    ),
    ("Explain attention like I'm completely new to ML.", "teaching"),
    ("I know linear algebra. Explain self-attention mathematically.", "teaching"),
    ("Walk me through the attention equation line by line.", "derivation"),
    ("Why do queries and keys need separate projections?", "teaching"),
    ("I understand dot products but not attention. Explain it from there.", "teaching"),
    ("Why doesn't the model lose word order?", "teaching"),
    ("What would break if positional encodings were removed?", "counterfactual"),
    (
        "Explain multi-head attention without jargon first, then technically.",
        "teaching",
    ),
    ("I'm confused about the mask in the decoder. Walk me through it.", "teaching"),
    (
        "Explain the difference between encoder attention and decoder attention "
        "using an example.",
        "teaching",
    ),
    (
        "Why is the feed-forward network needed if attention already mixes "
        "information?",
        "teaching",
    ),
    ("What is the single most important insight of this paper?", "interpretation"),
    ("What should an undergraduate remember from this paper?", "teaching"),
    (
        "What should a researcher be skeptical about in this paper?",
        "critical_reasoning",
    ),
)

_GENERIC_QUESTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Who wrote {title}?", "metadata", ("title",)),
    ("What is {title} trying to accomplish?", "metadata", ("abstract", "introduction")),
    ("What venue or date is stated in the paper?", "metadata", ("title",)),
    ("What problem domain does this work address?", "metadata", ("abstract",)),
    ("What problem motivated the work?", "motivation", ("introduction",)),
    (
        "What limitations of prior approaches do the authors identify?",
        "motivation",
        ("introduction", "related work"),
    ),
    (
        "Why do the authors argue that a new method is needed?",
        "motivation",
        ("introduction",),
    ),
    ("What is the central method?", "method", ("method",)),
    ("Explain the proposed method step by step.", "method", ("method",)),
    (
        "What are the main architectural components?",
        "architecture",
        ("method", "architecture"),
    ),
    ("What changes relative to prior work?", "comparison", ("method", "related work")),
    ("What is the paper's key equation?", "equation", ("method",)),
    ("Define every term in the main objective.", "equation", ("method",)),
    ("Why is the main mathematical operation used?", "equation", ("method",)),
    (
        "How do the authors move from one equation to the next?",
        "derivation",
        ("method",),
    ),
    ("What assumptions are required by the derivation?", "derivation", ("method",)),
    ("What happens to the method in a simple special case?", "derivation", ("method",)),
    ("Explain the core idea intuitively.", "intuition", ("method",)),
    (
        "Give an analogy for the main method and state where it breaks down.",
        "intuition",
        ("method",),
    ),
    ("Why should the method work?", "intuition", ("method",)),
    ("What is the paper's core insight?", "intuition", ("abstract", "method")),
    ("Derive the key equation and label inferred steps.", "derivation", ("method",)),
    ("What is the computational complexity?", "complexity", ("method", "experiments")),
    ("What is the memory complexity?", "complexity", ("method", "experiments")),
    (
        "What inductive bias does the method introduce?",
        "critical_reasoning",
        ("method",),
    ),
    (
        "What assumptions are implicit rather than stated?",
        "critical_reasoning",
        ("method",),
    ),
    ("Which datasets were used?", "experiment", ("experiments",)),
    ("Which baselines were used?", "experiment", ("experiments",)),
    ("Which metrics were used?", "experiment", ("experiments",)),
    ("What are the main results?", "result", ("experiments", "results")),
    ("Which result most directly supports the central claim?", "result", ("results",)),
    (
        "Were the experimental comparisons controlled?",
        "critical_reasoning",
        ("experiments",),
    ),
    ("What components were ablated?", "ablation", ("ablation", "experiments")),
    ("Which ablated component mattered most?", "ablation", ("ablation",)),
    ("What does the ablation establish?", "ablation", ("ablation",)),
    ("What does the ablation fail to establish?", "ablation", ("ablation",)),
    ("What optimizer was used?", "reproduction", ("training", "experiments")),
    ("What learning rate was used?", "reproduction", ("training", "experiments")),
    ("What batch size was used?", "reproduction", ("training", "experiments")),
    ("What hardware was used?", "reproduction", ("training", "experiments")),
    ("How long did training take?", "reproduction", ("training", "experiments")),
    (
        "Which implementation details are missing?",
        "reproduction",
        ("method", "experiments"),
    ),
    (
        "What would be needed to reproduce the main result?",
        "reproduction",
        ("method", "experiments"),
    ),
    (
        "Which limitations do the authors state?",
        "limitation",
        ("limitations", "conclusion"),
    ),
    (
        "Which limitations can be inferred from experimental scope?",
        "limitation",
        ("experiments", "conclusion"),
    ),
    (
        "What conditions could cause the method to fail?",
        "limitation",
        ("method", "limitations"),
    ),
    ("How general are the experiments?", "limitation", ("experiments",)),
    (
        "Does the evidence support the main claim?",
        "critical_reasoning",
        ("results", "conclusion"),
    ),
    (
        "What alternative explanation could fit the results?",
        "critical_reasoning",
        ("results",),
    ),
    (
        "Which conclusions are stronger than the experiments justify?",
        "critical_reasoning",
        ("results", "conclusion"),
    ),
    (
        "What experimental confounders might exist?",
        "critical_reasoning",
        ("experiments",),
    ),
    (
        "What additional experiment would strengthen the main claim?",
        "critical_reasoning",
        ("experiments",),
    ),
    (
        "Did the paper prove that its method always works?",
        "false_premise",
        ("results", "conclusion"),
    ),
    (
        "Did the paper invent every concept used by the method?",
        "false_premise",
        ("related work",),
    ),
    (
        "Did the paper evaluate every task where the method could be used?",
        "false_premise",
        ("experiments",),
    ),
    (
        "What important claim cannot be answered from this paper alone?",
        "insufficient_evidence",
        ("conclusion",),
    ),
    (
        "Which question would require another source to answer responsibly?",
        "insufficient_evidence",
        ("references",),
    ),
    ("Explain the main idea simply.", "teaching", ("method",)),
    ("Explain the main idea using linear algebra.", "teaching", ("method",)),
    ("Explain the method to someone who knows basic ML.", "teaching", ("method",)),
    ("Explain the method mathematically.", "teaching", ("method",)),
    ("Walk me through the most important equation.", "teaching", ("method",)),
    ("Why should I care about this work?", "teaching", ("abstract", "conclusion")),
    ("Give me the intuition first, then the math.", "teaching", ("method",)),
    (
        "I don't get what the central method section is saying.",
        "user_style",
        ("method",),
    ),
    (
        "Why did the authors do this instead of the main prior approach?",
        "user_style",
        ("method", "related work"),
    ),
    ("What's actually new here?", "user_style", ("method", "related work")),
    (
        "Can you walk me through this paper?",
        "user_style",
        ("abstract", "method", "experiments"),
    ),
    ("Is the main result actually impressive?", "user_style", ("results",)),
    ("How would I implement this?", "user_style", ("method", "experiments")),
    (
        "What should I take away from the main figure or table?",
        "user_style",
        ("experiments",),
    ),
    (
        "Create a concise implementation checklist.",
        "reproduction",
        ("method", "experiments"),
    ),
    (
        "List the dataset, architecture, loss, optimizer, metrics, and "
        "hyperparameters.",
        "extraction",
        ("method", "experiments"),
    ),
    (
        "Which claims are directly stated and which require inference?",
        "provenance",
        ("results", "conclusion"),
    ),
    (
        "What terminology should I learn before reading this paper?",
        "prerequisites",
        ("abstract", "method"),
    ),
    (
        "What is the strongest reason to be skeptical of this work?",
        "critical_reasoning",
        ("experiments", "limitations"),
    ),
    (
        "What follow-up project would test the paper's weakest assumption?",
        "extension",
        ("method", "limitations"),
    ),
    (
        "How could this method be extended without changing its central idea?",
        "extension",
        ("method", "conclusion"),
    ),
    (
        "What evidence would falsify the central claim?",
        "critical_reasoning",
        ("results", "conclusion"),
    ),
    (
        "Summarize the paper's contribution without overstating it.",
        "summary",
        ("abstract", "conclusion"),
    ),
)


def generate_paper_questions(
    paper_id: str,
    title: str,
    *,
    count: int = 60,
    section_titles: tuple[str, ...] = (),
) -> tuple[QuestionCandidate, ...]:
    """Generate 40--80 deterministic, untrusted candidates for one paper."""
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("paper_id must contain non-whitespace text.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must contain non-whitespace text.")
    if isinstance(count, bool) or not isinstance(count, int) or not 40 <= count <= 80:
        raise ValueError("count must be an integer in [40, 80].")
    if not isinstance(section_titles, tuple) or not all(
        isinstance(item, str) for item in section_titles
    ):
        raise TypeError("section_titles must be a tuple of strings.")
    if "attention is all you need" in title.casefold():
        templates = tuple(
            (question, kind, ()) for question, kind in ATTENTION_QUESTIONS
        )
    else:
        grouped: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
        for template in _GENERIC_QUESTIONS:
            grouped.setdefault(template[1], []).append(template)
        balanced = []
        offset = 0
        while len(balanced) < len(_GENERIC_QUESTIONS):
            for group in grouped.values():
                if offset < len(group):
                    balanced.append(group[offset])
            offset += 1
        templates = tuple(balanced)
    candidates = []
    for question, kind, sections in templates[:count]:
        candidates.append(
            QuestionCandidate.create(
                paper_ids=(paper_id,),
                question=question.format(title=title.strip()),
                question_type=kind,
                expected_sections=sections,
                review_status="proposed",
                metadata={
                    "candidate_only": True,
                    "generator": "paper_question_templates_v1",
                    "trusted_gold": False,
                    "paper_section_titles": list(section_titles),
                },
            )
        )
    return tuple(candidates)


def generate_prompt_variations(
    candidate: QuestionCandidate,
) -> tuple[QuestionCandidate, ...]:
    """Propose semantically related phrasings; never approve them automatically."""
    if not isinstance(candidate, QuestionCandidate):
        raise TypeError("candidate must be QuestionCandidate.")
    base = candidate.question.rstrip("?.")
    prompts = (
        f"Could you explain {base[:1].lower() + base[1:]}?",
        f"Walk me through this: {candidate.question}",
        f"In plain language, {base[:1].lower() + base[1:]}?",
        f"Give a rigorous answer with citations: {candidate.question}",
    )
    return tuple(
        QuestionCandidate.create(
            paper_ids=candidate.paper_ids,
            question=prompt,
            question_type=candidate.question_type,
            expected_sections=candidate.expected_sections,
            required_concepts=candidate.required_concepts,
            prohibited_claims=candidate.prohibited_claims,
            review_status="proposed",
            parent_question_id=candidate.question_id,
            metadata={
                "candidate_only": True,
                "variation_requires_human_approval": True,
            },
        )
        for prompt in prompts
    )


def question_type_counts(
    candidates: tuple[QuestionCandidate, ...],
) -> dict[str, int]:
    """Return deterministic question-type counts."""
    return dict(sorted(Counter(item.question_type for item in candidates).items()))
