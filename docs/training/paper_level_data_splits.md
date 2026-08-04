# Paper-level data splits

Grounded scholarly examples are split by paper, not merely by prompt. Every
example derived from a paper—including prompt variations—must remain in that
paper's split. This prevents near-identical facts and passages from appearing
in both training and evaluation data.

## Deterministic assignment

`assign_paper_splits` orders papers by a SHA-256 value derived from the seed and
paper ID, then assigns train, validation, or test according to configured
fractions. Manual paper assignments take precedence and are validated. Changing
input order does not change the result.

For multi-paper examples, all connected papers are coalesced into one split.
If Paper A is compared with Paper B, neither paper may appear in another split.
The dataset schema rejects cross-split examples and examples whose recorded
split disagrees with their papers.

Prompt variations retain a parent question/example identifier. Validation
rejects a variation split apart from its parent target.

## Recommended evaluation design

- Reserve entire papers as held-out validation and test sources.
- Keep all chunks, questions, corrections, and prompt variations from a paper
  together.
- Include conceptually related but non-identical held-out questions.
- Record manual assignments and the split seed in the artifact.
- Report results per paper as well as in aggregate.

With only a few papers, split counts can be sparse or empty. This is reported
as a limitation rather than repaired by leaking questions from the same paper
across splits.

