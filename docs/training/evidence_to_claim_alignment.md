# Evidence-to-claim alignment

LocalML Scholar now constructs grounded answers from prevalidated claims rather
than generating unrestricted prose and retrofitting citations.

The 1.2.5 pipeline is:

```text
question → local retrieval → structured candidate facts → atomic claims
         → support/relevance validation → sufficiency decision → answer plan
         → citation-first sentences → unsupported-language validation
```

`SupportedClaim` records the normalized claim, source type, stable evidence IDs,
display citation labels, support status, confidence, relevance, qualifiers, and
validation failures. Ordinary paper-grounded answers may use only `explicit` and
policy-approved `inferred_valid` claims. External background is labelled and
excluded from paper-only answers; unsupported and uncertain claims are omitted.

The relevance gate classifies claims as direct, supporting, optional, or
irrelevant. Question-type rules prevent true but unrelated passages from entering
answers. The sufficiency gate returns `sufficient`, `partially_sufficient`,
`insufficient`, or `external_source_required`. Only sufficient plans are eligible
for autonomous training.

Numerical claims require exact values, compatible units, and aligned nearby
entities. Named entities such as authors, datasets, models, optimizers, and
metrics must occur in the cited evidence. These deterministic checks are
conservative lexical safeguards; semantic Codex critics still review the same
claim graph.

The source implementation is
`src/localml_scholar/training_data/claim_alignment.py`. Authored regression tests
are in `tests/test_claim_alignment.py`.
