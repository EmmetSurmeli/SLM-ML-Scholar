# Retrieval query normalization

Package 1.2.6 defines one query-only lexical policy shared by `SearchQuery`,
BM25/TF-IDF query construction, evidence selection, essential-concept
extraction, and sufficiency diagnostics.

The source index remains lossless: document terms are not deleted. Query
normalization case-folds and tokenizes with the existing deterministic lexical
tokenizer, removes common English function words, preserves technical tokens
and numbers, and removes duplicates without changing their first-seen order.
It also removes question-only intent terms such as `used`, `main`, and
`strongest`, and applies two conservative aliases: `idea -> method` and
`skeptical -> limitations`.

For example:

```text
What does the ablation establish?
-> (ablation, establish)
```

`question_concepts` then separates essential concepts from optional intent
words. Here `ablation` is essential, `establish` is an optional conclusion
intent, and the question intent is `ablation`. Stopwords such as `what`, `does`,
and `the` cannot create an evidence match.

Evidence eligibility requires an essential concept match. Sufficiency then
checks essential coverage, retrieval score, substantive content, and—when the
candidate specifies expected sections—section/topic compatibility in the same
passage. A table-of-contents heading plus unrelated text cannot satisfy the
gate. Pilot preselection additionally checks that extractive claims cite the
expected section and that numeric/complexity questions contain a direct answer
signal.

This remains a lexical heuristic rather than semantic entailment. Synonyms,
mathematical equivalence, extraction errors, and specialized morphology can
still cause false positives or false abstentions; those limitations are
reported rather than hidden by lower thresholds.
