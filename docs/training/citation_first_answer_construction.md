# Citation-first answer construction

An `AnswerPlan` lists direct, supporting, qualification, inference, and omitted
claim IDs before any final answer text exists. Its citation plan binds every
selected claim to one or more current evidence labels.

The composer emits `AnswerSentence` objects. Each contains exact text, claim IDs,
and citation labels. Final text is a deterministic rendering of those objects;
the Codex review provider's `corrected_answer` is retained for audit but discarded
as a prose source. This prevents a later rewrite from adding facts or moving a
citation to the wrong sentence.

After composition, the unsupported-language detector checks that rendered text
equals the planned claims and flags new numbers, named entities, causal claims,
comparisons, superiority language, or other changed factual content. Connective
prose and fixed abstention messages are the only uncited exemptions.

Multi-citation claims retain every required label. When evidence is replaced,
the system rebuilds the claim graph and resolves labels from stable evidence
identities rather than preserving old retrieval ranks.

Diagnostics expose:

```text
Question
Evidence E1, E2
Validated claims K1 → E1, K2 → E2
Answer plan K1 + K2
Final answer sentence [C1] sentence [C2]
```

The dashboard provides expandable sentence → claim → evidence traces for recent
diagnostic records.
