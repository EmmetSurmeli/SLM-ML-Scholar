# Failure-specific repair

Milestone 12A.5 replaces unrestricted answer rewriting with bounded repairs over
the structured claim graph.

- Evidence repair reretrieves, constrains sections, expands exact terms, and can
  replace the evidence set.
- Claim repair removes unsupported claims or narrows broad claims.
- Citation repair remaps current evidence labels and supports multiple passages.
- Completeness repair adds a required concept only when validated evidence exists.
- Recomposition deterministically renders the corrected answer plan.

Each attempt records before/after claim metrics, repair types, evidence/answer/
citation changes, and an outcome: fixed, unchanged, worsened, or introduced a new
failure. A repair is successful only when hard failures decrease without creating
another hard failure. The reporting layer calculates success separately by repair
type.

A repaired answer is not automatically accepted. It must pass evidence,
claim-support, claim-citation completeness, sentence traceability, critic
agreement, provenance, duplicate, leakage, and confidence gates. Controlled
diagnostics remain mandatory before full autonomous curation.
