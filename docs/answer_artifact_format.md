# Grounded answer artifact format

Grounded answers use deterministic, UTF-8, non-pickle JSON. The current
`answer_format_version` is `1`.

```json
{
  "answer_format_version": 1,
  "package_version": "0.9.0",
  "artifact_type": "grounded_answer",
  "answer": {}
}
```

The nested answer stores:

- the exact question and answer method;
- final answer text plus raw/processed generation when applicable;
- segmented claims and attached labels;
- every selected exact evidence slice and structured citation;
- sufficiency diagnostics;
- abstention and fallback state;
- citation/support validation;
- index, corpus, evidence, tokenizer, and checkpoint identities where
  applicable;
- complete retrieval, selection, acceptance, context, and generation
  configuration metadata.

`EvidenceItem.selected_text_sha256` protects the included source text.
`AnswerValidation.evidence_hash` binds the ordered labels, evidence IDs, chunk
IDs, text hashes, and citations. The index hash binds the exact immutable
retrieval snapshot.

`save_grounded_answer` writes atomically. `load_grounded_answer` constructs and
validates a fresh object before returning it. Unknown versions, missing keys,
extra keys, invalid hashes, inconsistent bindings, or malformed values are
errors. Supplying the index to the loader additionally reruns claim and
citation validation against exact document ranges:

```python
from localml_scholar import RetrievalIndex, load_grounded_answer

index = RetrievalIndex.load("outputs/local_documents/index.json")
answer = load_grounded_answer(
    "outputs/local_documents/answer.json",
    index=index,
)
```

Artifacts are not forward-compatible by assumption. A schema migration must
be explicit rather than silently accepting unknown state.
