# Evaluation benchmark format

Evaluation artifacts are UTF-8 JSON envelopes with:

- `artifact_type`
- `format_version`
- recorded `package_version`
- canonical `payload`
- `payload_sha256`

Benchmark payloads also contain `benchmark_version`, `index_sha256`, exact
`document_hashes`, questions, and `benchmark_sha256`. Load verifies the
canonical hash before constructing typed objects. Supplying an index also
rejects changed source hashes, unknown chunk/section IDs, and invalid ranges.

## Question record

Every `BenchmarkQuestion` records:

- deterministic `question_id`, exact `paper_id`, question, type, and audience
- answerability: `paper_answerable`, `external_sources_required`,
  `unanswerable`, or `ambiguous`
- paper sufficiency: `sufficient`, `partially_sufficient`, `insufficient`, or
  `external_required`
- expected and forbidden sections
- exact graded `gold_evidence` and acceptable alternate chunk IDs
- required/optional concept groups and aliases
- prohibited claims, expected numbers/identifiers, abstention reasons, and
  completeness requirements
- optional core answer and required reviewer notes
- `proposed`, `approved`, `edited`, or `rejected` state

Answerability/sufficiency pairs are validated for contradictions. Approved
paper-answerable questions require exact evidence and reviewer notes.
Proposed questions never enter `EvaluationRunner`.

## Review decisions

The `review-benchmark` command accepts an object keyed by question ID:

```json
{
  "bq_example": {
    "status": "edited",
    "gold_notes": "Checked against section 3.2.",
    "edits": {
      "audience_level": "undergraduate",
      "paper_sufficiency": "sufficient",
      "gold_evidence": [
        {
          "chunk_id": "chk_example",
          "section_id": "sec_example",
          "start_character": null,
          "end_character": null,
          "relevance_grade": 3
        }
      ],
      "required_concepts": [
        {"concept": "score scaling", "aliases": ["scaled logits"]}
      ]
    }
  }
}
```

Omitted questions retain their current state and are never auto-approved.

