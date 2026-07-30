# Human-approved correction dataset

`CorrectionExample` is a future grounded-instruction-data record containing
the question, canonical audience level, approved evidence chunk IDs, one cited
`StructuredAnswerTarget`, corrected and original answers, failure labels,
citations, paper ID, exact source hash, and human review ID.

Export is intentionally strict:

1. The benchmark question must be `approved` or `edited`.
2. The review must have a substantive label: `correct`,
   `partially_correct`, or `incorrect`.
3. A human corrected answer and exact corrected evidence are required.
4. Every corrected chunk must be approved benchmark evidence.
5. Benchmark and paper source hashes must still match the supplied index.
6. Pending, `should_abstain`, and `benchmark_problem` records are excluded.

Export with:

```bash
python3 experiments/export_grounded_corrections.py \
  --benchmark outputs/evaluation/approved_benchmark.json \
  --index outputs/paper/index.json \
  --reviews outputs/evaluation/human_reviews.json \
  --output outputs/evaluation/corrections.json \
  --preview outputs/evaluation/corrections.md
```

The JSON is atomically written and hash-checked. It is not consumed by a
trainer in Milestone 11.5. A later milestone must separately validate
provenance/licensing, define splits, and compare learned behavior against the
trusted deterministic renderer.

