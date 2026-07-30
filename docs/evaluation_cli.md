# Evaluation CLI

All inputs are local. No command calls an external model, web search, or hosted
grader.

Generate generic source-linked candidates:

```bash
python3 -m localml_scholar.evaluation.cli generate-candidates \
  --index outputs/paper/index.json \
  --document-id DOCUMENT_ID \
  --output outputs/evaluation/candidates.json \
  --review-report outputs/evaluation/candidates.md
```

Create the 33-question untrusted Attention-paper starter:

```bash
python3 -m localml_scholar.evaluation.cli attention-starter \
  --index outputs/paper/index.json \
  --document-id DOCUMENT_ID \
  --output outputs/evaluation/attention_candidates.json
```

Apply explicit review decisions:

```bash
python3 -m localml_scholar.evaluation.cli review-benchmark \
  --candidates outputs/evaluation/candidates.json \
  --decisions local_review_decisions.json \
  --output outputs/evaluation/approved_benchmark.json
```

Run retrieval-only or extractive evaluation:

```bash
python3 -m localml_scholar.evaluation.cli run \
  --benchmark outputs/evaluation/approved_benchmark.json \
  --index outputs/paper/index.json \
  --method extractive --retriever bm25 --top-k 5 \
  --output outputs/evaluation/run.json
```

Generative modes additionally require `--checkpoint`; the matching tokenizer
identity is read from that checkpoint and recorded. They are never silently
substituted.

Render reports and a review queue:

```bash
python3 -m localml_scholar.evaluation.cli report \
  --run outputs/evaluation/run.json \
  --benchmark outputs/evaluation/approved_benchmark.json \
  --output outputs/evaluation/summary.md

python3 -m localml_scholar.evaluation.cli build-review-queue \
  --run outputs/evaluation/run.json \
  --benchmark outputs/evaluation/approved_benchmark.json \
  --pass-sample-fraction 0.10 --seed 0 \
  --output outputs/evaluation/review_queue.json
```

Compare two exact benchmark runs:

```bash
python3 -m localml_scholar.evaluation.cli compare \
  --baseline outputs/evaluation/old.json \
  --candidate outputs/evaluation/new.json \
  --output outputs/evaluation/comparison.json \
  --markdown outputs/evaluation/comparison.md
```

`--resume` accepts only an identical run ID, benchmark/index identity, and
configuration. Completed question records are reused exactly.

