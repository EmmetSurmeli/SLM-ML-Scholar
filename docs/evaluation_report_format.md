# Evaluation report format

The evaluation run is the machine-readable source of truth. It stores raw
ranked results, raw answer artifacts, stage grades, failure labels, root-cause
diagnostics, configuration, source/index/benchmark identities, and aggregate
metrics.

Markdown renderers provide:

- benchmark review state and untrusted-candidate warning
- system summary with retrieval, sufficiency, answer, citation, and audience
  sections
- per-paper question/audience distributions, metrics, wrong sections, missing
  concepts, common failures, best/worst IDs, and review backlog
- per-question-type aggregates
- every failed question with answer, evidence, gold evidence, missing
  concepts, failures, likely cause, and suggested inspection
- review-queue status
- regression deltas and every regressed question
- correction-dataset preview

Zero denominators are defined as `0.0` for aggregate groups with no applicable
records; the question-level metric retains its own documented convention.
Reports only display computed values. Human-reviewed rates are produced from
adjudicated `HumanReviewRecord` objects and never infer pending labels.

Automatic grades are transparent heuristics. The report explicitly states
that they do not prove semantic correctness.

