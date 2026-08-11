# Topic-aware question generation

Generic benchmark templates are proposals, not evidence that a paper can
answer them. Package 1.2.6 checks local source text and recovered headings
before creating topic-dependent candidates.

The current deterministic signal families cover ablation, architecture,
complexity, reproduction details, experiments, explicit limitations,
derivations, and methods. Ablation requires an explicit ablation/variation
section heading; prose containing `variant` or `without` is not enough.
Architecture requires a structural architecture or encoder/decoder heading,
and complexity requires a complexity/IO-analysis heading or a direct phrase
such as `memory complexity` or `time complexity`. Inferred-limitations prompts
are suppressed from the stable training path, and derivations, historical
impact, generic result/extraction prompts, cross-paper synthesis, and
figure/table interpretation remain deferred for the pilot.

Each generated `QuestionCandidate` records:

- its question type and expected sections;
- whether topic preflight was applied;
- how many templates had already been suppressed;
- its expected answerability pool.

Paper health is checked first. An unhealthy paper generates no autonomous
questions. Content-adaptive section questions use real or inferred titles and
are never fabricated as repeated `Untitled section` prompts.

Inspect the current corpus without calling Codex:

```bash
python3 -m localml_scholar.training_data.cli question-eligibility-report
```

The report separates eligible counts and suppressed-template counts per paper.
These are deterministic candidate diagnostics, not a claim that every retained
question has a correct final answer. The pilot applies a second retrieval and
direct-answer preselection pass, followed by sufficiency, claim validation, and
reviewer gates.
