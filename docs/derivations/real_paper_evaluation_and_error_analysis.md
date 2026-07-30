# Real-paper evaluation and error analysis

## 1. Grounded is not equivalent to correct

A citation has several independent properties. Its syntax may parse; its label
may exist; its source range may match an immutable chunk; the passage may
support the attached sentence; and the sentence may still fail to answer the
question. LocalML Scholar therefore never uses “citation valid” as a synonym
for “answer correct.”

The evaluator reports citation validity, support, relevance, precision,
recall, and claim coverage separately. These are deterministic diagnostics,
not proof of semantic correctness.

## 2. Stage-wise evaluation

For a question \(q\), retrieval produces ranked chunks
\((r_1,\ldots,r_k)\). Evidence selection and sufficiency decide whether an
answer should be attempted. The answer stage is graded for interrogative fit,
required content, prohibited content, numbers, and completeness. Citation
grading checks exact bindings and question relevance. Audience grading then
examines a rendering of the same factual target.

Keeping these stages separate answers useful questions: Did the right passage
fail to rank? Was adequate evidence rejected? Did answer construction omit a
known fact? Was a valid citation attached to an irrelevant answer?

## 3. Human-approved gold benchmarks

Deterministic scholarly extraction can propose useful questions and candidate
evidence. It cannot declare its own proposal correct. Candidate state is
therefore `proposed`; only explicit human `approved` or `edited` state is used
by official runs. Benchmarks bind paper content hashes, index hashes, exact
chunk IDs, ranges, concepts, and reviewer notes.

## 4. Question types and section policy

Evidence expectations depend on the question. Metadata normally favors title
or explicit metadata; motivation favors abstract/introduction; methods favor
architecture/method sections; experiments favor experiment/result sections.
References and author contributions are not globally discarded because some
questions legitimately need them. The policy applies context-specific
expected, forbidden, and boilerplate diagnostics.

## 5. Answer relevance

Transparent interrogative checks are combined with required-concept and
question-term signals. A “who” answer should contain an expected identifier or
person-like entity; “when” should contain an expected date; dataset questions
should identify a dataset; and “how” or “why” should contain mechanism or
rationale language. These checks catch obvious mismatches, but names and
mechanisms cannot be recognized perfectly without semantic judgment.

## 6. Required-concept recall

For manually specified concept groups,

\[
\operatorname{required\ recall}
=
\frac{\text{required concepts present in a supported cited claim}}
{\text{number of required concepts}}.
\]

A concept may use its canonical phrase or a reviewed alias. The evaluator also
records present-but-uncited and present-but-unsupported concepts instead of
hiding them in the scalar.

## 7. Prohibited claims

Gold annotations may list claims that must not be accepted, such as universal
superiority or a false historical premise. Detection uses normalized,
transparent phrase rules and reports the exact matched annotation. It is
conservative and cannot find arbitrary paraphrases.

## 8. One factual target, three audiences

`StructuredAnswerTarget` stores cited core, supporting, equation, assumption,
qualification, and limitation points. Beginner, undergraduate, and researcher
renderers select different depths from this structure. Shared claims retain
the same citation labels. The factual-basis check verifies that the core terms
and citations survive every rendering; no renderer invents an analogy.

## 9. Failure taxonomy

A response may fail in multiple places simultaneously. The evaluator retains
all applicable labels rather than forcing a single class. Root-cause
attribution checks stage groups in a stable order, records secondary causes,
and lowers confidence when several stages are plausible. Its next action is a
debugging recommendation, not a causal proof.

## 10. Selective human review

Human attention goes to all failures, low-confidence cases, method
disagreements, risky question types, numerical contradictions, benchmark
ambiguities, citation-valid/relevance disagreements, and a deterministic pass
sample. Sampling uses a local seeded PRNG. Pending records do not affect
human-reviewed pass metrics.

## 11. Regression analysis

Two runs are comparable only when they use the same benchmark hash and exact
question IDs. Comparison retains configuration and package versions, metric
deltas, improved/regressed IDs, new/resolved failures, and answer, evidence,
and citation changes. Aggregate improvement never removes a question-level
regression. No significance is claimed without justified sampling assumptions.

## 12. Training-data export

Correction records are future instruction-data candidates, not automatic
training events. Export requires an approved benchmark question, a substantive
human label, a corrected answer, exact approved gold chunks, and the matching
paper hash. Benchmark-problem and pending records are excluded.

## 13. Limitations

Relevance and style are heuristic; concepts are manually authored; benchmarks
may be biased; a five-paper set is small; graders can produce false positives;
there is no semantic entailment model or external historical verification;
source extraction can be wrong; and reviewer disagreement is not resolved by
the software. These limitations prohibit semantic-correctness claims.

## 14. Source mapping

- schemas and hashes: `evaluation/schemas.py`,
  `evaluation/serialization.py`
- candidate/approval workflow: `evaluation/benchmark.py`,
  `evaluation/attention_starter.py`
- retrieval equations and policy: `evaluation/retrieval_grading.py`
- sufficiency, concepts, relevance, claims, citations:
  `evaluation/answer_grading.py`
- structured target and audience rendering: `evaluation/audience.py`
- taxonomy and attribution: `evaluation/failures.py`
- queue and correction export: `evaluation/human_review.py`
- runner and aggregates: `evaluation/runner.py`
- regression comparison: `evaluation/comparison.py`
- Markdown: `evaluation/reports.py`
- validation tests: `tests/test_real_paper_evaluation_*.py`

