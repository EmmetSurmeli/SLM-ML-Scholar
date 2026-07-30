# Paper-specific scholarly analysis

This document maps the deterministic Milestone 11 extraction policy to source
code and tests. It describes provenance-preserving heuristics, not a claim of
semantic understanding.

## 1. Why paper analysis differs from generic QA

Generic QA returns passages or answers for one question. Scholarly analysis
must retain relationships among sections, symbols, equations, assumptions,
methods, experiments, numerical results, and limitations. LocalML Scholar
therefore represents every extracted fact as:

\[
e=(v,n,r,m,c,s),
\]

where \(v\) is the retained value, \(n\) its conservative normalization,
\(r=[a,b)\) an exact source range, \(m\) the disclosed extraction method,
\(c\) a categorical confidence label, and \(s\) its validation state. The
source invariant is:

\[
\operatorname{Document.text}[a:b]
=
\operatorname{e.source\_text}.
\]

The UTF-8 hash of that substring is stored in the citation and revalidated on
artifact load.

## 2. Paper sections

Section roles are ordered labels inferred from normalized headings. A rule
fires only when a public phrase matches; multiple roles remain possible. An
unmatched section is `unknown`, never discarded. Confidence is categorical:
it is not a calibrated posterior probability.

Implementation: `scholarly/sections.py`. Validation:
`test_scholarly_metadata_sections.py`.

## 3. Mathematical blocks

Candidates come from textual delimiters (`$`, `$$`, `\(`, `\[`, and represented
equation environments) plus narrow operator-heavy lines. Candidate intervals
are ordered by start, detector priority, length, and method. A candidate is
retained only when it does not overlap an earlier retained interval and lies
inside one canonical section.

Normalization changes whitespace and unambiguous minus glyphs only. It does not
rename variables, execute LaTeX, infer parentheses, simplify expressions, or
claim equivalence.

Implementation: `scholarly/equations.py`.

## 4. Notation glossaries

Notation extraction records exact occurrences of Latin, indexed, Greek,
Unicode, and common LaTeX-represented symbols. Operator commands such as
`\sum`, `\frac`, and `\log` are excluded from variable entries. Definitions
come from explicit patterns such as “where \(x\) denotes,” “let \(x\) be,” and
“define \(x\) as.”

A definition is selectable only when all retained candidates provide one
normalized defining text. Conflicting candidates remain attached and the
symbol is marked ambiguous. Symbols without candidates are listed as
unresolved.

## 5. Assumptions and claims

Assumptions default to explicit phrases (`we assume`, `suppose that`, `subject
to`, and narrow named conditions). Inferred prerequisites are disabled by
default and, when enabled, are labeled ambiguous and non-explicit.

Claims are limited to explicit contribution, theoretical, empirical,
comparative, limitation, and future-work cues. Qualifiers such as “on average,”
“under these settings,” and “may” are retained. Citation linkage does not prove
the claim.

## 6. Methodology and experiments

Methods, optimizers, objectives, architectures, preprocessing, hardware,
software, datasets, metrics, baselines, and hyperparameters are extracted with
category-specific patterns. Repeated occurrences are retained because identical
names can belong to different experiment scopes. Experiments are grouped only
inside explicitly classified experiment or ablation sections.

No field is borrowed from another section silently. Conflicting
hyperparameters retain every raw value and source scope.

## 7. Numerical result parsing

The parser recognizes signed integers/decimals, percentages, and an optional
`±` uncertainty. A number becomes a result only inside a result, experiment, or
ablation section and in a sentence that names a supported metric or contains an
explicit result verb such as “achieves.” Identifier digits (`F1`), table/
figure/equation labels, and training-count units are excluded. Raw text, parsed
value, percent unit, uncertainty, and sentence scope are retained.

The system does not turn relative changes into absolute scores or infer visual
table coordinates.

## 8. Reproduction checklists

Each predefined item is classified:

- `found`: cited values exist;
- `ambiguous`: cited candidates require interpretation;
- `conflicting`: multiple incompatible scoped values exist;
- `not_found`: no matching evidence exists in the analyzed document.

Risk flags follow explicit missing/conflict rules. They are document
completeness observations, not proof that reproduction is impossible.

## 9. Cross-paper comparison

For dimension \(d\), each paper contributes a set of normalized cited values
\(V_{p,d}\). Equality of all nonempty sets yields `shared`; differing sets
yield `different`; an empty set yields `missing`. Numerical results become
`incomparable` when datasets or metrics differ. No score or superiority
ranking is produced for an incomparable dimension.

## 10. Research gaps

Direct candidates originate in explicit limitations and future-work claims.
System candidates may originate in a deterministic missing-ablation rule.
Every candidate records `system_inference`, cited basis, and the caution:
external literature search was not performed, so novelty is not established.
Question templates are planning aids, not factual claims.

## 11. Evaluation

For predicted field set \(P\) and expected set \(G\):

\[
\operatorname{precision}=\frac{|P\cap G|}{|P|},\qquad
\operatorname{recall}=\frac{|P\cap G|}{|G|},
\]

\[
F_1=
\frac{2\,\operatorname{precision}\operatorname{recall}}
{\operatorname{precision}+\operatorname{recall}}.
\]

Empty predicted and expected sets receive precision, recall, and \(F_1\) of
one. Exact-value accuracy is one only when \(P=G\). Classification accuracy is
the fraction of authored keys with exact labels. Citation coverage is the
fraction of serialized records containing at least one source citation.

The authored fixtures test implementation behavior only. They do not estimate
performance on arbitrary papers.

## 12. Limitations

- equations are detected from extracted text only;
- there is no visual layout, OCR, or figure understanding;
- table parsing is intentionally narrow;
- section roles and confidence are heuristic;
- notation and cross-section dependencies may remain ambiguous;
- reference parsing has no external resolver;
- comparisons depend on explicitly extracted settings;
- research-gap candidates have no literature-wide novelty check;
- source linkage is not an entailment proof.

## 13. Source mapping

- models and citations: `scholarly/models.py`, `source.py`
- paper/metadata/references: `paper.py`
- section roles: `sections.py`
- equations/notation/definitions: `equations.py`
- scholarly fields/tables/experiments: `extraction.py`
- summaries/checklists/comparisons/gaps: `artifacts.py`
- equation-aware reranking: `retrieval.py`
- top-level API: `pipeline.py`
- atomic artifacts: `serialization.py`
- metrics: `evaluation.py`
- CLI: `cli.py`
- fixtures: `tests/fixtures/scholarly/`
- tests: `tests/test_scholarly_*.py`
- experiments: `experiments/*scholarly*`,
  `evaluate_reproduction_checklists.py`, `evaluate_paper_comparison.py`, and
  `evaluate_research_gap_candidates.py`
