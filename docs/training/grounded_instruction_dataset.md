# Grounded instruction dataset

The Milestone 12A.1 dataset contains versioned, provenance-aware
`GroundedInstructionExample` records. It supports single- and multi-turn
prompts, one or multiple papers, abstention, derivation, simplification,
critique, comparison, and ordinary paper question answering.

## Approval boundary

Question generation, answer runs, heuristic diagnostics, prompt variations,
and correction generation remain untrusted. Human and automated decisions use
permanently distinct statuses. The lower-level `build_dataset` API defaults to
`human-only`; the site/CLI defaults to `human-and-audited`. Explicit
`include-codex-approved` export is available, but Codex approval never becomes
human approval. Rejected, ambiguous, benchmark-problem, pending, circular, and
test-only records are excluded.

Every selected record receives trust weight metadata (1.0 human, 0.9 audited
Codex, or 0.6 unaudited Codex), a stable duplicate-cluster ID, and one
paper-level split. One highest-trust representative per duplicate cluster is
exported by default.

## Grounded target

The factual target is separate from the instruction profile. It groups facts,
equations, derivation steps, assumptions, qualifications, limitations,
unresolved items, and prohibited claims. Each target fact uses one provenance:

- `paper_explicit`: stated by the selected paper and bound to a citation;
- `mathematical_inference`: a derived mathematical step not stated verbatim;
- `external_knowledge`: information outside the supplied papers;
- `uncertain`: unresolved or insufficiently supported content.

A `paper_explicit` fact without a citation is structurally invalid. This keeps
inferred derivations from being represented as quotations from the paper.

## Artifact identity

The JSON artifact includes a format version, dataset version, examples,
paper-to-split map, composition metrics, warnings, and SHA-256 over canonical
content. Loading verifies the hash and schema invariants. The schema records
the immutable interaction ID, selected source evidence, conversation turns,
instruction profile, review label, reviewer metadata, and final corrected
answer.

## Diversity diagnostics

`dataset-report` counts task types, papers, optional audience metadata,
provenance categories, review labels, multi-turn examples, multi-paper
examples, abstentions, and derivations. It warns about narrow paper/task
coverage, missing difficult behaviors, and dominance by one paper. These
warnings guide review; they do not score semantic quality.

## Legal and quality requirements

An approved answer does not grant rights to train on its source. Dataset users
must separately track paper license, provenance, transformations, and permitted
uses. Citation syntax does not prove that a claim is entailed; human review is
still responsible for correctness.
