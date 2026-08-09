# Codex review pipeline

The autonomous curator requires a genuine structured Codex review provider.
The default provider invokes the installed `codex exec` command with ephemeral,
read-only execution, approval disabled, and a strict JSON output schema. It
instructs every pass not to browse and supplies only passages from the indexed
local corpus. If Codex is unavailable or returns malformed output, the run is
suspended rather than silently substituting deterministic heuristics.

## Separated passes

1. **Answerer** selects supported facts and may correct the answer, evidence
   IDs, structured target, or abstention state.
2. **Evidence critic** sees the question, target, and raw passages, but not the
   answer or answerer confidence. It judges evidence sufficiency only.
3. **Answer critic** sees the answer and passages with retrieval ranks/scores
   removed. It judges correctness, relevance, completeness, unsupported claims,
   and instruction following.
4. **Citation critic** sees claim-to-passage mappings but no final decision. It
   validates citation support and relevance.
5. **Final adjudicator** sees raw evidence, deterministic diagnostics, and the
   three focused critic records. It accepts, requests repair, rejects, or marks
   the example uncertain.

These are logically separated passes, not statistically independent models.
Their input/output hashes, reviewer identity, version, structured results, and
rationales are preserved per accepted example.

## Evidence-first repair

A repairable failure triggers local retrieval before prose correction when the
evidence critic reports weak evidence. The next answerer pass receives the
required corrections, then deterministic and all Codex checks run again from
scratch. The default maximum is two repairs. Exhaustion rejects the example.
The system never searches the public internet or invents missing facts; it must
abstain, qualify, reject, or identify an external-source requirement.

The provider contract is injectable, so tests use deterministic fakes without
claiming they are real Codex review. Production `codex_curated` status is only
available when the configured provider reports itself available.
