# Adaptive instruction profiles

An `InstructionProfile` is a deterministic interpretation of how a user wants
an answer presented. It replaces a mandatory static audience selector in the
Paper Training Lab. The canonical `beginner`, `undergraduate`, and `researcher`
labels remain optional metadata for regression comparisons.

## Inputs and precedence

`infer_instruction_profile` considers, in increasing precedence:

1. explicit opt-in stored preferences;
2. recent local user turns;
3. the current prompt;
4. explicit API overrides.

The result records desired depth, mathematical depth, assumed background,
style, format, verbosity, analogy/derivation/critique/comparison requests,
whether the previous response should be simplified, constraints, detected
signals, and an interpretation confidence. Conflicting current instructions
override older conversational cues. Unknown overrides fail rather than being
silently ignored.

This is deliberately rule-based. It is reproducible, inspectable, and does not
call an LLM or use a hidden classifier. Confidence describes the strength of
instruction signals, not answer correctness.

## Evidence independence

Instruction interpretation and evidence selection are separate functions:

```text
question + selected papers ──→ retrieval/evidence/sufficiency
question + recent turns ─────→ InstructionProfile
                                      ↓
                        presentation of supported facts
```

A beginner explanation can omit secondary detail, but it cannot change source
facts, add unsupported claims, relax citations, or turn insufficient evidence
into a confident answer. Similarly, requesting a formal derivation does not
make a missing derivation paper-explicit.

## Multi-turn state

`ConversationContext` stores selected papers, ordered user/assistant turns,
and optional preferences. Repeated prompts such as “simplify that” are resolved
against recent turns. The Review Service keeps contexts in memory by default.
Persistence must be explicitly enabled.

## Current limitation

The interpreter recognizes a bounded vocabulary of common instructions. It
does not infer every pragmatic nuance, and its profile does not itself rewrite
the deterministic extractive baseline into polished prose. Human review should
correct both factual and presentation failures.

