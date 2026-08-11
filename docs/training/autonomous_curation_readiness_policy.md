# Autonomous curation readiness policy

The full 14-paper/840-candidate run is blocked until controlled diagnostics
demonstrate reliability. Candidate yield is not a readiness metric, and the
0.97 acceptance threshold must not be weakened.

## Controlled sequence

1. Freeze and diagnose the original failed run without mutating it.
2. Run 50 candidates with seed 42, stratified across papers, question types,
   and known failure categories.
3. If every gate passes, run a second deterministic 100–150 candidate batch
   with a different seed.
4. Start a full run only after both controlled batches show no systematic
   failure.

The first controlled run targets evidence validation at least 95%, citation
structure at least 98%, citation support and relevance at least 95%, hard
disagreement at most 10%, overall disagreement at most 15%, malformed reviewer
output at most 2%, and zero leakage, stale IDs, or source-hash mismatch.

## Full-run gates

`full_run_readiness` returns ready only when:

- citation structural validity is at least 98%;
- citation support/relevance are each at least 95%;
- evidence validation is at least 95%;
- hard disagreement is at most 10%;
- overall disagreement is at most 15%;
- repair success is positive when repairs occur;
- leakage, stale evidence IDs, and source-hash mismatches are zero;
- malformed reviewer output is at most 2%;
- no question type shows a systematic unresolved failure.

If the 50-candidate run stops, the second batch and full run must not start.
Training Milestone 12B remains blocked until reliable accepted examples exist
and their source licensing/provenance is suitable for training.
## Additional 1.2.5 gates

Full-run readiness additionally requires at least 95% claim-citation
completeness and at least 70% repair success, alongside the existing evidence,
citation, disagreement, malformed-output, stale-ID, source-hash, and leakage
gates. A 100–150 candidate diagnostic may run only after the 50-candidate run
passes every hard gate. The 840-candidate run and Milestone 12B remain forbidden
until readiness is true.
