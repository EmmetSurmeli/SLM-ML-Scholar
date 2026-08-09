# Milestone 12A.2 calibration-readiness audit

Date: 2026-08-08. Scope: the 1.2.1 local review service, automatic-review
artifacts, CLI, trust export, browser UI, and versioned compatibility paths.

## Checks and findings

- Nested automatic-review batches, immutable interaction snapshots, deterministic
  reviewer gates, and local atomic JSON persistence were reusable.
- The prior calibration report used coarse bins and agreement/Brier thresholds;
  it did not emphasize false approvals, coverage, or integrity gates.
- The existing one-item rerun replaced a historical review in place. The new
  historical migration path is append-only and retains original/new snapshots,
  timestamps, links, and hashes. The legacy single-item rerun remains available
  for active drafts.
- Human outcomes were embedded in `review_policy.json`; version 1.2.2 reads them
  alongside dedicated calibration pairs and deduplicates by review ID.
- Calibration approval previously flowed through correction review. Version
  1.2.2 makes calibration and training approval explicitly separate.
- Existing Codex audit provenance was sound. The browser now displays the
  effective `audited_codex_approved` trust state after confirmation while
  retaining the serialized producer label.
- The UI already normalized legacy missing `paper_ids`; the new cards use the
  same safe helper, avoiding the earlier `item.paper_ids.map` failure.
- Current local reviewer profiles remain correlated deterministic
  configurations, not independent judges.

## Added verification

Focused tests cover bucket boundaries, deterministic coverage-seeking samples,
false-approval readiness, integrity blockers, one-click decisions, duplicate
finalization prevention, before/after edit hashes, acquisition deduplication,
and the default bulk lock. Existing compatibility tests cover second-pass gates,
calibration activation, review service behavior, and versioned artifacts.

Final verification: Ruff lint and formatting passed, JavaScript syntax and
`git diff --check` passed, all new CLI help paths ran, the package imported as
1.2.2, and the complete suite reported `791 passed in 10.23s`. An isolated
50-pair activation smoke moved from `calibration_active` to
`auto_approval_enabled` only after the explicit command, with no coverage or
integrity failures. The real corpus was inspected read-only and remained at 6
papers, 128 automatic reviews, 0 calibration pairs, 0 human-approved examples,
and `calibration_required`.

## Remaining limitations

The score is heuristic, the local paper set is small, and correlated rule
profiles cannot establish semantic correctness independently. Readiness metrics
are only as representative as the reviewed sample. No model is trained, no
paper is fetched, and no capability claim follows from workflow completion.
