# ADR-0024: Real-Slice Calibration with a Domain-Gap Honesty Gate

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Synthetic phantoms reconstruct well (~0.54 Dice); real public CT slices are far
harder (~0.30) because of anatomical complexity, modality mismatch (CT ≠ USCT),
boundary ambiguity, and registration error. A single mean-Dice headline would
either overclaim on real data or bury the honest physics boundaries.

## Decision

Treat real CT/MRI slices as **calibration targets, not USCT**. Score Dice **by
region** (`diceByRegion`) — fluid/fat/soft-tissue/bone — so easy classes (fluid,
fat) and hard ones (soft tissue, bone) are visible, and liver-vs-spleen stays
explicitly impossible from acoustic class alone. Compute a **domain-gap score**
(`scoreDomainGap`) from registration error, boundary complexity, class
imbalance, and missing acoustic equivalents, and a **honesty gate**
(`classifyRealSliceResult`): `headline` only when registration ≤ 12 px, gap ≤
0.30, and mean Dice ≥ 0.45; otherwise `researchOnly` or `exclude`. The benchmark
report is split into **synthetic / real / governance** sections; the headline
separates speed from anatomical fidelity.

## Consequences

### Positive
- No accidental overclaiming; real-slice results are research-only until they earn headline inclusion.
- Region-level Dice exposes honest physics boundaries.

### Negative / Trade-offs
- V0 registration is a centroid proxy; landmark/intensity registration is future work.

## Alternatives Considered
- Single mean-Dice headline (rejected: conflates speed with real fidelity, invites overclaiming).
- Excluding real slices entirely (rejected: feasibility signal is valuable, if gated).

## References
- `packages/metabiohacker/src/calibration/*`; `examples/sonic-ct/benchmark.mjs`.
  See ADR-0018 (governance), ADR-0007 (uncertainty-first).
