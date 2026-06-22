# ADR-0021: Patient State Graph and Rule-Based Contradiction Detection

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Multimodal observations must be auditable: which inputs supported a
reconstruction, which conflicted, and which came from where. A flat list cannot
express support/conflict/temporal-adjacency relationships.

## Decision

Assemble observations into a typed [`PatientStateGraph`](../../packages/metabiohacker/src/graph/types.ts)
of nodes (patient, observation, body_site, specimen, …) and edges
(`has_observation`, `measures_site`, `from_specimen`, `requires_review`,
`supports`, `conflicts_with`, `temporally_near`). `detectContradictions` (V0,
rule-based) flags low-quality inputs (medium), large same-test value
disagreements ≥2× (high, review), and any review-required modality left
unflagged (high). `applyContradictionPenalty` lowers multimodal agreement and
raises uncertainty — and dents safety for high-severity conflicts — but never
changes the acoustic residual (the physics).

## Consequences

### Positive
- Every output can answer "what supported / conflicted with this?".
- Contradictions reduce confidence instead of being silently averaged away.

### Negative / Trade-offs
- Rule-based detection is shallow; learned contradiction scoring is future work.

## Alternatives Considered
- Averaging all evidence (rejected: hides conflicts).
- LLM-only contradiction detection (rejected for V0: non-deterministic, costly).

## References
- `packages/metabiohacker/src/graph/*`, `src/fusion/contradictionPenalty.ts`.
  See ADR-0007 (uncertainty-first) and ADR-0022 (run ledger).
