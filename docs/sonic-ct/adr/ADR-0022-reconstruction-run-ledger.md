# ADR-0022: Reconstruction Run Ledger for Reproducibility and Audit

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

A medical signal system cannot be a black box. Every reconstruction must be
reproducible and explainable: from raw observations through priors, graph,
contradictions, routing decisions, score, and safety state to the final
UI packet.

## Decision

Each run produces a [`ReconstructionLedger`](../../packages/metabiohacker/src/ledger/types.ts)
capturing the frozen engine identity (binary hash + version + `frozen: true`),
the input observations, prior, patient state graph, contradictions, routing
decisions, multimodal score, and a safety block. Component `stableHash`es
(sorted-key SHA-256) plus a top-level `ledgerHash` make the run tamper-evident;
`verifyRunLedger` recomputes them and checks the engine-frozen and
diagnostic-language-blocked invariants. `createReconstructionPacket` derives the
safe UI view (confidence, uncertainty, evidence, review status) with mandatory
uncertainty overlay and blocked diagnosis language.

## Consequences

### Positive
- Reproducible, verifiable runs; Darwin candidates compared by ledger hash, not vibes.
- Human-review and output-mode (`research`/`clinical_review`) are derived, not optional.

### Negative / Trade-offs
- Ledgers embed full observations/graph — storage cost; dedup/ruVector indexing is future work.

## Alternatives Considered
- Logging free-text only (rejected: not verifiable).
- Signing with asymmetric keys (deferred: hashing is enough for V0 reproducibility).

## References
- `packages/metabiohacker/src/ledger/*`, `src/output/reconstructionPacket.ts`.
  See ADR-0003 (preserve raw evidence), ADR-0018 (governance).
