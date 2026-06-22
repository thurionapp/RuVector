# ADR-0019: The Product Is a Medical Signal Operating System, Not an AI Doctor

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

It is easy to describe a multimodal medical system as an "AI doctor." That framing
is both inaccurate and dangerous: it implies autonomous diagnosis (which we
explicitly do not do — ADR-0018) and obscures what the system actually is. The
preceding ADRs already define the parts — a frozen physics engine (ADR-0014), a
typed observation graph (ADR-0015), established standards (ADR-0016), governed
fusion patterns (ADR-0017), and a hard governance boundary (ADR-0018). This ADR
names the whole.

## Decision

MetaBioHacker is a **medical signal operating system**, not an AI doctor. The
architecture is layered:

- **Inputs** — acoustic, imaging, labs, waveforms, pathology, clinical notes, and
  wearables, ingested as typed observations (ADR-0015) over established standards
  (ADR-0016).
- **Core** — frozen physics engines (e.g. `sonic_ct`) plus deterministic
  validators. This layer is the trusted, non-mutating truth source and is never
  evolved by the search loop.
- **Learning layer** — evolves the reconstruction harness: reconstruction policy,
  model routing, confidence, priors, and explanation (ADR-0014). It refines *how*
  signals are reconstructed and explained; it never rewrites physics or anatomy.
- **Output** — an uncertainty-aware patient state graph (ADR-0015), where every
  value carries provenance, units, uncertainty, and consent scope.

The OS framing is deliberate: like an operating system, the core provides trusted,
stable primitives, and policy/applications evolve on top under explicit
constraints. No layer emits an autonomous clinical decision.

Acceptance test: adding multimodal context improves reconstruction confidence
**or** temporal stability by ≥10% while preserving full provenance, consent scope,
uncertainty, and a human-review path for any clinical claim. This mirrors the
harness acceptance gate (ADR-0014) and extends it to the multimodal whole.

## Consequences

### Positive

- A single, accurate framing aligns engineering, product, and governance, and
  rules out "AI doctor" expectations by construction.
- The layered split keeps the trusted core stable while allowing the learning
  layer to improve under hard, measurable gates.

### Negative / Trade-offs

- The OS metaphor sets an expectation of platform-grade stability and contracts
  between layers, which is more upfront design than a monolithic app.
- Positioning as infrastructure rather than a finished diagnostic product may be
  a harder near-term market story.

## Alternatives Considered

- **Position as an "AI doctor" / autonomous diagnostic product**: rejected —
  inaccurate and crosses the SaMD boundary (ADR-0018).
- **A single fused model with no layer boundaries**: rejected — collapses the
  trusted core into the mutable layer and loses provenance and auditability
  (ADR-0015).

## References

- ADR-0014 (Freeze Physics, Evolve Harness — the learning layer)
- ADR-0015 (Patient State Graph — the output)
- ADR-0016 (Medical Standards Architecture — the inputs)
- ADR-0017 (Multimodal Fusion Patterns)
- ADR-0018 (Governance / SaMD Boundary)
