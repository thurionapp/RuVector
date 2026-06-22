# ADR-0010: Organ Identity Inferred from Anatomical Priors, Not Speed

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

The reconstruction emits five acoustic classes and deliberately never assigns
organ names from speed (ADR-0009). But a useful workbench still wants to say
"this looks like the liver." Since the speed value cannot supply that, identity
must come from somewhere else: where a structure sits in the body, which side it
is on, how big it is, whether it lies posteriorly, and whether it persists across
slices. These are the same cues a reader uses, and they are independent of the
ambiguous speed signal.

## Decision

Organ identity is a **separate inference layer driven by anatomical priors**,
implemented in `crates/sonic-ct/src/organ.rs`. The `Organ` enum hypothesises
eight structures (`Liver`, `Spleen`, `KidneyLeft`, `KidneyRight`, `Aorta`,
`Heart`, `LungLeft`, `LungRight`). `detect_organs(labels, n, nz)` runs over the
**reconstructed soft-tissue distribution** — the segmenter's `Tissue::Organ`
voxels — never the ground-truth phantom, so it is genuine inference.

Each organ has a `Spec` prior: cranio-caudal zone `z`, `side` (left/right/
central), `posterior` flag, and `expected_frac`. From these the detector derives
sub-scores and sets an evidence bitmask of `EV_ZONE`, `EV_SIDE`, `EV_SIZE`,
`EV_ADJACENCY`, and `EV_CONSISTENCY`. Every result is an `OrganHypothesis`
carrying `{ organ, confidence, evidence, volume_frac }`. Confidence is a weighted
blend of zone, side, size, adjacency, and slice-consistency scores, clamped to
`[0, 0.97]`, and is exactly `0.0` when no matching tissue is found.

**Acceptance:** every organ label is a hypothesis carrying an evidence vector and
a confidence; identity is never read off a speed value.

## Consequences

### Positive

- Identity rests on spatial priors that are actually discriminative, decoupled
  from the non-discriminative speed signal.
- The evidence bitmask makes each call explainable (surfaced per ADR-0012).
- Operating on the reconstruction (not the phantom) keeps results honest.

### Negative / Trade-offs

- Priors are hand-tuned constants in `SPECS`; they encode an assumed body layout
  and will mislabel atypical or pathological anatomy.
- Side/zone gating is heuristic, so confidence is indicative, not calibrated.

## Alternatives Considered

- **Speed-only identity**: rejected by ADR-0009 — physically impossible.
- **Learned organ classifier**: more powerful but opaque and unvalidated; the
  prior-based layer keeps every decision auditable for a research tool.

## References (to the real code)

- `crates/sonic-ct/src/organ.rs` (`Organ`, `detect_organs`, `Spec`/`SPECS`,
  `EV_ZONE`/`EV_SIDE`/`EV_SIZE`/`EV_ADJACENCY`/`EV_CONSISTENCY`, `OrganHypothesis`)
- `crates/sonic-ct/src/types.rs` (`Tissue::Organ` input class)
