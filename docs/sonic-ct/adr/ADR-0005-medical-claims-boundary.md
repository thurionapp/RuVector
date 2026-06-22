# ADR-0005: Research/Simulation Only — No Diagnostic Claims

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

`sonic_ct` produces tissue maps, segmentations, and an "anomaly" flag that
superficially resemble clinical imaging outputs. It is a research-grade
simulator with no clinical validation, no regulatory clearance, and no real
acquisition path. Anything that reads like a diagnosis would be both wrong and
irresponsible. We need an explicit boundary that keeps the project firmly in the
research/simulation domain.

## Decision

`sonic_ct` is **research and simulation only and makes no diagnostic claims.**
This is enforced structurally, not just by disclaimer:

- All inputs are synthetic. `Phantom::build` (`phantom.rs`) generates a
  deterministic abdomen phantom (fat→muscle→organ shells + a spine bone) from a
  SplitMix64 seed; there is no patient data path into the pipeline.
- The hardware boundary is a **mock**, not a device. `butterfly.rs` exposes
  `MockButterflyEmbeddedBackend` whose `name()` is `"mock-butterfly-embedded"`;
  there is **no public raw-hardware SDK**, so no real Butterfly acquisition
  exists or is implied.
- Outputs are framed as algorithm quality, not findings. `metrics.rs` reports
  Dice and MAE against ground truth; `memory.rs::check_coherence` produces a
  `CoherenceReport` whose `anomaly` flag is an *anatomical-rule consistency*
  signal over labels, not a clinical anomaly.
- Subject identifiers in `ScanRecord` are pseudonymous (`patient_id`) and the
  code comments mandate "never raw PII"; no PHI flows anywhere.

## Consequences

### Positive

- The "no diagnosis" stance is backed by the absence of any real-data or
  real-hardware path, not just wording.
- Mock naming makes provenance unambiguous in logs and serialized scans.
- Pseudonymous-only identifiers keep the project clear of PHI handling.

### Negative / Trade-offs

- Synthetic-only inputs mean nothing here generalises to real anatomy without a
  separate, validated effort.
- The `anomaly` flag may still be misread as clinical; naming and docs must keep
  reinforcing the boundary.

## Alternatives Considered

- **Pursue clinical framing/validation now**: out of scope, unvalidated, and
  legally hazardous for a simulator.
- **Drop anomaly/coherence outputs entirely**: loses a useful research signal;
  better to keep it and label it precisely as a consistency check.

## References (to the real code)

- `crates/sonic-ct/src/phantom.rs` (synthetic `Phantom::build`)
- `crates/sonic-ct/src/butterfly.rs` (`MockButterflyEmbeddedBackend::name`)
- `crates/sonic-ct/src/memory.rs` (`check_coherence`, `CoherenceReport`,
  pseudonymous `ScanRecord::patient_id`)
- `crates/sonic-ct/src/metrics.rs` (`QualityReport`)
