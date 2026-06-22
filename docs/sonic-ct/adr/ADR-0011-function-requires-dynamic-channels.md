# ADR-0011: Organ Function Requires Dynamic/Multiparametric Channels

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Identifying *where* an organ is (ADR-0010) is not the same as knowing *how it is
doing*. There is a strong temptation to read function — "the kidney is well
perfused", "the liver is stiff" — off the single static speed-of-sound map we
already have. That is unfounded. A speed map is one snapshot of a bulk acoustic
property. Function is a dynamic, multiparametric quantity: perfusion, tissue
stiffness/elastography, flow, motion, longitudinal change over repeated scans,
and multiparametric quantitative ultrasound (QUS). None of those signals is
present in a static speed reconstruction, and inventing them would be a fabricated
clinical claim contrary to ADR-0005.

## Decision

Organ **function is unavailable unless the corresponding dynamic or
multiparametric channels exist**. Functional readouts (perfusion, stiffness,
flow, motion, longitudinal change, multiparametric QUS) are derived only from
their own acquired channels; a static speed map is explicitly not treated as a
proxy for any of them.

These channels are **not yet implemented** — this ADR is forward-looking and
records the constraint before any function feature is built, so the boundary is
designed in rather than retrofitted. Until a channel is acquired and validated,
the UI must render that functional dimension as **"not measured"**, never as a
guessed or interpolated value. The current build exposes only static structural
channels and the acoustic-class/organ-hypothesis layers; no functional value is
emitted today, which is the correct state given no functional channel exists.

**Acceptance:** any functional dimension without its acquired channel renders as
"not measured" and is never inferred from the static speed map.

## Consequences

### Positive

- Prevents the most seductive overclaim — reading physiology off structure.
- Establishes the channel-presence contract before the feature exists, so future
  function work inherits an honest default.
- Keeps the static pipeline cleanly scoped to structure and identity.

### Negative / Trade-offs

- The workbench currently shows no functional information at all, which may
  disappoint users expecting "organ health" output.
- Adding genuine function later requires real dynamic/multiparametric acquisition
  and its own validation effort — non-trivial.

## Alternatives Considered

- **Estimate function from the static speed map**: rejected — no functional
  signal is present; would be fabrication.
- **Hide functional fields entirely**: rejected — explicit "not measured" is more
  honest and signals the intended, unbuilt capability.

## References (to the real code)

- `crates/sonic-ct/src/types.rs` (`Tissue` — structural classes only; no
  functional fields)
- `examples/sonic-ct/src/hud/Hud.jsx` (channel selector / `CHANNELS`; structural
  channels only, no functional readouts)
