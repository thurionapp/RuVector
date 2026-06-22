# ADR-0012: Explainability Is Mandatory for Organ Detection

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Organ identity comes from a prior-driven inference layer (ADR-0010) that produces
a confidence number per organ. A bare percentage is not enough: a "78% liver"
badge invites trust without justification and hides why the call was made or what
would overturn it. For a research tool committed to honesty (ADR-0005) and to
first-class uncertainty (ADR-0007), every hypothesis must be able to answer "why
this label?" and "what would invalidate it?".

## Decision

Organ detection **must expose its evidence**. The `OrganHypothesis` evidence
bitmask from `crates/sonic-ct/src/organ.rs` (`EV_ZONE`, `EV_SIDE`, `EV_SIZE`,
`EV_ADJACENCY`, `EV_CONSISTENCY`) is surfaced verbatim in the HUD. The
`OrganPanel` in `examples/sonic-ct/src/hud/Hud.jsx` renders each hypothesis with
its confidence bar and a hover tooltip built by `evidenceText`, which decodes the
mask against the `EVIDENCE` map in `engine.js`:

- expected z-zone (`EV_ZONE`)
- correct side (`EV_SIDE`)
- plausible size (`EV_SIZE`)
- posterior adjacency (`EV_ADJACENCY`)
- consistent across slices (`EV_CONSISTENCY`)

The panel also carries the standing disclaimer that identity is "inferred from
shape, z-position, adjacency, landmarks — **not from speed alone**". The evidence
flags are the invalidation contract too: a hypothesis missing `EV_CONSISTENCY` or
`EV_SIZE` openly shows which support is absent, so a reader sees what a stronger
or weaker call would require. The `EVIDENCE` comment ties the JS labels to the
Rust `EV_*` constants, keeping the surfaced explanation in lock-step with the
detector.

**Acceptance:** every hypothesis shows its evidence flags, its confidence, and
the "not from speed alone" disclaimer.

## Consequences

### Positive

- Each organ call is auditable down to the specific priors that supported it.
- Missing evidence is visible, communicating fragility instead of hiding it.
- JS/Rust coupling of the evidence labels prevents drift between layers.

### Negative / Trade-offs

- Evidence is boolean per prior, not a graded contribution, so the "why" is
  coarse-grained.
- The UI must keep the `EVIDENCE` map and `EV_*` constants in sync by convention;
  a mismatch would mislabel evidence.

## Alternatives Considered

- **Show confidence only**: rejected — opaque, invites unjustified trust.
- **Free-text rationale per hypothesis**: rejected — harder to keep faithful to
  the actual scoring than a decoded bitmask tied to the constants.

## References (to the real code)

- `examples/sonic-ct/src/hud/Hud.jsx` (`OrganPanel`, `evidenceText`,
  "not from speed alone" note)
- `examples/sonic-ct/src/engine.js` (`EVIDENCE` map)
- `crates/sonic-ct/src/organ.rs` (`EV_*` constants, `OrganHypothesis.evidence`)
