# ADR-0026: Full-Waveform Inversion (Forward + Adjoint Gradient)

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Straight-ray time-of-flight reconstruction (SART/Landweber, ADR-0004/0025) uses
only first-arrival travel times and ignores diffraction/refraction. Full-waveform
inversion (FWI) — fitting the *whole* recorded waveform with a wave-physics model
— is the documented state-of-the-art frontier (research map) for higher
resolution.

## Decision

Add a transparent, dependency-free 2-D FWI reference in `fwi.rs`:

- **Forward model** — explicit finite-difference solution of the scalar acoustic
  wave equation `∂ₜ²p = κ ∇²p + f` (`κ = c²`), Ricker source, CFL-stable step, a
  damping sponge for the boundaries.
- **Adjoint-state gradient** — back-propagate the receiver residual through the
  (self-adjoint) wave operator and correlate with the forward field:
  `∂χ/∂κ(x) = Σ_t λ(x,t) ∇²p(x,t)`.
- **Inversion** — gradient descent on `κ` with source/receiver-footprint muting,
  gradient smoothing, and a backtracking step.
- **Frequency continuation** — `invert_multiscale` chains low → high frequency
  stages (each its own `FwiConfig` + observed set), smoothing the model between
  stages. Low frequencies recover the smooth, long-wavelength background first and
  keep the higher-frequency stages out of local minima (cycle-skipping).

**Correctness is proven by an adjoint-vs-finite-difference gradient check**
(cosine > 0.85) — the gold-standard FWI test — plus an inversion that reduces the
data misfit ≥ 15% and recovers a centrally-concentrated velocity anomaly. A third
test shows frequency continuation lowers the inclusion-region error below
single-scale FWI at matched iteration count.

## Consequences

### Positive
- A verified wave-equation forward/adjoint engine — the foundation for
  higher-resolution reconstruction beyond ray-based TOF.

### Negative / Trade-offs
- Single-frequency, unregularised FWI overshoots amplitude and mislocates the
  brightest pixel on small/underdetermined problems. Frequency continuation
  (now implemented) improves the inclusion-region error over single-scale FWI,
  but **Tikhonov/TV regularisation, source encoding, and 3-D are still the next
  steps** and are NOT yet implemented — claims are limited to misfit reduction +
  anomaly localisation + relative multiscale improvement, not quantitative
  clinical recovery (ADR-0013/0018).

## Alternatives Considered
- Staying ray-based only (rejected: caps resolution; FWI is the SOTA direction).
- A third-party FWI library (rejected: dependency-free reference is auditable).

## References
- `crates/sonic-ct/src/fwi.rs`, `tests/fwi.rs`. See ADR-0004 (baseline),
  ADR-0025 (method comparison), and the research map.
