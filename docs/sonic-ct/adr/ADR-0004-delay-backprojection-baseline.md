# ADR-0004: Delay-and-Backprojection as the Reconstruction Baseline

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

We need a reconstruction stage that is (a) cheap enough to run in the WASM demo,
(b) honest about being a baseline rather than a state-of-the-art solver, and
(c) a stepping stone towards iterative least-squares and, eventually,
full-waveform inversion (FWI). The simplest defensible time-of-flight method is
delay-and-backprojection; the natural generalisation is SART (Simultaneous
Algebraic Reconstruction Technique), where one sweep *is* backprojection.

## Decision

`reconstruction.rs` solves the linear tomography system `A s = t` with SART,
where `s` is per-cell slowness (1/c), `A_ij` is the length ray `i` spends in
cell `j` (from `ray.rs` path integration), and `t` is the interior travel time
(measured travel time minus the exterior water leg, computed against
`WATER_SPEED = 1480` m/s). `reconstruct_speed` and `reconstruct_attenuation`
share the generic `sart` solver, differing only in the right-hand side.

`ReconConfig::iters` controls sweeps; **`iters == 1` is exactly the
delay-backprojection baseline**, and additional sweeps move the estimate towards
the least-squares solution. The default is 6 sweeps with relaxation 0.9.
Measured on the synthetic corpus (~96×96 grid, 180 elements, ~8000 valid
measurements): speed MAE lands at ~28–31 m/s.

## Consequences

### Positive

- One code path covers both the baseline (`iters=1`) and an iterative refinement
  (`iters>1`), so the baseline is always available for comparison.
- Pure linear algebra over sparse ray-cell pairs — fast, dependency-free, small.
- Honest, well-understood error characteristics (~28–31 m/s MAE).

### Negative / Trade-offs

- Straight-ray TOF ignores diffraction and refraction; small high-contrast
  structures blur. Concretely, **bone Dice is ~0** because the small, fast spine
  is smeared by the straight-ray model.
- SART converges slowly; the speed/quality trade-off is fixed by `iters`.
- This is explicitly *not* FWI; it is the documented predecessor to it.

## Alternatives Considered

- **Filtered backprojection (FBP)**: assumes parallel/fan geometry and straight
  rays too; no easier and less flexible than SART for a transmission ring.
- **Full-waveform inversion now**: the correct long-term answer and the
  documented next step, but far heavier and premature before the contract and
  metrics stabilise (ADR-0001).

## References (to the real code)

- `crates/sonic-ct/src/reconstruction.rs` (`sart`, `reconstruct_speed`,
  `reconstruct_attenuation`, `ReconConfig`)
- `crates/sonic-ct/src/ray.rs` (supersampled straight-ray path integration)
- `crates/sonic-ct/src/types.rs` (`WATER_SPEED`)
- `crates/sonic-ct/src/metrics.rs` (`mae_speed`, `dice_all`)
