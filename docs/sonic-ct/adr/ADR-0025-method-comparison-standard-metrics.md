# ADR-0025: Benchmark Reconstruction Against Recognised Methods + Standard Metrics

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Claiming progress toward SOTA requires comparison against recognised baselines
on a recognised target with standard metrics — not just internal
baseline-vs-evolved deltas.

## Decision

Add the standard **Shepp–Logan** head phantom (`shepp_logan.rs`) and compare
three recognised reconstruction algorithms — **backprojection** (single sweep),
**SART** (algebraic, relaxed), and **Landweber** (gradient descent on
`‖A s − t‖²`) — via `reconstruct_speed_with(..., Method)`, scored with standard
image-quality metrics **RMSE / PSNR / SSIM** (`metrics.rs`). The
`sonic_ct_methods` binary emits a deterministic comparison table to
`docs/sonic-ct/METHOD-BENCHMARK.md`.

Measured (ground-truth speed): on both Shepp–Logan and the abdomen phantom,
backprojection < SART < Landweber on every metric (abdomen RMSE 130 → 99 → 51
m/s; SSIM 0.22 → 0.60 → 0.92), at increasing compute cost (≈4 / 28 / 100 ms).
**SART stays the production default** (best fidelity-per-millisecond);
**Landweber is the higher-fidelity option** when latency budget allows.

## Consequences

### Positive
- Defensible, reproducible "benched against others" claims on a standard phantom.
- A drop-in higher-fidelity method (Landweber) is now available.

### Negative / Trade-offs
- Landweber needs more iterations; full-waveform inversion remains the longer-term frontier (ADR-0004).

## Alternatives Considered
- Filtered backprojection (rejected: ill-suited to limited-angle transmission TOF).
- Only internal baseline-vs-evolved comparison (rejected: not externally grounded).

## References
- `crates/sonic-ct/src/{shepp_logan,reconstruction,metrics}.rs`,
  `src/bin/methods.rs`, `docs/sonic-ct/METHOD-BENCHMARK.md`.
