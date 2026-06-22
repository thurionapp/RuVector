# ADR-0001: Simulation-First Architecture

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

USCT research normally couples physics, reconstruction, and hardware tightly,
which makes it impossible to iterate on algorithms without a transducer ring and
a tank. We have no hardware and no public raw-acquisition SDK. We need a way to
develop and quantitatively evaluate the full pipeline — phantom, acquisition,
reconstruction, segmentation, metrics — purely in software, deterministically,
and in a form that compiles to a tiny WASM artifact for an in-browser demo.

## Decision

The core crate `crates/sonic-ct/` is a pure-Rust, **zero-dependency**,
deterministic simulator. Everything flows through one entry point,
`pipeline::run_with_model` (see `pipeline.rs`), which builds a deterministic
phantom (`phantom.rs`, SplitMix64-seeded), simulates a transmission acquisition
(`acquisition.rs::simulate`), reconstructs speed and attenuation
(`reconstruction.rs`), segments (`segmentation.rs`), and scores against ground
truth (`metrics.rs`). The result is a single `Scene` struct holding phantom,
ring, acquisition, reconstructions, segmentation, and a `QualityReport`.

Because there is ground truth (`Phantom::labels`, `phantom.speed`), every run is
self-scoring: `dice_all` + `mae_speed` give Dice-per-class, mean Dice, and speed
MAE without any external dataset.

## Consequences

### Positive

- Deterministic, reproducible runs (fixed seeds) — regressions are detectable.
- Zero dependencies keep the WASM artifact at ~31 KB and the build trivial.
- Ground-truth phantoms make the whole pipeline quantitatively measurable.
- Hardware can be added later behind a trait (ADR-0002) without disturbing core.

### Negative / Trade-offs

- The simulator uses straight-ray time-of-flight, not full-wave physics, so
  results are optimistic relative to real diffracting media (see ADR-0004).
- Synthetic phantoms cannot capture the variability of real anatomy.
- "Self-scoring" measures algorithm consistency, not clinical accuracy.

## Alternatives Considered

- **Full-waveform forward solver first**: physically faithful but far heavier,
  non-trivial to compile to a small WASM target, and premature before the
  end-to-end contract is settled.
- **Replay recorded RF datasets**: no licensed raw SDK exists, and it couples
  development to one acquisition rig.

## References (to the real code)

- `crates/sonic-ct/src/pipeline.rs` (`run`, `run_with_model`, `Scene`)
- `crates/sonic-ct/src/phantom.rs` (`Phantom::build`, `SplitMix64`)
- `crates/sonic-ct/src/metrics.rs` (`dice_all`, `mae_speed`, `QualityReport`)
- `crates/sonic-ct/Cargo.toml` (detached workspace, no dependencies)
