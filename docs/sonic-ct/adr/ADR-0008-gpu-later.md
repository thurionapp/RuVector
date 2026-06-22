# ADR-0008: CPU/WASM Now, GPU Later

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

USCT reconstruction is ultimately compute-bound, and the long-term roadmap
(full-waveform inversion, 3-D vertical sweeps) will need GPU acceleration to be
practical. But the immediate goals are correctness, reproducibility, and a tiny
in-browser demo. Reaching for GPU now would add dependencies, a toolchain, and
non-determinism before the algorithms and data contracts are stable. This is a
**forward-looking** decision: **no GPU backend exists yet.**

## Decision

`sonic_ct` runs on scalar CPU today and targets `wasm32-unknown-unknown` for the
demo. The WASM crate `crates/sonic-ct-wasm/` is a raw **C-ABI cdylib (no
wasm-bindgen)** that compiles to ~31 KB, exporting `sct_run` plus scalar getters
and `*_ptr` buffer accessors that JS reads directly from WebAssembly memory. The
core stays zero-dependency and free of SIMD/threading intrinsics, so the same
code runs natively and in the browser identically and deterministically.

GPU is explicitly deferred. The structures that will benefit are already shaped
to make a later GPU path additive rather than a rewrite: the SART solver
(`reconstruction.rs`) is sparse ray-cell accumulation that maps naturally to a
parallel kernel, and `volume3d.rs::SweepPlan` is a **stub** describing a vertical
multi-slice sweep — the future workload that would justify GPU. When FWI and 3-D
sweeps land, a GPU backend can sit behind the existing solver/sweep interfaces.

## Consequences

### Positive

- Tiny (~31 KB), dependency-free, deterministic artifact that runs anywhere a
  WebAssembly runtime exists; no GPU/toolchain requirement to use or test.
- Native and WASM behaviour are identical, simplifying verification.
- The deferral keeps the door open: SART and `SweepPlan` are GPU-friendly shapes.

### Negative / Trade-offs

- Scalar CPU caps throughput; larger grids, more sweeps, and 3-D sweeps are slow.
- No SIMD even on native targets today, leaving performance on the table.
- `SweepPlan` is a stub — 3-D acquisition/reconstruction is not implemented.

## Alternatives Considered

- **GPU (wgpu/CUDA) now**: premature; adds heavy dependencies and
  non-determinism before algorithms and the data contract stabilise (ADR-0001).
- **wasm-bindgen + SIMD demo**: larger artifact and more build surface than the
  raw C-ABI cdylib needs for the current scalar pipeline.

## References (to the real code)

- `crates/sonic-ct-wasm/src/lib.rs` (`sct_run`, scalar getters, `*_ptr`
  accessors; raw C-ABI, no wasm-bindgen)
- `crates/sonic-ct/src/reconstruction.rs` (`sart` — GPU-friendly sparse kernel)
- `crates/sonic-ct/src/volume3d.rs` (`SweepPlan` — vertical-sweep stub)
