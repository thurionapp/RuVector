# ADR-0003: Preserve Raw RF Before Any AI Processing

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

In USCT the raw radio-frequency (RF) channel data is the only lossless record of
an acquisition. Once it has been beamformed, reconstructed, or fed through a
learned model, information is discarded irreversibly and provenance is hard to
re-establish. We want the data contract and storage formats designed for raw
capture from the start, even though the current simulator does not synthesise
full RF waveforms. This is a **forward-looking** decision: we are fixing the
shape now, not claiming we capture RF today.

## Decision

Define `RawRfFrame` in `butterfly.rs` as a **shape/contract placeholder**:

```rust
pub struct RawRfFrame {
    pub source: usize,       // transmitting element
    pub channels: usize,     // receive channels
    pub samples: usize,      // samples per channel
    pub sample_rate: f32,    // Hz
}
```

It records the `channels × samples` framing and sample rate of a real capture
but carries no waveform buffer yet — the simulator produces reduced
`Measurement`s (travel time, attenuation) in `acquisition.rs`, not RF. The type
exists so that downstream code, the `AcquisitionBackend` seam (ADR-0002), and
the portable container format are designed around raw frames from day one.

The container intent is realised by `memory.rs`: `AcousticMemory` round-trips
through a compact `.rvf`-style binary (`to_bytes`/`from_bytes`), making scans
portable, auditable artifacts that can later embed raw frames alongside
derived products.

## Consequences

### Positive

- Storage, provenance, and the backend trait are all designed for raw capture;
  adding waveforms later is additive, not a rewrite.
- The honest "placeholder" framing avoids over-claiming capability.

### Negative / Trade-offs

- `RawRfFrame` is currently unused by the simulated pipeline — dead-ish weight
  until a backend fills it, and its fields may shift once real RF is captured.
- No actual RF persistence exists yet; the `.rvf` container stores embeddings
  and quality provenance (`ScanRecord`), not waveforms.

## Alternatives Considered

- **Add `RawRfFrame` only when hardware lands**: avoids unused code but forces a
  later contract/storage redesign across the backend seam.
- **Store only reconstructions**: smallest, but discards the lossless record and
  blocks reprocessing with better solvers (e.g. FWI, ADR-0004).

## References (to the real code)

- `crates/sonic-ct/src/butterfly.rs` (`RawRfFrame`)
- `crates/sonic-ct/src/acquisition.rs` (`Measurement` — current reduced product)
- `crates/sonic-ct/src/memory.rs` (`AcousticMemory::to_bytes`/`from_bytes`,
  `ScanRecord`)
