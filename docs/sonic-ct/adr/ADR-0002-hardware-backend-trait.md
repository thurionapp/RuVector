# ADR-0002: Hardware Acquisition Behind a Backend Trait

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

We want hardware to be addable later (ADR-0001 is simulation-first), but the
reconstruction core must not learn anything about a specific device. There is
**no public raw-hardware SDK** for the Butterfly Ultrasound-on-Chip modules, so
we cannot integrate against a real driver today. What we can do is fix the seam:
define the contract the core consumes for acquisition, and provide a simulated
implementation of it so the rest of the pipeline is written against the seam
from day one.

## Decision

Define `AcquisitionBackend` in `butterfly.rs`:

```rust
pub trait AcquisitionBackend {
    fn name(&self) -> &str;                                  // provenance
    fn acquire(&self, phantom: &Phantom, ring: &Ring) -> Acquisition;
}
```

The core depends only on the returned `Acquisition` (`acquisition.rs`), never on
how it was produced. Today the only implementer is
`MockButterflyEmbeddedBackend`, which delegates to `acquisition::simulate`. Its
hardware shape is described by `ButterflyEmbeddedConfig` (default 40 modules ×
64 channels = 2560 elements, ~3 MHz centre frequency) — values drawn from public
prototype figures, not from a licensed SDK. `name()` returns
`"mock-butterfly-embedded"` so provenance is honest in any logged scan.

A future licensed backend implements the same trait and produces an
`Acquisition` (eventually from real `RawRfFrame`s, see ADR-0003) without
touching `reconstruction.rs`, `segmentation.rs`, or `pipeline.rs`.

## Consequences

### Positive

- Core/reconstruction is decoupled from any device; one trait is the only seam.
- Provenance is explicit via `name()` — a scan records which backend made it.
- `ButterflyEmbeddedConfig` documents target hardware geometry without faking an
  SDK that does not exist.

### Negative / Trade-offs

- The trait was shaped by the simulator, so a real device may need contract
  revisions (sample rate, framing) once `RawRfFrame` is actually populated.
- `acquire(phantom, ring)` is phantom-driven; a hardware backend would ignore
  `phantom` and read the device, so the signature is slightly simulation-biased.

## Alternatives Considered

- **Hard-code `simulate` calls in the pipeline**: simplest, but bakes the
  simulator into the core and blocks any future device.
- **Vendor SDK adapter now**: impossible — no public raw SDK exists.

## References (to the real code)

- `crates/sonic-ct/src/butterfly.rs` (`AcquisitionBackend`,
  `MockButterflyEmbeddedBackend`, `ButterflyEmbeddedConfig`)
- `crates/sonic-ct/src/acquisition.rs` (`Acquisition`, `simulate`)
- `crates/sonic-ct/src/pipeline.rs` (consumes `Acquisition`, not a backend)
