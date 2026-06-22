# ADR-0006: DICOMweb / FHIR as Planned External Adapters

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

If `sonic_ct` ever interoperates with imaging infrastructure, the lingua franca
is DICOM/DICOMweb for pixel data and FHIR for clinical context. We want to
record the intended integration boundary so that internal representations stay
compatible with those standards — without pulling heavy DICOM/FHIR dependencies
into a zero-dependency core, and without pretending these adapters exist. This
is a **forward-looking** decision: **DICOMweb and FHIR are not yet implemented.**

## Decision

DICOMweb and FHIR support are **planned external adapters**, deliberately *not*
part of `crates/sonic-ct/`. The core stays zero-dependency (ADR-0001) and
standards-agnostic; any DICOM/FHIR mapping will live in a separate adapter crate
or service that translates to/from the core's existing structures:

- A reconstructed scan is a `Scene` (`pipeline.rs`) with `Grid` speed/attenuation
  maps and a `Segmentation` — these map cleanly to DICOM multi-frame pixel data
  with a per-class segmentation overlay when an adapter is written.
- Provenance and identity already exist in a standards-friendly shape:
  `ScanRecord` (`memory.rs`) carries a stable `id`, a pseudonymous `patient_id`,
  a `timestamp`, and quality fields (`mean_dice`, `mae`) — the seeds of DICOM
  patient/study/series identifiers and a FHIR `ImagingStudy`/`Observation`.
- The `.rvf`-style container (`to_bytes`/`from_bytes`) is the current portable
  format; a DICOMweb adapter would sit beside it, not replace the core.

No DICOM or FHIR code, types, or network calls exist in the repository today.

## Consequences

### Positive

- The core stays small and dependency-free; standards complexity is quarantined
  to a future adapter boundary.
- Existing structures (`Grid`, `Segmentation`, `ScanRecord`) already align with
  the eventual mapping, reducing future rework.

### Negative / Trade-offs

- No interoperability today — scans cannot be exported to a PACS or FHIR server.
- The clean-mapping assumption is unverified until an adapter is actually built;
  some fields (units, geometry, coding systems) may need adjustment then.

## Alternatives Considered

- **Build DICOMweb/FHIR into the core now**: violates zero-dependency and
  research-only scope (ADR-0001, ADR-0005) for unbuilt interop.
- **Invent a bespoke interchange format only**: the `.rvf`-style container already
  covers portability; standards adapters are the right external layer when needed.

## References (to the real code)

- `crates/sonic-ct/src/pipeline.rs` (`Scene` — the export source of truth)
- `crates/sonic-ct/src/grid.rs` (`Grid` — pixel-data analogue)
- `crates/sonic-ct/src/segmentation.rs` (`Segmentation` — overlay analogue)
- `crates/sonic-ct/src/memory.rs` (`ScanRecord`, `to_bytes`/`from_bytes`)
