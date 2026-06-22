# ADR-0020: Canonical Observation as the Multimodal Ingest Boundary

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

MetaBioHacker must ingest many artifact types — MRI/CT/ultrasound (DICOM), labs
(CSV/FHIR/LOINC), waveforms (EKG/EEG), pathology/Pap/cytology reports, wearables,
and acoustic scans. A "one model eats everything" design loses provenance and
units and invites silent misreads.

## Decision

Every artifact is parsed by a modality-specific **adapter** into one canonical
[`MedicalObservation`](../../packages/metabiohacker/src/ingest/types.ts) carrying
`modality`, `sourceFormat`, optional `codeSystem`/`code` (LOINC/SNOMED/…),
`value`/`unit`, `bodySite`/`specimenType`, `derivedFeatures`, `uncertainty`,
`qualityScore`, `consentScope`, a `provenance` block (sourceSystem, sourceId,
parserVersion, content hash), and `humanReviewRequired`. Adapters share the
`IngestAdapter` contract (`name`, `parse`). V0 ships `labAdapter` (CSV→LOINC),
`imagingAdapter` (DICOM sidecar JSON), and `pathologyAdapter` (always review).

## Consequences

### Positive
- Uniform downstream code; FHIR/DICOM/LOINC/SNOMED/OMOP stay at the edges.
- Provenance + uncertainty travel with every datum.

### Negative / Trade-offs
- Adapters must be written per source; V0 imaging reads a sidecar, not raw DICOM.

## Alternatives Considered
- Per-modality bespoke pipelines (rejected: no shared provenance/audit).
- Storing raw artifacts only (rejected: not queryable or fusable).

## References
- `packages/metabiohacker/src/ingest/*`; FHIR Observation/DiagnosticReport,
  DICOM, LOINC. See ADR-0016 (standards) and ADR-0015 (patient state graph).
