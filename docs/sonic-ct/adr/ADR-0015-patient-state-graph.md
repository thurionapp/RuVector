# ADR-0015: Patient Data as a Graph of Typed Observations

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

The tempting design for a multimodal medical system is "one big model that eats
everything" — feed images, labs, notes, and waveforms into a single representation
and read out a state. That design loses the two things clinical and research use
depend on most: *provenance* (where each number came from, in what units, under
what consent) and *uncertainty* (how much to trust it). It also makes timelines,
audit, and consent enforcement nearly impossible after the fact.

A person is better modelled as an accumulating timeline of discrete, typed
observations from many procedures, each self-describing.

## Decision

We model patient data as a graph of **typed observations**, not a single fused
blob. Each procedure adds a normalized event with a timestamp, units, provenance,
uncertainty, consent scope, and clinical context. The core type:

```ts
type MedicalObservation = {
  patientId: string;
  eventTime: string; // ISO 8601
  modality:
    | "acoustic" | "mri" | "ct" | "ultrasound" | "xray"
    | "ekg" | "eeg" | "lab" | "pathology" | "cytology"
    | "wearable" | "clinicalNote";
  sourceFormat: string; // e.g. "DICOM", "FHIR", "HL7v2", "CSV", "rvf"
  codeSystem: "LOINC" | "SNOMED" | "ICD" | "CPT" | "RxNorm";
  code: string;
  name: string;
  value: number | string | null;
  unit: string | null;
  rawUri: string; // pointer to the original artifact, never discarded
  derivedFeatures: Record<string, number>;
  uncertainty: { kind: "ci95" | "stddev" | "categorical"; value: number };
  provenance: {
    device?: string;
    operator?: string;
    pipeline?: string; // engine/version that produced derived values
    ingestedAt: string;
  };
  consentScope: string[]; // e.g. ["research", "clinical-care", "screening"]
};
```

Observations are nodes; edges link them by patient, time, anatomy, and derivation
(a derived feature edges back to its `rawUri`). The graph is append-only.

Key principle: **never lose provenance.** Raw artifacts are referenced by
`rawUri`, not summarized away, so any derived value can be traced to its source
and pipeline version.

Acceptance: every observation carries provenance, uncertainty, and a consent
scope; an observation missing any of the three is rejected at ingestion.

## Consequences

### Positive

- Timelines, audit, and consent filtering fall out naturally from the structure.
- Each modality keeps its native semantics (units, coding system) instead of
  being flattened into a lossy shared space.
- Uncertainty travels with every value, supporting the uncertainty-first stance
  (ADR-0007).

### Negative / Trade-offs

- More schema discipline and per-modality mapping work than a single fused model.
- Cross-modality reasoning happens above the graph rather than inside one model,
  which is more explicit but more code.

## Alternatives Considered

- **Single fused model over raw inputs**: rejected — destroys provenance,
  consent scope, and per-value uncertainty.
- **Flat per-modality tables**: rejected — loses the derivation and temporal edges
  that make longitudinal reasoning and audit tractable.

## References

- ADR-0007 (Uncertainty-First AI)
- ADR-0016 (Medical Standards Architecture — code systems and source formats)
- ADR-0003 (Preserve Raw RF Before AI — the same never-lose-the-source principle)
