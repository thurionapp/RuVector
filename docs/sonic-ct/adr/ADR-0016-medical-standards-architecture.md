# ADR-0016: Adopt Established Medical Standards Rather Than Bespoke Formats

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

The patient state graph (ADR-0015) requires every observation to declare a
`sourceFormat` and a `codeSystem`. If we invent our own codes and containers,
nothing we produce interoperates with hospitals, labs, registries, or research
networks, and we re-create decades of standards work badly. The mature path is to
map each modality onto an established standard and reserve our own structures for
similarity search, where no clinical standard applies.

## Decision

We adopt established standards per concern rather than bespoke formats:

- **Imaging** — DICOM (Digital Imaging and Communications in Medicine) for pixel
  data, geometry, and study/series identity. See
  [DICOM](https://www.dicomstandard.org/). This aligns with the planned DICOMweb
  adapter boundary (ADR-0006).
- **Clinical exchange** — HL7 FHIR for transporting observations, imaging studies,
  and clinical context between systems. See [HL7](https://www.hl7.org/).
- **Lab / observation codes** — LOINC (maintained by the Regenstrief Institute)
  for identifying labs and measurements. See [LOINC](https://loinc.org/).
- **Clinical concepts** — SNOMED CT (SNOMED International) for clinical findings
  and concepts. See [SNOMED International](https://www.snomed.org/).
- **Research-scale analytics** — OMOP CDM, the OHDSI open community standard for
  observational data. Analytics and cohort export target OMOP. See
  [OHDSI Data Standardization](https://www.ohdsi.org/data-standardization/).
- **Similarity search** — a RuVector index over reports, images, derived
  features, and prior cases. This is our own layer for retrieval, *not* a clinical
  standard, and it stores pointers and embeddings, not the system of record.

These map directly to the `codeSystem` enum in `MedicalObservation`
(LOINC | SNOMED | ICD | CPT | RxNorm) and to `sourceFormat` values (DICOM, FHIR,
HL7v2, etc.).

Regulatory and standards posture is deliberately conservative: we reference the
official bodies (DICOM, HL7, Regenstrief/LOINC, SNOMED International, OHDSI) and
the [FDA](https://www.fda.gov/) for device context, and we do not claim
certification or conformance we have not demonstrated.

Acceptance: each supported modality has a defined standard mapping, and analytics
export to OMOP CDM.

## Consequences

### Positive

- Outputs are portable to PACS, FHIR servers, labs, and OHDSI research networks.
- Coding systems are externally governed, so we inherit their vocabularies and
  updates instead of maintaining our own.

### Negative / Trade-offs

- Standards are large and version-sensitive; mapping and conformance is real work.
- Some standards (SNOMED CT, certain DICOM toolchains) carry licensing or
  governance obligations we must track.

## Alternatives Considered

- **A single bespoke interchange format**: rejected — zero interoperability and
  duplicates standards work; the `.rvf`-style container already covers internal
  portability (ADR-0006).
- **Standards for imaging only**: rejected — labs, concepts, and analytics need
  LOINC/SNOMED/OMOP to be useful at research scale.

## References

- [DICOM](https://www.dicomstandard.org/)
- [HL7 / FHIR](https://www.hl7.org/)
- [LOINC (Regenstrief Institute)](https://loinc.org/)
- [SNOMED International](https://www.snomed.org/)
- [OHDSI / OMOP CDM](https://www.ohdsi.org/data-standardization/)
- [FDA](https://www.fda.gov/)
- ADR-0006 (DICOMweb / FHIR as Planned External Adapters)
- ADR-0015 (Patient State Graph — `codeSystem` / `sourceFormat`)
