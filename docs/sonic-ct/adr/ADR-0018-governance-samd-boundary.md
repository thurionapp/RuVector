# ADR-0018: Governance and the Software-as-a-Medical-Device Boundary

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Multimodal fusion (ADR-0017) and reconstruction confidence (ADR-0014) produce
outputs that *could* influence clinical decisions. The moment software is intended
to inform diagnosis, treatment, triage, or patient management, it may meet the
definition of Software as a Medical Device (SaMD) and fall under regulatory
oversight. We need an explicit boundary so the project knows, by construction,
which side of that line it operates on.

## Decision

MetaBioHacker stays **outside autonomous diagnosis**. It operates only in:

- **Research mode** — dataset construction, model and harness evaluation,
  validation; no patient-facing clinical output.
- **Wellness mode** — general, non-diagnostic monitoring framed against the
  person's own baseline.
- **Clinical Decision Support (CDS) with mandatory professional review** — outputs
  surfaced to a qualified human who makes the decision; the software never decides.

If MetaBioHacker influences diagnosis, treatment, triage, or management, it may
become SaMD and must enter a regulated pathway before any such use.

We reference regulatory guidance conservatively, without inventing dates or
figures:

- The FDA's total-product-lifecycle approach to AI/ML-enabled SaMD, including
  Good Machine Learning Practice (GMLP), Predetermined Change Control Plans
  (PCCP), and transparency principles. The FDA has issued GMLP and PCCP guidance;
  see
  [FDA — AI in Software as a Medical Device](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device).
- [Health Canada](https://www.canada.ca/en/health-canada.html) guidance on
  machine-learning-enabled medical devices.
- Ontario's Personal Health Information Protection Act
  ([PHIPA](https://www.ontario.ca/laws/statute/04p03)) for handling of personal
  health information.

Acceptance: no autonomous diagnosis and no treatment recommendation without a
regulated pathway; consent, audit, and human review are enforced for any output
that could bear on a clinical decision.

## Consequences

### Positive

- A clear, declared boundary makes scope decisions and reviews simple: anything
  that crosses into diagnosis/treatment/triage/management is gated.
- Aligns the project with recognized regulatory framings (GMLP, PCCP, PHIPA)
  early, reducing rework if a regulated pathway is later pursued.

### Negative / Trade-offs

- Excludes the most autonomous (and commercially tempting) use cases until and
  unless a regulated pathway is undertaken.
- Mandatory human review adds latency and operational cost to clinical-adjacent
  flows.

## Alternatives Considered

- **Ship autonomous diagnostic features and address regulation later**: rejected
  — unsafe and likely non-compliant; risks patient harm and enforcement.
- **Avoid all clinical-adjacent output**: rejected — CDS with mandatory review is
  valuable and stays within the boundary when properly gated.

## References

- [FDA — Artificial Intelligence in Software as a Medical Device](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)
- [Health Canada](https://www.canada.ca/en/health-canada.html)
- [Ontario PHIPA](https://www.ontario.ca/laws/statute/04p03)
- ADR-0005 (Medical Claims Boundary)
- ADR-0017 (Multimodal Fusion Patterns)
