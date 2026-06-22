# ADR-0017: Typed Multimodal Fusion Patterns for Monitoring and Research

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Once observations live in a typed graph (ADR-0015), the value is in combining
modalities — anatomy with chemistry, structure with function, imaging with
pathology. But "fusion" can quietly drift into autonomous diagnosis. We want a
small set of *named, typed* fusion patterns that are explicitly scoped to
monitoring and research, each declaring its provenance, uncertainty, and a path
to a human reviewer. The governance boundary (ADR-0018) is what these patterns
operate inside.

## Decision

We define typed fusion patterns that combine modalities for **monitoring and
research only — never autonomous diagnosis**:

- **(a) Anatomy + chemistry** — ultrasound/MRI structure combined with blood
  labs (e.g. liver reconstruction with enzymes, bilirubin, platelets) to flag
  trends, not to diagnose disease.
- **(b) Heart function** — EKG + echo + troponin/BNP + wearable trend, fused to
  monitor functional change over time.
- **(c) Brain & sleep** — EEG + sleep study + HRV + medications + reported
  symptoms, combined for longitudinal monitoring.
- **(d) Cancer-screening research** — imaging + pathology + cytology + genomics.
  This is **high regulatory risk** and is restricted to dataset construction and
  validation only, not patient-facing output.
- **(e) Women's health** — Pap + HPV + cytology history + ultrasound + hormones,
  fused for longitudinal screening context, **not** autonomous interpretation.

Guiding principle: **the strongest signal is deviation from the person's own
baseline.** Fusion patterns are built to surface change against an individual's
history rather than to render a population-level verdict.

Acceptance: every fusion output declares the provenance of each contributing
modality, an aggregate uncertainty, and an explicit human-review path; outputs
that cannot declare all three are not emitted.

## Consequences

### Positive

- A finite, named set of patterns is auditable and easy to govern, versus
  open-ended "feed everything in" fusion.
- Baseline-relative framing reduces false alarms and keeps outputs interpretable.
- Provenance and uncertainty per modality satisfy ADR-0007 and ADR-0015.

### Negative / Trade-offs

- Hand-defining patterns is less flexible than a learned end-to-end fuser and
  needs maintenance as modalities are added.
- The highest-value research pattern (d) is deliberately the most restricted,
  limiting near-term product surface.

## Alternatives Considered

- **One end-to-end multimodal model producing a diagnosis**: rejected — crosses
  the SaMD/diagnosis boundary (ADR-0018) and loses per-modality provenance.
- **Population thresholds instead of personal baselines**: rejected — less
  sensitive to individual change and more prone to context-free alerts.

## References

- ADR-0015 (Patient State Graph — typed observations the patterns consume)
- ADR-0018 (Governance / SaMD Boundary — the limits these patterns operate within)
- ADR-0007 (Uncertainty-First AI)
- ADR-0005 (Medical Claims Boundary)
