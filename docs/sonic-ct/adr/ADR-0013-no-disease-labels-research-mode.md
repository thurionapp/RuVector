# ADR-0013: No Disease Labels — Research Mode Only

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

The workbench produces organ hypotheses (ADR-0010) with explainable evidence
(ADR-0012). The next slippery step would be to attach a *state* judgement that
reads like a diagnosis — "enlarged", "lesion", "abnormal kidney". That crosses
from research signal into clinical assertion. The project has no clinical
validation, no regulatory clearance, and synthetic-only inputs (ADR-0005). The
U.S. FDA draws a conservative line between general-wellness/informational tools
and software intended to diagnose, treat, or prevent disease (Software as a
Medical Device). We must stay firmly on the research side of that line, in both
output content and wording.

## Decision

The system **outputs organ state vectors and hypotheses, never diagnoses**, and
asserts no disease names. The persistent disclaimer in `Badge` from
`examples/sonic-ct/src/hud/Hud.jsx` — "research only — not diagnostic" — stays
visible at all times. Limitations are communicated through quality flags, not
diagnostic language: the `QF_LABELS` in `Hud.jsx` ("Bone shadowing", "Sparse path
coverage", "Boundary uncertainty", "Gas artifact"), populated from
`qualityFlags` in `engine.js` (`sct_quality_flag`), tell the user where the
reconstruction is unreliable rather than what is wrong with the subject.

UI copy uses research-mode vocabulary — "research only", "requires validation",
"hypothesis", "inferred", "not measured" (ADR-0011) — and never names a disease
or states a finding. Following the FDA's general-wellness-vs-diagnosis
distinction conservatively, we make no quantitative clinical claims and cite no
performance numbers.

**Acceptance:** the UI uses research/hypothesis/validation language only; no
disease name is asserted and no diagnostic claim is made.

## Consequences

### Positive

- Keeps the project unambiguously on the research side of the SaMD boundary.
- Quality flags turn limitations into honest, actionable context instead of
  silent failure or pseudo-diagnosis.
- Consistent "hypothesis/research" wording resists scope creep toward clinical
  framing.

### Negative / Trade-offs

- The tool cannot offer the "is this normal?" answer some users will want.
- Maintaining the boundary is an ongoing review discipline — any new label or
  copy string must be checked against the no-diagnosis rule.

## Alternatives Considered

- **Emit normal/abnormal or disease labels**: rejected — unvalidated, legally
  hazardous, and outside research scope (ADR-0005).
- **Drop quality flags to avoid clinical resemblance**: rejected — they are
  limitation signals, not findings, and removing them would reduce honesty.

## References

- `examples/sonic-ct/src/hud/Hud.jsx` (`Badge` "research only — not diagnostic",
  `QF_LABELS`, `OrganPanel` hypothesis framing)
- `examples/sonic-ct/src/engine.js` (`qualityFlags` from `sct_quality_flag`)
- ADR-0005 (research/simulation-only boundary)
- FDA, "Artificial Intelligence-Enabled Software as a Medical Device":
  https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device
