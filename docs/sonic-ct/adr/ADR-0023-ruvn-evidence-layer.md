# ADR-0023: ruvn as the Evidence-Intelligence Layer (Claim Gate)

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

Darwin tests which modality/prior combinations improve reconstruction, but
"improves the metric" is not "is scientifically supported". We need a research
truth filter that grades the evidence behind each modality, prior, and
user-facing claim. [`@ruvnet/ruvn`](https://github.com/ruvnet/ruvn) is a research
agent harness (scout → search → source-grader → synthesizer → fact-checker →
citer) that returns a graded, cited dossier and grades sources A/B/C/D
(synthesis only from A/B). It is explicitly a research tool, not medical advice.

## Decision

Integrate ruvn as an **evidence layer off the reconstruction hot path**, behind
an [`EvidenceProvider`](../../packages/metabiohacker/src/evidence/types.ts)
interface. A deterministic `CachedEvidenceProvider` (committed evidence cache)
powers tests and offline runs; an optional `RuvnEvidenceProvider` shells out to
the `ruvn` CLI (requires `OPENROUTER_API_KEY`) for nightly refresh / new-modality
review. `evidenceGate` enforces the hard rule: a modality, prior, or claim ships
only when the dossier grade is **A or B** and citations are present; pathology,
biopsy, Pap, HPV, and cytology always force human review. Evidence grade and
unsupported-claim count become additional Darwin Pareto objectives.

ruvn is **never** called in the hot path and is **not** a hard dependency
(proprietary license, heavy LLM/web runtime).

## Consequences

### Positive
- No medical claim ships without graded, cited support; science is auditable.
- Fast hot path; evidence refreshed asynchronously.

### Negative / Trade-offs
- The real provider needs network + OpenRouter; the cache can go stale (mitigated by refresh + timestamps).

## Alternatives Considered
- Calling ruvn inline (rejected: latency/cost/flakiness in the hot path).
- Bundling ruvn as a hard dep (rejected: proprietary license + heavy deps).

## References
- `packages/metabiohacker/src/evidence/*`. See ADR-0014 (Darwin), ADR-0018
  (governance), ADR-0013 (no disease labels).
