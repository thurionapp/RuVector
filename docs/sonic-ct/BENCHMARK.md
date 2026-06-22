# MetaBioHacker reconstruction benchmark

Frozen engine: `sonic_ct_serve`. Only the harness config differs between rows.
Reports are split so reconstruction **speed** is never conflated with real
anatomical **fidelity**.

## 1. Synthetic phantom benchmark

Statistics over 12 reproducible synthetic phantoms (mean ± 95% CI).

| Config | Dice (95% CI) | Acoustic residual | Latency (ms) |
|--------|---------------|-------------------|--------------|
| baseline | 0.543 ± 0.002 | 0.028 | 412 |
| evolved | 0.545 ± 0.004 | 0.028 | 172 |

**Evolved vs baseline:** Dice +0.4%, **latency 140.0% faster**, residual −0.1%.

## 2. Real public slice benchmark (region-level)

Real CT slices (Wikimedia Commons, fetched on demand, not committed) are
calibration targets — **not** ultrasound-CT. Intensity is banded into the five
acoustic classes as a proxy ground truth. Region-level Dice + a domain-gap score
gate headline inclusion.

| Slice | fluid | fat | soft tissue | bone | domain gap | inclusion |
|-------|-------|-----|-------------|------|-----------|-----------|
| real-abdomen | 0.708 | 0.513 | 0.000 | 0.000 | 0.600 | **exclude** |
| real-thorax | 0.532 | 0.120 | 0.039 | 0.468 | 0.499 | **researchOnly** |

Domain gap < 0.30 → headline · 0.30–0.60 → research only · > 0.60 → excluded.

## 3. Governance & safety benchmark

- Acoustic residual is invariant to multimodal/contradiction layers (physics frozen).
- Pathology/biopsy/Pap/HPV/cytology force human review.
- User-facing claims require ruvn evidence grade **A/B** with citations (acoustic USCT grades **C → research-only**).
- Reconstruction run ledgers verify end-to-end (tamper-evident).

## Headline (honest wording)

> The Darwin-evolved reconstruction harness achieved about **140% faster runtime at equal synthetic-phantom Dice**.
> On real public CT slices, Dice remained **research stage (~0.300)**, showing the expected domain
> gap between controlled acoustic phantoms and real anatomical images.
> No diagnostic claims are emitted; the multimodal layer only adjusts priors, uncertainty, routing, and review state.
