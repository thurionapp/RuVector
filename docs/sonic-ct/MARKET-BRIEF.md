# sonic_ct — Market & Product-Wedge Brief

> Honesty note: All market sizes here are **illustrative; definitions vary** and
> should not be treated as forecasts. This brief is about *sequencing* — which
> wedge is credible first — not about asserting a total addressable market.
> `sonic_ct` itself is research/simulation software making **no diagnostic
> claim**, and this brief keeps wellness and diagnostic positioning strictly
> separated.

## Thesis: Don't Sell "Replace MRI" — Sell the First Credible Wedge

The temptation with a full-body acoustic scanner is to position it as an
MRI/CT alternative. That framing is a regulatory and clinical-evidence trap:
diagnostic-equivalence claims demand large prospective trials, device clearance,
and reimbursement pathways that take years and capital most early efforts do not
have. The credible strategy is to enter through a **general-wellness** wedge that
needs no diagnostic claim, accumulate evidence and data assets, and only then
climb toward regulated clinical modules. The wedges below are ordered by
time-to-revenue and regulatory friction, lowest first.

## Wedge 1 — Body-Composition & Longitudinal Wellness Scanning

The strongest first commercial wedge. A ring scanner that produces quantitative
acoustic maps can report **body-composition and structural-trend metrics** —
fat envelope distribution, lean/muscle structure, organ-region size proxies —
tracked *longitudinally*. This is squarely the territory of established wellness
imaging (DEXA-style body composition, full-body MRI screening startups) but with
a non-ionizing, water-bath-coupled, potentially lower-cost ultrasound modality.

Why it fits `sonic_ct`'s grain:
- The reconstruction → segmentation → metrics pipeline already produces
  per-tissue maps and quantitative summaries.
- The **acoustic memory** (`memory.rs`) is purpose-built for longitudinal
  tracking: per-subject timelines, cosine-drift between earliest/latest scans,
  and portable `.rvf` archives make "show me how my composition changed over six
  months" a native operation, not an add-on.
- Outputs are *trends and proportions*, not diagnoses — the natural home for
  general-wellness claims.

Illustrative market context (definitions vary): preventive/wellness imaging and
body-composition services are a meaningful and growing consumer-health segment,
but figures depend heavily on whether one counts DEXA, MRI screening, fitness
testing, or all three. Treat any single dollar number with suspicion.

## Wedge 2 — Preventive-Imaging Membership / Spa Model

The delivery vehicle for Wedge 1. Rather than per-scan medical billing, a
**membership** model (annual or quarterly scans bundled with a longitudinal
dashboard) mirrors the executive-physical and "wellness spa" market that
full-body-MRI screening companies have already validated. The recurring-revenue
shape rewards exactly the longitudinal-tracking capability `sonic_ct` centers on,
and it keeps the customer relationship in the consumer-wellness lane where
general-wellness claims are permissible. Key discipline: the membership product
must present **trends and educational context**, and must not drift into implied
diagnosis or screening claims without the evidence and clearance to back them.

## Wedge 3 — Research Platform & Tooling

A parallel, lower-capital wedge that monetizes the simulator itself rather than a
clinical service. `sonic_ct` is, today, a clean, dependency-free, WASM-portable
USCT simulator with:
- procedural, deterministic **phantoms** (a reproducible benchmark corpus),
- a **reconstruction benchmark harness** (SART baseline, MAE/Dice metrics),
- a transparent segmentation/training loop, and
- a portable archive format.

That is precisely the toolkit USCT and FWI researchers, device teams, and
algorithm groups need: shared phantoms, reproducible baselines, and a
reconstruction leaderboard. A tooling/benchmark offering (open core plus
supported builds, hosted benchmark, or licensed phantom/eval suites) can generate
early revenue and credibility, seed an academic user base, and de-risk the
clinical roadmap by attracting the very FWI expertise the product needs.
Illustrative sizing: scientific-imaging tooling is a niche but sticky market;
value is in workflow lock-in and benchmark authority, not unit volume.

## Wedge 4 — Clinical Specialty Modules (After Evidence)

Only *after* the wellness wedge has accumulated longitudinal data and the physics
has climbed to FWI-grade fidelity should specialty clinical modules follow — e.g.
musculoskeletal assessment, hepatic or other organ-focused quantification — each
gated behind its own evidence and clearance. These are **diagnostic** outputs and
must be developed, validated, labelled, and (where applicable) cleared
independently of the wellness product. The architectural rule that keeps physics
reconstruction separate from AI segmentation, and preserves raw RF evidence,
exists precisely so a future regulated module can be validated on auditable data
without re-engineering the platform.

## Wedge 5 — Butterfly-Adjacent Embedded Software Tooling

`butterfly.rs` is explicitly a **mock boundary, not an SDK** — there is no public
raw-hardware SDK for Butterfly Ultrasound-on-Chip modules, and the code says so.
The adjacent-but-honest opportunity is *software tooling around* low-cost
Ultrasound-on-Chip hardware: a clean acquisition-backend contract
(`AcquisitionBackend`), raw-RF data-format design, simulation-to-hardware
parity testing, and reconstruction pipelines that a future licensed backend could
plug into. This positions `sonic_ct` as the reconstruction/tooling layer for an
ecosystem of inexpensive arrayed probes without overclaiming a partnership or
integration that does not exist. Should an embedded raw-data path become
available, the boundary is already designed for it.

## Regulatory Posture

| Dimension | Posture |
|---|---|
| Claim class (initial) | **General wellness** — composition/trend education only; no diagnosis, no screening claim |
| Output separation | Wellness outputs and any future diagnostic outputs kept in **distinct products/labels**; never co-mingled in one report |
| AI/ML lifecycle | Treat the learned components as an **AI/ML medical-device lifecycle** problem when (and only when) diagnostic: versioned models, locked vs. adaptive distinction, change control, validation datasets, performance monitoring |
| Evidence preservation | **Raw evidence preserved** (RF-frame data contract designed from day one) so any future regulated claim is auditable and reproducible |
| Physics vs. AI boundary | Physics reconstruction kept separate from AI segmentation, so the quantitative substrate can be validated independently of the classifier |
| Hardware claims | Butterfly boundary labelled as a **mock**; no implied hardware certification or SDK partnership |

## Sequencing Summary

| Wedge | Time-to-revenue | Reg. friction | Depends on |
|---|---|---|---|
| 1. Body-composition / longitudinal wellness | Near | Low (general wellness) | Current pipeline + memory |
| 2. Preventive-imaging membership/spa | Near | Low–medium | Wedge 1 + dashboard |
| 3. Research platform / tooling | Now | Minimal | Simulator as-is |
| 4. Clinical specialty modules | Far | High (diagnostic) | FWI fidelity + evidence + clearance |
| 5. Butterfly-adjacent embedded tooling | Medium | Low (software) | Licensed raw-data backend |

**Bottom line.** Lead with wellness-grade body-composition longitudinal scanning
delivered through a membership model, fund and de-risk it with a research-tooling
wedge, design honestly around (not on top of a fictional) Butterfly SDK, and earn
the right to diagnostic specialty modules through evidence — keeping wellness and
diagnostic claims, and physics and AI, cleanly separated throughout.
