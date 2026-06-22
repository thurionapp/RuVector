# sonic_ct — SPARC Analysis

> SPARC = **S**pecification, **P**seudocode, **A**rchitecture, **R**efinement,
> **C**ompletion. This document analyzes the `sonic_ct` USCT simulator against
> its real, implemented modules. `sonic_ct` is research/simulation software,
> makes **no diagnostic claim**, and the Butterfly Embedded boundary is a mock,
> not a hardware SDK.

---

## S — Specification

### Inputs

| Input | Source module | Notes |
|---|---|---|
| Ring geometry | `geometry.rs` (`Ring::new`) | N elements on a circle, radius = `ring_frac × half_FOV`; inward normals |
| Tissue speed map | `phantom.rs` → `Grid` (`types.rs`, `grid.rs`) | Speed-of-sound (m/s); procedural abdomen phantom or external map |
| Tissue attenuation map | `phantom.rs` → `Grid` | Acoustic attenuation (Np/m), co-registered with speed |
| Ground-truth labels | `phantom.rs` → `Grid` | Per-cell `Tissue` class (water/fat/muscle/organ/bone) |
| Source/receiver plan | `acquisition.rs` (`AcquisitionConfig`) + `Ring::fan_receivers` | Fan width, min angular separation, samples/cell, timing noise |
| Optional raw RF frames | `butterfly.rs` (`RawRfFrame`) | Data-*contract* shape only (channels × samples); simulator does not synthesize waveforms |

### Outputs

| Output | Module |
|---|---|
| Projection measurements (TOF, attenuation, validity, ray geometry) | `acquisition.rs` (`Acquisition`, `Measurement`) |
| Speed-of-sound reconstruction (m/s) | `reconstruction.rs` (`reconstruct_speed`) |
| Attenuation reconstruction (Np/m) | `reconstruction.rs` (`reconstruct_attenuation`) |
| Segmentation mask + per-cell uncertainty | `segmentation.rs` (`Segmentation`) |
| Dice (per-class + mean), MAE metrics | `metrics.rs` (`QualityReport`) |
| Inspection images (PGM) | `grid.rs` (`to_pgm`) |
| Acoustic-memory archive (`.rvf`-style, NSW index) | `memory.rs` (`AcousticMemory`, `to_bytes`/`from_bytes`) |

### Hard Constraints

1. **No diagnostic claim.** Outputs are research/quantitative, not clinical
   findings. Enforced as a documentation and labelling invariant (`lib.rs`).
2. **No fake Butterfly SDK.** `butterfly.rs` is a *mock* `AcquisitionBackend`;
   it must never present as a licensed hardware integration.
3. **Preserve raw evidence.** The `RawRfFrame` contract and the portable
   archive format exist so raw/intermediate data stays auditable from day one.
4. **Physics ≠ AI.** Reconstruction (physics inverse problem) and segmentation
   (AI classification) are separate modules and must remain swappable
   independently. The segmenter consumes reconstructions; it never alters them.
5. **Determinism / dependency-free.** Phantom and pipeline are reproducible
   (seeded PRNGs) and build to `wasm32-unknown-unknown`.

---

## P — Pseudocode (End-to-End Pipeline)

```text
function run(cfg, model):                      # pipeline.rs::run_with_model
    validate(cfg)                              # reject out-of-range config

    # --- Acquisition layer ---
    phantom  = Phantom.build(cfg.phantom)      # phantom.rs: seeded speed/atten/labels
    ring     = Ring.new(cfg.elements,
                        half_fov * cfg.ring_frac)   # geometry.rs

    acq = []                                    # acquisition.rs::simulate
    slowness = 1 / phantom.speed                # linear travel-time integral
    for source in ring.elements:
        for receiver in ring.fan_receivers(source, fan, min_sep):
            if receiver <= source: continue     # de-dup reciprocal pairs
            ray = Ray.between(grid, src_pos, rcv_pos)     # ray.rs (DDA cells)
            tt  = ray.integrate(slowness) + exterior_water_leg
            att = ray.integrate(attenuation)
            valid = ray spends > 50% length in tissue
            acq.append(Measurement{tt, att, ray, valid, ...})
    if acq.valid_count == 0: return error(NoMeasurements)

    # --- Physics layer (SART) ---
    speed = SART(acq, init=1/WATER_SPEED,       # reconstruction.rs
                 rhs = travel_time - exterior_water_leg)   # solves A·s = t
    speed = 1/slowness, clamped to [SPEED_MIN, SPEED_MAX]
    atten = SART(acq, init=0, rhs = attenuation)
    # 1 SART sweep == delay-backprojection baseline; more sweeps -> least squares

    # --- AI layer (segmentation, kept separate from physics) ---
    seg = segment(speed, model)                 # segmentation.rs
        for each cell c:
            label = model.classify(c)           # piecewise speed-band
            uncertainty = exp(-margin_to_boundary / margin_scale)

    # --- Clinical-workflow layer (metrics) ---
    dice      = dice_all(seg.labels, phantom.labels)       # metrics.rs
    mae_speed = mean_abs_diff(speed, phantom.speed)
    quality   = {mae_speed, dice, mean_dice, measurements}

    # --- Governance layer (memory + coherence) ---
    embedding = speed.embedding(k)              # grid.rs: k×k, mean-centred, L2
    coherence = check_coherence(seg.labels)     # memory.rs: anatomical rules
    memory.insert(ScanRecord{id, patient_id, ts, embedding, dice, mae})
    archive = memory.to_bytes()                 # .rvf-style portable container

    return Scene{phantom, ring, acq, speed, atten, seg, quality}
```

Offline training loop (`model.rs`, `bin/train.rs`): coordinate-ascent over the
segmentation band boundaries to maximize mean Dice on a corpus of
reconstruction/ground-truth pairs — produces `SegModel::tuned()`.

---

## A — Architecture (Five Layers → Real Modules)

| Layer | Responsibility | Modules |
|---|---|---|
| **1. Acquisition** | Geometry, ray tracing, transmission simulation, hardware boundary, raw-RF contract | `geometry.rs`, `ray.rs`, `acquisition.rs`, `butterfly.rs`, `phantom.rs` |
| **2. Physics** | Inverse problem: TOF/attenuation reconstruction (SART), grid math, types/constants | `reconstruction.rs`, `grid.rs`, `types.rs` |
| **3. AI** | Tissue segmentation, per-cell uncertainty, reproducible model training | `segmentation.rs`, `model.rs` |
| **4. Clinical Workflow** | Quality metrics (Dice/MAE), inspection imagery, end-to-end orchestration, UI | `metrics.rs`, `pipeline.rs`, `grid::to_pgm`, `crates/sonic-ct-wasm/` (raw C-ABI, ~31 KB), `examples/sonic-ct/` (React Three Fiber) |
| **5. Governance** | Acoustic memory (NSW index, longitudinal tracking, FWI warm-start), anatomical graph-coherence, portable `.rvf` archive, 3-D sweep scaffolding | `memory.rs`, `volume3d.rs` |

**Boundaries that matter:**
- `AcquisitionBackend` (Layer 1) decouples physics from data source — a licensed
  hardware backend can replace `MockButterflyEmbeddedBackend` untouched.
- Layer 2 (physics) emits *only* property maps; Layer 3 (AI) consumes them and
  never writes back — preserving the physics/AI separation constraint.
- Layer 5 (governance) observes outputs (embeddings, coherence) without altering
  reconstructions, keeping evidence and audit trails intact.

The WASM crate is intentionally a **raw C-ABI** module (no `wasm-bindgen`),
keeping the browser artifact tiny (~31 KB) and the JS glue explicit; the React
Three Fiber UI in `examples/sonic-ct/` is a *consumer* of the core and holds no
reconstruction logic.

---

## R — Refinement Roadmap

| Stage | Status | Description |
|---|---|---|
| 1. TOF SART | **Done** | Straight-ray travel-time + attenuation; 1 sweep = delay backprojection (`reconstruction.rs`) |
| 2. Finite-difference wave propagation | Planned | Replace ray integral with an FD acoustic forward solver behind Layer 1/2 boundary |
| 3. Adjoint FWI | Planned | Adjoint-state gradients of waveform misfit; the documented fix for bone Dice ≈ 0 |
| 4. Frequency continuation + source encoding | Planned | Multiscale low→high schedule (cycle-skip mitigation) + encoded sources for compute reduction |
| 5. Learned sparse completion | Planned | AI measurement/image completion for sparse rings, layered *on top of* physics, evidence preserved |
| 6. 3-D vertical sweep | Stub (`volume3d.rs`) | Promote `SweepPlan` to full stacked-slice 3-D reconstruction with inter-slice regularization |
| 7. DICOMweb / FHIR adapters | Planned | Standards-based export at the clinical-workflow layer |
| 8. QMS / validation harness | Planned | AI/ML-lifecycle change control, validation datasets, performance monitoring (gating any diagnostic claim) |

Sequencing rationale: stages 2–4 raise *physics fidelity* (the biggest measured
gap — bone Dice ≈ 0 from straight-ray blur); stages 5–6 raise *coverage and
efficiency*; stages 7–8 raise *clinical/regulatory readiness*. The acoustic
memory's `warm_start` already anticipates stage 3 by retrieving the nearest
prior reconstruction as an FWI starting model to reduce cycle-skipping.

---

## C — Completion Criteria Checklist

**Implemented (current baseline):**
- [x] Deterministic procedural phantom (speed/attenuation/labels)
- [x] Ring geometry + fan acquisition with reciprocal de-duplication
- [x] SART speed + attenuation reconstruction (clamped to physical bounds)
- [x] Transparent speed-band segmentation with per-cell uncertainty
- [x] Reproducible coordinate-ascent model training (`SegModel::tuned`)
- [x] Dice (per-class + mean) and MAE metrics
- [x] PGM inspection images
- [x] Acoustic memory: NSW index, patient timelines, longitudinal drift, warm-start
- [x] Anatomical graph-coherence anomaly check
- [x] Portable `.rvf`-style archive (round-trips via `to_bytes`/`from_bytes`)
- [x] Mock Butterfly `AcquisitionBackend` + `RawRfFrame` data contract
- [x] WASM raw C-ABI build (~31 KB) + React Three Fiber UI
- [x] No-diagnostic-claim / no-fake-SDK invariants documented in `lib.rs`

**Measured baseline (simulator self-report, not external claims):**
- [x] Tuned mean Dice ~0.63 (vs ~0.30 default), MAE ~28–31 m/s, ~8000 measurements
- [ ] Bone Dice > 0 — **open**, blocked on wave physics (straight-ray blur)

**Remaining for higher fidelity / clinical readiness:**
- [ ] Finite-difference wave forward solver
- [ ] Adjoint-state FWI with frequency continuation + source encoding
- [ ] Learned sparse completion (physics-preserving, evidence-preserving)
- [ ] 3-D stacked-slice reconstruction with inter-slice regularization
- [ ] DICOMweb / FHIR export adapters
- [ ] QMS / validation harness and AI/ML lifecycle controls
- [ ] Wellness vs. diagnostic output separation enforced at product layer

**Invariant gates (must hold at every stage):** no diagnostic claim · no fake
Butterfly SDK · raw evidence preserved · physics reconstruction separate from AI
segmentation · deterministic, dependency-free core that builds to WASM.
