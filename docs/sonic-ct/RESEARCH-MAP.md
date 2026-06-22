# sonic_ct — State-of-the-Art Research Map for Computational Ultrasound Tomography

> Scope note: This document surveys the research landscape that `sonic_ct` draws
> from and aims toward. External results are described as **directions reported
> in the literature**, not as exact, attributable numbers. Where a number appears
> it is from `sonic_ct`'s own measured simulator (clearly labelled), not a
> third-party claim. `sonic_ct` is simulation/research software and makes **no
> diagnostic claim**.

Ultrasound Computed Tomography (USCT) reconstructs maps of acoustic tissue
properties — speed of sound, attenuation, and (in advanced variants) density —
from many transmit/receive paths through the body. Unlike conventional B-mode
ultrasound, which forms reflectivity images from backscatter, USCT exploits the
*transmission* geometry of a surrounding transducer ring (or bowl) to solve an
inverse problem for quantitative material properties. The field spans a ladder
of physical fidelity, from straight-ray travel-time tomography up to full wave
physics. `sonic_ct` sits at the bottom rung today and is architected to climb.

## 1. Time-of-Flight (TOF) Transmission Tomography

The entry point — and what `sonic_ct` implements now. Each transmit/receive pair
yields a first-arrival travel time. Under a straight-ray (high-frequency,
geometric-acoustics) approximation, travel time is a line integral of *slowness*
(1/c) along the ray, giving a linear system `A s = t`, where `A_ij` is the length
ray `i` spends in cell `j`. `sonic_ct` solves this with **SART** (Simultaneous
Algebraic Reconstruction Technique) in `reconstruction.rs`; a single SART sweep
is equivalent to the classic delay-backprojection baseline, and additional sweeps
move toward the least-squares solution. Attenuation is recovered by swapping the
right-hand side for integrated amplitude loss.

**Strengths:** fast, robust, convex, dependency-free, and a faithful baseline for
soft-tissue *speed* contrast. **Limits:** the straight-ray assumption ignores
refraction and diffraction, so it blurs sharp, high-contrast boundaries. In
`sonic_ct` this shows up concretely: measured mean Dice with the tuned threshold
model is ~0.63 (up from ~0.30 with literature defaults), MAE ~28–31 m/s over
~8000 valid measurements, but **bone Dice ≈ 0** — straight rays bend strongly
around and refract through cortical bone, smearing it out. The literature is
unanimous that recovering hard, refractive structures requires wave-physics
methods. This is the documented motivation for the FWI roadmap.

## 2. Full-Waveform Inversion (FWI)

FWI is the SOTA for high-fidelity USCT. Rather than reducing each trace to one
travel time, it fits the *entire recorded waveform* by iteratively updating the
property model until simulated and observed wavefields match. Key machinery
reported across the geophysics and medical-USCT literature:

- **Adjoint-state gradients.** The gradient of the data misfit with respect to
  the model is computed by cross-correlating the forward wavefield with a
  back-propagated *adjoint* (residual) wavefield. This gives a full-model
  gradient at the cost of roughly two wave simulations per source — the
  enabling trick that makes FWI tractable at all.
- **Frequency continuation (multiscale).** Inversion proceeds from low to high
  frequencies. Low frequencies recover smooth, large-scale structure and are
  far less prone to local minima; higher frequencies then sharpen detail. This
  coarse-to-fine schedule is the standard FWI workflow.
- **Cycle-skipping.** FWI's central failure mode: if the starting model
  mispredicts a phase by more than half a wavelength, the optimizer locks onto
  the wrong cycle and converges to a wrong-but-plausible model. Low-frequency
  data, good starting models (e.g. a TOF result), and envelope/optimal-transport
  misfits are the reported mitigations. `sonic_ct`'s TOF output and the
  acoustic-memory warm-start mechanism are designed to provide exactly such
  starting models.
- **Source encoding.** Simultaneously firing many encoded sources and inverting
  the superposition dramatically cuts the number of forward simulations per
  iteration; the literature reports large compute reductions at the cost of
  managing crosstalk noise.
- **Brain/skull FWI** (e.g. work in the style of Guasch and colleagues). The
  skull is the hard case: strong speed contrast, attenuation, and aberration.
  Reported results indicate that 3-D acoustic FWI can reconstruct
  through-skull speed-of-sound maps of brain tissue from a surrounding array —
  a direction directly relevant to whole-body transmission scanning.
- **Musculoskeletal / vortex-encoded FWI.** More recent reported work applies
  FWI to limbs and musculoskeletal targets and uses *vortex* (orbital-angular-
  momentum-style) encoded illumination to reduce the compute burden of
  many-source acquisition while preserving reconstruction quality. The theme is
  the same: keep the wave physics, cut the simulation count.

For `sonic_ct`, FWI is the documented fix for the bone-Dice failure and the
gateway to quantitative density/elasticity. The architecture deliberately keeps
the forward operator (ray → wave) swappable behind the acquisition/physics
boundary so an FWI engine can replace SART without disturbing acquisition or AI
layers.

## 3. Sparse Acquisition + AI Reconstruction

Dense rings with thousands of paths are expensive and slow. A major MICCAI-era
direction (e.g. APS-USCT-style "adaptive/sparse" pipelines) trains neural
networks to reconstruct high-quality property maps from **deliberately
sparse** acquisitions — fewer elements, fewer angles, or fewer firings — by
learning the data prior that a sparse linear solver lacks. Reported approaches
combine learned sinogram/measurement completion with image-domain refinement.
The relevance to `sonic_ct`: a "learned sparse completion" stage (on the
refinement roadmap) would let a Butterfly-style ring with limited channel count
approximate the coverage of a dense research scanner. Critically, `sonic_ct`'s
governance stance keeps any such learned completion as an *enhancement on top of*
the physics reconstruction, with raw evidence preserved, rather than a black box
that replaces it.

## 4. Regularization with Structural Priors

Tomographic inversion is ill-posed; regularization injects prior knowledge.
Beyond generic Tikhonov/total-variation smoothing, the literature reports
**structural priors** that borrow edges from a co-registered modality — for
example using an EIT (electrical impedance tomography) or optical reconstruction
to guide where ultrasound speed boundaries should fall (joint/multimodal
inversion, structurally-guided TV). The shared theme is that one modality's
spatial structure constrains another's, sharpening boundaries without inventing
detail. `sonic_ct`'s anatomical **graph-coherence** check in `memory.rs` is a
lightweight cousin of this idea: it encodes anatomical rules (e.g. "bone must not
touch the water bath") and flags reconstructions that violate them, providing a
prior-as-validator today and a hook for prior-as-regularizer later.

## 5. Learned Segmentation + Uncertainty Quantification

Once properties are reconstructed, tissue must be classified, and clinical use
demands knowing *where the model is unsure*. The literature couples learned
segmentation with uncertainty quantification (Bayesian/ensemble/evidential
methods, calibrated confidence maps). `sonic_ct` keeps this stage intentionally
**transparent**: `segmentation.rs` is an auditable piecewise speed-band
classifier — every label is explained by a speed band — and every cell carries an
uncertainty derived from its margin to the nearest decision boundary. The bands
are *fitted* by reproducible coordinate ascent (`model.rs`) rather than hand-set,
which is why the tuned model roughly doubles mean Dice versus defaults. This is a
deliberate, honest floor: a glass-box classifier with real uncertainty, leaving
room to swap in a learned, calibrated segmenter once evidence justifies it — and
keeping the physics reconstruction strictly separate from the AI segmentation.

## How sonic_ct Maps to the Literature

| Capability area | sonic_ct today | Reported SOTA direction | Gap / roadmap |
|---|---|---|---|
| Forward physics | Straight-ray TOF + amplitude integral (`acquisition.rs`, `ray.rs`) | Finite-difference / pseudo-spectral wave propagation | Add FD wave kernel behind the physics boundary |
| Speed reconstruction | SART (1 sweep = delay backprojection) | Adjoint-state FWI with frequency continuation | Adjoint gradients; multiscale schedule |
| Hard-structure (bone/skull) | Bone Dice ≈ 0 (straight-ray blur) | Through-skull / MSK FWI (Guasch-style, vortex-encoded) | FWI is the documented fix |
| Acquisition efficiency | Dense ring, fan sweep, ~8000 meas. | Source/vortex encoding; AI sparse reconstruction | Source encoding + learned completion |
| Regularization | Anatomical graph-coherence validator | Multimodal/EIT-guided structural priors | Promote validator to regularizer |
| Segmentation | Glass-box speed-band + margin uncertainty (tuned Dice ~0.63) | Learned segmentation + calibrated UQ | Optional learned segmenter, kept separate from physics |
| Memory / longitudinal | NSW vector index, `.rvf` archive, FWI warm-start (`memory.rs`) | Population priors; warm-started inversion | Use warm-start to seed FWI, mitigate cycle-skipping |
| Volume | 2-D slices; vertical-sweep stub (`volume3d.rs`) | 3-D / 4-D acoustic FWI | 3-D solver + inter-slice regularization |

**Bottom line.** `sonic_ct` is an honest TOF/SART baseline with a transparent AI
layer and a memory substrate, deliberately structured so each rung of the
research ladder — wave physics, adjoint FWI, frequency continuation, source
encoding, learned sparse completion, 3-D — can be added without rewriting the
layers around it. The single most important documented gap is the leap from
straight-ray TOF to wave-based FWI, which is what unlocks hard-structure
fidelity that the current Dice metrics show is missing.
