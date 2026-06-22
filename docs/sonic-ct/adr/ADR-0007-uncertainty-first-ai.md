# ADR-0007: Uncertainty-First, Auditable Segmentation

- Status: Accepted
- Date: 2026-06-21
- Deciders: sonic_ct core team

## Context

Reconstructed speed maps are noisy and blurred (ADR-0004), so any tissue
labelling will be wrong in places — especially near class boundaries and around
small structures. For a research tool that must stay honest (ADR-0005), an
opaque high-accuracy network that hides *where* it is unsure is the wrong
trade-off. We want every label to be explainable and to carry a calibrated-ish
confidence, so downstream consumers can down-weight uncertain regions.

## Decision

Segmentation is a transparent, auditable **speed-band classifier**, not a neural
network (`segmentation.rs`). `SegModel` is an ordered list of
`(upper_speed_bound, Tissue)` bands plus a `margin_scale`; `classify` assigns
the first band a speed falls under, so every label is explained by one threshold.

Crucially, `segment` emits a **per-cell uncertainty** grid alongside labels:

```rust
uncertainty = exp(-margin / margin_scale)
```

where `margin` is the distance (m/s) from the cell's speed to the nearest finite
band boundary. A cell sitting on a decision boundary gets uncertainty ≈ 1; a
cell deep inside a band approaches 0. Thresholds are learned by coordinate
ascent (`model.rs::train`, exposed as `SegModel::tuned()`), which lifts mean
Dice from ~0.30 (literature-default thresholds) to ~0.63 on the synthetic
corpus. The WASM demo ships `tuned()` so the live output reflects the trained
model, and surfaces the uncertainty grid to the UI.

## Consequences

### Positive

- Every label is auditable: one band boundary explains it.
- Uncertainty is first-class and propagates to the WASM/UI layer, so unreliable
  regions (boundaries, small bone) are visibly flagged.
- Training is interpretable (coordinate ascent over thresholds), with a measured
  ~2× mean-Dice improvement.

### Negative / Trade-offs

- A 1-D speed-band model cannot use spatial context, so it cannot recover
  structures the reconstruction has already blurred (bone Dice ~0).
- `exp(-margin/scale)` is a heuristic confidence, not a calibrated probability.
- Accuracy ceiling is bounded by reconstruction quality, not the classifier.

## Alternatives Considered

- **Opaque CNN segmenter**: potentially higher Dice, but unexplainable and
  without honest per-cell uncertainty — contrary to ADR-0005.
- **Hard labels with no uncertainty**: simpler, but hides exactly the
  near-boundary errors that matter most for an honest research tool.

## References (to the real code)

- `crates/sonic-ct/src/segmentation.rs` (`SegModel`, `classify`,
  `boundary_margin`, `segment`, `Segmentation::uncertainty`)
- `crates/sonic-ct/src/model.rs` (`train` coordinate ascent, `evaluate`)
- `crates/sonic-ct/src/segmentation.rs` (`SegModel::default` vs `tuned`)
- `crates/sonic-ct-wasm/src/lib.rs` (`uncertainty` buffer exported to JS)
