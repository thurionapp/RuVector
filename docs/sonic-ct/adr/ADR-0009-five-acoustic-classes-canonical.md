# ADR-0009: Five Acoustic Classes as the Canonical Reconstruction Output

- Status: Accepted
- Date: 2026-06-22
- Deciders: MetaBioHacker core team

## Context

The reconstruction pipeline recovers a speed-of-sound field and segments it into
tissue labels (ADR-0007). It is tempting to relabel those segments with organ
names — "liver", "kidney" — because the output looks anatomical. That would be
physically unsound. Speed of sound is a bulk acoustic property; soft organs sit
in a narrow band and overlap heavily. Liver and spleen, for example, have nearly
identical speeds, so no threshold over a speed value can separate them. Asserting
organ identity from speed alone would manufacture confidence the physics does not
support, contradicting the honesty stance of ADR-0005.

## Decision

The canonical reconstruction output is **five acoustic classes defined purely by
speed of sound**: water, fat, muscle, organ (soft tissue), and bone. This is
fixed by the `Tissue` enum in `crates/sonic-ct/src/types.rs` (`Water=0`, `Fat=1`,
`Muscle=2`, `Organ=3`, `Bone=4`, `Tissue::COUNT == 5`), and the segmenter assigns
these and only these via the speed bands in `SegModel` (`segmentation.rs`).

`Tissue::Organ` is the soft-tissue parenchyma class — its doc comment notes "e.g.
liver, kidney" as examples only, never as an assigned identity. The HUD reinforces
this: `CLASS_LABELS` in `Hud.jsx` renders the class as "Soft tissue", and the
acoustic-class legend note states organ identity is inferred separately. No code
path maps a speed value to an organ name. Organ identity, where attempted, is a
strictly separate inference layer (ADR-0010).

**Acceptance:** no organ label is ever derived directly from a speed value; the
reconstruction emits exactly the five `Tissue` classes.

## Consequences

### Positive

- Output stays faithful to what speed of sound can actually distinguish.
- The class set is small, stable, and serializable as a `u8` wire value shared
  with the WASM/UI layer.
- Cleanly separates the physically-grounded layer from the inferential one.

### Negative / Trade-offs

- Users wanting "organ maps" do not get them from the speed map alone; identity
  requires the separate, prior-driven layer with explicit uncertainty.
- A single "Organ" class lumps all soft parenchyma together, so the speed map
  cannot visually distinguish adjacent organs.

## Alternatives Considered

- **Direct speed→organ labels**: rejected — overlapping speeds make liver vs.
  spleen unrecoverable; fabricates unsupported certainty.
- **More speed sub-classes for organs**: rejected — finer thresholds do not exist
  in the underlying physics; would be arbitrary and misleading.

## References (to the real code)

- `crates/sonic-ct/src/types.rs` (`Tissue` enum, `COUNT`, `nominal_speed`)
- `crates/sonic-ct/src/segmentation.rs` (`SegModel` speed bands, `classify`)
- `examples/sonic-ct/src/hud/Hud.jsx` (`CLASS_LABELS`, acoustic-class legend note)
