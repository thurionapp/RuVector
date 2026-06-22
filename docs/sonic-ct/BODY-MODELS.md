# Body models for the Sonic Chamber ghost

The visual ghost body is a **visual/anatomical prior only**. The Rust procedural
phantom (`crates/sonic-ct`) remains the single source of physics truth for
acoustic classes, reconstruction, confidence, and Dice scoring. A supplied GLB
never feeds the reconstruction.

## Architecture

```
GLB visual anatomy ─▶ React Three Fiber display layer (BodyModel.jsx)
procedural phantom ─▶ Rust reconstruction + metrics (truth)
organ hypotheses   ─▶ alignment between GLB organ meshes and phantom regions
```

`examples/sonic-ct/src/scene/BodyModel.jsx` loads `GLB_URL` (under `/public`,
e.g. `/models/torso.glb`) and falls back to a procedural ghost when it is empty
or the asset fails to load (wrapped in an error boundary). `applyGhostMaterials`
overrides every mesh with a translucent ghost material and tints named organ
meshes via `src/organ_manifest.json`.

To use a real model: drop `torso.glb` in `examples/sonic-ct/public/models/`,
set `GLB_URL = "/models/torso.glb"`, and update the organ name map in the
manifest.

## Shipped default

The app ships with **CesiumMan** (`public/models/human.glb`) from the
[Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets),
licensed **CC-BY 4.0** (Cesium). It is loaded as the translucent outer-body
ghost shell and styled at runtime via `applyGhostMaterials`; the procedural
internal organ glows are rendered inside it. Replace this file (and the manifest
organ name map) with a higher-fidelity licensed model — e.g. Zygote organs —
for production fidelity. Attribution must be preserved per CC-BY.

## Sourced model options (researched)

| Need | Service | License notes |
|---|---|---|
| Accurate visible anatomy | **Zygote** | Medically precise commercial anatomy library; per-asset commercial license. Best fidelity for organs/ribs/spine/vessels. |
| Web-embedded anatomy | **BioDigital** Human Viewer API | Embedded interactive anatomy; API/licensing rather than owning meshes. |
| Parametric outer body | **SMPL / Meshcapade** | Realistic body shape/pose from scans; exports GLB/FBX. Good for the outer ghost shell. |
| Open research anatomy | **Z-Anatomy** (from BodyParts3D) | CC BY-SA 4.0; Blender file, retopologised, exportable to glTF. Good for non-commercial research prototypes — preserve attribution + ShareAlike. |
| Open mesh dataset | **BodyParts3D / Anatomography** | CC BY-SA 2.1 Japan; 382 segmented body-part meshes. |
| Concept props (not anatomy) | Meshy / Tripo / Rodin | Chambers, housings, UI props only — never the anatomy source of truth. |

Recommended stack: **Meshcapade/SMPL** outer shell + **Zygote** internal organs
for production fidelity; **Z-Anatomy** (CC BY-SA) for an open demo. Keep the
"research only — not diagnostic" banner visible regardless of model.

### GLB pipeline
1. Acquire licensed/open anatomy GLB or FBX.
2. Clean + decimate in Blender; split organs into named meshes.
3. Export GLB with Draco compression.
4. Update `organ_manifest.json` name map.
5. `useGLTF` loads it; `applyGhostMaterials` styles it at runtime.
6. Align to the scanner coordinate system; map organ meshes to phantom regions.

Sources: [Z-Anatomy](https://github.com/Z-Anatomy) · [BodyParts3D](https://github.com/Kevin-Mattheus-Moerman/BodyParts3D) · [Zygote](https://www.zygote.com/) · [BioDigital](https://www.biodigital.com/) · [Meshcapade/SMPL](https://meshcapade.com/).
