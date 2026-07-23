# AOVs — Cryptomatte, data passes & camera data

SAM 3D Body renders posed human meshes; every AOV below is geometry the rasterizer
already computes per fragment, surfaced for downstream comp and 3D reconstruction.
This is Phase 1: the **portable single-plane path** (any OFX host) plus camera-data
delivery. True multi-plane single-node output (Nuke/Natron) is a follow-up
(see *Delivery* below).

## The passes

| Output AOV | channels | space / units | notes |
|---|---|---|---|
| **Beauty** | RGBA | linear | neutral-grey clay mesh + coverage alpha (default) |
| **Depth (Z)** | Z→RGB, A=coverage | camera, metres, +Z fwd | metric depth of the frontmost surface |
| **Position (XYZ)** | XYZ, A=coverage | camera = world, metres | per-pixel surface position (camera at origin) |
| **Normal (XYZ)** | XYZ, A=coverage | camera, unit | oriented toward camera; for relighting |
| **Pref** | XYZ, A=coverage | canonical rest, **normalised [0,1]³** | pose/shape-invariant body coord for locator/texture pinning; positive & shot-invariant (normalised by the fixed canonical bbox) |
| **ST** | U,V→RG, A=coverage | texture UV, [0,1] | per-vertex UV, **always available**: a real `uv` asset block if present, else a deterministic **cylindrical unwrap** computed at load |
| **CryptoObject00** | RGBA | — | Cryptomatte ranks 0,1 = (id0,cov0,id1,cov1) |
| **CryptoObject01** | RGBA | — | Cryptomatte ranks 2,3 |

**Data passes are point-sampled, not anti-aliased.** Averaging depth / position /
normal across a silhouette edge would invent surfaces that don't exist, so each
pixel takes the value of the nearest covered subsample. Silhouette softness lives
in the Cryptomatte *coverage* channels, not in blended data. When comping data
AOVs, keep filtering nearest and matte with the coverage/alpha.

## Selecting a pass (portable path)

Set the **Output AOV** param. The inference result is cached per frame, so pulling
several passes is cheap: clone the node (or duplicate with a different Output AOV)
and each extra pass reuses the cached `FrameResult` — only the pixel write differs.

## Cryptomatte

Standard Psyop Cryptomatte (`MurmurHash3_32` name→float, `uint32_to_float32`
conversion). Per-person silhouettes are ranked front-to-back with occlusion, packed
two ranks per RGBA layer. IDs are **per-frame deterministic**: `person_00`,
`person_01`, … by depth order (nearest first). This is single-frame correct; on a
sequence the numbering can swap between frames — temporal ID locking is Phase 3.

**Cryptomatte needs multiple layers + metadata at once**, which shapes how each
host consumes it:

- **Natron / Fusion (multi-plane):** `CryptoObject00` and `01` arrive as layers on
  one node — feed the Cryptomatte node directly. ✅ Works.
- **Nuke:** the enum path cannot deliver this. Cryptomatte requires ≥2 RGBA levels
  *simultaneously* (each level = 2 ranks; AA/overlap needs 4+), plus the
  `cryptomatte/<key>/manifest|hash|conversion` **stream metadata** the gizmo keys
  off. A single RGBA output carries one level and OFX passes no metadata to Nuke.
  So Cryptomatte does **not** work from one Nuke node. Two real options:
  1. **Do the crypto isolation upstream in Natron/Fusion** (multi-plane), or
  2. **Assemble it in Nuke:** render each level via the enum path
     (`outputAov = CryptoObject00`, then `01`), Shuffle them into the
     `CryptoObject00/01` channel layout, and attach the manifest with a
     `ModifyMetaData` node writing the standard `cryptomatte/<key>/…` keys (the
     manifest string comes from the **Bake camera data** button — it is fully
     determined by *Max people*, no render needed). Then the Cryptomatte gizmo
     works. Fiddly; option 1 is cleaner.

The scalar data AOVs (Depth/Position/Normal/Pref/ST) have no such constraint —
they are self-contained single planes and work fine via the enum path in Nuke.

## Camera data (world↔NDC m44f)

Click **Bake camera data** to fill the read-only fields:

- **Camera intrinsics** — `focal cx cy W H`.
- **world→NDC (m44f)** — 16 row-major floats; camera/world → NDC in [-1,1]³
  (Y-up, Z maps [near,far]→[-1,1]).
- **NDC→world (m44f)** — its inverse. Unproject a pixel:
  `q = NDC→world · [ndc_x, ndc_y, ndc_z, 1]; P = q.xyz / q.w`.

These are computed from frame size + focal/FOV params (no inference), which is why
they're baked via a button rather than set during render (OFX forbids setting
params in `render()`). In Nuke, expression-link the 16 floats onto a `Camera`/`Axis`.

**Minimal 3D-reconstruction set:** *Depth + world↔NDC* reconstructs metric position
for every covered pixel; **Position** is the exact convenience pass (no reprojection
error); **Normal** is a required separate pass (not reliably recoverable from depth
derivatives); **Cryptomatte** adds per-person separation.

## What the passes enable

- **2.5D relighting** — `albedo · Σ lights·N`; point/spot falloff & shadows use P.
- **Per-person isolation** — Cryptomatte mattes for grades / relights / holdouts.
- **Deep holdouts (Phase 2)** — front/back-Z → `DeepFromImage` → `DeepHoldout`.
- **Locator/texture pinning & 2.5D remapping** — the normalised Pref is a stable
  positive body coordinate you can look up/compare across pose, frames and shots;
  pair it with **Normal** for a surface frame (→ a per-point 4×4 of position +
  rotation + scale, curve/Kalman-smoothed for a usable tracked locator). **ST**
  gives a 2D surface parametrisation for projecting/reading textures and remapping
  surface features. The cylindrical unwrap has a back seam and pole compression
  (it is not a seam-free atlas); a baked artist `uv` asset supersedes it when present.
- **ControlNet / video-diffusion conditioning** — Normal, Depth, ST (DensePose-like
  IUV) and per-person masks are the geometry-conditioning channels such models train
  on. Temporal stability (Phase 3) matters for the *video* case.

## Two delivery paths

1. **Multi-plane single node — Natron (and Natron-family hosts).** The plugin
   opts into the multi-plane extension, so Natron exposes all AOV planes
   (`Depth`, `Position`, `Normal`, `Pref`, `ST`, `CryptoObject00/01`) on **one
   node** — Shuffle/Cryptomatte read them directly. Implemented against the
   vendored `src/nuke/fnOfxExtensions.h` constants + the raw plane suite, with the
   action interceptor delegating to the C++ Support library (no SDK patch).
   Natron's GPL `ofxsMultiPlane` helper is deliberately **not** used (license
   incompatible with the SAM License).

   **Nuke does not do this.** Nuke's OFX host supports only a fixed, hard-coded
   plane set (Colour, Motion Vectors, Stereo Disparity) and never adopted the
   arbitrary-plane extension or the `getClipComponents` action — so custom AOV
   planes cannot be surfaced as extra layers on one Nuke node. This is a Nuke
   limitation, not a plugin bug (verified: Nuke 16.0v8 loads the plugin, renders
   the pipeline, but shows only `rgba`). In Nuke, use path 2.
2. **Portable single plane (every host, incl. Nuke & Flame).** The **Output AOV**
   enum routes one pass to the RGBA output; clone the node per pass (inference is
   cached per frame). This is the supported Nuke path and the universal fallback.

**Autodesk Flame** (validated for the base plugin, 2027/Linux): treat like Nuke —
its OFX host is not expected to expose arbitrary multi-plane layers, so use the
**Output AOV** enum for the scalar passes, and deliver **Cryptomatte via an EXR
round-trip** (Flame reads Cryptomatte EXRs natively). AOV validation in Flame is
still **to do** — the enum path is portable and should work; multi-plane will not.

> **Verification note:** the multi-plane path is **validated in Natron** (2.6,
> arm64/macOS): the plugin loads as `com.tokgan.Sam3dBody v0.3`, exposes all seven
> AOV planes with correct names/channels (`Depth[Z]`, `Position[XYZ]`,
> `Normal[XYZ]`, `Pref[XYZ]`, `ST[UV]`, `CryptoObject00/01[RGBA]`) alongside
> `Color[RGBA]`, every channel is individually selectable downstream (Shuffle),
> the full pipeline renders end-to-end (CoreML) to EXR, and requesting a specific
> AOV plane (Depth, routed via Shuffle) exercises the `clipGetImagePlane` render
> path and emits distinct, plane-specific pixels (idiff vs beauty: 100% differ).
> Values' metric correctness rests on the C++ unit tests + rasterizer, not yet an
> in-host pixel readout. **Nuke 16.0v8 (macOS) is validated for the portable
> path:** the plugin loads (`OFXcom.tokgan.Sam3dBody_v0`), renders the full
> pipeline, and the **Output AOV** enum yields distinct correct passes (Beauty /
> Depth / Normal differ over exactly the person's coverage area). Nuke does **not**
> expose the multi-plane layers (host limitation, see above) — use the enum path
> there. Run Nuke headless with `-i` (interactive login license); plain `-t`
> requests a render-only license the login token doesn't cover.

## Roadmap

- **Phase 2:** deep front/back-Z holdout planes (`DeepFromImage` → `DeepHoldout`).
- **Phase 3:** temporal per-person ID tracking + EXR-sequence dataset export.
