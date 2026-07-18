# M8 Multi-Person Report

**Milestone M8** — verify and harden multi-person handling in
`src/Sam3dBodyPipeline.cpp`: N detected people, each posed, composited into one
RGBA in correct depth order (nearer occludes farther where they overlap).

## Result

**Multi-person renders correctly. No pipeline fix was required** — the existing
depth-ordered over-composite in `Sam3dBodyPipeline::Run` is correct on a real
multi-person photo. The only changes were to the offline harness
(`tools/pipeline_smoke.cpp`), never to the compositing logic.

## Image used

COCO val2017 `000000004134.jpg` (640×425) — two men shaking hands in the
foreground (their arms/hands **overlap** in the center of the frame), plus two
smaller people standing in the background. This is a deliberate depth-order test:
the overlap between the two foreground figures exercises near-occludes-far, and
the background figures exercise foreground-occludes-background.

Staged at `build/multiperson_src.jpg`. Detector: `person_detector.onnx`
(ssdlite320), score threshold **0.40**, `max_people = 4`.

Outputs:
- `build/multiperson.png` — grey mesh RGBA (straight alpha) at frame resolution.
- `build/multiperson_over.png` — the same mesh composited over the source photo
  for a visual depth-order check.

## People detected: 4 (all posed + meshed, 18439 verts each)

Depth order is by `cam.cam_t[2]` (metric z, camera at origin, +z forward → larger
z = farther). Painted far→near, so the nearest silhouette lands on top:

| paint order | person | box [x0,y0,x1,y1] | score | cam_t.z (m) | hands |
|:-----------:|:------:|:------------------|:-----:|:-----------:|:-----:|
| 1 (farthest)| 3 | 234,134,300,263 | 0.420 | **11.011** | no |
| 2           | 2 | 7,110,81,260    | 0.499 | **6.745**  | no |
| 3           | 0 | 231,58,581,425  | 0.998 | **1.776**  | yes |
| 4 (nearest) | 1 | 37,44,283,425   | 0.996 | **1.705**  | yes |

- Persons 0 (right man) and 1 (left man) are the two foreground figures; person 1
  is fractionally nearer (1.705 < 1.776 m), so at the handshake overlap the left
  man's forearm/hand correctly occludes the right man's.
- Persons 2 and 3 are the two smaller background figures (larger z), rendered
  smaller by the perspective solve and correctly drawn *behind* the foreground
  men — they show through only in the gaps between the foreground silhouettes.
- Composited alpha coverage 44.4% of the frame; silhouette bbox [14,53,577,424].

## How overlaps composite

Each person is rasterized independently to a full-frame premultiplied RGBA (its
own per-triangle z-buffer resolves that person's *self*-occlusion). The frame is
then assembled as a **depth-ordered 2.5D layer composite**:

1. `order` = person indices sorted by `cam_t[2]` **descending** (far first).
2. For each person far→near: `Render(...)` (premultiplied) then a standard
   premultiplied Porter-Duff **"over"** onto the accumulator
   (`OverComposite`: `acc = fg + acc·(1−fg.a)`).
3. Because layers are painted far→near, the nearer person's silhouette is applied
   last and occludes the farther person wherever they overlap. Alpha is composited
   in premultiplied space, so overlaps are **not** double-counted (the accumulated
   alpha is a proper union coverage, ≤ 1).
4. When the caller wants straight alpha (`premultiply == false`), the accumulator
   is un-premultiplied once at the end.

This is a per-person layer composite, not a single fused 3D z-buffer, so at an
overlap the nearer person occludes the farther one by whole silhouette (the
intended behavior for this deliverable). Inter-penetrating meshes at the exact
same depth would not be z-merged per-pixel, but that is out of scope here.

## Timing (CPU / macOS CoreML fallback, `HASTUR_PIPELINE_TIMING=1`)

| stage | person 0 | person 1 | person 2 | person 3 |
|:------|---------:|---------:|---------:|---------:|
| body-regressor | 20.8 s | 23.3 s | 24.3 s | 23.5 s |
| hand-refine    | 56.6 s | 50.5 s | 0.01 s | 0.01 s |
| pose-corrective| 0.48 s | 0.49 s | 0.41 s | 0.55 s |
| mhr-mesh       | 0.05 s | 0.03 s | 0.03 s | 0.03 s |

detector 0.36 s · raster+composite (all 4) 0.18 s · **total ≈ 3.4 min** for 4
people (the two foreground people also run the M7 hand refiner, ~50–57 s each).
This is CPU/CoreML-fallback cost as expected; CUDA accelerates it.

## Single-person regression

Re-ran the seated-yoga smoke with defaults (`max_people = 1`,
`build/regression_single.png`): output is **byte-identical (same MD5,
`041957bf…`) to the committed baseline `build/smoke_out.png`** — single-person +
hands behavior is unchanged.

## Harness changes (tools only — pipeline untouched)

`tools/pipeline_smoke.cpp`:
- Optional env overrides `HASTUR_SMOKE_MAXPEOPLE` and `HASTUR_SMOKE_SCORE` to
  drive `max_people` / detector score threshold without recompiling. Defaults are
  unchanged (max_people 1, score 0.4), so single-person runs are identical.
- Also writes an `<out>_over.png` RGB composite (grey mesh over the source frame)
  for the visual depth-order check.

`src/Sam3dBodyPipeline.cpp` and all other engine sources: **unchanged.** Full
build green (`cmake --build build -j`); ORT isolation intact (the `.ofx` links
only `@rpath/libonnxruntime_hastur.1.dylib`).
