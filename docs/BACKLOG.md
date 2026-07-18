# Hastur backlog / roadmap

The plan of record for porting SAM 3D Body to a cross-platform OFX plugin. Milestone-per-PR, reusing the humbaba
scaffold. See the repo README for the short version.

## Decisions (2026-07-18)

1. **0.1.0 = full parity**: multi-person + hands, all three OS (CoreML / CUDA / CPU). Build order de-risks internally
   (single-person body-only end-to-end first, then hands → multi-person → all platforms).
2. **MHR mesh = C++ FK + LBS (Eigen) from the start** — extracted static assets + the neural pose-corrective MLP as a
   small ONNX. A one-off traced mesh-ONNX is produced **only as a numerical oracle** in `tools/`, never shipped.
3. **Detector = clean ONNX swap** (RTMDet / YOLO / torchvision FRCNN); only person bounding boxes are consumed, so we
   drop the 2.4 GB export-hostile Detectron2 ViTDet-H.
4. **Model-less weights** — ship the converter/fetch tooling; the user pulls the HF-gated Meta weights with their own
   token and exports locally. Hastur redistributes no Meta weights. SAM no-reverse-engineering clause + embedded
   DINOv3 terms are a legal gate before the public flip (MHR + Detectron2 are Apache-2.0).

## Tensor contract

Frozen in [`src/MeshTypes.h`](../src/MeshTypes.h) — the synchronization surface for all tracks. Stage boundaries:
detector → boxes; crop → `(image[3,512,512], ray_cond[2,32,32], condition_info[3])`; body regressor →
`(pred[519], pred_cam[3], hand_logits[2,2], hand_box[2,4])`; MHR → `(verts[18439,3], joints[127,3], keypoints[70,3])`;
camera → `(focal, cam_t[3])`; rasterizer → `RGBA[H,W,4]`.

## Milestones

| M | Title | Type | Gate |
|---|-------|------|------|
| M0 | Bootstrap: scaffold from humbaba, freeze `MeshTypes.h` | impl | scaffold builds a passthrough bundle |
| M1 | Detector swap (clean ONNX) | EXPERIMENT | bbox IoU/recall vs ViTDet-H |
| M2 | MHR in C++ (asset extraction, FK+LBS, pose-corrective ONNX) | EXPERIMENT | verts RMSE vs oracle < ε |
| M3 | Backbone+decoder+heads ONNX export (body-only, fixed 512) | EXPERIMENT | `pred[519]`+`pred_cam` parity; CoreML partitions logged |
| M4 | C++ scaffold: OFX skeleton, OrtSessionManager, CropAffine, CameraSolver (**SYNC**) | impl | plugin loads in a host |
| M5 | Body-only end-to-end, single person, macOS/CoreML | impl+exp | grey-RGBA vs `render` (alpha IoU + shaded-L1) |
| M6 | Software rasterizer hardening | impl | rasterizer parity golden |
| M7 | Hands (lazy refiner, C++ wrist gate, flip/merge) | EXPERIMENT | hand-joint parity; gate correctness |
| M8 | Multi-person (bbox loop + depth-ordered composite) | impl | multi-person golden |
| M9 | Linux CUDA + Windows CUDA ports | impl | GPU smoke green all 3 OS |
| M10 | Packaging + model-less installer + release workflow + legal sign-off | impl | signed bundles; fetch verified |
| M11 | **0.1.0 public release** | impl | tagged release, host matrix, repo public |

## Multi-agent tracks (fan out after M0 freezes contracts)

**A** detector export (M1) · **B** backbone/decoder export (M3) · **C** MHR C++ LBS + asset extraction (M2) ·
**D** software rasterizer (M6, unit-testable vs Python-dumped meshes) · **E** OFX scaffold + CI/isolation/packaging
(M4/M9/M10) · **F** CameraSolver/CropAffine geometry ports. Sync points: (1) M0 freeze; (2) M4 conform engine
`Run()` signatures; (3) M5 first full-chain integration (ray_cond/condition_info ↔ backbone seam — highest risk);
(4) rasterizer parity before M8.

## Open risks

- Detector bbox shift → pose drift (M1): feed identical boxes to isolate the effect.
- CoreML silent CPU fallback on `GridSample`/`atan2`/scatter (M3): read partition logs, fix all shapes.
- MHR coordinate/unit fidelity in C++ (M2): validate per-vertex vs the traced oracle.
- ray_cond / CLIFF geometry port (M5): unit-test C++ vs Python within 1e-4 before integration.
- Rasterizer ≠ pyrender (180° X flip, cam_t convention, premult) (M6): match on dumped meshes.
- Hand gate / L-R flip-merge (M7): joint parity vs reference "full".
- Multi-session VRAM (M9): shared `Ort::Env` + CUDA `gpu_mem_limit` + shared arena + fp16 + lazy loads.
- Legal gate (M10/M11): SAM no-RE clause + embedded DINOv3 terms; model-less keeps weights out of the bundle.

## Prerequisites (before M2/M3)

- Complete the HF-gated download of `facebook/sam-3d-body-dinov3` `model.ckpt` (~1.7 GB bf16) with an HF token;
  `assets/mhr_model.pt` (664 MB) is already present in the local clone.
- Clean Python export env (the clone ships a conda `env/`); neutralize the local `mps`/`tokgan` edits for tracing.
- Local ground truth: `/Users/sam/Documents/github/sam-3d-body` (commit `c259bfc`).
