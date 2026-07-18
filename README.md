# Hastur — SAM 3D Body as an OpenFX plugin

A hardware-accelerated **OpenFX** plugin that runs Meta's **[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)**
human-mesh-recovery pipeline through **ONNX Runtime** — the **CoreML** execution provider on Apple Silicon and the
**CUDA** execution provider on Linux/Windows (NVIDIA), with automatic CPU fallback. It reconstructs posed 3D human
mesh(es) from a single frame and renders them **in neutral grey with a coverage alpha, at the input-frame resolution**.

> **Status: early development (M0 — bootstrap).** Private until the **0.1.0** release, then public. This is the
> SAM-3D-Body counterpart to [humbaba](https://github.com/samhodge-tokgan/humbaba) (which does the same for the
> DepthAnything3 depth model) and reuses its cross-platform ORT/OFX scaffold.

- **Input:** RGB(A) frame buffer (sRGB display-referred or ACEScg working space).
- **Output:** an **RGBA** render — neutral-grey shaded humanoid mesh(es) over a transparent (coverage-alpha)
  background, at the input resolution. This is the C++/ORT equivalent of the reference
  `Renderer.__call__(..., return_rgba=True)`.
- **Acceleration:** ONNX Runtime — CoreML EP (macOS), CUDA EP (Linux/Windows), CPU fallback everywhere.

## Pipeline

```
frame ─▶ person detector (clean ONNX; bboxes only)
      ─▶ per-person crop (TopdownAffine 512², ImageNet norm) + camera conditioning
      ─▶ SAM-3D-Body regressor  (DINOv3-H+ backbone ─▶ SAM decoder ─▶ MHR head + camera head)
      ─▶ (optional, wrist-gated) hand refiner decoder
      ─▶ MHR body model  (C++ FK + LBS + blendshapes ─▶ posed 18,439-vertex mesh)
      ─▶ software rasterizer  (perspective pinhole, neutral-grey Lambert, coverage alpha)
      ─▶ RGBA at input resolution (multi-person = depth-ordered composite)
```

The heavy ViT networks (backbone, decoder, heads, hand refiner) run in **ONNX Runtime**; the orchestration, the
MHR **forward-kinematics + linear-blend-skinning** mesh generation, and the **renderer** are native C++ so the whole
thing runs headless inside a host process on three platforms.

## Models (model-less install)

Hastur ships **no model weights**. Meta's SAM-3D-Body weights are gated on Hugging Face and covered by the custom
**SAM License**; you download them yourself (with your own HF token) and run the provided `tools/` scripts to export
the ONNX graphs and extract the MHR static assets into the plugin bundle's `Contents/Resources`. See
[`docs/MODELS.md`](docs/MODELS.md) (TODO) and `tools/`.

## Licensing

- **This plugin (code):** Apache-2.0 (see [`LICENSE`](LICENSE)).
- **MHR** (body model) and **Detectron2** (reference detector): Apache-2.0.
- **SAM 3D Body** weights: **SAM License** (Meta) — commercial use is permitted, but redistribution passes the
  license through and clause 1.b.iv restricts reverse-engineering; treat local ONNX conversion as a derivative work
  and obtain sign-off before shipping weights. **DINOv3** (embedded in the checkpoint) has its own gated terms.
  Because Hastur is **model-less**, it never redistributes Meta weights — the user generates them locally.

## Roadmap

Development is milestone-per-PR (see [`docs/BACKLOG.md`](docs/BACKLOG.md) for the full plan). Highlights:
M0 bootstrap → M1 detector swap → M2 MHR in C++ → M3 backbone/decoder export → M4 C++ scaffold → M5 body-only
end-to-end (macOS) → M6 rasterizer → M7 hands → M8 multi-person → M9 Linux/Windows CUDA → M10 packaging + legal →
**M11 = 0.1.0 public release** (multi-person + hands, all three OS).

## Building

Requires CMake ≥ 3.20 and a C++17 compiler; the OpenFX SDK and per-platform ONNX Runtime are fetched by CMake. Same
invocation on every platform (toolchain differs) — see [`docs/LINUX.md`](docs/LINUX.md) / [`docs/WINDOWS.md`](docs/WINDOWS.md).

```sh
cmake -S . -B build -DHASTUR_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Produces `build/Sam3dBody.ofx.bundle` (OFX layout `Contents/{MacOS,Linux-x86-64,Win64,Frameworks,Resources}`).
