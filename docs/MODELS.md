# Models — model-less install & local generation

Hastur ships **no model weights**. Meta's SAM-3D-Body checkpoints are gated on Hugging Face and covered by the custom
**SAM License** (with an embedded **DINOv3** under its own gated terms); the ONNX graphs and MHR assets Hastur runs are
**derived** from those weights, which we must not redistribute. So instead of *downloading* a model set, you
**generate** one locally from your own licensed copy of the checkpoints, with a single wire-up script.

---

## 1. Prerequisites

- **Accept the licenses and download the checkpoints yourself** (your own HF token): the SAM-3D-Body DINOv3 checkpoint
  directory, containing:
  ```
  <sam-3d-body>/checkpoints/sam-3d-body-dinov3/model.ckpt              (~2.0 GB)
  <sam-3d-body>/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt     (~0.7 GB)
  ```
- **A Python env** with `torch`, `onnx`, `onnxruntime`, `torchvision`, `roma`, and the **`sam-3d-body` package
  importable** (the reference checkout provides it). Pass it via `--python`.
- **The Hastur `tools/` directory** (this repo). The export scripts live there; the installer-injected
  `convert_models.sh` locates them via `$HASTUR_TOOLS_DIR` (or `--tools-dir`) when run from inside a bundle.
- **~7 GB free disk and patience.** The body + hand ViT exports each trace a DINOv3-H+ backbone and fold to fp16 —
  **multiple hours on CPU**, with several GB of scratch. The detector / MHR-asset / pose-corrective steps are quick.

---

## 2. Install flow

1. **Install the plugin.** Run the platform installer — on macOS, `Sam3dBody-<ver>-macos-arm64.pkg` (built by
   [`packaging/make_pkg.sh`](../packaging/make_pkg.sh); installs into `~/Library/OFX/Plugins`, no admin) — or
   `cmake --install build`. This stages the plugin only; **no weights**. The macOS pkg also injects
   `convert_models.sh` into the bundle's `Contents/`.

2. **Generate the model set** from your gated checkpoints:

   ```sh
   tools/convert_models.sh \
       --ckpt-dir  /path/to/sam-3d-body/checkpoints/sam-3d-body-dinov3 \
       --out-dir   ~/Library/OFX/Plugins/Sam3dBody.ofx.bundle/Contents/Resources \
       --python    /path/to/torch-env/bin/python
   ```

   Windows:

   ```powershell
   .\tools\convert_models.ps1 `
       -CkptDir C:\path\sam-3d-body\checkpoints\sam-3d-body-dinov3 `
       -OutDir  "$env:CommonProgramFiles\OFX\Plugins\Sam3dBody.ofx.bundle\Contents\Resources" `
       -Python  C:\path\torch-env\Scripts\python.exe
   ```

   Running the injected copy from an installed bundle (tools not alongside it):

   ```sh
   HASTUR_TOOLS_DIR=/path/to/Hastur/tools \
     ~/Library/OFX/Plugins/Sam3dBody.ofx.bundle/Contents/convert_models.sh \
       --ckpt-dir /path/to/checkpoints/sam-3d-body-dinov3 \
       --python   /path/to/torch-env/bin/python
   # --out-dir defaults to the sibling Contents/Resources.
   ```

   `convert_models` is **idempotent** — an existing target file is skipped (`--force` regenerates). Flags:
   `--skip-hand` (body-only; skips the second multi-hour export), `--pose-fp16` (halves `pose_corrective.onnx`; IO
   stays fp32 so the C++ hook is unchanged), `--sam3d-root` (override the `sam_3d_body` package / checkpoints base).

3. **The plugin finds the models** by scanning, in order: the *Model directory* plugin param → `$HASTUR_MODEL_DIR`
   (colon/`;`-separated) → the bundle's `Contents/Resources`. If you wrote to a custom `--out-dir`, point the plugin
   at it with `HASTUR_MODEL_DIR`.

---

## 3. What gets generated

`convert_models` runs these tools **in order** to populate `--out-dir`:

| # | File (+ sidecar) | Tool & invocation | Approx size | Gated ckpt? |
|---|---|---|---|---|
| 1 | `person_detector.onnx` + `.json` | `export_detector.py --model ssdlite --out …` | ~14 MB | no (torchvision, BSD) |
| 2 | `mhr_assets.bin` + `.manifest.json` | `extract_mhr_assets.py --out … --manifest …` | ~35 MB | yes |
| 3 | `pose_corrective.onnx` | `export_pose_corrective.py --out …` (`--fp16` optional) | ~670 MB fp32 / ~335 MB fp16 | yes |
| 4 | `sam3dbody_body.onnx` | `export_sam3dbody.py --mode faithful --out …` | ~1.8 GB fp16 | yes — **hours** |
| 5 | `sam3dbody_hand.onnx` + `.json`, `mhr_wrist.bin` | `export_hand.py --out …` | ~1.8 GB fp16, 72 B | yes — **hours** |

Files 1–4 are **required** (body-only pipeline). File 5 is **optional**: without `sam3dbody_hand.onnx` the pipeline
runs body-only, and without `mhr_wrist.bin` the hand-merge falls back to the box-size gate only. The `SAM3D_ROOT` env
that the tools use to resolve the checkpoints and import `sam_3d_body` is derived from `--ckpt-dir` (its grandparent)
or set explicitly via `--sam3d-root`. See [`docs/MHR_ASSETS.md`](MHR_ASSETS.md) for the field-by-field provenance of
`mhr_assets.bin`, and each `tools/*_REPORT.md` for the per-stage export/parity studies.

### Optional mirror

If you host a **pre-generated** set on a mirror **you** control (e.g. an internal artifact store you populated with
`convert_models` output), `tools/fetch_models.{sh,ps1}` can pull it into `Contents/Resources`:

```sh
HASTUR_MODELS_BASE_URL=https://your-mirror/path tools/fetch_models.sh
```

With no `HASTUR_MODELS_BASE_URL` set, `fetch_models` just prints the generate-locally guidance — there is **no
Hastur-hosted release** of these derived weights. An optional per-file `<name>.sha256` sidecar on the mirror, if
present, is verified.

---

## 4. Legal posture

- **Hastur code:** Apache-2.0.
- **MHR** (body model) and the **detector** (torchvision `ssdlite`, BSD-3-Clause): permissively licensed.
- **SAM-3D-Body weights:** the **SAM License** (Meta). Commercial use is permitted, but redistribution passes the
  license through and its **no-reverse-engineering** clause applies; treat the local ONNX conversion as a derivative
  work of weights **you** are licensed for. The embedded **DINOv3** carries its own gated terms.
- Because Hastur is **model-less**, it never redistributes Meta weights or anything derived from them. **You** own the
  gated checkpoints and the local conversion; the generated ONNX/asset files stay on your machine (or a mirror you
  control). Do not redistribute them.
