<!-- Copyright the Hastur authors. SPDX-License-Identifier: LicenseRef-SAM-License -->
# SAM 3 tracker ONNX export — prior-effort reference (imported AS-IS)

> **Status: raw scratch, checked in verbatim for reference.** These scripts are the
> exact working set from the skylab export box (`skylab:/root/sam3_export/`) that produced
> the SAM 3 tracker's `G1.onnx`–`G5.onnx` graphs the C++ tracker engine loads. They are
> **not** yet tidied to this repo's export-script conventions (no per-file SPDX headers,
> hardcoded skylab paths, `probe*/debug*/reverse_check` scratch left in). Committed so
> concurrent work can build on the previous effort rather than redo it. Productizing this
> into a single clean `export_sam3_tracker.py` (à la `tools/export_detector.py`) is a
> follow-up.

## What this exports

The SAM 3 video **tracker** is split into five per-frame ONNX graphs with **host-owned
recurrent memory** (the C++ `Sam3TrackerEngine` / `Sam3MultiTracker` drive them; see
`src/Sam3TrackerEngine.cpp`, which loads `G1.onnx`…`G5.onnx`):

- **G1** — shared ViT-L/14 image backbone → tracker FPN feats + position embeds
  (real-valued RoPE swapped in for ONNX-clean complex RoPE; G1.onnx is ~1.8 GB).
- **G2** — person detector seed (frame-0 mask prompt).
- **G3** — memory-attention read.
- **G4** — mask decoder (emits `pred_masks_high_res`, iou, obj-ptr, object-score-logits).
- **G5** — memory encoder (fuses pixel feats + predicted mask → new memory frame).

Plus `const_*.npy` sidecars (baked language-feature / temporal-pos / no-mem constants the
engine also loads). **The `.onnx`, `.onnx.data`, `golden.pt`, and `const_*.npy` binaries are
NOT in this repo** — they are gitignored and far too large (G1 alone ~1.8 GB); they live on
skylab next to these scripts. This directory is the *code that regenerates them*.

## Files

Top level — the export pipeline:
- `common.py` — builds the fp32 SAM 3 model, frame loader, shared paths (`EXPORT`, `DEV`).
- `graphs.py` — the five `nn.Module` wrappers (G1Backbone, …) that get traced.
- `capture.py` — runs the golden fp32 tracker on the soccer clip, captures per-graph example
  IO + golden per-frame masks (writes `golden.pt`).
- `export.py`, `export_g1.py`, `export_g2.py` — the actual `torch.onnx.export` calls
  (legacy-TS then dynamo fallback; G1 and G2 needed their own scripts).
- `validate.py`, `validate_full.py`, `final_checks.py`, `reverse_check.py`,
  `inspect_io.py`, `probe1/2/3.py`, `debug_g2.py` — parity / IO-shape / sanity probes.

`engine/` — the **Python reference tracker** the C++ engine was validated bit-close against:
- `engine_ref.py` — single-object forward-only recurrent tracker (1:1 mirror of the C++
  `Sam3TrackerEngine`; documents the config: num_maskmem=7, mem_dim=64, hidden_dim=256,
  image_size=1008, backbone_stride=14, …).
- `multi_ref.py`, `bidir_ref.py`, `split_bidir.py` — multi-object + bidirectional variants
  (mirror `Sam3MultiTracker` / the bidirectional prepass in `Sam3TrackerPlugin.cpp`).
- `prep*.py`, `compare_*.py`, `parity*.py`, `dump_lang.py` — driver / golden-compare harness.

## Provenance / license

Imported from `skylab:/root/sam3_export/` (mtime 2026-07-26). Derived from the SAM 3 model,
so it carries the repo's `LicenseRef-SAM-License` like the other exporters
(`tools/export_detector.py` et al.). Runtime deps: torch + the SAM 3 checkpoint + onnxruntime
(GPU) — none captured here; see skylab.
