#!/usr/bin/env bash
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# convert_models.sh — generate the Sam3dBody plugin's model set LOCALLY from your
# own copy of Meta's gated SAM-3D-Body checkpoints.
#
# Hastur is MODEL-LESS by design: it ships NO Meta weights. Instead you (the user)
# download the gated SAM-3D-Body checkpoints yourself and run this one script to
# EXPORT/EXTRACT the plugin's runtime assets into a target model directory (by
# default the installed bundle's Contents/Resources, which the plugin scans first).
#
# It runs the tools/ export scripts IN ORDER to populate --out-dir with:
#     person_detector.onnx (+ .json)   person / bbox detector      (fast, no ckpt)
#     mhr_assets.bin        (+ manifest) MHR static buffers (C++ FK+LBS)
#     pose_corrective.onnx             MHR pose-corrective MLP
#     sam3dbody_body.onnx              SAM-3D-Body ViT regressor   (SLOW, hours)
#     sam3dbody_hand.onnx (+ .json)    hand-decoder 2nd pass       (SLOW, hours)
#     mhr_wrist.bin                    wrist-twist rest rotations (from export_hand)
#
# It is IDEMPOTENT: a target file that already exists is skipped (use --force to
# regenerate). This REPLACES the download role of fetch_models.sh for Hastur.
#
# ─── PREREQUISITES ──────────────────────────────────────────────────────────
#   * You have accepted Meta's SAM License + the DINOv3 gated terms on Hugging
#     Face and downloaded the checkpoints (model.ckpt + assets/mhr_model.pt) with
#     your OWN HF token. Hastur never redistributes these.
#   * A Python env with torch / onnx / onnxruntime and the sam-3d-body package
#     importable (the reference checkout provides it). Pass it via --python.
#   * ~7 GB of free disk and PATIENCE: the body + hand ViT exports each trace a
#     DINOv3-H+ backbone and fold to fp16 — expect MULTIPLE HOURS and several GB
#     of scratch on CPU. The detector / MHR-asset / pose-corrective steps are quick.
#
# Usage:
#   tools/convert_models.sh \
#       --ckpt-dir  /path/to/sam-3d-body/checkpoints/sam-3d-body-dinov3 \
#       --out-dir   ~/Library/OFX/Plugins/Sam3dBody.ofx.bundle/Contents/Resources \
#       --python    /path/to/torch-env/bin/python
#
#   # From a Hastur source checkout, --out-dir + --python + --ckpt-dir all default
#   # to the reference layout, so a bare `tools/convert_models.sh` just works there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── defaults (reference dev layout) ────────────────────────────────────────
DEFAULT_SAM3D_ROOT="/Users/sam/Documents/github/sam-3d-body"
CKPT_DIR=""
SAM3D_ROOT="${SAM3D_ROOT:-}"       # sam-3d-body checkout (python pkg + checkpoints base)
OUT_DIR=""
PYTHON=""
FORCE=0
POSE_FP16=0
SKIP_HAND=0

usage() { sed -n '2,60p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dir)    CKPT_DIR="$2"; shift 2;;
    --sam3d-root)  SAM3D_ROOT="$2"; shift 2;;
    --out-dir)     OUT_DIR="$2"; shift 2;;
    --python)      PYTHON="$2"; shift 2;;
    --tools-dir)   SCRIPT_DIR="$2"; shift 2;;
    --force)       FORCE=1; shift;;
    --pose-fp16)   POSE_FP16=1; shift;;
    --skip-hand)   SKIP_HAND=1; shift;;
    -h|--help)     usage 0;;
    *) echo "unknown arg: $1" >&2; usage 2;;
  esac
done

# ─── locate the export tools (export_*.py + mhr_*.py) ───────────────────────
# In a Hastur source checkout they sit next to this script. If this script was
# injected into an installed bundle's Contents/ (see packaging/make_pkg.sh), the
# python tools are NOT alongside it — point at them with $HASTUR_TOOLS_DIR or
# --tools-dir (a Hastur checkout's tools/ dir).
if [[ ! -f "$SCRIPT_DIR/export_detector.py" && -n "${HASTUR_TOOLS_DIR:-}" ]]; then
  SCRIPT_DIR="$HASTUR_TOOLS_DIR"
fi
TOOLS="$SCRIPT_DIR"
if [[ ! -f "$TOOLS/export_detector.py" ]]; then
  echo "error: cannot find the Hastur export tools (export_detector.py …) in:" >&2
  echo "         $TOOLS" >&2
  echo "       Run this from a Hastur source checkout's tools/ dir, or set" >&2
  echo "       HASTUR_TOOLS_DIR=/path/to/Hastur/tools (or pass --tools-dir)." >&2
  exit 1
fi

# ─── resolve the checkpoints + the sam-3d-body root ─────────────────────────
# The export tools resolve the weights from $SAM3D_ROOT/checkpoints/sam-3d-body-
# dinov3/{model.ckpt,assets/mhr_model.pt} and import the sam_3d_body package from
# $SAM3D_ROOT. --ckpt-dir is that checkpoints dir; SAM3D_ROOT is its grandparent
# (the sam-3d-body checkout) unless overridden.
if [[ -z "$SAM3D_ROOT" ]]; then
  if [[ -n "$CKPT_DIR" ]]; then
    SAM3D_ROOT="$(cd "$CKPT_DIR/../.." && pwd)"
  else
    SAM3D_ROOT="$DEFAULT_SAM3D_ROOT"
  fi
fi
[[ -n "$CKPT_DIR" ]] || CKPT_DIR="$SAM3D_ROOT/checkpoints/sam-3d-body-dinov3"

TOOL_CKPT="$SAM3D_ROOT/checkpoints/sam-3d-body-dinov3/model.ckpt"
TOOL_MHR="$SAM3D_ROOT/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

if [[ ! -d "$SAM3D_ROOT/sam_3d_body" ]]; then
  echo "error: sam_3d_body python package not found under SAM3D_ROOT:" >&2
  echo "         $SAM3D_ROOT" >&2
  echo "       Point --sam3d-root at your sam-3d-body checkout." >&2
  exit 1
fi
if [[ ! -f "$TOOL_CKPT" || ! -f "$TOOL_MHR" ]]; then
  echo "error: gated checkpoints not found where the export tools resolve them:" >&2
  echo "         $TOOL_CKPT" >&2
  echo "         $TOOL_MHR" >&2
  echo "       The tools require the standard layout" >&2
  echo "         <sam3d-root>/checkpoints/sam-3d-body-dinov3/{model.ckpt,assets/mhr_model.pt}" >&2
  echo "       Download the gated SAM-3D-Body weights into that path (your HF token)." >&2
  exit 1
fi

# ─── locate the python interpreter ──────────────────────────────────────────
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$SAM3D_ROOT/env/bin/python" ]]; then PYTHON="$SAM3D_ROOT/env/bin/python"
  else PYTHON="$(command -v python3 || command -v python || true)"; fi
fi
[[ -n "$PYTHON" && -x "$(command -v "$PYTHON" 2>/dev/null || echo "$PYTHON")" ]] || {
  echo "error: no usable python. Pass --python /path/to/torch-env/bin/python" >&2; exit 1; }

# ─── resolve the output (model) directory ───────────────────────────────────
# Default: if this script sits inside an installed bundle's Contents/, write to the
# sibling Contents/Resources; else auto-detect an installed Sam3dBody.ofx.bundle.
find_resources() {
  local here_parent; here_parent="$(basename "$SCRIPT_DIR")"
  if [[ "$here_parent" == "Contents" && "$(dirname "$SCRIPT_DIR")" == *.ofx.bundle ]]; then
    echo "$SCRIPT_DIR/Resources"; return 0
  fi
  local d
  for d in "$HOME/Library/OFX/Plugins" "/Library/OFX/Plugins" \
           "$HOME/OFX/Plugins" "/usr/OFX/Plugins" "/usr/local/OFX/Plugins"; do
    [[ -d "$d/Sam3dBody.ofx.bundle" ]] && { echo "$d/Sam3dBody.ofx.bundle/Contents/Resources"; return 0; }
  done
  return 1
}
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(find_resources)" || {
    echo "error: no --out-dir and could not find an installed Sam3dBody.ofx.bundle." >&2
    echo "       Pass --out-dir <bundle>/Contents/Resources (or any model dir; then" >&2
    echo "       point the plugin at it via \$HASTUR_MODEL_DIR or the model-dir param)." >&2
    exit 1
  }
fi
mkdir -p "$OUT_DIR"
[[ -w "$OUT_DIR" ]] || { echo "error: '$OUT_DIR' is not writable (try sudo)." >&2; exit 1; }

export SAM3D_ROOT

# ─── banner ─────────────────────────────────────────────────────────────────
cat <<EOF
== Hastur model generation (model-less; user-generated from gated weights) ==
  python      : $PYTHON
  tools       : $TOOLS
  sam3d-root  : $SAM3D_ROOT
  checkpoints : $TOOL_CKPT
                $TOOL_MHR
  out-dir     : $OUT_DIR
  force       : $FORCE   pose-fp16: $POSE_FP16   skip-hand: $SKIP_HAND

  LEGAL: these assets are DERIVED from Meta's gated SAM-3D-Body weights (SAM
  License, no-reverse-engineering clause) with embedded DINOv3 (own gated terms).
  You generate them from YOUR licensed copy; do not redistribute them.

  TIME/DISK: the body + hand ViT exports take ~HOURS on CPU and several GB of
  scratch each; the detector / MHR-asset / pose-corrective steps are quick.
EOF

step=0
run_step() {  # run_step <primary-out> <label> -- <cmd...>
  local out="$1" label="$2"; shift 2; [[ "$1" == "--" ]] && shift
  step=$((step+1))
  if [[ -s "$OUT_DIR/$out" && "$FORCE" -ne 1 ]]; then
    echo ""; echo "-- [$step] $label -> $out  [SKIP: exists] --"
    return 0
  fi
  echo ""; echo "-- [$step] $label -> $out --"
  ( set -x; "$@" )
  [[ -s "$OUT_DIR/$out" ]] || { echo "error: $label did not produce $OUT_DIR/$out" >&2; exit 1; }
}

# ─── the ordered export pipeline ────────────────────────────────────────────
# 1. detector — clean torchvision Faster R-CNN R50-FPN v2 ONNX at 1280 (no gated
#    weights involved). Stable ~1.0 confidence on real people (no flicker) and
#    separates close figures far better than the old ssdlite320. Set
#    HASTUR_DETECTOR / HASTUR_DETECTOR_SIZE to override (see export_detector.py).
run_step person_detector.onnx "person detector (frcnn_r50_v2 @ ${HASTUR_DETECTOR_SIZE:-1280})" -- \
  "$PYTHON" "$TOOLS/export_detector.py" --model "${HASTUR_DETECTOR:-frcnn_r50_v2}" \
    --size "${HASTUR_DETECTOR_SIZE:-1280}" \
    --out "$OUT_DIR/person_detector.onnx"

# 2. MHR static assets — flat binary for the C++ FK+LBS.
run_step mhr_assets.bin "MHR static assets" -- \
  "$PYTHON" "$TOOLS/extract_mhr_assets.py" \
    --out "$OUT_DIR/mhr_assets.bin" \
    --manifest "$OUT_DIR/mhr_assets.manifest.json"

# 3. pose-corrective MLP — standalone ONNX (fp32 by default; --pose-fp16 halves it,
#    IO stays fp32 so the C++ hook is unchanged).
POSE_ARGS=(--out "$OUT_DIR/pose_corrective.onnx")
[[ "$POSE_FP16" -eq 1 ]] && POSE_ARGS+=(--fp16)
run_step pose_corrective.onnx "pose-corrective MLP" -- \
  "$PYTHON" "$TOOLS/export_pose_corrective.py" "${POSE_ARGS[@]}"

# 4. body regressor — fp16 ONNX (SLOW: DINOv3-H+ backbone trace + fp16 fold).
run_step sam3dbody_body.onnx "SAM-3D-Body regressor (faithful)" -- \
  "$PYTHON" "$TOOLS/export_sam3dbody.py" --mode faithful \
    --out "$OUT_DIR/sam3dbody_body.onnx"

# 5. hand decoder — fp16 ONNX; also emits mhr_wrist.bin + the json sidecar (SLOW).
if [[ "$SKIP_HAND" -eq 1 ]]; then
  echo ""; echo "-- hand decoder SKIPPED (--skip-hand): plugin runs body-only --"
else
  run_step sam3dbody_hand.onnx "hand-decoder 2nd pass + mhr_wrist.bin" -- \
    "$PYTHON" "$TOOLS/export_hand.py" \
      --out "$OUT_DIR/sam3dbody_hand.onnx"
fi

# ─── summary ────────────────────────────────────────────────────────────────
echo ""
echo "== done. model set in: $OUT_DIR =="
for f in person_detector.onnx person_detector.json mhr_assets.bin \
         mhr_assets.manifest.json pose_corrective.onnx sam3dbody_body.onnx \
         sam3dbody_hand.onnx sam3dbody_hand.json mhr_wrist.bin; do
  if [[ -e "$OUT_DIR/$f" ]]; then
    printf "  [ok]      %-28s %s\n" "$f" "$(du -h "$OUT_DIR/$f" | cut -f1)"
  else
    printf "  [missing] %-28s\n" "$f"
  fi
done
echo ""
echo "The plugin resolves these from Contents/Resources, \$HASTUR_MODEL_DIR, or the"
echo "'Model directory' param. If you wrote to a custom --out-dir, set HASTUR_MODEL_DIR."
