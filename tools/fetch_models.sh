#!/usr/bin/env bash
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# fetch_models.sh — stage the SAM 3D Body ONNX models + mesh assets into the
# installed Sam3dBody.ofx.bundle's Contents/Resources.
#
# ─── Hastur is MODEL-LESS ───────────────────────────────────────────────────
# Hastur ships NO Meta weights. The runtime assets are DERIVED from Meta's gated
# SAM-3D-Body checkpoints (SAM License, no-reverse-engineering clause + embedded
# DINOv3 gated terms), which we must NOT redistribute. So the PRIMARY path is to
# GENERATE them locally from your own licensed copy of the checkpoints:
#
#     tools/convert_models.sh --ckpt-dir <checkpoints> --out-dir <Resources> --python <env>
#
# This script therefore does NOT download from a Hastur release (there is none —
# it would redistribute derived Meta weights). It exists only to:
#   (a) point you at convert_models (the default, no-arg behaviour), and
#   (b) OPTIONALLY pull the pre-generated set from a mirror YOU control/host
#       (e.g. an internal artifact store you populated with convert_models output)
#       via $HASTUR_MODELS_BASE_URL — the same mirror hook humbaba used.
#
# Usage:
#   fetch_models.sh [RESOURCES_DIR]                       # prints the generate-locally guidance
#   HASTUR_MODELS_BASE_URL=https://your-mirror/path fetch_models.sh [RESOURCES_DIR]
#       # pull user-generated assets from your own mirror into Contents/Resources
#
# With no RESOURCES_DIR it locates the installed bundle in the standard OFX dirs.
set -euo pipefail

# Optional mirror hook (a mirror YOU host with convert_models output). Empty by
# default: there is no Hastur-hosted release of these derived weights.
BASE="${HASTUR_MODELS_BASE_URL:-}"

# The user-generated model set. No pinned checksums: these files are generated on
# YOUR machine (or your mirror), not distributed by Hastur, so their hashes are
# per-user. A mirror may host an optional sha256 sidecar (<name>.sha256) which,
# when present, is verified.
MODELS=(
  person_detector.onnx
  person_detector.json
  sam3dbody_body.onnx
  sam3dbody_hand.onnx
  sam3dbody_hand.json
  mhr_assets.bin
  mhr_assets.manifest.json
  pose_corrective.onnx
  mhr_wrist.bin
)

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

find_resources() {
  if [ -n "${1:-}" ]; then
    case "$1" in
      */Contents/Resources) echo "$1"; return 0 ;;
      *.ofx.bundle) echo "$1/Contents/Resources"; return 0 ;;
      *) echo "$1"; return 0 ;;
    esac
  fi
  local dirs=(
    "$HOME/Library/OFX/Plugins" "/Library/OFX/Plugins"          # macOS
    "$HOME/OFX/Plugins" "/usr/OFX/Plugins" "/usr/local/OFX/Plugins"  # Linux
  )
  local d
  for d in "${dirs[@]}"; do
    if [ -d "$d/Sam3dBody.ofx.bundle" ]; then
      echo "$d/Sam3dBody.ofx.bundle/Contents/Resources"; return 0
    fi
  done
  return 1
}

RES="$(find_resources "${1:-}")" || {
  echo "error: could not find Sam3dBody.ofx.bundle in the standard OFX dirs." >&2
  echo "       pass the bundle's Contents/Resources path as an argument." >&2
  exit 1
}

# ─── primary path: no mirror configured -> generate locally ─────────────────
if [ -z "$BASE" ]; then
  cat >&2 <<EOF
Hastur is model-less: there is no download of Meta-derived weights.

Generate the model set LOCALLY from your own gated SAM-3D-Body checkpoints:

  tools/convert_models.sh \\
      --ckpt-dir  /path/to/sam-3d-body/checkpoints/sam-3d-body-dinov3 \\
      --out-dir   "$RES" \\
      --python    /path/to/torch-env/bin/python

(Windows: tools/convert_models.ps1.) See docs/MODELS.md for the full flow.

To instead pull a set you already generated onto a mirror you host, set
HASTUR_MODELS_BASE_URL=https://your-mirror/path and re-run this script.
EOF
  exit 2
fi

# ─── optional mirror path: pull user-generated assets from your own host ────
if ! mkdir -p "$RES" 2>/dev/null || [ ! -w "$RES" ]; then
  echo "error: '$RES' is not writable. Re-run with sudo (system install), e.g.:" >&2
  echo "       sudo $0 ${1:-}" >&2
  exit 1
fi

echo "Staging user-generated models into: $RES"
echo "Mirror: $BASE"
for name in "${MODELS[@]}"; do
  dest="$RES/$name"
  echo "  [get]  $name"
  tmp="$dest.part"
  hdr=()
  [ -n "${GITHUB_TOKEN:-}" ] && hdr=(-H "Authorization: token ${GITHUB_TOKEN}")
  # A json/manifest may legitimately be absent for a body-only set; skip 404s softly.
  if ! curl -fL --retry 3 --progress-bar "${hdr[@]}" -o "$tmp" "$BASE/$name"; then
    rm -f "$tmp"
    echo "  [warn] $name not on mirror (skipping)"; continue
  fi
  # Verify against an OPTIONAL <name>.sha256 sidecar if the mirror hosts one.
  if want_sha="$(curl -fsSL "${hdr[@]}" "$BASE/$name.sha256" 2>/dev/null)"; then
    want_sha="$(echo "$want_sha" | awk '{print $1}')"
    got_sha="$(sha256 "$tmp")"
    if [ "$got_sha" != "$want_sha" ]; then
      rm -f "$tmp"
      echo "  error: checksum mismatch for $name (want $want_sha got $got_sha)" >&2
      exit 1
    fi
    echo "  [ok]   $name (sha256 verified)"
  else
    echo "  [ok]   $name (no sha256 sidecar; unverified)"
  fi
  mv -f "$tmp" "$dest"
done
echo "Done. Model staging complete for $RES"
