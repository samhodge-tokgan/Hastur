#!/usr/bin/env bash
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# fetch_models.sh — download/stage the SAM 3D Body ONNX models + mesh assets into the
# installed Sam3dBody.ofx.bundle's Contents/Resources.
#
# NOTE: Hastur is MODEL-LESS by design. The models below are USER-GENERATED locally
# (exported from the SAM 3D Body checkpoints via tools/, or pulled from a private
# mirror you control) — they are NOT committed and there is no public release tag yet.
# The manifest checksums/sizes are placeholders (TODO) to be pinned once the export
# pipeline produces stable weights. The download plumbing (gh / GITHUB_TOKEN / curl,
# HF mirror override) is kept from humbaba so a gated release can drop straight in.
#
# Usage:
#   fetch_models.sh [RESOURCES_DIR]     # explicit Contents/Resources target
#   HASTUR_MODELS_TAG=models-v1 fetch_models.sh
#   HASTUR_MODELS_BASE_URL=https://host/path fetch_models.sh   # mirror / HF override
#
# With no argument it locates the installed bundle in the standard OFX dirs.
set -euo pipefail

TAG="${HASTUR_MODELS_TAG:-models-v1}"
REPO="samhodge-tokgan/Hastur"
BASE="${HASTUR_MODELS_BASE_URL:-https://github.com/${REPO}/releases/download/${TAG}}"

# Manifest: "<release-asset-name> <installed-filename> <sha256> <bytes>"
# TODO(models): fill in the real sha256 + byte size once the models are pinned. Until
# then the checksum gate below will (intentionally) refuse a mismatched download.
MODELS=(
  "person_detector.onnx person_detector.onnx TODO 0"
  "sam3dbody_body.onnx sam3dbody_body.onnx TODO 0"
  "sam3dbody_hand.onnx sam3dbody_hand.onnx TODO 0"
  "mhr_assets.bin mhr_assets.bin TODO 0"
  "pose_corrective.onnx pose_corrective.onnx TODO 0"
)

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

find_resources() {
  # $1 may be an explicit Resources dir OR a bundle OR empty (auto-detect).
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

if ! mkdir -p "$RES" 2>/dev/null || [ ! -w "$RES" ]; then
  echo "error: '$RES' is not writable. Re-run with sudo (system install), e.g.:" >&2
  echo "       sudo $0 ${1:-}" >&2
  exit 1
fi

echo "Installing models into: $RES"
echo "Source: $BASE"
for entry in "${MODELS[@]}"; do
  read -r asset name want_sha want_bytes <<<"$entry"
  dest="$RES/$name"
  if [ "$want_sha" = "TODO" ]; then
    echo "  [todo] $name — no pinned checksum yet (model-less; stage it locally)"
    continue
  fi
  if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$want_sha" ]; then
    echo "  [skip] $name (already present, checksum OK)"
    continue
  fi
  echo "  [get]  $asset -> $name (${want_bytes} bytes)"
  tmp="$dest.part"
  # Prefer the GitHub CLI when present (handles auth for a gated/private release);
  # otherwise curl the download URL (a GITHUB_TOKEN is optional, for auth/rate limits).
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --output "$tmp" --clobber
  else
    hdr=()
    [ -n "${GITHUB_TOKEN:-}" ] && hdr=(-H "Authorization: token ${GITHUB_TOKEN}")
    curl -fL --retry 3 --progress-bar "${hdr[@]}" -o "$tmp" "$BASE/$asset"
  fi
  got_sha="$(sha256 "$tmp")"
  if [ "$got_sha" != "$want_sha" ]; then
    rm -f "$tmp"
    echo "  error: checksum mismatch for $asset" >&2
    echo "         expected $want_sha" >&2
    echo "         got      $got_sha" >&2
    exit 1
  fi
  mv -f "$tmp" "$dest"
  echo "  [ok]   $name verified"
done
echo "Done. Model staging complete for $RES"
