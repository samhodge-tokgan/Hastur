#!/usr/bin/env bash
# Run the AOV multi-plane smoke against an arch-matched, isolated Natron.
#
# Defaults to the isolated arm64 install at ~/hastur-test/Natron.app (see
# docs/AOVS.md); override any of these via env:
#   NATRON            NatronRenderer binary
#   OFX_PLUGIN_PATH   dir holding Sam3dBody.ofx.bundle          (default: build/)
#   HASTUR_MODEL_DIR  dir with the 4 model/asset files          (default: ~/hastur_models)
#   HASTUR_INPUT      a person plate                            (default: build/smoke_input.png)
#   HASTUR_RENDER=1   also render the beauty pass end-to-end
#
# Exit 0 = all AOV planes declared (and, with HASTUR_RENDER=1, a frame rendered).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NATRON="${NATRON:-$HOME/hastur-test/Natron.app/Contents/MacOS/NatronRenderer}"
export OFX_PLUGIN_PATH="${OFX_PLUGIN_PATH:-$REPO/build}"
export HASTUR_MODEL_DIR="${HASTUR_MODEL_DIR:-$HOME/hastur_models}"
export HASTUR_INPUT="${HASTUR_INPUT:-$REPO/build/smoke_input.png}"

if [[ ! -x "$NATRON" ]]; then
  echo "NatronRenderer not found at: $NATRON" >&2
  echo "Set NATRON=/path/to/NatronRenderer (must match the plugin arch)." >&2
  exit 127
fi
# --no-settings keeps the run hermetic (ignores your personal Natron prefs).
exec "$NATRON" --no-settings --clear-openfx-cache -t "$REPO/tests/natron/smoke_aovs.py" < /dev/null
