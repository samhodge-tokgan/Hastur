#!/usr/bin/env bash
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# make_pkg.sh — build a macOS installer for the Sam3dBody OFX plugin (project
# "Hastur").
#
# What it does:
#   1. ad-hoc codesign the ISOLATED bundle (the privately-renamed
#      libonnxruntime_hastur dylib first, then the .ofx binary) — matches the
#      isolation CMake already applied. No Apple Developer ID yet; notarization is
#      deferred until one exists.
#   2. inject tools/convert_models.sh into the staged bundle's Contents/ (AFTER
#      signing, so it rides in the payload but outside the code-signature seal).
#   3. pkgbuild a component pkg + productbuild a distribution that installs into
#      ~/Library/OFX/Plugins (the per-user OFX path; no admin needed).
#
# Hastur is MODEL-LESS: this installer bundles the plugin + convert_models.sh but
# NO model weights. After installing, the user runs convert_models to generate the
# ONNX/asset set from their own gated SAM-3D-Body checkpoints (see docs/MODELS.md).
#
# Usage (from the repo root, after `cmake --build build`):
#   packaging/make_pkg.sh
#   packaging/make_pkg.sh --bundle build/Sam3dBody.ofx.bundle \
#       --out dist/Sam3dBody-0.1.0-macos-arm64.pkg --version 0.1.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IDENTIFIER="com.tokgan.openfx.Sam3dBody"
VERSION="0.1.0"
ARCH="arm64"
# Per-user OFX path. Relative install-location + enable_currentUserHome in the
# distribution => the payload lands in ~/Library/OFX/Plugins on the target machine.
INSTALL_LOCATION="Library/OFX/Plugins"
BUNDLE="$REPO_ROOT/build/Sam3dBody.ofx.bundle"
OUT=""
CODESIGN_ID="-"        # "-" = ad-hoc
PKG_SIGN_ID=""         # empty = unsigned pkg
CONVERT_SCRIPT="$REPO_ROOT/tools/convert_models.sh"
EXTRA_CONTENTS=()      # additional files to drop into Contents/ AFTER signing

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --identity) CODESIGN_ID="$2"; shift 2;;
    --pkg-identity) PKG_SIGN_ID="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    --arch) ARCH="$2"; shift 2;;
    --extra-contents) EXTRA_CONTENTS+=("$2"); shift 2;;
    --no-convert-script) CONVERT_SCRIPT=""; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -d "$BUNDLE" ]] || {
  echo "error: bundle not found: $BUNDLE" >&2
  echo "       build it first:  cmake -S . -B build -DHASTUR_WITH_ONNX=ON && cmake --build build -j" >&2
  exit 2
}
[[ -n "$OUT" ]] || OUT="$REPO_ROOT/dist/Sam3dBody-${VERSION}-macos-${ARCH}.pkg"

BIN="$BUNDLE/Contents/MacOS/Sam3dBody.ofx"
[[ -f "$BIN" ]] || { echo "error: plugin binary missing: $BIN" >&2; exit 2; }

# Coherence guard: the plugin is model-less; a bundled Resources model set would
# mean weights got staged in. Warn (don't fail) so a deliberate self-contained
# build is still packageable.
if compgen -G "$BUNDLE/Contents/Resources/*.onnx" >/dev/null 2>&1; then
  echo "warning: $BUNDLE/Contents/Resources contains *.onnx — Hastur is model-less;" >&2
  echo "         the installer normally ships NO weights (user generates them)." >&2
fi

echo "== codesign (identity: $CODESIGN_ID) =="
# Sign inner-to-outer: bundled dylib(s) first, the plugin binary last (signing the
# main executable seals the bundle, so Contents/ must hold only normal bundle
# structure at this point — loose extras are injected AFTER, into the staged copy).
# The dylib is privately renamed by the isolation step, so glob rather than hardcode.
if [[ -d "$BUNDLE/Contents/Frameworks" ]]; then
  while IFS= read -r -d '' dylib; do
    echo "  sign $dylib"
    codesign --force --sign "$CODESIGN_ID" --timestamp=none "$dylib"
  done < <(find "$BUNDLE/Contents/Frameworks" -type f -name '*.dylib' -print0)
fi
codesign --force --sign "$CODESIGN_ID" --timestamp=none "$BIN"
codesign --verify --verbose=2 "$BIN" || true

echo "== stage =="
STAGE="$(mktemp -d)"
COMP_PKG="$(mktemp -d)/component.pkg"
DIST_XML="$(mktemp -t sam3dbody-dist).xml"
trap 'rm -rf "$STAGE" "$(dirname "$COMP_PKG")" "$DIST_XML"' EXIT
cp -R "$BUNDLE" "$STAGE/"
STAGED_CONTENTS="$STAGE/$(basename "$BUNDLE")/Contents"

# Inject convert_models.sh (the model-generation entrypoint) + any extras into the
# already-signed staged bundle's Contents/. They ride in the payload but outside the
# signature seal — fine for an ad-hoc/unsigned distribution.
inject() {
  local f="$1"
  [[ -f "$f" ]] || { echo "extra-contents file not found: $f" >&2; exit 2; }
  echo "  add $(basename "$f") -> Contents/"
  cp "$f" "$STAGED_CONTENTS/"
  chmod +x "$STAGED_CONTENTS/$(basename "$f")" 2>/dev/null || true
}
[[ -n "$CONVERT_SCRIPT" ]] && inject "$CONVERT_SCRIPT"
for f in "${EXTRA_CONTENTS[@]:-}"; do [[ -n "$f" ]] && inject "$f"; done

mkdir -p "$(dirname "$OUT")"

echo "== pkgbuild (component) =="
PKG_ARGS=(--root "$STAGE" --identifier "$IDENTIFIER" --version "$VERSION"
          --install-location "/$INSTALL_LOCATION")
pkgbuild "${PKG_ARGS[@]}" "$COMP_PKG"

echo "== productbuild (distribution -> ~/$INSTALL_LOCATION) =="
# enable_currentUserHome + a relative-under-home layout installs into the invoking
# user's ~/Library/OFX/Plugins with no admin prompt.
cat > "$DIST_XML" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>Sam3dBody OFX Plugin (Hastur)</title>
    <domains enable_anywhere="false" enable_currentUserHome="true" enable_localSystem="false"/>
    <options customize="never" require-scripts="false" hostArchitectures="$ARCH"/>
    <choices-outline>
        <line choice="default"/>
    </choices-outline>
    <choice id="default" title="Sam3dBody">
        <pkg-ref id="$IDENTIFIER"/>
    </choice>
    <pkg-ref id="$IDENTIFIER" version="$VERSION" onConclusion="none">component.pkg</pkg-ref>
</installer-gui-script>
EOF

PB_ARGS=(--distribution "$DIST_XML" --package-path "$(dirname "$COMP_PKG")")
if [[ -n "$PKG_SIGN_ID" ]]; then
  PB_ARGS+=(--sign "$PKG_SIGN_ID")
fi
productbuild "${PB_ARGS[@]}" "$OUT"

echo "== done =="
ls -lh "$OUT"
echo "Installs $(basename "$BUNDLE") into ~/$INSTALL_LOCATION (per-user, no admin)."
echo "It bundles convert_models.sh but NO model weights — after install, run:"
echo "  HASTUR_TOOLS_DIR=<Hastur>/tools \\"
echo "    ~/$INSTALL_LOCATION/$(basename "$BUNDLE")/Contents/convert_models.sh \\"
echo "    --ckpt-dir <checkpoints> --python <torch-env>/bin/python"
if [[ "$CODESIGN_ID" == "-" ]]; then
  echo "NOTE: ad-hoc signed + unsigned pkg. Users may need to allow it in System"
  echo "Settings > Privacy & Security, or run: xattr -dr com.apple.quarantine \"$OUT\""
fi
