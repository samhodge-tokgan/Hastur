#!/usr/bin/env bash
# Build a macOS .pkg installer for the Sam3dBody OFX plugin (project "Hastur").
#
# Ad-hoc signs the bundled dylib + plugin binary (no Apple Developer ID yet), then
# packages the .ofx.bundle to install into /Library/OFX/Plugins (the system OFX
# path scanned by hosts). Notarization is deferred until a Developer ID is available.
#
# Usage:
#   packaging/make_pkg.sh --bundle build/Sam3dBody.ofx.bundle \
#       --out dist/Sam3dBody-0.1.0.pkg [--identity "Developer ID Installer: ..."]
set -euo pipefail

IDENTIFIER="com.tokgan.openfx.Sam3dBody"
VERSION="0.1.0"
INSTALL_LOCATION="/Library/OFX/Plugins"
BUNDLE=""
OUT=""
CODESIGN_ID="-"        # "-" = ad-hoc
PKG_SIGN_ID=""         # empty = unsigned pkg
EXTRA_CONTENTS=()      # files to drop into the bundle's Contents/ AFTER signing

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --identity) CODESIGN_ID="$2"; shift 2;;
    --pkg-identity) PKG_SIGN_ID="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    --extra-contents) EXTRA_CONTENTS+=("$2"); shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$BUNDLE" && -d "$BUNDLE" ]] || { echo "need --bundle <dir>"; exit 2; }
[[ -n "$OUT" ]] || { echo "need --out <pkg>"; exit 2; }

BIN="$BUNDLE/Contents/MacOS/Sam3dBody.ofx"

echo "== codesign (identity: $CODESIGN_ID) =="
# Sign inner-to-outer: the bundled dylib(s) first, then the plugin binary last.
# Signing the bundle's main executable puts codesign into bundle-seal mode, so the
# Contents/ tree must contain ONLY normal bundle structure at this point — any loose
# file (e.g. fetch_models.sh) is unsignable "nested code" and fails the sign. Such
# extras are injected into the staged copy AFTER signing (see --extra-contents below).
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
trap 'rm -rf "$STAGE"' EXIT
cp -R "$BUNDLE" "$STAGE/"

# Inject extra files (e.g. the model fetch script) into the staged, already-signed
# bundle's Contents/. They ride along in the pkg payload but are intentionally outside
# the code-signature seal — fine for an ad-hoc/unsigned distribution.
if [[ ${#EXTRA_CONTENTS[@]} -gt 0 ]]; then
  STAGED_CONTENTS="$STAGE/$(basename "$BUNDLE")/Contents"
  for f in "${EXTRA_CONTENTS[@]}"; do
    [[ -f "$f" ]] || { echo "extra-contents file not found: $f" >&2; exit 2; }
    echo "  add $(basename "$f") -> Contents/"
    cp "$f" "$STAGED_CONTENTS/"
  done
fi

mkdir -p "$(dirname "$OUT")"

echo "== pkgbuild =="
PKG_ARGS=(--root "$STAGE" --identifier "$IDENTIFIER" --version "$VERSION"
          --install-location "$INSTALL_LOCATION")
if [[ -n "$PKG_SIGN_ID" ]]; then
  PKG_ARGS+=(--sign "$PKG_SIGN_ID")
fi
pkgbuild "${PKG_ARGS[@]}" "$OUT"

echo "== done =="
ls -lh "$OUT"
echo "Installs $(basename "$BUNDLE") into $INSTALL_LOCATION"
if [[ "$CODESIGN_ID" == "-" ]]; then
  echo "NOTE: ad-hoc signed + unsigned pkg. Users may need to allow it in System"
  echo "Settings > Privacy & Security, or run: xattr -dr com.apple.quarantine \"$OUT\""
fi
