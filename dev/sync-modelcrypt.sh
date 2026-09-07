#!/usr/bin/env bash
# Check that the vendored crypto still matches Rotobot-Next.
#
# Sealing and opening must agree byte for byte. A divergence shows up at a
# customer site as "this model artifact is corrupt", which is a long way from
# the actual cause, so it is worth a cheap check in CI.
#
#   dev/sync-modelcrypt.sh [path-to-Rotobot-Next]     # verify
#   dev/sync-modelcrypt.sh [path] --fix               # re-copy from upstream
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$HERE/../Rotobot-Next}"
FIX=0; [ "${2:-}" = "--fix" ] && FIX=1

[ -d "$SRC/src/crypto" ] || {
  echo "ERROR: $SRC does not look like Rotobot-Next (no src/crypto)." >&2
  echo "  usage: $0 [path-to-Rotobot-Next] [--fix]" >&2
  exit 2
}

rc=0
for f in license/Crypto.h license/Crypto.cpp crypto/modelcrypt.h crypto/modelcrypt.cpp; do
  up="$SRC/src/$f"; mine="$HERE/src/$f"
  # The vendored copy carries a provenance banner the original does not, so
  # compare from the first line the two have in common.
  if diff -q <(sed '/^\/\/ =\{10,\}$/,/^\/\/ =\{10,\}$/d' "$mine") "$up" >/dev/null 2>&1; then
    printf '  %-26s in sync\n' "$f"
  else
    printf '  %-26s DRIFTED\n' "$f"
    rc=1
    if [ "$FIX" = 1 ]; then
      banner=$(sed -n '1,/^\/\/ =\{10,\}$/p' "$mine" | head -n -0)
      { sed -n '1,/^\/\/ =\{10,\}$/p' "$mine"; cat "$up"; } > "$mine.new"
      mv "$mine.new" "$mine"
      printf '  %-26s re-copied\n' "$f"
      rc=0
    fi
  fi
done

if [ "$rc" != 0 ]; then
  echo
  echo "The vendored crypto has drifted from $SRC." >&2
  echo "Edit the ORIGINAL, then re-run with --fix." >&2
fi
exit $rc
