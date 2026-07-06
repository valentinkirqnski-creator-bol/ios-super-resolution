#!/usr/bin/env bash
# Vendors LibRaw source into ios/vendor/LibRaw so the Xcode target can compile
# it directly (no CMake, no prebuilt binaries). Run once before `xcodegen`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/../vendor"
DEST="$VENDOR/LibRaw"
TAG="0.21.2"

if [ -d "$DEST/src" ]; then
  echo "LibRaw already present at $DEST"
  exit 0
fi

mkdir -p "$VENDOR"
echo "Cloning LibRaw $TAG ..."
git clone --depth 1 --branch "$TAG" https://github.com/LibRaw/LibRaw.git "$DEST"

# Drop parts we don't compile (examples/samples pull in optional deps).
rm -rf "$DEST/samples" "$DEST/doc" "$DEST/bin" 2>/dev/null || true

echo "LibRaw vendored at $DEST"
