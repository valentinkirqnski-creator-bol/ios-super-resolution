#!/usr/bin/env bash
# Builds Resources/PWCNetFlow.mlpackage (the model core/neural_flow.mm loads
# at runtime) by running tools/neural_flow/convert_coreml.py. Needs a real
# Core ML toolchain to produce the final weight blob, which only ships on
# macOS -- see tools/neural_flow/README.md for why. Run before `xcodegen
# generate` (CI does this), same as scripts/fetch_libraw.sh.
#
# Deliberately non-fatal: "Use Neural Flow" is an experimental, off-by-default
# Settings toggle (Config::use_neural_flow), and neural_flow.mm already falls
# back to the classical alignment path per-frame when the model isn't
# bundled. A conversion hiccup here (network blip, a future dependency
# update) shouldn't take down the whole app build over a feature most users
# have never turned on.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
TOOLDIR="$ROOT/tools/neural_flow"
OUT="$ROOT/Resources/PWCNetFlow.mlpackage"

if [ -d "$OUT" ]; then
  echo "PWCNetFlow.mlpackage already present at $OUT"
  exit 0
fi

fail() {
  echo "WARNING: neural flow model build failed ($1) -- continuing without it." >&2
  echo "The app still builds; 'Use Neural Flow' just falls back to classical alignment." >&2
  exit 0
}

python3 -m venv "$TOOLDIR/.venv" || fail "venv creation"
# shellcheck disable=SC1091
source "$TOOLDIR/.venv/bin/activate" || fail "venv activation"

pip install -q --upgrade pip || fail "pip upgrade"
pip install -q torch==2.7.0 coremltools==9.0 "numpy==1.26.4" pillow gdown || fail "dependency install"

mkdir -p "$TOOLDIR/pretrained_networks"
WEIGHTS="$TOOLDIR/pretrained_networks/pwcnet-network-default.pth"
if [ ! -f "$WEIGHTS" ]; then
  gdown "https://drive.google.com/uc?id=1s11Ud1UMipk2AbZZAypLPRpnXOS9Y1KO" -O "$WEIGHTS" || fail "weights download"
fi

(cd "$TOOLDIR" && python3 convert_coreml.py) || fail "conversion"

if [ ! -d "$TOOLDIR/PWCNetFlow.mlpackage" ]; then
  fail "conversion produced no output"
fi

mkdir -p "$ROOT/Resources"
mv "$TOOLDIR/PWCNetFlow.mlpackage" "$OUT"
echo "Built $OUT"
