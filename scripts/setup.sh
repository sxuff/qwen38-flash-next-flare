#!/bin/bash
# Build Flare and Ignition at the pinned commit with the recipe patch applied.
set -euo pipefail
. "$(dirname "$0")/config.sh"

missing=()
for p in build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing packages: ${missing[*]}"
  echo "Install them with: sudo apt-get update && sudo apt-get install -y ${missing[*]}"
  exit 1
fi
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
command -v nvcc >/dev/null || { echo "CUDA toolkit not found (nvcc); install CUDA 13 under /usr/local/cuda."; exit 1; }
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v cargo >/dev/null || { echo "Rust not found; install 1.88 or newer from https://rustup.rs"; exit 1; }

if [ ! -d "$FLARE_SRC/.git" ]; then
  git clone "$FLARE_REPO" "$FLARE_SRC"
fi
git -C "$FLARE_SRC" checkout --quiet "$FLARE_COMMIT"

PATCH="$RECIPE_DIR/patches/qwen38-topk-range.patch"
if git -C "$FLARE_SRC" apply --check "$PATCH" 2>/dev/null; then
  git -C "$FLARE_SRC" apply "$PATCH"
  echo "Applied $(basename "$PATCH")"
elif git -C "$FLARE_SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "$(basename "$PATCH") is already applied"
else
  echo "$(basename "$PATCH") does not apply to $FLARE_SRC; is it at $FLARE_COMMIT?"
  exit 1
fi

cd "$FLARE_SRC"
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
echo "Built $("$FLARE_BIN" --version) at ${FLARE_COMMIT:0:7}"
