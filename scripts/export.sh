#!/bin/bash
# Export the runtime package into Flare's model cache: Ignition tunes kernels, compiles forwards, captures CUDA graphs and
# signs the package with a local development key. About 1.5 h the first time on one GB10; about 30 min again once the
# kernel caches are warm. Disk: besides the 102 GB checkpoint, the package takes about 107 GB (100 GiB).
# Run it from a login with an unlimited locked-memory limit (see the README), with nothing else on the GPU.
set -uo pipefail
. "$(dirname "$0")/config.sh"
export PATH=/usr/local/cuda/bin:$PATH

CACHE=${FLARE_PACKAGE_CACHE:-$HOME/.cache/flare/packages}
mkdir -p "$CACHE" "$(dirname "$PACKAGE_FILE")"
free_gb=$(df -BG --output=avail "$CACHE" | tail -1 | tr -dc 0-9)
echo "Profile $PROFILE: $FLARE_SRC at ${FLARE_COMMIT:0:12}, export flags: $EXPORT_FLAGS"
if [ "$free_gb" -lt 120 ]; then
  echo "Only ${free_gb} GB free. The package needs about 100 GB besides the checkpoint; it fails near the end otherwise."
  exit 1
fi
if [ "$(ulimit -l)" != unlimited ]; then
  echo "Locked-memory limit is $(ulimit -l) KB, not unlimited. See 'Prepare the host' in the README."
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "Another process is using the GPU; stop it first (stop-flare.sh stops Flare)."
  exit 1
fi

# A DFlash2 export fetches the drafter at the revision Hyperflare pins; check it against the recorded sha256 first.
if [[ "$EXPORT_FLAGS" == *dflash2* ]]; then
  "$FLARE_SRC/.venv/bin/hf" download "$DRAFTER" --revision "$DRAFTER_REVISION" config.json model.safetensors >/dev/null
  drafter=$("$FLARE_SRC/.venv/bin/hf" download "$DRAFTER" --revision "$DRAFTER_REVISION" model.safetensors 2>/dev/null | tail -1)
  got=$(sha256sum < "$drafter" | cut -d' ' -f1)
  [ "$got" = "$DRAFTER_SHA256" ] || { echo "DFlash2 drafter sha256 is $got, expected $DRAFTER_SHA256"; exit 1; }
  echo "DFlash2 drafter verified: $DRAFTER@${DRAFTER_REVISION:0:8} sha256 ${got:0:16}..."
fi

cd "$FLARE_SRC"
log=$(mktemp)
start=$(date +%s)
# Extra arguments pass through, e.g. --reuse-weights to skip rehashing the weights of an earlier export.
# EXPORT_FLAGS comes from the profile in config.sh; flare-mtp-mixed is v1's export unchanged.
.venv/bin/ignition export qwen3.8-flash --gpus 0 $EXPORT_FLAGS --verbose "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
entry=$(sed -n 's/^Exported cache entry: //p' "$log" | tail -1)
rm -f "$log"
echo "Export finished with status $status after $(( ($(date +%s) - start) / 60 )) min"
if [ "$status" -eq 0 ] && [ -n "$entry" ]; then
  echo "$entry" > "$PACKAGE_FILE"
  echo "Package: $entry (recorded in $PACKAGE_FILE)"
fi
exit $status
