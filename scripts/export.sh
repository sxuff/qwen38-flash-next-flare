#!/bin/bash
# Export the runtime package: Ignition tunes kernels, compiles forwards, captures CUDA graphs and signs the package with a
# local development key. About 2 h the first time on one GB10; about 20 min again once the kernel caches are warm.
# Disk: besides the 102 GB checkpoint it writes a ~80 GB staging copy and the ~108 GB package (see clean-staging.sh).
# Run it from a login with an unlimited locked-memory limit (see the README), with nothing else on the GPU.
set -uo pipefail
. "$(dirname "$0")/config.sh"
export PATH=/usr/local/cuda/bin:$PATH

mkdir -p "$(dirname "$PACKAGE")"
if [ -e "$PACKAGE" ]; then
  echo "$PACKAGE already exists; Ignition will not overwrite a package."
  exit 1
fi
free_gb=$(df -BG --output=avail "$(dirname "$PACKAGE")" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 200 ]; then
  echo "Only ${free_gb} GB free. The export needs about 200 GB besides the checkpoint; it fails near the end otherwise."
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

cd "$FLARE_SRC"
start=$(date +%s)
.venv/bin/ignition export qwen3.8-flash --gpus 0 --max-concurrency 8 --image-concurrency 4 \
  --mtp-draft-tokens 5 --speculative-token-capacity 48 --mixed-prefill \
  --output "$PACKAGE" --verbose
status=$?
echo "Export finished with status $status after $(( ($(date +%s) - start) / 60 )) min"
exit $status
