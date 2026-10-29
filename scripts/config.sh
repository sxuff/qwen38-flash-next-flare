# Shared settings for the recipe scripts. Every value can be overridden from the environment.

# Flare and Ignition, pinned to the commit this recipe was measured on.
FLARE_REPO=${FLARE_REPO:-https://github.com/NVIDIA/flare-local-inference.git}
FLARE_COMMIT=${FLARE_COMMIT:-ca1f960edbf67d624f2d2948d2c111e8a6c8eb3c}
FLARE_SRC=${FLARE_SRC:-$HOME/flare-local-inference}
FLARE_BIN=${FLARE_BIN:-$FLARE_SRC/flare/target/release/flare}

# NVIDIA's NVFP4 checkpoint and the package exported from it. Ignition exports into Flare's model cache
# (~/.cache/flare/packages); export.sh records the new cache entry's path in PACKAGE_FILE.
CHECKPOINT=${CHECKPOINT:-nvidia/Qwen3.8-Flash-Next-FP4-Local}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}
PACKAGE_FILE=${PACKAGE_FILE:-$HOME/.local/state/qwen38-flash-next-flare/package}
PACKAGE=${PACKAGE:-$(cat "$PACKAGE_FILE" 2>/dev/null || true)}

# Serving. Three drafts every round measured faster than adaptive verification for agents. The throughput scheduler and
# a prompt priority of 16 start each new turn sooner. PREFILL_PRIORITY becomes FLARE_PREFILL_PRIORITY, which
# patches/flare-prefill-priority.patch reads; 1 is upstream's pricing.
SERVED_NAME=${SERVED_NAME:-qwen38-flash-next-flare}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVE_FLAGS=${SERVE_FLAGS:---no-mtp-adaptive-verification --mtp-draft-tokens 3 --scheduler-policy throughput}
PREFILL_PRIORITY=${PREFILL_PRIORITY:-16}
RUN_LOG=${RUN_LOG:-$HOME/flare-run.log}

# Prefix cache on disk, off by default: with the eviction patch the GPU's 64 snapshot slots already keep 85% of each prompt
# cached for eight agents, and the disk store writes about 30 GB per benchmark run. A limit such as 20G turns it on.
PREFIX_CACHE_DISK=${PREFIX_CACHE_DISK:-$HOME/.cache/flare/prefix-cache}
PREFIX_CACHE_DISK_LIMIT=${PREFIX_CACHE_DISK_LIMIT:-0}

# Optional systemd --user unit that also uses the GPU: stopped before Flare starts, started again by stop-flare.sh.
STOP_SERVICE=${STOP_SERVICE:-}

RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

require_package() {
  if [ -z "$PACKAGE" ] || [ ! -d "$PACKAGE" ]; then
    echo "No exported package${PACKAGE:+ at $PACKAGE}. Run export.sh first, or set PACKAGE to a cache entry."
    exit 1
  fi
}

# Free unified memory in GiB; Flare needs about 100 back before it can load.
mem_available_gib() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_for_memory() {
  for _ in $(seq 1 120); do [ "$(mem_available_gib)" -ge 100 ] && return 0; sleep 1; done
  return 1
}
