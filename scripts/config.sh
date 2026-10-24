# Shared settings for the recipe scripts. Every value can be overridden from the environment.

# Flare and Ignition, pinned to the commit this recipe was measured on.
FLARE_REPO=${FLARE_REPO:-https://github.com/NVIDIA/flare-local-inference.git}
FLARE_COMMIT=${FLARE_COMMIT:-1fd1f917fcf2b0a48068e1f9e58be8db2165b383}
FLARE_SRC=${FLARE_SRC:-$HOME/flare-local-inference}
FLARE_BIN=${FLARE_BIN:-$FLARE_SRC/flare/target/release/flare}

# NVIDIA's NVFP4 checkpoint and the package exported from it.
CHECKPOINT=${CHECKPOINT:-nvidia/Qwen3.8-Flash-Next-FP4-Local}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}
PACKAGE=${PACKAGE:-$HOME/flare-packages/qwen38-gb10-c8-i4-mtp5-cap48-mixed-topk.safetensors}

# Serving. Three drafts every round measured faster than adaptive verification for agents.
SERVED_NAME=${SERVED_NAME:-qwen38-flash-next-flare}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVE_FLAGS=${SERVE_FLAGS:---no-mtp-adaptive-verification --mtp-draft-tokens 3}
RUN_LOG=${RUN_LOG:-$HOME/flare-run.log}

# Optional systemd --user unit that also uses the GPU: stopped before Flare starts, started again by stop-flare.sh.
STOP_SERVICE=${STOP_SERVICE:-}

RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Free unified memory in GiB; Flare needs about 100 back before it can load.
mem_available_gib() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_for_memory() {
  for _ in $(seq 1 120); do [ "$(mem_available_gib)" -ge 100 ] && return 0; sleep 1; done
  return 1
}
