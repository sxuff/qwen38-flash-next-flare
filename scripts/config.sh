# Shared settings for the recipe scripts. Every value can be overridden from the environment.

# Which measured configuration to build and serve (README, "Which config should I run?"):
#   MODE=agents  the fastest at 8 concurrent agents (the default)
#   MODE=single  the fastest for one request at a time
# Each mode names a PROFILE below; PROFILE can also be set directly.
MODE=${MODE:-agents}
case "$MODE" in
  agents) DEFAULT_PROFILE=flare-mtp-mixed ;;
  single) DEFAULT_PROFILE=flare-mtp-mixed ;;
  *) echo "unknown MODE=$MODE (agents or single)" >&2; return 2 2>/dev/null || exit 2 ;;
esac
PROFILE=${PROFILE:-$DEFAULT_PROFILE}

# Engine profiles, as measured in results/v2 (bench/v2/arms.json has the same flags).
#   flare-mtp-mixed       v1 recipe and v2 arm A: Flare ca1f960 + patches/, MTP, mixed prefill
#   hyperflare-mtp-mixed  v2 arm B: Hyperflare 8a887202 + patches/hyperflare/, MTP, mixed prefill
#   hyperflare-mtp        v2 arm C: as B with separate prefill and verification graphs
#   hyperflare-dflash2    v2 arm D: as C with the DFlash2 drafter instead of the MTP module
case "$PROFILE" in
  flare-mtp-mixed)
    P_REPO=https://github.com/NVIDIA/flare-local-inference.git; P_COMMIT=ca1f960edbf67d624f2d2948d2c111e8a6c8eb3c
    P_SRC=$HOME/flare-local-inference; P_PATCHES=patches; P_CLI=flare
    P_EXPORT="--mtp-draft-tokens 5 --speculative-token-capacity 48 --mixed-prefill"
    P_SERVE="--no-mtp-adaptive-verification --mtp-draft-tokens 3 --scheduler-policy throughput" ;;
  hyperflare-mtp-mixed|hyperflare-mtp|hyperflare-dflash2)
    P_REPO=https://github.com/NVIDIA/hyperflare-inference.git; P_COMMIT=8a887202a0ee55d456ddcf7510161e22865505bc
    P_SRC=$HOME/hyperflare-inference; P_PATCHES=patches/hyperflare; P_CLI=flare-cli
    P_SERVE="--no-mtp-adaptive-verification --mtp-draft-tokens 3 --scheduler-policy throughput"
    case "$PROFILE" in
      hyperflare-mtp-mixed) P_EXPORT="--mtp-draft-tokens 5 --speculative-token-capacity 48 --mixed-prefill" ;;
      hyperflare-mtp)       P_EXPORT="--mtp-draft-tokens 5 --speculative-token-capacity 48" ;;
      hyperflare-dflash2)   P_EXPORT="--speculative-method dflash2 --mtp-draft-tokens 6 --speculative-token-capacity 56"
                            P_SERVE="--no-mtp-adaptive-verification --mtp-draft-tokens DFLASH2_DEPTH --scheduler-policy throughput" ;;
    esac ;;
  *) echo "unknown PROFILE=$PROFILE" >&2; return 2 2>/dev/null || exit 2 ;;
esac

# Engine source, pinned to the commit the profile was measured on. Hyperflare is flare-local-inference renamed;
# its `flare` command moved from flare/ to flare-cli/.
FLARE_REPO=${FLARE_REPO:-$P_REPO}
FLARE_COMMIT=${FLARE_COMMIT:-$P_COMMIT}
FLARE_SRC=${FLARE_SRC:-$P_SRC}
PATCH_DIR=${PATCH_DIR:-$P_PATCHES}
FLARE_CLI_DIR=${FLARE_CLI_DIR:-$P_CLI}
FLARE_BIN=${FLARE_BIN:-$FLARE_SRC/$FLARE_CLI_DIR/target/release/flare}

# NVIDIA's NVFP4 checkpoint and the package exported from it. Ignition exports into Flare's model cache
# (~/.cache/flare/packages); export.sh records the new cache entry's path in PACKAGE_FILE, one per profile.
CHECKPOINT=${CHECKPOINT:-nvidia/Qwen3.8-Flash-Next-FP4-Local}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}
STATE_HOME=${STATE_HOME:-$HOME/.local/state/qwen38-flash-next-flare}
if [ "$PROFILE" = flare-mtp-mixed ]; then PACKAGE_FILE=${PACKAGE_FILE:-$STATE_HOME/package}   # v1's file, unchanged
else PACKAGE_FILE=${PACKAGE_FILE:-$STATE_HOME/package-$PROFILE}; fi
PACKAGE=${PACKAGE:-$(cat "$PACKAGE_FILE" 2>/dev/null || true)}
EXPORT_FLAGS=${EXPORT_FLAGS:---max-concurrency 8 --image-concurrency 4 $P_EXPORT}

# DFlash2 drafter, pinned by Hyperflare itself (ignition/recipes/qwen38/draft_config.py); export.sh checks its sha256.
DRAFTER=incoai/Qwen3.8-Flash-Next-DFlash2
DRAFTER_REVISION=a22a1744e85c17364dcc54a6a420af7135d4c7f6
DRAFTER_SHA256=d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f

# Serving. Three drafts every round measured faster than adaptive verification for agents. The throughput scheduler and
# a prompt priority of 16 start each new turn sooner. PREFILL_PRIORITY becomes FLARE_PREFILL_PRIORITY, which
# flare-prefill-priority.patch reads; 1 is upstream's pricing.
SERVED_NAME=${SERVED_NAME:-qwen38-flash-next-flare}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVE_FLAGS=${SERVE_FLAGS:-$P_SERVE}
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
    echo "No exported package${PACKAGE:+ at $PACKAGE} for profile $PROFILE. Run export.sh first, or set PACKAGE to a cache entry."
    exit 1
  fi
}

# Free unified memory in GiB; Flare needs about 100 back before it can load.
mem_available_gib() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
wait_for_memory() {
  for _ in $(seq 1 120); do [ "$(mem_available_gib)" -ge 100 ] && return 0; sleep 1; done
  return 1
}
