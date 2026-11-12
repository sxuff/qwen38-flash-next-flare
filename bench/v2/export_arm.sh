#!/bin/bash
# Export one v2 Hyperflare package: export_arm.sh ARM [ignition export flags...]
#   B: --max-concurrency 8 --image-concurrency 4 --mtp-draft-tokens 5 --speculative-token-capacity 48 --mixed-prefill
#   C: the same without --mixed-prefill
#   D: --max-concurrency 8 --image-concurrency 4 --speculative-method dflash2 --mtp-draft-tokens 6 --speculative-token-capacity 56
# Like scripts/export.sh: needs an unlimited locked-memory limit (run it inside the `tmux -L flare` server) and an
# idle GPU. Unlike it, it does not ask for 120 GB free: Hyperflare stores weights by content, so a package whose
# prepared weights match an earlier export adds about 1 GB. A watchdog stops the export if free disk on the cache's
# filesystem falls below MIN_FREE_GB, so a layout change cannot fill the system disk; nothing is deleted.
set -uo pipefail
ARM=$1; shift
HF_SRC=${HF_SRC:-$HOME/v2/hyperflare}
STATE=${STATE:-$HOME/v2/state}
CACHE=${FLARE_PACKAGE_CACHE:-$HOME/.cache/flare/packages}
MIN_START_GB=${MIN_START_GB:-40}
MIN_FREE_GB=${MIN_FREE_GB:-10}
LOG=$HOME/v2/logs/export-$ARM.log
mkdir -p "$STATE" "$(dirname "$LOG")"
export PATH=/usr/local/cuda/bin:$PATH
free_gb() { df -BG --output=avail "$CACHE" | tail -1 | tr -dc 0-9; }
[ "$(ulimit -l)" = unlimited ] || { echo "locked-memory limit is $(ulimit -l), not unlimited"; exit 1; }
[ "$(free_gb)" -ge "$MIN_START_GB" ] || { echo "only $(free_gb) GB free, need $MIN_START_GB to start"; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "the GPU is in use"; exit 1; fi

cd "$HF_SRC"
{ echo "arm=$ARM commit=$(git rev-parse HEAD) start=$(date -Is) free_gb=$(free_gb)"; git status --porcelain; } > "$LOG"
.venv/bin/ignition export qwen3.8-flash --gpus 0 --verbose "$@" >> "$LOG" 2>&1 &
pid=$!
low=0
while kill -0 $pid 2>/dev/null; do
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    echo "WATCHDOG: free disk $(free_gb) GB < $MIN_FREE_GB GB at $(date -Is); stopping the export" >> "$LOG"
    pkill -TERM -P $pid; kill -TERM $pid; low=1; break
  fi
  sleep 15
done
wait $pid; status=$?
[ $low = 1 ] && status=99
entry=$(sed -n 's/^Exported cache entry: //p' "$LOG" | tail -1)
echo "EXPORT_EXIT=$status end=$(date -Is) free_gb=$(free_gb) entry=${entry:-none}" >> "$LOG"
[ "$status" -eq 0 ] && [ -n "$entry" ] && echo "$entry" > "$STATE/$ARM.package"
exit $status
