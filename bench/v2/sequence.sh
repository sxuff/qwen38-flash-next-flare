#!/bin/bash
# Run a list of v2 cells in order, one fresh engine per line: sequence.sh PLAN_FILE
# Each plan line: ARM WORKLOAD RUN [LEVELS...]   e.g.  "A agents 1 1 2 4 8"   "B single 2"
# Lines are run in the order given (interleave arms, as v1's sweeps did with O M M O, so drift over the day does not
# favour one arm). A line whose receipt already exists is skipped, so a stopped sequence can be restarted.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/v2/raw}
while read -r arm workload run levels; do
  [ -z "${arm:-}" ] || [ "${arm:0:1}" = "#" ] && continue
  if [ "$workload" = agents ]; then tag="$arm-agents-L${levels// /-}-r$run"; set -- --levels $levels; else tag="$arm-$workload-r$run"; set --; fi
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag (done)"; continue; }
  python3 bench/v2/driver.py --arm "$arm" --workload "$workload" --runs 1 --first-run "$run" --out "$OUT" "$@" < /dev/null \
    || { echo "SEQUENCE_FAILED at $tag"; exit 1; }
done < "$1"
echo SEQUENCE_DONE
