#!/bin/bash
# After a successful export, Ignition's package cache keeps a ~80 GB staging copy that the package does not use.
# The package's own blobs are hardlinked into its folder, so they have two links; blobs with a single link are staging.
# A later export reuses staging blobs and finishes faster, so keep them if you plan to export again.
# Lists by default; deletes only with --delete.
set -euo pipefail
BLOBS=${BLOBS:-$HOME/.cache/flare/packages/blobs}
count=$(find "$BLOBS" -type f -links 1 | wc -l)
size=$(find "$BLOBS" -type f -links 1 -printf "%s\n" | awk '{s += $1} END {printf "%.1f", s / 1e9}')
echo "$count unreferenced blobs, $size GB, in $BLOBS"
if [ "${1:-}" = --delete ]; then
  find "$BLOBS" -type f -links 1 -delete
  echo "Deleted."
fi
