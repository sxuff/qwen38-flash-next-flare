#!/bin/bash
# First use only. The package is signed with your own local development key, which Flare asks you to approve once.
# Run this in an interactive terminal, check the fingerprint, type the namespace when asked, and stop it with Ctrl-C once
# the model starts loading. Flare remembers the approval in ~/.config/flare/trust.toml.
set -euo pipefail
. "$(dirname "$0")/config.sh"
"$FLARE_BIN" inspect --json "$PACKAGE" | python3 -c '
import json, sys
p = json.load(sys.stdin)["package"]["publisher"]
print("Signing key", p["fingerprint"], "namespace", p["namespace"])'
exec "$FLARE_BIN" run "$PACKAGE" --gpus 0 --host 127.0.0.1 --port 8009
