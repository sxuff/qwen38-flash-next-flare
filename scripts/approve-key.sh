#!/bin/bash
# First use only. The package is signed with your own local development key, which Flare asks you to approve once; the same
# answer agrees to the NVIDIA Open Model Agreement that the package carries from the checkpoint.
# Run this in an interactive terminal, check the fingerprint, type the namespace when asked, and stop it with Ctrl-C once
# the model starts loading. Flare remembers the approval in ~/.config/flare/trust.toml.
set -euo pipefail
. "$(dirname "$0")/config.sh"
require_package
"$FLARE_BIN" inspect --json "$PACKAGE" | python3 -c '
import json, sys
p = json.load(sys.stdin)["package"]
print("Signing key", p["publisher"]["fingerprint"], "namespace", p["publisher"]["namespace"])
if p.get("license"):
    print("License", p["license"]["name"], p["license"].get("link", ""))'
exec "$FLARE_BIN" run "$PACKAGE" --gpus 0 --host 127.0.0.1 --port 8009
