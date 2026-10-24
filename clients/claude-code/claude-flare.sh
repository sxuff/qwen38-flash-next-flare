#!/bin/bash
# Run Claude Code against Flare on the Spark over an SSH tunnel (macOS and Linux).
#   claude-flare.sh [claude args...]   start Flare if needed, open the tunnel, run claude
#   claude-flare.sh --stop             stop Flare on the Spark
SPARK=${FLARE_SPARK:-user@spark}                        # key-based SSH
SPARK_LOGIN=${FLARE_SPARK_LOGIN:-$SPARK}                # a login with PAM limits
RECIPE_DIR=${FLARE_RECIPE_DIR:-'~/qwen38-flash-next-flare'}
LOCAL_PORT=${FLARE_LOCAL_PORT:-18000}
SETTINGS="$(cd "$(dirname "$0")" && pwd)/settings.json"

if [ "${1:-}" = --stop ]; then
  exec ssh "$SPARK" "bash $RECIPE_DIR/scripts/stop-flare.sh"
fi

healthy() { ssh -o BatchMode=yes "$SPARK" 'curl -sf http://127.0.0.1:8000/health' >/dev/null 2>&1; }
port_open() { nc -z 127.0.0.1 "$LOCAL_PORT" 2>/dev/null; }

if ! healthy; then
  echo "Flare is not running. Starting it on the Spark."
  # The tmux server keeps its memlock limit after Flare stops, so usually no password is needed.
  ssh -o BatchMode=yes "$SPARK" "bash $RECIPE_DIR/scripts/start-flare.sh --no-create"; rc=$?
  if [ $rc -eq 3 ]; then
    echo "First start since boot: log in once so Flare gets the unlimited memlock limit."
    ssh -t "$SPARK_LOGIN" "bash $RECIPE_DIR/scripts/start-flare.sh"; rc=$?
  fi
  [ $rc -eq 0 ] || { echo "Could not start Flare."; exit 1; }
  printf "Waiting for Flare to load"
  for _ in $(seq 1 48); do healthy && break; printf .; sleep 5; done
  healthy || { echo; echo "Flare did not become healthy; see the run log on the Spark."; exit 1; }
  echo " ready."
fi

tunnel=
if ! port_open; then
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "$LOCAL_PORT:127.0.0.1:8000" "$SPARK" &
  tunnel=$!
  for _ in $(seq 1 60); do port_open && break; sleep 0.3; done
  port_open || { echo "SSH tunnel failed to open."; exit 1; }
fi
trap '[ -n "$tunnel" ] && kill "$tunnel"' EXIT

# Variables inherited from another Claude session override the profile; run claude without them.
unset_args=()
for name in $(env | grep -oE '^(ANTHROPIC|CLAUDE)[A-Z0-9_]*'); do unset_args+=(-u "$name"); done
env "${unset_args[@]}" claude --settings "$SETTINGS" "$@"
