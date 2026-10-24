#!/bin/bash
# Stop Flare. The `tmux -L flare` server and its keep window stay up, so the next start-flare.sh needs no new login.
# If STOP_SERVICE is set, that service is started again once the memory is back.
. "$(dirname "$0")/config.sh"
pkill -TERM -f "/flare run " || true
for _ in $(seq 1 90); do pgrep -f "/flare run " >/dev/null || break; sleep 1; done
wait_for_memory || true
if [ -n "$STOP_SERVICE" ]; then
  systemctl --user start "$STOP_SERVICE"
  echo "Flare stopped; $STOP_SERVICE is $(systemctl --user is-active "$STOP_SERVICE")."
else
  echo "Flare stopped."
fi
