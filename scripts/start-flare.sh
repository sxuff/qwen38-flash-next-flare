#!/bin/bash
# Start, or restart, Flare with the recipe package.
# Flare runs in pane flare:0 of a dedicated `tmux -L flare` server. Its `keep` window holds the server open, so the server
# keeps the unlimited locked-memory limit of the login that created it. Creating the server needs a regular login (console
# or OpenSSH; Tailscale SSH skips PAM limits). Once it exists, this works from any login.
#   start-flare.sh              start or restart Flare, creating the server if needed
#   start-flare.sh --no-create  exit 3 instead of creating the server
set -e
. "$(dirname "$0")/config.sh"
CMD="$FLARE_BIN run '$PACKAGE' --gpus 0 --served-model-name $SERVED_NAME --host $HOST --port $PORT $SERVE_FLAGS 2>&1 | tee $RUN_LOG; echo FLARE_EXITED >> $RUN_LOG"

spid=$(tmux -L flare display-message -p -t flare '#{pid}' 2>/dev/null || true)
if [ -n "$spid" ]; then
  lim=$(awk '/locked memory/{print $4}' "/proc/$spid/limits")
  if [ "$lim" != unlimited ]; then
    echo "The flare tmux server's locked-memory limit is $lim, not unlimited."
    echo "Stop it (tmux -L flare kill-server) and run this again from a console or OpenSSH login."
    exit 1
  fi
else
  [ "${1:-}" = --no-create ] && exit 3
  if [ "$(ulimit -l)" != unlimited ]; then
    echo "Locked-memory limit is $(ulimit -l) KB, not unlimited. Log in from the console or OpenSSH and run this again."
    exit 1
  fi
  tmux -L flare new-session -d -s flare -n flare 'sleep infinity'
fi
windows=$(tmux -L flare list-windows -t flare -F '#{window_index}')
echo "$windows" | grep -qx 9 || tmux -L flare new-window -d -t flare:9 -n keep 'sleep infinity'
echo "$windows" | grep -qx 0 || tmux -L flare new-window -d -t flare:0 -n flare 'sleep infinity'

[ -n "$STOP_SERVICE" ] && systemctl --user stop "$STOP_SERVICE"
# A running Flare exits first: its unified memory comes back some seconds later, and the new one needs it all.
pkill -TERM -f "/flare run " || true
for _ in $(seq 1 90); do pgrep -f "/flare run " >/dev/null || break; sleep 1; done
wait_for_memory || true
tmux -L flare set-option -w -t flare:0 remain-on-exit on
tmux -L flare respawn-pane -k -t flare:0 "$CMD"
echo "Flare is starting on http://$HOST:$PORT (log: $RUN_LOG; attach: tmux -L flare attach -t flare:0)."
