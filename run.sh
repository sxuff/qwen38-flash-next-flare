#!/usr/bin/env bash
# run.sh - one-command setup and launch for this recipe on one NVIDIA GB10.
#
#   git clone https://github.com/sxuff/qwen38-flash-next-flare && cd qwen38-flash-next-flare && ./run.sh
#
# Every pin and flag below comes from this repo (mostly scripts/config.sh) or its README; the
# comment next to each value names its source. run.sh drives the repo's own scripts in the order
# the README gives them (setup -> download -> export -> approve key -> start) and changes no setting.
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

# ===================== recipe config (per repo) =============================
RECIPE_NAME="qwen38-flash-next-flare"
. scripts/config.sh                              # FLARE_*, CHECKPOINT*, PACKAGE*, SERVED_NAME, HOST, PORT, SERVE_FLAGS ...
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH   # same PATH as scripts/setup.sh:15
MODELS_DIR="${MODELS_DIR:-$HOME/models}"         # unused: Flare reads the checkpoint from the Hugging Face cache
SRC_DIR="${SRC_DIR:-$(dirname "$FLARE_SRC")}"    # Flare is built in FLARE_SRC (scripts/config.sh:6)
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-flash-next-flare}"  # same folder as PACKAGE_FILE (scripts/config.sh:13)
NEEDS_TOOLS="git curl sha256sum sha1sum python3 tmux uv cargo"  # uv and cargo: scripts/setup.sh:18-19 (no uv version is pinned in this repo)
NEED_MEM_BYTES=107374182400                      # 100 GiB available before Flare can load: scripts/config.sh:44-47
ENGINE_KIND=process
CONTAINER_NAME=""
READY_URL="http://127.0.0.1:${PORT}/health"      # clients/claude-code/claude-flare.sh:15
READY_TIMEOUT=1200                               # README.md:175 says ready in about 30 s; a margin, not a measured value
EXPECTED_LINE="83.0 tok/s total output with 8 concurrent agents (README.md:43,59; no receipt JSON in this repo)"
RECEIPT_PATH="README.md (figure: results/figures/eight-agents.png)"

# Checkpoint files at the pinned revision (scripts/config.sh:11-12). The repo pins the revision but
# no per-file checksums, so these come from the Hub's own metadata at that revision (Hub check
# 2026-11-11): sha256 = the Git LFS object id, gitsha1 = the git blob id of the small files.
# Format: <kind:hash> <bytes> <path-in-repo>
CHECKPOINT_FILES="
gitsha1:f473a27636acad7a1128a8f5db50339419f17613 166 .gitattributes
gitsha1:44aa98ae0cae81b2fb531268253c445e48f7b7fc 29021 README.md
gitsha1:c0c686f9c38d70d179fb7b5f5aa7530bc913dda3 8952 chat_template.jinja
gitsha1:1efa9baa6f332ef821a9468eddd0414ee9bfefb0 75217 config.json
gitsha1:023756cfadf88e5bf69eefeee3e172f38c448d64 202 generation_config.json
gitsha1:a494e019ca1502219fd0128658b979e5f05ae8e8 3353259 merges.txt
sha256:5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1 47280974 model.safetensors.index.json
gitsha1:2ea84a437d448ff71b08df68fdd949d5cc4ebb64 390 preprocessor_config.json
sha256:ebf3648c5f4cd1f24e79789e7803498dc4e74cb52515cc515b2d22cf8161953b 5000643504 tensors/model-00001-of-00021.safetensors
sha256:79a643fc5c024089cecd32ca27dc43e671cbec70ff3628b728909b7656ac43ec 5002792096 tensors/model-00002-of-00021.safetensors
sha256:c45f942203db4c7f5a1071700b93b908495cc790271acddfff39091c2b953923 5002927400 tensors/model-00003-of-00021.safetensors
sha256:071ab9cf963ca0f1a6a194b4cec485f5de18ef9804da915b2e6abd0eb0c707af 5003194672 tensors/model-00004-of-00021.safetensors
sha256:f80d17870eb1d36917257494770984953476583c57737f61fdbd341c957b9715 5003151448 tensors/model-00005-of-00021.safetensors
sha256:3e9183d4ebb267fa92429184d5132d302cb1a98fd049149e5d01dccd4ca65795 5002892148 tensors/model-00006-of-00021.safetensors
sha256:aa1ad959c4d62e0bc4600fd0b57cc2939b463e9fb53dc01020e188552b52ce5a 5002907708 tensors/model-00007-of-00021.safetensors
sha256:0b18844027be74b84b240f23bf3a744bc29b35de0bd384f41c40cd85e6339326 5003103756 tensors/model-00008-of-00021.safetensors
sha256:2d0b6c66790c17f8b1fdcf442f541778c8f0a381aca26c11b9d7cfcf7c96f2f0 5002976316 tensors/model-00009-of-00021.safetensors
sha256:3ed9bd73745aebf6dc136f437b67d6b9fc3d38d25ad84300b399164bc87a72b7 5002834136 tensors/model-00010-of-00021.safetensors
sha256:4936ff28c6cc8b66653ddb9cae28bc284569a1a2f7116819a017dc1a0e517ed8 5002685828 tensors/model-00011-of-00021.safetensors
sha256:dd7df5a6ff73e9960412243f744eab9ec35f4be40da5bb158dc4ba8796b33742 5002912920 tensors/model-00012-of-00021.safetensors
sha256:28a455c8ea4ebaa847fb4cbed2bc1de9eff0b501dd3f848a3a3cfc38778e8500 5003237060 tensors/model-00013-of-00021.safetensors
sha256:6941676e685abf9817d9c48a17fdeda20b2b72d16217555252aa4c602b7f98b8 4845006312 tensors/model-00014-of-00021.safetensors
sha256:2a1129b5acfc24a45bf402aa63dcef08b531aedfdaf5233c65fd3d12dc303fdd 4860574920 tensors/model-00015-of-00021.safetensors
sha256:635e97fbad78fa988647d407e2b58e55d551ef94d798c9b6fb5550de25e86429 4950031028 tensors/model-00016-of-00021.safetensors
sha256:bf60d564d89a805b67f8be5f2847a1b1567a70dbc23d19754395c127568474ad 4950030896 tensors/model-00017-of-00021.safetensors
sha256:c5bde22f9364e82eb25973456795592fdcb4b158bc2095898979a2566bf4c562 4950030896 tensors/model-00018-of-00021.safetensors
sha256:da3dc3b4f428f5b9dd53362691043d0242f48d03adca9b7b553bd9f3a1e96350 4950030904 tensors/model-00019-of-00021.safetensors
sha256:4978bb6b8480cfc02d3c8bc84ba86202889f5f10d090bb17c21fee663cea7683 4950030904 tensors/model-00020-of-00021.safetensors
sha256:48203838f2c76921bddbf1cc51d20a23858c128821a21add55fa1778e5e25283 2700016864 tensors/model-00021-of-00021.safetensors
sha256:0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 12809320 tokenizer.json
gitsha1:5de744b3fca2129d7186979ae47c06be33903243 17928 tokenizer_config.json
gitsha1:3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe 385 video_preprocessor_config.json
gitsha1:0aa0ce0658d60ac4a5d609f4eadb0e8e43514176 6722759 vocab.json
"
HF_HUB_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
CKPT_CACHE="$HF_HUB_DIR/models--${CHECKPOINT//\//--}"
SNAPSHOT="$CKPT_CACHE/snapshots/$CHECKPOINT_REVISION"
WEIGHTS=""   # the checkpoint goes through weights_step below, into the Hugging Face cache that Ignition reads

# Disk: the remaining checkpoint bytes, plus the 120 GB that export.sh insists on before exporting
# (scripts/export.sh:13) when no exported package exists yet. README.md:111: about 230 GB in all.
CKPT_TOTAL=$(awk 'NF==3 {s+=$2} END {print s+0}' <<< "$CHECKPOINT_FILES")
_ckpt_have=0; [ -d "$CKPT_CACHE/blobs" ] && _ckpt_have=$(du -sb "$CKPT_CACHE/blobs" | cut -f1)
package_ready() { [ -n "$PACKAGE" ] && [ -d "$PACKAGE" ]; }
if package_ready; then ENGINE_DISK_BYTES=0   # already exported: the checkpoint is no longer needed (README.md:111)
else ENGINE_DISK_BYTES=$(( (CKPT_TOTAL > _ckpt_have ? CKPT_TOTAL - _ckpt_have : 0) + 120 * 1073741824 )); fi

# Locked memory: Flare needs an unlimited memlock limit from a PAM login (console or OpenSSH, not
# Tailscale SSH), unless the 'flare' tmux server already exists with one
# (README.md:115-128,175; scripts/start-flare.sh:20-33).
memlock_ok() {
  local spid; spid=$(tmux -L flare display-message -p -t flare '#{pid}' 2>/dev/null || true)
  if [ -n "$spid" ]; then [ "$(awk '/locked memory/{print $4}' "/proc/$spid/limits")" = unlimited ]
  else [ "$(ulimit -l)" = unlimited ]; fi
}

flare_built() {
  [ -x "$FLARE_BIN" ] && [ "$(git -C "$FLARE_SRC" rev-parse HEAD 2>/dev/null)" = "$FLARE_COMMIT" ] || return 1
  local p; for p in "$REPO_DIR"/patches/*.patch; do git -C "$FLARE_SRC" apply --reverse --check "$p" 2>/dev/null || return 1; done
}

engine_step() {
  if memlock_ok; then ok "locked memory: unlimited"
  else fail "locked-memory limit is $(ulimit -l) KB, not unlimited. See 'Prepare the host' in the README, then run ./run.sh from the console or OpenSSH (Tailscale SSH skips PAM limits)."; fi
  if flare_built; then ok "Flare built at ${FLARE_COMMIT:0:12} with the recipe patches: $FLARE_BIN"; return; fi
  doit bash scripts/setup.sh   # pinned commit + patches/*.patch, uv sync --locked, cargo build --release
}

# Each checkpoint file is checked against CHECKPOINT_FILES once; a marker then skips the hashing.
verify_checkpoint() {
  local kind_hash size path f got
  while read -r kind_hash size path; do
    [ -z "${kind_hash:-}" ] && continue
    f="$SNAPSHOT/$path"
    [ "$(stat -L -c %s "$f" 2>/dev/null)" = "$size" ] || { echo "    size mismatch or missing: $path"; return 1; }
    case "$kind_hash" in
      sha256:*)  got="sha256:$(sha256sum < "$f" | cut -d' ' -f1)" ;;
      gitsha1:*) got="gitsha1:$( { printf 'blob %s\0' "$size"; cat "$f"; } | sha1sum | cut -d' ' -f1)" ;;
      *) echo "    unknown hash kind: $kind_hash"; return 1 ;;
    esac
    [ "$got" = "$kind_hash" ] || { echo "    checksum mismatch: $path (expected $kind_hash, got $got)"; return 1; }
  done <<< "$CHECKPOINT_FILES"
}

weights_step() {
  local mark="$STATE_DIR/checkpoint-${CHECKPOINT_REVISION:0:12}.verified"
  if [ "$CHECKPOINT" != nvidia/Qwen3.8-Flash-Next-FP4-Local ] || [ "$CHECKPOINT_REVISION" != b9ce18a6887e18c089fdea8509a33e63429a3300 ]; then
    fail "CHECKPOINT or CHECKPOINT_REVISION was overridden; the checksums in run.sh cover nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6 only"
  elif [ -f "$mark" ] && [ -d "$SNAPSHOT" ]; then ok "checkpoint verified earlier: $SNAPSHOT"
  elif package_ready; then ok "checkpoint not needed: the package is already exported"
  elif [ "$CHECK" = 1 ]; then
    plan "download $CHECKPOINT@${CHECKPOINT_REVISION:0:12} (33 files, $(human "$CKPT_TOTAL")) into $HF_HUB_DIR and verify every file"
  else
    # The download of scripts/download.sh, but with the hf CLI that Flare's own uv.lock pins
    # (huggingface-hub 1.33.0 at ca1f960) instead of an unpinned uvx. hf reads HF_TOKEN from the environment.
    "$FLARE_SRC/.venv/bin/hf" download "$CHECKPOINT" --revision "$CHECKPOINT_REVISION" --max-workers 16
    printf '    hash  33 files (a few minutes)\n'
    verify_checkpoint || die "checkpoint verification failed; nothing was deleted ($SNAPSHOT)"
    mkdir -p "$STATE_DIR"; date -u +%FT%TZ > "$mark"; ok "checkpoint verified: $SNAPSHOT"
  fi
  # Export the runtime package: about 1.5 h the first time (scripts/export.sh:2-5).
  if package_ready; then ok "exported package: $PACKAGE"
  elif [ "$CHECK" = 1 ]; then plan "bash scripts/export.sh   (records the package in $PACKAGE_FILE)"
  else bash scripts/export.sh; PACKAGE=$(cat "$PACKAGE_FILE"); package_ready || die "export.sh did not record a package"; fi
  # First use only: you approve your own development key and the NVIDIA Open Model Agreement
  # yourself (scripts/approve-key.sh:2-5). run.sh never answers that prompt for you.
  if [ ! -x "$FLARE_BIN" ] || ! package_ready; then plan "check that the package's signing key is approved in ~/.config/flare/trust.toml"; return; fi
  local fp; fp=$("$FLARE_BIN" inspect --json "$PACKAGE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["package"]["publisher"]["fingerprint"])')
  if grep -qF "$fp" "$HOME/.config/flare/trust.toml" 2>/dev/null; then ok "signing key approved: $fp"
  else fail "first use: approve the package's key and licence yourself with 'bash scripts/approve-key.sh' (README.md:164; Ctrl-C once the model loads), then run ./run.sh again"; fi
}

serve_step() {
  if [ "$CHECK" = 1 ]; then plan "start Flare with scripts/start-flare.sh (tmux -L flare, pane flare:0): $SERVE_FLAGS, prefill priority $PREFILL_PRIORITY, port $PORT"; return; fi
  RUN_LOG="$LOGFILE" bash scripts/start-flare.sh
  local pid=""; for _ in $(seq 1 60); do pid=$(pgrep -n -f "^$FLARE_BIN run " || true); [ -n "$pid" ] && break; sleep 1; done
  [ -n "$pid" ] || die "Flare did not start; see $LOGFILE (tmux -L flare attach -t flare:0)"
  echo "$pid" > "$PIDFILE"; ok "started (pid $pid; watch it with: tmux -L flare attach -t flare:0)"
}

stop_override() {
  doit bash scripts/stop-flare.sh   # leaves the tmux server up, so later starts work from any login (README.md:175)
  [ "$CHECK" = 1 ] || rm -f "$PIDFILE"
}

curl_test() {     # README.md:188-192, plus "top_k": 1 for greedy (README.md:195: temperature 0 is refused)
  cat <<EOT
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "${SERVED_NAME}",
  "messages": [{"role": "user", "content": "Explain the ISCO of a Kerr black hole in two sentences."}],
  "max_tokens": 512, "top_k": 1
}'
EOT
}
# ===================== end of recipe config =================================

# ---------------------------------------------------------------------------
# Shared machinery (identical in every sxuff recipe repo).
# ---------------------------------------------------------------------------
CHECK=0; ACTION=run
usage() {
  cat <<EOF
Usage: ./run.sh [--check] [--stop] [--help]

Sets up and launches ${RECIPE_NAME} on one NVIDIA GB10 (SM121).

  (no flag)  check the machine, build or pull the pinned engine, download the
             pinned weights, verify SHA-256, start the server, print a test.
             Safe to re-run: finished steps are skipped.
  --check    print what would happen; change nothing.
  --stop     stop the server this script started.
  --help     show this text.

Environment:
  HF_TOKEN     Hugging Face token, read from the environment only, never saved.
  MODELS_DIR   weights        (default: ${MODELS_DIR})
  SRC_DIR      engine source  (default: ${SRC_DIR})
  STATE_DIR    pid and log    (default: ${STATE_DIR})
  PORT         server port    (default: ${PORT})

Expected: ${EXPECTED_LINE}
Receipt:  ${RECEIPT_PATH}
EOF
}
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --stop)  ACTION=stop ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a (see ./run.sh --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    ok    %s\n' "$*"; }
plan() { printf '    would %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }
CHECK_FAILED=0   # under --check, a failed pre-flight check is reported and the plan continues
fail() { if [ "$CHECK" = 1 ]; then printf '    FAIL  %s\n' "$*"; CHECK_FAILED=1; else die "$*"; fi; }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
doit() { if [ "$CHECK" = 1 ]; then plan "$*"; else "$@"; fi; }   # mutate, or describe under --check

PIDFILE="${STATE_DIR}/server.pid"
LOGFILE="${STATE_DIR}/server.log"
STOP_CMD="cd ${REPO_DIR} && ./run.sh --stop"

server_running() {
  if [ "$ENGINE_KIND" = docker ]; then
    [ -n "$(docker ps -q --filter "name=^${CONTAINER_NAME}\$" 2>/dev/null)" ]
  else
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
  fi
}

stop_step() {   # may be overridden in the config block (e.g. to call scripts/stop.sh)
  if ! server_running; then ok "nothing started by this script is running"; return; fi
  if [ "$ENGINE_KIND" = docker ]; then doit docker stop "$CONTAINER_NAME"
  else
    local pid; pid="$(cat "$PIDFILE")"; doit kill "$pid"
    if [ "$CHECK" = 0 ]; then
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; rm -f "$PIDFILE"
    fi
  fi
  ok "stopped"
}
declare -f stop_override >/dev/null && stop_step() { stop_override; }

if [ "$ACTION" = stop ]; then say "Stopping ${RECIPE_NAME}"; stop_step; exit 0; fi
[ "$CHECK" = 1 ] && say "--check: nothing will be changed"

# git checkout of an exact commit, idempotent. Usage: git_pin <url> <sha> <dir>
git_pin() {
  local url="$1" sha="$2" dir="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then ok "source at ${sha:0:12}: $dir"; return; fi
  if [ ! -d "$dir/.git" ]; then doit git clone --quiet "$url" "$dir"; fi
  if [ "$CHECK" = 1 ]; then plan "check out $sha in $dir"; return; fi
  [ -z "$(git -C "$dir" status --porcelain)" ] || die "$dir has local changes; refusing to move it to $sha"
  git -C "$dir" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet --detach "$sha"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "could not check out $sha in $dir"
  ok "source at ${sha:0:12}: $dir"
}

# docker image by digest, idempotent. Usage: image_pin <repo@sha256:...>
image_pin() {
  if docker image inspect "$1" >/dev/null 2>&1; then ok "image present: $1"; else doit docker pull "$1"; fi
}

# --- 1. machine ------------------------------------------------------------
say "Machine"
[ "$(uname -m)" = aarch64 ] || die "expected aarch64 (GB10), got $(uname -m)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
GPU="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
case "$GPU" in *GB10*", 12.1") ok "GPU: $GPU" ;; *) die "measured on an NVIDIA GB10 (SM121); found: ${GPU:-none}" ;; esac
MISSING=""; for t in $NEEDS_TOOLS; do command -v "$t" >/dev/null || MISSING="$MISSING $t"; done
if [ -n "$MISSING" ]; then fail "missing tool(s):$MISSING"; else ok "tools: $NEEDS_TOOLS"; fi
ALREADY=0; server_running && ALREADY=1
MEM_AVAIL=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 ))
if [ "$ALREADY" = 1 ]; then ok "server already running; memory check skipped"
elif [ "$MEM_AVAIL" -lt "$NEED_MEM_BYTES" ]; then
  fail "need $(human "$NEED_MEM_BYTES") available memory, have $(human "$MEM_AVAIL"). Stop other models first (nvidia-smi, docker ps)."
else ok "memory: $(human "$MEM_AVAIL") available, need $(human "$NEED_MEM_BYTES")"; fi
if [ "$ALREADY" = 1 ]; then ok "port ${PORT}: served by this recipe"
else
  # The server binds 127.0.0.1 (or the host network). A listener on 127.0.0.1, a wildcard or ::1
  # blocks it; one bound to another interface address does not, and is only reported.
  PORT_ALL="$(ss -ltnH "sport = :${PORT}" 2>/dev/null || true)"
  PORT_CLASH="$(printf '%s\n' "$PORT_ALL" | awk '{print $4}' | grep -E '^(127\.0\.0\.1|0\.0\.0\.0|\*|\[::\]|\[::1\]|\[::ffff:127\.0\.0\.1\]):' || true)"
  PORT_DOCKER="$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -E "(127\.0\.0\.1|0\.0\.0\.0|\[::\]|:::?):${PORT}->" || true)"
  if [ -n "$PORT_CLASH$PORT_DOCKER" ]; then
    printf '    port %s is taken:\n%s\n%s\n' "$PORT" "$PORT_CLASH" "$PORT_DOCKER"
    fail "free port ${PORT} or run: PORT=<other> ./run.sh"
  elif [ -n "$PORT_ALL" ]; then
    ok "port ${PORT}: free on 127.0.0.1 (also bound elsewhere: $(printf '%s' "$PORT_ALL" | awk '{print $4}' | tr '\n' ' '))"
  else ok "port ${PORT}: free"; fi
fi

# --- 2. disk ---------------------------------------------------------------
say "Disk"
nearest_dir() { local d="$1"; while [ ! -d "$d" ]; do d="$(dirname "$d")"; done; echo "$d"; }
NEED_DISK=$ENGINE_DISK_BYTES
while read -r sha size repo rev rpath dest; do
  [ -z "${sha:-}" ] && continue
  have=0; [ -f "$MODELS_DIR/$dest" ] && have=$(stat -c %s "$MODELS_DIR/$dest")
  [ "$have" -lt "$size" ] && NEED_DISK=$(( NEED_DISK + size - have ))
done <<< "$WEIGHTS"
FREE_DISK=$(df -B1 --output=avail "$(nearest_dir "$MODELS_DIR")" | tail -1 | tr -d ' ')
if [ "$FREE_DISK" -ge "$NEED_DISK" ]; then ok "need $(human "$NEED_DISK") more, have $(human "$FREE_DISK")"
else fail "need $(human "$NEED_DISK") more disk under $MODELS_DIR, have $(human "$FREE_DISK"). Nothing was deleted."; fi

# --- 3. engine -------------------------------------------------------------
say "Engine"
engine_step

# --- 4. weights: pinned revision + SHA-256 -----------------------------------
hf_fetch() {   # repo rev path dest. The token reaches curl on stdin: never in argv or a file.
  { [ -n "${HF_TOKEN:-}" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; true; } |
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - --config - \
         --output "$4.part" "https://huggingface.co/$1/resolve/$2/$3"
  mv "$4.part" "$4"
}
default_weights_step() {
  while read -r sha size repo rev rpath dest; do
    [ -z "${sha:-}" ] && continue
    local f="$MODELS_DIR/$dest" mark="$MODELS_DIR/$dest.sha256-ok"
    if [ -f "$mark" ] && [ "$(cat "$mark")" = "$sha $size" ] && [ "$(stat -c %s "$f" 2>/dev/null)" = "$size" ]; then
      ok "verified earlier: $dest"; continue
    fi
    if [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" != "$size" ]; then
      if [ "$CHECK" = 1 ]; then plan "download $repo@${rev:0:12} $rpath ($(human "$size"))"; continue; fi
      mkdir -p "$(dirname "$f")"; printf '    get   %s (%s)\n' "$rpath" "$(human "$size")"
      hf_fetch "$repo" "$rev" "$rpath" "$f"
    elif [ "$CHECK" = 1 ]; then plan "verify SHA-256 of $dest"; continue; fi
    local got; got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" != "$sha" ]; then mv "$f" "$f.sha256-mismatch"; die "SHA-256 mismatch for $dest: expected $sha, got $got (kept as $f.sha256-mismatch)"; fi
    echo "$sha $size" > "$mark"; ok "sha256 ok: $dest"
  done <<< "$WEIGHTS"
}
declare -f weights_step >/dev/null || weights_step() { default_weights_step; }
say "Weights"
weights_step

# --- 5. serve --------------------------------------------------------------
say "Server"
if [ "$ALREADY" = 1 ]; then ok "already running"
elif [ "$CHECK" = 1 ]; then serve_step
else
  mkdir -p "$STATE_DIR"; serve_step
  printf '    wait  for %s (up to %ss; log: %s)\n' "$READY_URL" "$READY_TIMEOUT" "$LOGFILE"
  t=0
  until curl -fsS -o /dev/null "$READY_URL" 2>/dev/null; do
    server_running || die "server exited during start-up; see $LOGFILE"
    [ "$t" -ge "$READY_TIMEOUT" ] && die "not ready after ${READY_TIMEOUT}s; see $LOGFILE"
    sleep 5; t=$((t + 5))
  done
  ok "ready after ${t}s"
fi

# --- 6. done ---------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  if [ "$CHECK_FAILED" = 1 ]; then say "Check finished with FAIL lines above; nothing was changed. ./run.sh would stop at the first one."; exit 1; fi
  say "Check finished; nothing was changed. Run ./run.sh to do it."; exit 0
fi
say "Test it"
curl_test
say "Stop it"
echo "$STOP_CMD"
echo
echo "Expected: ${EXPECTED_LINE} - receipt: ${RECEIPT_PATH}"
