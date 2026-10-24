#!/usr/bin/env python3
"""Agent-style concurrency benchmark for an OpenAI-compatible server (chat API, streaming).

Each session is a 6-turn code review: a shared ~5K-token agent system prompt with tool
definitions, then one fresh ~1.5K-token source file per turn and a reply capped at 256
tokens. Sessions share only the system prompt; every session, level and repeat reads
different files, so no level reuses another's cache. Levels 1, 2, 4, 8 x 2 repeats.

The files come from the Flare checkout set up by scripts/setup.sh (FLARE_SRC), including its
.venv, which is how the recipe's results were measured; the pinned commit and uv lockfile make
the workload identical on another machine.
"""
import json, os, threading, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
SRC = Path(os.environ.get("FLARE_SRC", Path.home() / "flare-local-inference"))
OUT = Path(__file__).resolve().parent / "turns.json"
TURNS, MAX_TOKENS, CHUNK_CHARS = 6, 256, 5200
LEVELS, REPS = [1, 2, 4, 8], 2

SYSTEM = ("You are a careful coding agent reviewing the Flare inference server for correctness bugs. "
          "The user shows you one source file at a time. After each file, list anything suspicious "
          "with the line it is on, say how confident you are, and name the file you want to read next. "
          "Use the tools when you need more context. Keep answers short and concrete.\n\n"
          "# Repository guidance (AGENTS.md)\n\n" + (SRC / "AGENTS.md").read_text())
TOOLS = [{"type": "function", "function": {"name": n, "description": d,
                                           "parameters": {"type": "object", "properties": p, "required": list(p)}}}
         for n, d, p in [
             ("read_file", "Read a file from the repository.", {"path": {"type": "string"}}),
             ("search", "Search the repository with a regular expression.", {"pattern": {"type": "string"}}),
             ("edit_file", "Replace text in a file.", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}),
             ("run_command", "Run a shell command in the repository.", {"command": {"type": "string"}})]]

files = sorted(p for p in SRC.rglob("*")
               if p.suffix in (".rs", ".py") and ".git" not in p.parts and p.stat().st_size > CHUNK_CHARS)
CHUNKS = []
for p in files:
    text = p.read_text(errors="replace")
    for i in range(0, len(text) - CHUNK_CHARS + 1, CHUNK_CHARS):
        CHUNKS.append((str(p.relative_to(SRC)), text[i:i + CHUNK_CHARS]))
next_chunk = [0]


MODEL, SAMPLING = "qwen38-flash-next-flare", {}


def post_stream(messages, seed):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": MAX_TOKENS, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}, **SAMPLING}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = first_count = finish = None
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            now = time.monotonic()
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta") or {}
                if any(v for k, v in delta.items() if k != "role"):
                    if first is None:
                        first, first_count = now, usage.get("completion_tokens")
                    last = now
    end = time.monotonic()
    comp = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    decode = None
    if first_count is None and comp:
        first_count = 1  # server reports usage only at the end; assume one token in the first chunk
    if first and last and last > first and comp and first_count is not None:
        decode = (comp - first_count) / (last - first)
    return {"t0": t0, "end": end, "ttft_s": (first - t0) if first else None, "latency_s": end - t0, "decode_tps": decode,
            "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached, "completion_tokens": comp,
            "finish_reason": finish}


def session(level, rep, sid, chunks, start_at, rows, lock):
    msgs = [{"role": "system", "content": SYSTEM}]
    while time.monotonic() < start_at:
        time.sleep(0.0005)
    for turn, (path, code) in enumerate(chunks, 1):
        intro = f"Review session L{level}-R{rep}-S{sid}. " if turn == 1 else ""
        msgs.append({"role": "user", "content": f"{intro}Contents of {path}:\n```\n{code}\n```"})
        r = post_stream(msgs, seed=1000 * level + 100 * rep + 10 * sid + turn)
        msgs.append({"role": "assistant", "content": "Noted. Show me the next file."})
        row = {"level": level, "rep": rep, "session": sid, "turn": turn, "file": path,
               "start_s": r["t0"] - start_at, "end_s": r["end"] - start_at}
        row.update({k: r[k] for k in ("ttft_s", "latency_s", "decode_tps", "prompt_tokens", "cached_tokens",
                                      "completion_tokens", "finish_reason")})
        row["new_prompt_tokens"] = (row["prompt_tokens"] or 0) - row["cached_tokens"]
        with lock:
            rows.append(row)


def run_level(level, rep, rows):
    lock = threading.Lock()
    start_at = time.monotonic() + 0.5
    threads = []
    for sid in range(1, level + 1):
        chunks = CHUNKS[next_chunk[0]: next_chunk[0] + TURNS]
        next_chunk[0] += TURNS
        th = threading.Thread(target=session, args=(level, rep, sid, chunks, start_at, rows, lock))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=LEVELS)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sampling-json", default="{}", help="extra sampling fields sent with every request")
    args = ap.parse_args()
    LEVELS, REPS, OUT = args.levels, args.reps, args.out
    BASE, MODEL, SAMPLING = args.base, args.model, json.loads(args.sampling_json)
    assert len(CHUNKS) >= sum(LEVELS) * REPS * TURNS, len(CHUNKS)
    # Warm the shared system prompt once, as a running agent fleet would have it cached.
    warm = post_stream([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Ready?"}], seed=1)
    print("warmup", {k: warm[k] for k in ("prompt_tokens", "cached_tokens", "latency_s")}, flush=True)
    rows = []
    for rep in range(1, REPS + 1):
        for level in LEVELS:
            n0 = len(rows)
            run_level(level, rep, rows)
            lv = rows[n0:]
            wall = max(r["end_s"] for r in lv)
            out_tps = sum(r["completion_tokens"] for r in lv) / wall
            print(f"rep {rep} level {level}: {len(lv)} turns, wall {wall:.1f}s, output {out_tps:.1f} tok/s", flush=True)
            time.sleep(5)
    OUT.write_text(json.dumps(rows, indent=1))
    print("ALL_DONE", len(rows), flush=True)
