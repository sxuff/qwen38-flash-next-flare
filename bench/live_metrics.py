#!/usr/bin/env python3
"""Passive Flare speed meter: samples /metrics every 2 s, or summarizes the samples.
  live_metrics.py sample [OUT]          append one JSON line per sample to OUT
  live_metrics.py summary [OUT] [SECS]  decode speed over the last SECS seconds (default: all)"""
import json, statistics, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/metrics"
KEYS = {"flare_generated_tokens_total": "gen", "flare_prefill_tokens_total": "pre", "flare_active_requests": "active",
        "flare_speculative_offered_drafts_total": "offered", "flare_speculative_accepted_drafts_total": "accepted",
        "flare_speculative_rounds_total": "rounds", "flare_requests_completed_total": "done"}

def read():
    totals = dict.fromkeys(KEYS.values(), 0.0)
    for line in urllib.request.urlopen(URL, timeout=5).read().decode().splitlines():
        if line.startswith("#") or not line.strip(): continue
        name, value = line.rsplit(" ", 1)
        key = KEYS.get(name.split("{", 1)[0])
        if key: totals[key] += float(value)
    return totals

def sample(out):
    with open(out, "a") as f:
        while True:
            try: row = {"t": time.time(), **read()}
            except Exception as error: row = {"t": time.time(), "error": str(error)}
            f.write(json.dumps(row) + "\n"); f.flush()
            time.sleep(2)

def summary(out, secs):
    rows = [json.loads(l) for l in Path(out).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    if secs: rows = [r for r in rows if r["t"] >= rows[-1]["t"] - secs]
    decode, busy = [], 0.0
    for a, b in zip(rows, rows[1:]):
        dt, dg, dp = b["t"] - a["t"], b["gen"] - a["gen"], b["pre"] - a["pre"]
        if dg > 0 or dp > 0: busy += dt
        # Pure decode intervals: tokens generated, no prompt read, same request count at both ends.
        if dg > 0 and dp == 0 and a["active"] == b["active"] and a["active"] >= 1:
            decode.append((int(a["active"]), dg / dt))
    first, last = rows[0], rows[-1]
    by = {}
    for n, v in decode: by.setdefault(n, []).append(v)
    offered, accepted = last["offered"] - first["offered"], last["accepted"] - first["accepted"]
    print(json.dumps({
        "window_s": round(last["t"] - first["t"]), "busy_s": round(busy),
        "generated_tokens": int(last["gen"] - first["gen"]), "prompt_tokens_read": int(last["pre"] - first["pre"]),
        "requests_completed": int(last["done"] - first["done"]), "active_now": int(last["active"]),
        "decode_tok_s_by_active_requests": {n: {"median": round(statistics.median(v), 1), "intervals": len(v)} for n, v in sorted(by.items())},
        "draft_acceptance": round(accepted / offered, 3) if offered else None,
        "tokens_per_round": round((last["gen"] - first["gen"]) / (last["rounds"] - first["rounds"]), 2) if last["rounds"] > first["rounds"] else None,
    }, indent=1))

if __name__ == "__main__":
    mode = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "live-metrics.jsonl"
    sample(out) if mode == "sample" else summary(out, float(sys.argv[3]) if len(sys.argv) > 3 else 0)
