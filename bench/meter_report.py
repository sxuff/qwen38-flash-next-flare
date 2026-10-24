#!/usr/bin/env python3
"""Full speed report for one Claude Code run from live_metrics.py samples plus Flare's current counters."""
import json, statistics as st, sys, time, urllib.request
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
errors = sum("error" in r for r in rows)
rows = [r for r in rows if "error" not in r]
pairs = list(zip(rows, rows[1:]))
act = [(a, b) for a, b in pairs if b["gen"] > a["gen"] or b["pre"] > a["pre"]]
start, end = act[0][0]["t"], act[-1][1]["t"]
first, last = rows[0], rows[-1]
fmt = lambda t: time.strftime("%H:%M:%S", time.localtime(t))

decode_t = prefill_t = idle_t = parallel_t = 0.0
dec, dec_by_active, pre_rates, by_hour = [], {}, [], {}
for a, b in pairs:
    dt = b["t"] - a["t"]
    if dt > 30 or a["t"] < start or b["t"] > end: continue  # sampler gap or outside the run
    dg, dp = b["gen"] - a["gen"], b["pre"] - a["pre"]
    if a["active"] > 1: parallel_t += dt
    if dp > 0: prefill_t += dt
    elif dg > 0: decode_t += dt
    else: idle_t += dt
    if dg > 0 and dp == 0 and a["active"] == b["active"] >= 1:
        rate = dg / dt
        dec_by_active.setdefault(int(a["active"]), []).append(rate)
        if a["active"] == 1:
            dec.append(rate); by_hour.setdefault(int((a["t"] - start) // 1800), []).append(rate)
    if dp > 2000: pre_rates.append(dp / dt)

counters = {}
for line in urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode().splitlines():
    if line.startswith("#") or not line.strip(): continue
    name, v = line.rsplit(" ", 1); n = name.split("{", 1)[0]
    counters[n] = counters.get(n, 0) + float(v)

gen, pre = last["gen"] - first["gen"], last["pre"] - first["pre"]
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print(json.dumps({
    "run": {"from": fmt(start), "to": fmt(end), "hours": round((end - start) / 3600, 2), "sampler_errors": errors},
    "requests": {"completed_in_run": int(last["done"] - first["done"]), "admitted_total": int(counters["flare_requests_admitted_total"]),
                 "cancelled_total": int(counters.get("flare_requests_cancelled_total", 0)), "failed_total": int(counters.get("flare_requests_failed_total", 0)),
                 "max_active": int(max(r["active"] for r in rows)), "minutes_with_2plus_active": round(parallel_t / 60, 1)},
    "tokens": {"generated": int(gen), "prompt_read": int(pre), "prefix_cache_reused_since_restart": int(counters.get("flare_cache_reused_tokens_total", 0))},
    "time_minutes": {"decoding": round(decode_t / 60, 1), "reading_prompts": round(prefill_t / 60, 1), "flare_idle": round(idle_t / 60, 1)},
    "decode_tok_s_one_request": {"median": round(st.median(dec), 1), "p10": round(q(dec, .1), 1), "p90": round(q(dec, .9), 1), "intervals": len(dec)},
    "decode_tok_s_by_active": {n: round(st.median(v), 1) for n, v in sorted(dec_by_active.items())},
    "decode_median_by_half_hour": {f"{fmt(start + 1800 * h)}": round(st.median(v), 1) for h, v in sorted(by_hour.items())},
    "prompt_read_tok_s": {"median": round(st.median(pre_rates)), "intervals": len(pre_rates)} if pre_rates else None,
    "draft_acceptance": round((last["accepted"] - first["accepted"]) / (last["offered"] - first["offered"]), 3),
    "tokens_per_round": round(gen / (last["rounds"] - first["rounds"]), 2),
}, indent=1))
