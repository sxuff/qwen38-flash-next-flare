#!/usr/bin/env python3
"""Summarize results/v2/raw into results/v2/summary.json: one row per arm and agent count, plus single-request cells.

Definitions (the same as v1's analysis):
  total output tok/s   completion tokens of all turns at that agent count / time until the last turn ended
  time to finish       time until the last of the level's turns ended (48 turns at 8 agents)
  first token          median and 90th percentile of each turn's time to first token
  longest pause        90th percentile of each turn's longest gap between tokens after the first (max_gap_s)
  prompt cached        cached prompt tokens / prompt tokens over the level's turns
  acceptance           accepted draft tokens per request verification round, over the whole run (engine counters)
  target step time     single-request decode: total decode time / verification rounds (one target step each)
  long prompt          the harness probes ask for one token, so their total request time is the time to first token
  peak memory          largest GPU memory nvidia-smi reported for the engine during the run
Cells report the mean over runs and keep every run's value.
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results/v2/raw")


def pct(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else None


def counter(delta, *needles):
    return sum(v for k, v in delta.items() if all(n in k for n in needles) and "{" not in k)


def acceptance(delta):
    rounds = counter(delta, "draft_request_rounds_total") or counter(delta, "verification_request_rounds_total")
    accepted = counter(delta, "accepted_draft_tokens_total")
    return accepted / rounds if rounds else None


def mean(v):
    v = [x for x in v if x is not None]
    return round(st.mean(v), 3) if v else None


cells = defaultdict(lambda: defaultdict(list))
for rec_path in sorted(RAW.glob("*.json")):
    if rec_path.name.endswith(("-turns.json", "-score.json")):
        continue
    rec = json.loads(rec_path.read_text())
    arm, sysinfo = rec["arm"], rec.get("system", {})
    common = {"run": rec["tag"], "acceptance": acceptance(rec["metrics_delta"]),
              "peak_engine_gpu_mib": sysinfo.get("peak_engine_gpu_mib"), "clock_gr_mhz": sysinfo.get("clock_gr_mhz"),
              "temp_c_max": (sysinfo.get("temp_c") or {}).get("max"), "power_w_median": (sysinfo.get("power_w") or {}).get("median"),
              "foreign_gpu_processes": sysinfo.get("foreign_gpu_processes"), "engine_errors": rec.get("engine_errors")}
    if rec["workload"] == "agents":
        turns = json.loads((RAW / rec["turns_file"]).read_text())
        for level in sorted({t["level"] for t in turns}):
            lv = [t for t in turns if t["level"] == level]
            wall = max(t["end_s"] for t in lv)
            ttft = [t["ttft_s"] for t in lv if t.get("ttft_s") is not None]
            cells[(arm, f"agents-{level}")]["runs"].append({**common,
                "turns": len(lv), "total_tok_s": sum(t["completion_tokens"] or 0 for t in lv) / wall, "finish_s": wall,
                "ttft_median_s": st.median(ttft), "ttft_p90_s": pct(ttft, .9),
                "longest_pause_p90_s": pct([t["max_gap_s"] for t in lv if t.get("max_gap_s") is not None], .9),
                "prompt_cached": sum(t["cached_tokens"] or 0 for t in lv) / max(1, sum(t["prompt_tokens"] or 0 for t in lv)),
                "truncated_turns": sum(t.get("finish_reason") == "length" for t in lv)})
    elif rec["workload"] == "single":
        score = RAW / f"{rec['tag']}-single-score.json"
        if not score.exists():
            continue
        s = json.loads(score.read_text())
        speed = [json.loads(l) for l in (RAW / f"{rec['tag']}-speed.jsonl").read_text().splitlines() if l.strip()]
        decode_s = sum((r["completion_tokens"] - 1) / r["decode_tps"] for r in speed if r.get("decode_tps"))
        rounds = counter(rec["metrics_delta"], "verification_request_rounds_total")
        # The probes ask for one token, so (as in v1) the request's total time is its time to first token.
        long32 = {k: v["elapsed_s"] for k, v in s["prefill"].items() if v["prompt_tokens"] > 30000}
        cells[(arm, "single")]["runs"].append({**common, "decode_tok_s": s["decode_tps"],
            "ttft_32k_s": st.mean(long32.values()) if long32 else None, "ttft_by_probe_s": {k: v["elapsed_s"] for k, v in s["prefill"].items()},
            "target_step_ms": 1000 * decode_s / rounds if rounds else None})
    elif rec["workload"] == "quality":
        score = RAW / f"{rec['tag']}-quality-score.json"
        if score.exists():
            q = json.loads(score.read_text())
            cells[(arm, "quality")]["runs"].append({**common, "by_seed": q["by_seed"],
                                                    "identical_outputs_across_seeds": q["identical_outputs_across_seeds"]})

summary = {}
for (arm, cell), d in sorted(cells.items()):
    runs = d["runs"]
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float)) and k not in ("turns",)]
    row = {"n_runs": len(runs), **{k: mean([r.get(k) for r in runs]) for k in keys}}
    if cell.startswith("agents"):
        row["total_tok_s_range"] = [round(min(r["total_tok_s"] for r in runs), 1), round(max(r["total_tok_s"] for r in runs), 1)]
    row["runs"] = runs
    summary.setdefault(arm, {})[cell] = row
out = RAW.parent / "summary.json"
out.write_text(json.dumps(summary, indent=1) + "\n")
for arm, c in summary.items():
    for cell, r in c.items():
        if cell.startswith("agents"):
            print(f"{arm:4} {cell:9} n={r['n_runs']} {r['total_tok_s']:6.1f} tok/s {r['total_tok_s_range']}  finish {r['finish_s']:6.1f}s  "
                  f"ttft {r['ttft_median_s']:5.2f}s  pause90 {r['longest_pause_p90_s']:5.2f}s  cached {r['prompt_cached']:.0%}  "
                  f"accept {r['acceptance'] or 0:.2f}  peak {r['peak_engine_gpu_mib']} MiB")
        elif cell == "single":
            print(f"{arm:4} single    n={r['n_runs']} decode {r['decode_tok_s']:.1f} tok/s  32K {r['ttft_32k_s']:.1f}s  "
                  f"step {r['target_step_ms'] or 0:.1f} ms  accept {r['acceptance'] or 0:.2f}")
        else:
            print(f"{arm:4} quality   {[x['by_seed'] for x in r['runs']]}")
