"""Drafter check: acceptance on the frozen harness's GSM8K + HumanEval prompts (one request at a time), per arm.

Acceptance length = 1 + accepted draft tokens per request verification round (the drafter card's definition: completion
tokens / verification steps). Per-position acceptance from flare_speculative_drafts_total{depth=..}.
"""
import json, collections
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "results/v2/raw"
for tag in ("A-quality-r1", "B-quality-r1", "D3-quality-r1", "B4-quality-r1", "D4-quality-r1"):
    f = RAW / f"{tag}.json"
    if not f.exists():
        print(tag, "missing"); continue
    m = json.loads(f.read_text())["metrics_delta"]
    acc = m.get("flare_accepted_draft_tokens_total", 0); rounds = m.get("flare_draft_request_rounds_total", 0)
    pos_a, pos_o = collections.Counter(), collections.Counter()
    for k, v in m.items():
        if k.startswith("flare_speculative_drafts_total{"):
            d = int(k.split('depth="')[1].split('"')[0])
            (pos_a if "accepted" in k else pos_o)[d] += v
    score = RAW / f"{tag}-quality-score.json"
    q = json.loads(score.read_text())["by_seed"] if score.exists() else None
    print(f"{tag:14} acceptance length {1 + acc / rounds:.2f}  per position "
          + " ".join(f"{d}:{pos_a[d] / pos_o[d]:.3f}" for d in sorted(pos_o)) + f"  quality {q}")
for arm in ("D4", "B4"):
    rows = [json.loads(f.read_text()) for f in sorted(RAW.glob(f"{arm}-single-r?-single-score.json"))]
    if rows:
        print(arm, "single decode", [round(r["decode_tps"], 1) for r in rows],
              "32K", [round(sum(v["elapsed_s"] for k, v in r["prefill"].items() if v["prompt_tokens"] > 30000) / 2, 2) for r in rows])
