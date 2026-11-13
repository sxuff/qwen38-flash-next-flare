#!/usr/bin/env python3
"""Score v2 harness outputs with the frozen flashnext-tf405 functions: GSM8K extraction and the HumanEval Docker sandbox.

  score_v2.py results/v2/raw/A-quality-r1        -> GSM8K and HumanEval per seed, determinism across seeds
  score_v2.py --single results/v2/raw/A-single-r1 -> decode (mean of 4 prompt medians) and long-prompt times

Same scoring code, fixtures and sandbox image as v1 (score_greedy.py in the TensorFold-to-Flare harness);
only the input paths differ.
"""
import argparse, json, statistics, sys
from pathlib import Path

H = Path.home() / "flare-install/tf405-flare/harness"
sys.path.insert(0, str(H))
IMAGE = "sha256:1f626b72bd7b20a470dae03ae18039049d50a0d496de20bf818b3b612c8699ee"
SEEDS = (405, 406)
manifest = json.loads((H / "rendered/manifest.json").read_text())
tasks = {t["task_id"]: t for t in manifest["tasks"]}
fixtures = {t["task_id"]: t for t in json.loads((H / "fixtures.json").read_text())["tasks"]}
ids = lambda suite: [t["task_id"] for t in manifest["tasks"] if t["suite"] == suite]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        t = tasks[r["task_id"]]
        assert r["prompt_sha256"] == t["prompt_sha256"] and r["prompt_token_ids_sha256"] == t["prompt_token_ids_sha256"], r["request_id"]
        assert r["max_tokens"] == t["max_tokens"] and not r.get("error"), r["request_id"]
    return rows


def quality(prefix):
    from score import gsm_number, candidate_program
    from sandbox import Sandbox
    box = Sandbox(IMAGE)
    assert box.canary().get("passed"), "sandbox canary failed"
    texts, verdicts = {}, {}
    for s in SEEDS:
        rows = load(f"{prefix}-quality-s{s}.jsonl")
        texts[s] = {r["task_id"]: r.get("output_text", "") for r in rows if r["suite"] == "quality" and not r.get("serial")}
        assert sorted(texts[s]) == sorted(ids("quality")), "missing quality rows"
        v = {}
        for tid, text in texts[s].items():
            task = fixtures[tid]
            if task["dataset"] == "gsm8k":
                got, want = gsm_number(text), gsm_number(task["reference_answer"])
                v[tid] = got is not None and got == want
            else:
                try:
                    v[tid] = bool(box.run(candidate_program(task, text))["passed"])
                except (SyntaxError, ValueError):
                    v[tid] = False
        verdicts[s] = v
    summary = {str(s): {d: sum(ok for t, ok in verdicts[s].items() if fixtures[t]["dataset"] == d)
                        for d in ("gsm8k", "humaneval")} for s in SEEDS}
    return {"sandbox_image": IMAGE, "by_seed": summary, "of": {"gsm8k": 50, "humaneval": 20},
            "identical_outputs_across_seeds": sum(texts[SEEDS[0]][t] == texts[SEEDS[1]][t] for t in texts[SEEDS[0]]),
            "per_item_seed405": verdicts[SEEDS[0]]}


def single(prefix):
    speed = load(f"{prefix}-speed.jsonl")
    res = {}
    for tid in ids("speed"):
        reps = [r for r in speed if r["task_id"] == tid and r["suite"] == "speed" and not r.get("serial")]
        assert len(reps) == 3 and all(r.get("decode_tps") for r in reps), tid
        res[tid] = {"decode_median_tps": statistics.median(r["decode_tps"] for r in reps),
                    "decode_reps_tps": [r["decode_tps"] for r in reps],
                    "ttft_median_s": statistics.median(r["ttft_seconds"] for r in reps)}
    pre = {r["task_id"]: {"prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_seconds"], "elapsed_s": r["elapsed_seconds"]}
           for r in load(f"{prefix}-prefill.jsonl") if r["suite"] == "prefill"}
    return {"decode_tps": statistics.mean(v["decode_median_tps"] for v in res.values()), "speed": res, "prefill": pre}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--single", action="store_true")
    a = ap.parse_args()
    out = single(a.prefix) if a.single else quality(a.prefix)
    Path(f"{a.prefix}-{'single' if a.single else 'quality'}-score.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_item_seed405", "speed")}, indent=1))
