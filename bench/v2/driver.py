#!/usr/bin/env python3
"""v2 measurement driver: one fresh engine start per run, the repo's own workloads, one JSON receipt per run.

  driver.py --arm A --workload agents --levels 1 2 4 8 --runs 3 --out results/v2/raw
  driver.py --arm D5 --workload single --runs 1 ...

Per run it starts Flare/Hyperflare with scripts/start-flare.sh (FLARE_BIN, PACKAGE, SERVE_FLAGS and PREFILL_PRIORITY
from the arm table in arms.json), waits for /health, then samples GPU clocks, temperature, power, memory and the
engine's /metrics every 2 s while the workload runs. Workloads are unchanged repo code:
  agents   bench/agents_bench.py (six-turn code review, greedy top-k 1, thinking on, 256-token replies)
  single   the frozen flashnext-tf405 harness: speed suite (4 prompts x 3 x 400 tokens) and prefill suite (32K probe)
  quality  the same harness's quality suite (GSM8K 50, HumanEval 20), seeds 405 and 406
The engine is stopped after every run and the GPU cools before the next one. The agent workload always reads its files
from the same Flare checkout (WORKLOAD_SRC), so every arm gets identical prompts.
"""
import argparse, json, os, re, subprocess, threading, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
WORKLOAD_SRC = Path(os.environ.get("WORKLOAD_SRC", HOME / "v2/flare-ca1f960"))
HARNESS = Path(os.environ.get("HARNESS", HOME / "flare-install/tf405-flare/harness"))
BASE = "http://127.0.0.1:8000"
MODEL = "qwen38-flash-next-flare"
KEEP = re.compile(r"(generated_tokens|prefill_tokens|active_requests|accepted|draft|rounds|step|duration|graph_launches"
                  r"|requests_completed|cache_reused|snapshot_evictions)")
COOL_MIN_S, COOL_MAX_S, COOL_TEMP_C = 90, 300, 50  # this GB10 idles at 47-50 C with the engine stopped


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def metrics_text():
    try:
        return urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
    except Exception as error:  # noqa: BLE001 - recorded, never fatal
        return f"# error {error}"


def parse(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = out.get(name, 0.0) + float(value)
            except ValueError:
                pass
    return out


def gpu():
    q = sh("nvidia-smi --query-gpu=clocks.gr,clocks.mem,clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active"
           " --format=csv,noheader,nounits").stdout.strip().split(", ")
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits").stdout.strip()
    mem = {k: int(v.split()[0]) * 1024 for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
           if k in ("MemAvailable", "MemTotal")}
    keys = ("clock_gr_mhz", "clock_mem_mhz", "clock_sm_mhz", "temp_c", "power_w", "throttle")
    row = {k: (float(v) if v.replace(".", "", 1).isdigit() else v) for k, v in zip(keys, q)}
    row["apps"] = [a for a in apps.splitlines() if a.strip()]
    row["mem_available_bytes"] = mem.get("MemAvailable")
    return row


class Sampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop, self.rows = path, threading.Event(), []

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop.is_set():
                row = {"t": time.time(), **gpu(),
                       "m": {k: v for k, v in parse(metrics_text()).items() if KEEP.search(k)}}
                self.rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.stop.wait(2)


def start_engine(arm, log):
    env = dict(os.environ, FLARE_BIN=arm["flare_bin"], PACKAGE=arm["package"], SERVE_FLAGS=arm["serve_flags"],
               PREFILL_PRIORITY=str(arm["prefill_priority"]), RUN_LOG=str(log))
    r = subprocess.run(["bash", str(REPO / "scripts/start-flare.sh"), "--no-create"], env=env, capture_output=True,
                       text=True)
    if r.returncode:
        raise SystemExit(f"start-flare.sh failed ({r.returncode}): {r.stdout}{r.stderr}")
    t0 = time.time()
    while time.time() - t0 < 1200:
        try:
            urllib.request.urlopen(BASE + "/health", timeout=5)
            return time.time() - t0
        except Exception:  # noqa: BLE001
            if "FLARE_EXITED" in Path(log).read_text(errors="replace") if Path(log).exists() else False:
                raise SystemExit(f"engine exited during start; see {log}")
            time.sleep(3)
    raise SystemExit(f"engine not ready after 1200 s; see {log}")


def stop_engine():
    sh(f"bash {REPO / 'scripts/stop-flare.sh'}")


def cool_down():
    t0 = time.time()
    while True:
        waited = time.time() - t0
        temp = gpu().get("temp_c")
        if waited >= COOL_MAX_S or (waited >= COOL_MIN_S and isinstance(temp, float) and temp <= COOL_TEMP_C):
            return {"cooldown_s": round(waited), "temp_after_cooldown_c": temp}
        time.sleep(10)


PREWARM = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import agents_bench as ab
ab.MODEL, ab.SAMPLING = sys.argv[2], {"top_k": 1}
out = []
for i in range(int(sys.argv[3])):
    path, code = ab.CHUNKS[-1 - i]  # from the end of the file list: never used by the benchmark's own sessions
    r = ab.post_stream([{"role": "system", "content": ab.SYSTEM},
                        {"role": "user", "content": f"Warm-up {i + 1}. Contents of {path}:\n```\n{code}\n```"}], seed=900 + i)
    out.append({"prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"], "ttft_s": r["ttft_s"]})
print(json.dumps(out))
"""


def prewarm(n):
    """Send n first-turn-like requests (the benchmark's system prompt and tools, files it never uses) before the
    workload. Hyperflare saves snapshots at earlier message boundaries only once it has timed a few prefill and snapshot
    steps, so a freshly started engine misses the shared system prompt on its first turns; a running server would not.
    Engine counters and the workload's own warm-up start after this."""
    r = subprocess.run(["python3", "-c", PREWARM, str(REPO / "bench"), MODEL, str(n)], capture_output=True, text=True,
                       env=dict(os.environ, FLARE_SRC=str(WORKLOAD_SRC)))
    if r.returncode:
        raise SystemExit(f"prewarm failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_agents(levels, out):
    cmd = ["python3", str(REPO / "bench/agents_bench.py"), "--levels", *map(str, levels), "--reps", "1",
           "--sampling-json", '{"top_k": 1}', "--model", MODEL, "--out", str(out)]
    r = subprocess.run(cmd, env=dict(os.environ, FLARE_SRC=str(WORKLOAD_SRC)), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def run_harness(suites, outdir, label):
    extra = '{"stream_options":{"include_usage":true,"continuous_usage_stats":true},"top_k":1}'
    eff = '{"temperature":0.7,"top_p":0.95,"top_k":1,"note":"top_k=1 is argmax; clamped temperature/top_p have no effect"}'
    log, code = "", 0
    for suite, seed in suites:
        out = outdir / f"{label}-{suite}{'' if seed is None else f'-s{seed}'}.jsonl"
        cmd = ["python3", "collect.py", "sse", "--manifest", "rendered/manifest.json", "--suite", suite, "--arm", label,
               "--model", MODEL, "--base-url", BASE, "--extra-json", extra, "--effective-sampling-json", eff,
               "--out", str(out)] + ([] if seed is None else ["--seed", str(seed)])
        r = subprocess.run(cmd, cwd=HARNESS, capture_output=True, text=True)
        log += f"== {suite} {seed}\n{r.stdout}{r.stderr}"
        code = code or r.returncode
    return code, log


def summarize_samples(rows, flare_pid_hint="/flare"):
    if not rows:
        return {}
    num = lambda k: [r[k] for r in rows if isinstance(r.get(k), float)]
    used = []
    for r in rows:
        for a in r["apps"]:
            parts = [p.strip() for p in a.split(",")]
            if len(parts) == 3 and flare_pid_hint in parts[1] and parts[2].isdigit():
                used.append(int(parts[2]))
    foreign = sorted({a for r in rows for a in r["apps"] if flare_pid_hint not in a})
    stat = lambda v: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)} if v else None
    return {"samples": len(rows), "clock_gr_mhz": stat(num("clock_gr_mhz")), "clock_mem_mhz": stat(num("clock_mem_mhz")),
            "temp_c": stat(num("temp_c")), "power_w": stat(num("power_w")),
            "throttle_reasons_seen": sorted({str(r.get("throttle")) for r in rows}),
            "peak_engine_gpu_mib": max(used) if used else None,
            "min_mem_available_gib": round(min(r["mem_available_bytes"] for r in rows) / 2**30, 2),
            "foreign_gpu_processes": foreign}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workload", choices=("agents", "single", "quality"), required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--first-run", type=int, default=1)
    ap.add_argument("--out", type=Path, default=REPO / "results/v2/raw")
    ap.add_argument("--arms", type=Path, default=Path(__file__).with_name("arms.json"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    # The harness collector runs from its own directory, so a relative --out must be made absolute here; v2's single
    # runs were saved under the harness directory before this line existed (results/v2/README.md).
    args.out = args.out.resolve()
    arm = json.loads(args.arms.read_text())[args.arm]
    if arm["package"].startswith("@"):  # "@B": the cache entry export_arm.sh recorded for that export
        arm["package"] = (HOME / "v2/state" / f"{arm['package'][1:]}.package").read_text().strip()
    args.out.mkdir(parents=True, exist_ok=True)
    traces = HOME / "v2/traces"
    traces.mkdir(parents=True, exist_ok=True)
    for k in range(args.first_run, args.first_run + args.runs):
        tag = f"{args.arm}-{args.workload}{'-L' + '-'.join(map(str, args.levels)) if args.workload == 'agents' else ''}-r{k}"
        log = traces / f"{tag}-engine.log"
        receipt = {"tag": tag, "arm": args.arm, "arm_config": arm, "workload": args.workload, "run": k,
                   "note": args.note, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "workload_src": str(WORKLOAD_SRC),
                   "workload_src_commit": sh(f"git -C {WORKLOAD_SRC} rev-parse HEAD").stdout.strip(),
                   "engine_version": sh(f"{arm['flare_bin']} --version").stdout.strip(), "gpu_before": gpu()}
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: starting engine", flush=True)
        receipt["ready_s"] = round(start_engine(arm, log), 1)
        if arm.get("prewarm"):
            receipt["prewarm"] = prewarm(arm["prewarm"])
        sampler = Sampler(traces / f"{tag}-samples.jsonl")
        before = metrics_text()
        sampler.start()
        t0 = time.time()
        if args.workload == "agents":
            out = args.out / f"{tag}-turns.json"
            code, text = run_agents(args.levels, out)
            receipt["turns_file"] = out.name
        elif args.workload == "single":
            code, text = run_harness([("speed", None), ("prefill", None)], args.out, tag)
        else:
            code, text = run_harness([("quality", 405), ("quality", 406)], args.out, tag)
        receipt["workload_s"] = round(time.time() - t0, 1)
        sampler.stop.set()
        sampler.join(10)
        after = metrics_text()
        receipt["exit_code"], receipt["workload_log"] = code, text[-6000:]
        b, a = parse(before), parse(after)
        receipt["metrics_delta"] = {k: a[k] - b.get(k, 0.0) for k in sorted(a) if KEEP.search(k) and a[k] != b.get(k, 0.0)}
        receipt["system"] = summarize_samples(sampler.rows)
        receipt["engine_errors"] = log.read_text(errors="replace").count("server_error") if log.exists() else None
        stop_engine()
        receipt.update(cool_down())
        (args.out / f"{tag}.json").write_text(json.dumps(receipt, indent=1))
        last = text.strip().splitlines()[-1][:160] if text.strip() else ""
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: exit {code}, {receipt['workload_s']} s; {last}", flush=True)
        if code:
            raise SystemExit(f"{tag}: workload failed with {code}; see {args.out / (tag + '.json')}")


if __name__ == "__main__":
    main()
