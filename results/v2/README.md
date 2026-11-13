# v2 results

Receipts for the v2 comparison in the main [README](../../README.md#v2-results-flare-recipe-vs-hyperflare-with-dflash2), measured Nov 12–13, 2026 on one ASUS Ascent GX10.

- [`pins.json`](pins.json): host, checkpoint and drafter pins with hashes, both engine commits and binary hashes, every patch's sha256, and each arm's package (cache entry, envelope sha256, signature check, speculative capacity).
- [`summary.json`](summary.json): one row per arm and cell, the mean over runs plus every run's value. Produced by `python3 bench/v2/summarize.py`.
- [`raw/`](raw): one receipt per run, `ARM-WORKLOAD[-LEVELS]-rN.json`, with the arm's flags, engine version, start time, GPU state before the run, engine counter deltas, clocks, temperature, power and peak memory over the run, cool-down, and the workload's log tail.
  - `*-turns.json`: every agent turn of an agent run, as `bench/agents_bench.py` records it.
  - `*-speed.jsonl`, `*-prefill.jsonl`, `*-quality-s405.jsonl`, `*-quality-s406.jsonl`: the frozen harness's rows, outputs included. The per-request event folders and the 2-second traces stay on the measuring host (about 80 MB).
  - `*-single-score.json`, `*-quality-score.json`: `bench/v2/score_v2.py` output.

Arms in file names: `A`, `B`, `C`, and `D4`/`D5`/`D6` for DFlash2 at 4, 5 and 6 drafts (`D5` and `D6` ran only the depth sweep), all on a freshly started engine; `Aw`, `Bw`, `D4w` are A, B and D4 on an engine first warmed by three first-turn requests (`prewarm` in each receipt records them). Run plans, in the order run: [`bench/v2/plans`](../../bench/v2/plans). Runs `D4-agents-L1-8-r1` and `-r2` are the depth sweep; the main plan's D runs are `D4-agents-L1-2-4-8-r1..3` and `D4-agents-L1-8-r4..5`.

Two things to know:

- The one-request (`single`) runs' harness files were first written under the harness directory, because `bench/v2/driver.py` passed a relative output path to a collector that runs from there. They were copied unchanged (`cp -p`, byte-compared) into `raw/` before scoring; the driver now resolves the path.
- `A-quality-r1` ran on Nov 12 while the checkpoint was downloading; quality is greedy and does not depend on speed. Every speed run came after the download, with no export running and no other process on the GPU.
