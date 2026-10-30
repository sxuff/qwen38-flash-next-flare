# Qwen3.8 Flash-Next on Flare, one DGX Spark or ASUS Ascent GX10

A **Flare** serving recipe for NVIDIA's NVFP4 Qwen3.8 Flash-Next on one NVIDIA GB10 with 128 GB unified memory. It rebuilds NVIDIA's runtime package from source with three changes the prebuilt package lacks: **greedy decoding** (top-k 1), **speculative capacity for eight concurrent agents**, and **mixed prefill**, then serves it with **three drafts per round** on a Flare runtime patched for agents: **oldest-first prefix-cache eviction** and **prompts first** in the scheduler.

**Built for agents.** With eight coding agents at once it produces **3.5×** the output of TensorFold running the EXL3 4.05 recipe, and reaches each agent's first token in 2.7 s instead of 63 s. One request at a time the engines are close, and quality is the same.

![8 concurrent coding agents](results/figures/eight-agents.png)

## The recipe

```text
Weights:          nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6 (NVFP4 experts, MXFP8 attention, QAD)
Runtime:          Flare 0.1.0 and Ignition, NVIDIA/flare-local-inference @ ca1f960, built from source
Patches:          top-k range 1-20 (top-k 1 is greedy; sampling defaults unchanged);
                  mixed-prefill export fix (upstream bug since Oct 24);
                  oldest-first prefix-snapshot eviction (upstream's order since Oct 23 costs 30% with 8 agents);
                  prompt priority setting (FLARE_PREFILL_PRIORITY)
Package:          8 request slots, 4 image slots, up to 5 MTP drafts,
                  speculative capacity 48 tokens, mixed prefill graphs
Serving:          3 drafts every round (--no-mtp-adaptive-verification --mtp-draft-tokens 3),
                  throughput scheduler (--scheduler-policy throughput), FLARE_PREFILL_PRIORITY=16
Context:          262,144 tokens
N-gram table:     26.8 GiB, read from disk with a 1 GiB RAM row cache
Prefix cache:     64 snapshot slots, about 1.8M KV tokens, GPU memory only
API:              OpenAI, Anthropic and Responses, 127.0.0.1:8000
Model ID:         qwen38-flash-next-flare
Signing:          your own local development key
Hardware:         one GB10, 128 GB unified memory
```

The weights are NVIDIA's own checkpoint; nothing is re-quantized. The package Ignition exports from it is about 107 GB (100 GiB) and runs with 76.5 GiB of fixed GPU allocations.

## Results

All runs on one GB10 with Qwen3.8-Flash-Next. This recipe's agent and single-request speed were measured Oct 28 and 29, 2026, and so were the Flare figures in the eight-agents card at the top; everything else, including quality, the other figures and TensorFold, Oct 22 to 24, when the recipe ran the previous Flare pin (1fd1f91) without runtime patches. TensorFold ran the published EXL3 4.05 recipe (`4.05bpw_h6_ng6`, MTP 6 at 0.70, BF16 KV, prompt kernels on).

### Several agents at once

Six-turn code-review sessions: a shared ~5K-token agent prompt with tool definitions, a new source file each turn, replies up to 256 tokens, thinking on. Greedy on both engines. This recipe's row is the mean of five runs; with eight agents, single runs ranged from 79.7 to 85.8 tok/s.

| Total output, tok/s | 1 agent | 8 agents | 48 turns finished in | Median first token, 8 agents |
|---|---|---|---|---|
| **Flare, this recipe** | **38.0** | **83.0** | **143 s** | **2.7 s** |
| Flare, previous recipe (1fd1f91) | 35.7 | 77.6 | 157 s | 7 s |
| TensorFold EXL3 4.05 | 29.9 | 23.9 | 480 s | 63 s |
| TensorFold NVFP4 (same weights as Flare) | 35.0 | 24.5 | 478 s | 65 s |

TensorFold serves one request at a time, so with eight agents the others wait in line and its prompt cache stops hitting (0% reuse). Flare batches all eight and keeps 85% of each prompt cached.

### What each runtime change adds

The same eight-agent sessions on Flare ca1f960 with three fixed drafts, adding one change at a time. Each row is the mean of two runs, except two: the throughput row ran in two separate sweeps, so it shows both means, and the last row is the mean of five runs.

| Flare ca1f960 | 8 agents, tok/s | 48 turns finished in | Median first token | Prompt cached |
|---|---|---|---|---|
| No runtime patches | 54.4 | 215 s | 10.9 s | 70% |
| + oldest-first eviction | 77.6 | 153 s | 7.1 s | 85% |
| + throughput scheduler | 82.1 to 83.0 | 148 s | 5.0 to 5.5 s | 85% |
| **+ prompt priority 16 (this recipe)** | **83.0** | **143 s** | **2.7 s** | 85% |

- **Eviction.** Upstream commit `2c99983e` (Oct 23) changed which prefix snapshot Flare drops first when its 64 slots fill: closely spaced checkpoints now go before the oldest ones. With eight agents that cuts the cached share of each prompt from 85% to 70%, and the unpatched runtime is 30% slower than the previous pin. [`flare-oldest-first-eviction.patch`](patches/flare-oldest-first-eviction.patch) restores the oldest-first order.
- **Throughput scheduler.** `--scheduler-policy throughput` is Flare's own preset; it turns off two of the scheduler's fairness prices (delivery fairness and decode age).
- **Prompt priority.** Flare's scheduler values a prompt token at 1/64 of an output token, so a new turn's prompt waits until its fairness price has risen. [`flare-prefill-priority.patch`](patches/flare-prefill-priority.patch) multiplies the value of prompt work by `FLARE_PREFILL_PRIORITY` (unset means 1, upstream's pricing). At 16 each turn starts decoding sooner: the median first token was 2.4 to 3.0 s in all five runs, against 4.3 to 6.1 s in the four runs with the throughput scheduler alone. Total output barely moves: single runs ranged from 79.7 to 85.8 tok/s with priority 16 and 80.8 to 85.2 without, so its gain in throughput is within run-to-run noise.
- **The cost of prompts first:** a reply already streaming pauses while another agent's prompt is processed. With eight agents, the 90th percentile of each reply's longest pause between tokens is 2.5 to 4.5 s, against 1.4 to 1.5 s without prompt priority. A long prompt does not freeze the others: with seven agents running and 8K- and 32K-token prompts arriving as an eighth client, the longest pause in two runs was 5.3 s at priority 16 and 8.8 s without priority, and the 32K prompt reached its first token in 16.1 s instead of 22.1 s. Priority 4 was as fast with eight agents (84.0 tok/s, two runs) and reached the first token in 3.1 to 3.4 s, but had the single longest pause in the long-prompt test, 11.8 s.

### Stock package vs this build

The same agent sessions on NVIDIA's prebuilt package and on this export, with the sampling the stock package allows (temperature 1.0, top-p 0.95, top-k 20) and adaptive drafting on both:

![Stock package vs custom build](results/figures/stock-vs-custom.png)

| Total output, tok/s | 1 agent | 8 agents | Decode per session, 8 agents |
|---|---|---|---|
| Prebuilt package | 33.3 | 63.5 | 10.4 |
| **This build** | **33.9** | **75.0** | **12.8** |

With one agent the two are the same. With eight, the prebuilt package's speculative graphs fit only 8 token rows, so it stops drafting; this build keeps drafting and produces 18% more. Serving with three fixed drafts added a little more on top (77.6 tok/s greedy, on the previous pin).

### One request at a time

![Single request](results/figures/single-request.png)

| | Decode, tok/s | 32K-token prompt to first token | GSM8K (50) | HumanEval (20) |
|---|---|---|---|---|
| **Flare, this recipe** | 52.5 | **13.4 s** | – | – |
| Flare, previous recipe (1fd1f91) | 50.7 | 14.1 s | 49 | 13 (10 on a second run) |
| TensorFold EXL3 4.05 | **75.7** | 20.0 s | 49 | 14 |
| TensorFold NVFP4 | 59.9 | 19.6 s | 49 | 12 |

Quality was scored on the previous recipe and not re-run on this one. Decode is the mean of four frozen prompts × three repeats of 400 tokens. The long-prompt probe asks for one token, so its total request time is its time to first token. TensorFold's times use its engine clock, Flare's a client clock including HTTP. Quality uses the harness's own GSM8K extraction and HumanEval Docker sandbox, greedy on both engines. With 70 items, differences of a few items are noise.

### Same weights, different engines

TensorFold also reads NVFP4. NVIDIA's checkpoint stores its expert scales compressed, so they were decoded into plain ModelOpt scales with Flare's own reference decoder; spot checks matched `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` byte for byte. On identical weights the single-request gap mostly closes: TensorFold's lead in single-stream decode comes largely from the EXL3 format, not the engine.

![Same weights](results/figures/same-weights.png)

### A real session: 4.8 hours of Claude Code

Claude Code on this recipe built a Kerr black hole visualizer, fully local. Measured from Flare's own counters every 2 s:

![Claude Code session](results/figures/claude-code-session.png)

- **40 tok/s** median decode with about 120K tokens of context.
- **94%** of 35.3M prompt tokens came from the prefix cache, so prompt processing took 25 minutes instead of about 4.8 hours.
- Each decode step took about 60 ms throughout; speed moved only with how many drafted tokens were accepted (1.4 to 3.1 per step).

## Requirements

- A DGX Spark or ASUS Ascent GX10 on Ubuntu 24.04 with the CUDA 13 toolkit (`nvcc`) under `/usr/local/cuda`.
- **Disk:** about 230 GB free: the 102 GB checkpoint, the 107 GB package and about 20 GB of kernel caches. Ignition hard-links its staging files into the package, so there is no second copy. Afterwards the checkpoint can go.
- **Time:** about 15 minutes to build, an hour or less to download, about 1.5 hours for the first export.
- [uv](https://docs.astral.sh/uv/) and [Rust](https://rustup.rs) 1.88 or newer.

## 1. Prepare the host

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

Flare locks its n-gram staging buffers in memory, which needs an unlimited locked-memory limit:

```bash
echo "$USER - memlock unlimited" | sudo tee /etc/security/limits.d/90-flare-memlock.conf
```

Log out and back in, then check that `ulimit -l` prints `unlimited`. The limit applies to logins that go through PAM: the console and OpenSSH. **Tailscale SSH skips PAM**, so a Tailscale SSH session keeps the default 8 MB and Flare fails at "register Engram staging". Run `export.sh` and the first `start-flare.sh` from the console or OpenSSH.

## 2. Build Flare and Ignition

```bash
git clone https://github.com/sxuff/qwen38-flash-next-flare ~/qwen38-flash-next-flare
cd ~/qwen38-flash-next-flare
bash scripts/setup.sh
```

This clones `NVIDIA/flare-local-inference` into `~/flare-local-inference`, checks out the pinned commit, applies the four patches in [`patches/`](patches), syncs Ignition's Python environment (rebuilding the patched native exporter) and builds the Flare runtime. Every path and pin lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment.

## 3. Download the checkpoint

```bash
bash scripts/download.sh
```

102 GB from `nvidia/Qwen3.8-Flash-Next-FP4-Local` at the pinned revision. The export would fetch it too; downloading first keeps the GPU free meanwhile.

## 4. Export the package

```bash
bash scripts/export.sh
```

Ignition loads the checkpoint, tunes kernels (about 40 min), compiles 135 forwards (about 25 min), warms them, captures 159 CUDA graphs and publishes a 107 GB package into Flare's model cache (`~/.cache/flare/packages`), signed with a development key it creates in `~/.config/flare`. `export.sh` records the new cache entry in `~/.local/state/qwen38-flash-next-flare/package`, where the other scripts find it. Nothing else may use the GPU meanwhile. The script checks free disk, the memlock limit and the GPU before it starts.

- If an export fails, run it again: kernel and forward caches make a rerun take about 30 minutes.
- To re-export after a change to the export flags or the Flare build, `bash scripts/export.sh --reuse-weights` skips rehashing the weights (about 17 minutes). The weights are shared between versions, so a second version adds only about 1 GB.
- `flare models list` shows the exported versions; `flare models remove MODEL@VERSION` and then `flare models gc` free an old one.
- Packages exported by the earlier pin (1fd1f91, a single `.safetensors` file) do not load in this Flare: it asks for a re-export.

## 5. Approve your signing key (first use)

```bash
bash scripts/approve-key.sh
```

Flare verifies every package's signature and asks once before trusting a new key. This prints the fingerprint of your own development key and starts Flare in the foreground; type `local-development` when asked, then stop it with Ctrl-C once the model starts loading. Flare remembers the approval.

## 6. Start the server

```bash
bash scripts/start-flare.sh
```

Flare runs in a dedicated tmux server (`tmux -L flare attach -t flare:0` to watch it) and is ready in about 30 seconds. A `keep` window holds that tmux server open with the memlock limit of the login that created it, so later starts and restarts work from any login, Tailscale SSH included. `bash scripts/stop-flare.sh` stops Flare and leaves the tmux server up.

If another server holds the GPU, set `STOP_SERVICE=<systemd --user unit>`: `start-flare.sh` stops it first and `stop-flare.sh` starts it again.

Flare serves with three drafts per round, the throughput scheduler and `FLARE_PREFILL_PRIORITY=16`, so each agent's new turn starts sooner (see [What each runtime change adds](#what-each-runtime-change-adds)). `SERVE_FLAGS` and `PREFILL_PRIORITY` change them; `PREFILL_PRIORITY=1` is upstream's scheduling.

Flare can also keep its prefix cache on disk, but the recipe leaves that off (measurements in [Notes](#notes)). `PREFIX_CACHE_DISK_LIMIT=20G` turns it on: every prefix snapshot is also written to `~/.cache/flare/prefix-cache` (`PREFIX_CACHE_DISK` moves it), and one evicted from the GPU is read back when its session returns. Flare keeps one subdirectory per package layout, so after a re-export the old ones can be deleted while Flare is stopped.

## 7. Use it

OpenAI-compatible:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen38-flash-next-flare",
  "messages": [{"role": "user", "content": "Explain the ISCO of a Kerr black hole in two sentences."}],
  "max_tokens": 512
}'
```

**Greedy decoding:** send `"top_k": 1`. Flare refuses `temperature: 0` by design and clamps request values into the package's ranges (temperature 0.7 to 1, top-p 0.8 to 0.95); with top-k 1 only the best token is kept, so those no longer matter. Greedy runs are not bit-for-bit repeatable: the model's scores are BF16, so the two best tokens sometimes tie exactly, and speculative decoding changes the arithmetic slightly between runs.

The Anthropic Messages API is served at `/v1/messages` on the same port, which is what Claude Code uses.

## 8. Claude Code

[`clients/claude-code`](clients/claude-code) holds a settings profile and launchers for Windows (`claude-flare.ps1`) and macOS or Linux (`claude-flare.sh`). They start Flare if needed, open an SSH tunnel from `localhost:18000` to the Spark and run `claude` with the profile.

```bash
export FLARE_SPARK=user@spark              # key-based SSH to the Spark
export FLARE_SPARK_LOGIN=user@spark.local  # a PAM login, used only for the first start after boot
bash clients/claude-code/claude-flare.sh
```

Every model slot (main, subagents, background) points at the one model, so Claude Code's parallel requests land on Flare's eight request slots.

## Benchmarks

- [`bench/agents_bench.py`](bench/agents_bench.py): the agent-session test above. `FLARE_SRC=~/flare-local-inference python3 bench/agents_bench.py --levels 1 8 --reps 1 --sampling-json '{"top_k": 1}' --out agents.json`. It draws its files from the Flare checkout made by `setup.sh`, so the workload is the same on any machine. It refuses to start while another process shares the GPU with the server (`--allow-busy-gpu` overrides), and flags any level during which one appears. Each turn also records its longest pause between tokens after the first (`max_gap_s`).
- [`bench/live_metrics.py`](bench/live_metrics.py) and [`bench/meter_report.py`](bench/meter_report.py): a passive speed meter for a real session, sampling Flare's `/metrics` every 2 s, and its report.
- Decode, long-prompt and quality figures come from the frozen `flashnext-tf405` harness of the TensorFold recipe: four speed prompts × three repeats, three long-prompt probes, 50 GSM8K and 20 HumanEval items.

## Notes

- **Greedy and the stock package.** NVIDIA's prebuilt package fixes every sampling parameter; requests can only set the seed. That is why this recipe re-exports with a top-k range.
- **Drafting with many agents.** The prebuilt package captured speculative graphs for at most 8 token rows, so with eight requests in a batch drafting switched off. This export captures widths 1 to 6 for batches up to 8 (48 rows): +18% total output with eight agents.
- **Benchmark hygiene.** Nothing else may use the server during a run; one stock-package run that shared it with a live Claude Code session came out at less than half speed. The agent workload also depends on the files in the checkout, so compare runs made on the same checkout.
- **Prefix cache on disk.** Measured Oct 28 on ca1f960 without the eviction patch, three fixed drafts, two runs each: eight agents produced 73.8 tok/s with a 20 GiB disk store and 56.4 without, with 84% vs 70% of each prompt cached. The eviction patch alone does better (77.6 tok/s, 85% cached) and writes nothing, while the disk store wrote about 30 GB per run of the agent benchmark to read back about 4 GB, so the recipe leaves it off. The two were not measured together. If you turn it on, check the drive's endurance rating: a busy agent fleet writes far more.
- **Tried on ca1f960 and not kept:** resident n-gram scales (`--embedding-table-resident-scales`: 32K prompts 0.7 s faster, but 3 GiB less KV cache and 81.0 vs 82.1 tok/s with eight agents), a re-export with mixed-prefill graphs at more prompt sizes (+2.5% with eight agents, but one of two rounds was slower), and keeping fewer, wider-spaced prompt checkpoints (76.3 vs 77.6 tok/s).
- **Fixed three drafts** beat adaptive verification and five fixed drafts on the agent workload (1 and 8 agents); five drafts was fastest only on single-stream decode.
- One export attempt failed during warm-up with a one-off CUDA illegal instruction (Xid 13); the next attempt, with the same cached kernels, succeeded.

## Licenses

This repository is MIT. [`patches/`](patches) modifies NVIDIA's [flare-local-inference](https://github.com/NVIDIA/flare-local-inference), licensed under Apache 2.0. The model weights are under the NVIDIA Open Model Agreement, which Flare shows when you first approve a package.
