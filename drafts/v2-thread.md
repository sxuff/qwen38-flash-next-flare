# v2 thread (draft, not posted)

Three posts. Attach `results/figures/v2-time-to-finish.png` to the first and `results/figures/v2-output-by-agents.png` to the second.

**1/3**

I tested NVIDIA's new Hyperflare engine and the DFlash2 drafter against my Flare recipe for Qwen3.8 Flash-Next NVFP4 on one GB10.

On a running server, Hyperflare with plain MTP is ~5% faster for 8 coding agents: 48 turns in 132 s vs 139 s. DFlash2 ties it.

**2/3**

Crossover at ~4 agents: below it all three tie, above it Hyperflare leads by 3-5%.

The catch: a freshly started Hyperflare skips caching the shared system prompt for its first sessions, so measured cold my recipe wins (137 s vs 145 s). DFlash2 only wins one-request decode (+5%).

**3/3**

Every run, pin and hash is in the repo, cold-start trap included, plus `./run.sh --mode agents` and `--mode single`. Same quality on GSM8K and HumanEval.

github.com/sxuff/qwen38-flash-next-flare
