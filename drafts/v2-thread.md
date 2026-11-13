# v2 thread (draft, not posted)

Three posts. Attach `results/figures/v2-time-to-finish.png` to the first and `results/figures/v2-output-by-agents.png` to the second.

**1/3**

I tested NVIDIA's new Hyperflare engine and the DFlash2 drafter against my Flare recipe for Qwen3.8 Flash-Next NVFP4 on one GB10.

For coding agents the older recipe still wins: 8 agents finish all 48 turns in 137 s, vs 144 s on Hyperflare with MTP and 149 s with DFlash2.

**2/3**

No crossover: Flare is ahead at 1, 2, 4 and 8 agents (tied at 1).

The gap is the prompt cache: 85% reused vs 77%, so Hyperflare's slowest first tokens come ~3x later. DFlash2 wins only one request at a time (+5% decode), and its acceptance per draft matches MTP on these weights.

**3/3**

Every run, pin and hash is in the repo now, plus `./run.sh --mode agents` and `--mode single`. Same quality on GSM8K and HumanEval.

github.com/sxuff/qwen38-flash-next-flare
