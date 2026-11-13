# v2 recipe update: X post (draft, not posted)

Attach `results/figures/v2-card.png` to the post.

**Post**

Updated my Qwen3.8 Flash-Next NVFP4 recipe for one DGX Spark / GX10.

New default: NVIDIA's Hyperflare engine + MTP. 8 coding agents finish 48 turns in 132 s vs 139 s on my Flare build, same quality.

One request at a time, DFlash2 is fastest: 55.5 tok/s.

github.com/sxuff/qwen38-flash-next-flare

**Reply (optional)**

Gotcha: a freshly restarted Hyperflare only caches a shared system prompt after its first few requests, so cold benchmarks make it look 5% slower. Warm it up, or keep it running.

Every run, pin and hash is in results/v2, plus ./run.sh --mode agents / single.
