# v2 recipe update: X post (draft, not posted)

One long post; attach `results/figures/v2-card.png`.

---

i almost told you nvidia's new engine was slower than my old recipe. it wasn't. it was a cold start.

qwen3.8 flash-next nvfp4 on one dgx spark / gx10, 8 coding agents doing 6-turn code reviews:

– hyperflare + mtp: 48 turns in 132 s
– hyperflare + dflash2: 134 s
– my v1 flare build: 139 s

so the new default is hyperflare + mtp. same gsm8k and humaneval scores.

the catch: a freshly restarted hyperflare doesn't cache the shared system prompt until it has timed a few requests. benchmark it cold and every agent re-reads ~6k tokens on turn one. it looks 5% slower and the slowest first tokens take 19 s. warm it with a few requests, or just leave it running, and it pulls ahead.

dflash2 is the pick when you send one request at a time: 55.5 tok/s vs 52.7. that's the drafter, not the extra draft: mtp with 4 drafts gains nothing over 3.

and no, nvfp4 doesn't break the drafter. it still beats mtp's acceptance by 7–10% on gsm8k / humaneval. it just stops helping on long agent contexts.

margins are small (5 runs, p ≈ 0.08 at 8 agents), so every run, pin and hash is in the repo. one line to run it:

./run.sh --mode agents

github.com/sxuff/qwen38-flash-next-flare
