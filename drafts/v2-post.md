# v2 recipe update: X post (draft, not posted)

One long post; attach `results/figures/v2-card.png`. Post it after PR #3 is merged: until then the repo link shows v1.

---

updated my qwen3.8 flash-next nvfp4 recipe for the dgx spark / gx10. v2 moves to nvidia's new hyperflare engine.

8 coding agents at once, 48 turns of code review:

→ hyperflare + mtp: 88.3 tok/s, all done in 132 s
→ hyperflare + dflash2: 88.5 tok/s, 134 s
→ my old flare build: 85.4 tok/s, 139 s

one agent: 37–38 tok/s on all three. one request at a time, dflash2 decodes fastest: 55.5 tok/s vs 52.7.

same gsm8k and humaneval scores everywhere.

the part that almost fooled me: restart hyperflare and it won't cache your shared system prompt until it has timed a few requests. my first benchmark ran it cold, every agent re-read 6k tokens on turn one, and it lost by 5%. warm it up, or just leave it running, and it wins.

two more things i checked so you don't have to:

· dflash2's edge is the drafter, not the extra draft. mtp with 4 drafts gains nothing over 3.
· nvfp4 doesn't break the drafter. it still beats mtp's acceptance by 7–10% on gsm8k / humaneval, and only fades on long agent contexts.

the margins are small (5 runs, p ≈ 0.08 at 8 agents), so every run, pin and hash is in the repo. one command:

./run.sh --mode agents

github.com/sxuff/qwen38-flash-next-flare
