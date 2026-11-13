# v2 recipe update: X post (draft, not posted)

One long post. Post it after PR #3 is merged: until then the repo link shows v1. Every number is arm B in results/v2/summary.json
(warmed engine for agents; frozen harness for one request).

---

updated my qwen3.8 flash-next nvfp4 recipe for the dgx spark / gx10. v2 runs on nvidia's new hyperflare engine with mtp drafts.

8 coding agents at once: 88.3 tok/s. 48 turns of code review done in 132 s, first token in ~2 s.

1 agent: 38 tok/s.

1 request: 51 tok/s decode, and a 32k-token prompt answered in under 14 s.

85% of every agent prompt comes straight from the prefix cache. greedy, thinking on, gsm8k 48–49/50, humaneval 13/20.

one tip: give it a few requests after a restart. hyperflare only starts caching a shared system prompt once it has timed some work, so the first agents after a cold start re-read it.

one command, every pin and hash in the repo:

./run.sh

github.com/sxuff/qwen38-flash-next-flare
