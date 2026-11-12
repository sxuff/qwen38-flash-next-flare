# Patches for Hyperflare

v2 arms B, C and D run NVIDIA/hyperflare-inference at `8a887202a0ee55d456ddcf7510161e22865505bc` (main on Nov 12, 2026).
Hyperflare is NVIDIA/flare-local-inference renamed (README rename in `d2b5c5f`, Nov 8; GitHub redirects the old name,
and v1's pin `ca1f960` is an ancestor of this commit, 250 commits earlier).

What happened to each of v1's four patches (in [`..`](..), still used by arm A on `ca1f960`):

| v1 patch | On Hyperflare 8a887202 | Here |
|---|---|---|
| `qwen38-topk-range.patch` | Still needed: the recipe's sampling ranges still have no top-k, so greedy (top-k 1) is impossible without it. Applies unchanged. | [`qwen38-topk-range.patch`](qwen38-topk-range.patch), regenerated |
| `flare-prefill-priority.patch` | Still needed: `helios/src/scheduler/utility.rs` changed only its licence header. Applies unchanged. | [`flare-prefill-priority.patch`](flare-prefill-priority.patch), regenerated |
| `flare-mixed-export-depth.patch` | Fixed upstream in `7303efb1` (Nov 2): Ignition's `GraphSignature` now gives speculative mixed graphs the depth `width - 1` itself and exports it, so the exporter's check passes. The Rust patch would still apply, but its branch could never run. | Not carried |
| `flare-oldest-first-eviction.patch` | Does not apply, and what it reverted is gone. `617ef92` (Nov 11) and `be74428` (Nov 2) rewrote the prefix cache: eviction now picks the candidate with the lowest `(use_epoch, role, consumed_tokens)`, which is least recently used first, like the patch; the dense-chain thinning of `2c99983e` no longer exists, and fixed per-request snapshot quotas became a shared pool. | Not carried; checked by arm B's prompt-cache hit rate against arm A |

`qwen38-mixed-verification-rows.patch` from the unmerged `flare-44ec239` branch is also upstream (`ignition/recipes/qwen38/decoder.py`, same condition).

DFlash2 itself needs no patch to build or serve on SM121; if that changes, the patch and the exact error go here.
