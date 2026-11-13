#!/usr/bin/env python3
"""Draw the v2 result card (results/figures/v2-card.png) from results/v2/summary.json, in the v1 card style.

  python bench/v2/card.py [summary.json] [out.png]
Left: total output at 1 and 8 agents on a warmed engine (arms Aw, Bw, D4w). Right: time to finish 48 turns at
8 agents, and one-request decode (frozen harness). Every number is read from summary.json.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from figures import BG, GRID, MUTED, TEXT, card, legend, style_axis  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/v2/summary.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/figures/v2-card.png"
d = json.loads(SUMMARY.read_text(encoding="utf-8"))
ARMS = [("Aw", "A", "Flare ca1f960 + MTP (v1 recipe)", "#3b88e6"),
        ("Bw", "B", "Hyperflare + MTP (new default)", "#2fa36b"),
        ("D4w", "D", "Hyperflare + DFlash2", "#e07a2e")]

fig = card("New default: Hyperflare + MTP", "Qwen3.8-Flash-Next NVFP4 · one DGX Spark / GX10 (GB10) · greedy decoding",
           None,
           ["Same 6-turn code-review sessions on every engine, warmed by three requests as a running server is; mean of "
            "five runs. Same GSM8K and HumanEval scores.",
            "Freshly restarted, the v1 recipe is faster (137 s vs 145 s): Hyperflare caches a shared system prompt only "
            "after its first few requests."],
           "Nov 13, 2026")
legend(fig, [{"label": f"{short} · {label}", "color": color} for _, short, label, color in ARMS], y=0.77)

ax = fig.add_axes([0.15, 0.2, 0.5, 0.48])
style_axis(ax)
h = 0.26
for i, level in enumerate((1, 8)):
    for j, (key, short, _, color) in enumerate(ARMS):
        v = d[key][f"agents-{level}"]["total_tok_s"]
        y = -(i + (j - 1) * h)
        ax.barh(y, v, height=h * 0.9, color=color)
        ax.text(1.5, y, short, va="center", fontsize=14, weight="bold", color="white")
        ax.text(v + 1.2, y, f"{v:.1f}", va="center", fontsize=15, weight="bold")
ax.set_yticks([0, -1], ["1 agent", "8 agents"], fontsize=17, color=TEXT)
ax.set_xlim(0, 100)
ax.set_xlabel("total output, tok/s", color=MUTED, fontsize=13)

x0 = 0.71
fig.add_artist(plt.Line2D([x0 - 0.015, x0 - 0.015], [0.2, 0.68], color=GRID, linewidth=1.5, transform=fig.transFigure))
fig.text(x0, 0.655, "8 agents: 48 turns done in", fontsize=15, color=MUTED)
for k, (key, short, label, color) in enumerate(ARMS[:2][::-1]):
    y = 0.575 - 0.075 * k
    fig.text(x0, y, f"{d[key]['agents-8']['finish_s']:.0f} s", fontsize=40 if k == 0 else 30, weight="bold")
    fig.text(x0 + 0.105, y + 0.008, f"■ {short}", fontsize=16, color=color, weight="bold")
fig.text(x0, 0.39, "One request: decode, tok/s", fontsize=15, color=MUTED)
for k, (key, short, color) in enumerate((("D4", "D", "#e07a2e"), ("A", "A", "#3b88e6"))):
    y = 0.31 - 0.075 * k
    fig.text(x0, y, f"{d[key]['single']['decode_tok_s']:.1f}", fontsize=40 if k == 0 else 30, weight="bold")
    fig.text(x0 + 0.105, y + 0.008, f"■ {short}", fontsize=16, color=color, weight="bold")
fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
