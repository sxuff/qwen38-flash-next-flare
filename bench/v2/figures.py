#!/usr/bin/env python3
"""Draw the v2 figures from results/v2/summary.json in the style of results/figures (dark card, 3200 x 1800).

  python bench/v2/figures.py [summary.json] [out_dir]
Writes v2-time-to-finish.png (grouped bars by agent count) and v2-output-by-agents.png (the crossover chart).
Segoe UI matches the v1 cards; any sans-serif works.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/v2/summary.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/figures"
BG, CARD, EDGE, TEXT, MUTED, GRID = "#0d0d0d", "#1c1c1b", "#2c2c2a", "#f2f2f0", "#a8a8a3", "#33332f"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "text.color": TEXT, "axes.edgecolor": GRID})


def card(title, kicker, subtitle, foot, date):
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=BG)
    fig.patches.append(FancyBboxPatch((0.018, 0.03), 0.964, 0.937, boxstyle="round,pad=0,rounding_size=0.018",
                                      transform=fig.transFigure, facecolor=CARD, edgecolor=EDGE, linewidth=1.2, zorder=-10))
    fig.text(0.058, 0.893, kicker, fontsize=15, color=MUTED)
    fig.text(0.058, 0.82, title, fontsize=38, weight="bold")
    if subtitle:
        fig.text(0.058, 0.77, subtitle, fontsize=16, color=TEXT)
    for i, line in enumerate(foot):
        fig.text(0.058, 0.115 - 0.027 * i, line, fontsize=11.5, color=MUTED)
    fig.text(0.942, 0.088, date, fontsize=11.5, color=MUTED, ha="right")
    return fig


def legend(fig, arms, x=0.058, y=0.715):
    for arm in arms:
        fig.patches.append(FancyBboxPatch((x, y), 0.0085, 0.015, boxstyle="round,pad=0,rounding_size=0.002",
                                          transform=fig.transFigure, facecolor=arm["color"], edgecolor="none"))
        t = fig.text(x + 0.014, y + 0.0075, arm["label"], fontsize=14, va="center")
        fig.canvas.draw()
        x += 0.014 + t.get_window_extent().width / fig.bbox.width + 0.022


def style_axis(ax):
    ax.set_facecolor(CARD)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=12, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def time_to_finish(data, arms, levels, foot, date, path):
    fig = card("Time to finish every turn", "Qwen3.8-Flash-Next NVFP4 · one GB10 · greedy decoding", None, foot, date)
    legend(fig, arms, y=0.77)
    ax = fig.add_axes([0.15, 0.2, 0.74, 0.52])
    style_axis(ax)
    n, h = len(arms), 0.8 / len(arms)
    top = max(data[a["key"]][f"agents-{l}"]["finish_s"] for a in arms for l in levels if f"agents-{l}" in data[a["key"]])
    for i, level in enumerate(levels):
        for j, arm in enumerate(arms):
            cell = data[arm["key"]].get(f"agents-{level}")
            if not cell:
                continue
            y = -(i + j * h - 0.4 + h / 2)
            ax.barh(y, cell["finish_s"], height=h * 0.92, color=arm["color"])
            ax.text(cell["finish_s"] + top * 0.012, y, f"{cell['finish_s']:.0f} s", va="center", fontsize=13, weight="bold")
    ax.set_yticks([-i for i in range(len(levels))], [f"{l} agent{'s' * (l > 1)}" for l in levels], fontsize=16, color=TEXT)
    ax.set_xlim(0, top * 1.12)
    ax.set_xlabel("seconds until the last turn ends (6 turns per agent)", color=MUTED, fontsize=12)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def output_by_agents(data, arms, levels, foot, date, path, note=None, cold=()):
    fig = card("Total output by agent count", "Qwen3.8-Flash-Next NVFP4 · one GB10 · greedy decoding", note, foot, date)
    legend(fig, arms)
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.47])
    style_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for arm in cold:  # the same arms on a freshly started engine: thin dashed lines, for contrast
        xs = [l for l in levels if f"agents-{l}" in data.get(arm["key"], {})]
        ax.plot(xs, [data[arm["key"]][f"agents-{l}"]["total_tok_s"] for l in xs], color=arm["color"], linewidth=1.8,
                linestyle=(0, (4, 3)), alpha=0.75)
    if cold:
        fig.text(0.058, 0.68, "Solid: engine warmed by three requests. Dashed: freshly started engine, one warm-up request.",
                 fontsize=13, color=MUTED)
    ends = []
    for arm in arms:
        xs = [l for l in levels if f"agents-{l}" in data[arm["key"]]]
        ys = [data[arm["key"]][f"agents-{l}"]["total_tok_s"] for l in xs]
        ax.plot(xs, ys, color=arm["color"], linewidth=4, marker="o", markersize=11, solid_capstyle="round")
        ends.append([ys[-1], ys[-1], xs[-1], arm["color"]])
    gap = 0.045 * max(e[0] for e in ends)  # keep end labels apart when lines finish close together
    ends.sort(key=lambda e: e[0])
    for lower, upper in zip(ends, ends[1:]):
        upper[1] = max(upper[1], lower[1] + gap)
    for value, y, x, color in ends:
        ax.text(x * 1.06, y, f"{value:.1f}", color=color, fontsize=15, weight="bold", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, [str(l) for l in levels], fontsize=14, color=TEXT)
    ax.set_xlim(levels[0] * 0.85, levels[-1] * 1.35)
    ax.set_ylim(0, None)
    ax.set_xlabel("concurrent agents", color=MUTED, fontsize=12)
    ax.set_ylabel("total output, tok/s", color=MUTED, fontsize=12)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    spec = json.loads((Path(__file__).with_name("figures.json")).read_text(encoding="utf-8"))
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    arms = [a for a in spec["arms"] if a["key"] in data]
    time_to_finish(data, arms, spec["levels"], spec["foot"], spec["date"], OUT / "v2-time-to-finish.png")
    cold = [c for c in spec.get("cold", []) if c["key"] in data]
    output_by_agents(data, arms, spec["levels"], spec["foot"], spec["date"], OUT / "v2-output-by-agents.png", spec.get("note"),
                     cold)
    print("wrote", OUT / "v2-time-to-finish.png", OUT / "v2-output-by-agents.png")
