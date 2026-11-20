"""Render image.png from scripts/figure-data.json populated by the Kaggle run.

The JSON is curated from artifacts/kaggle/<run>/metrics.json and manifest.json;
this script only draws the figure, it never invents benchmark values.
Run from the repository root with `uv run python scripts/render_figure.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
FROZEN_COLOR = "#d4a017"
ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "scripts" / "figure-data.json").read_text(encoding="utf-8"))

SYSTEMS = DATA["systems"]
# Sort by NDCG@10 so the strongest systems read first.
ORDER = sorted(SYSTEMS, key=lambda n: -SYSTEMS[n]["ndcg_at_10"])
DISPLAY = {name: SYSTEMS[name].get("display", name) for name in ORDER}
FROZEN = DATA.get("frozen_system")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
fig.subplots_adjust(left=0.075, right=0.80, top=0.85, bottom=0.30, wspace=0.30)

# Panel 1: ranking quality on the frozen test split.
x = np.arange(len(ORDER))
width = 0.38
axes[0].bar(
    x - width / 2,
    [SYSTEMS[n]["ndcg_at_10"] for n in ORDER],
    width,
    color=COLORS[0],
    label="NDCG@10",
)
axes[0].bar(
    x + width / 2,
    [SYSTEMS[n]["mrr_at_10"] for n in ORDER],
    width,
    color=COLORS[1],
    label="MRR@10",
)
axes[0].set(
    xticks=x,
    xticklabels=[DISPLAY[n] for n in ORDER],
    ylim=(0, max(1.05, max(s["ndcg_at_10"] for s in SYSTEMS.values()) * 1.15)),
    title="SciFact test split ranking quality",
)
axes[0].tick_params(axis="x", rotation=45)
for j, label in enumerate(axes[0].get_xticklabels()):
    label.set_ha("right")
    if ORDER[j] == FROZEN:
        label.set_fontweight("bold")
axes[0].legend(fontsize=9)

# Mark the frozen final system: bold tick label above, plus a star and a
# small annotation floating above its taller quality bar.
if FROZEN in SYSTEMS:
    i = ORDER.index(FROZEN)
    top = max(SYSTEMS[FROZEN]["ndcg_at_10"], SYSTEMS[FROZEN]["mrr_at_10"])
    axes[0].plot(
        i - width / 2,
        top + 0.03,
        marker="*",
        color=FROZEN_COLOR,
        markersize=15,
        markeredgecolor="#7a5b00",
        zorder=5,
    )
    axes[0].annotate(
        "(frozen)",
        (i - width / 2, top + 0.065),
        ha="center",
        fontsize=9,
        color="#8a6d00",
        clip_on=False,
    )

# Panel 2: median end-to-end per-query latency, log scale because reranking
# is ~100x slower than first-stage retrieval.
lat = [max(SYSTEMS[n].get("median_ms", 0.0), 1e-3) for n in ORDER]
axes[1].bar(x, lat, color=COLORS[2])
axes[1].set_yscale("log")
axes[1].set(
    xticks=x,
    xticklabels=[DISPLAY[n] for n in ORDER],
    ylabel="Median per-query latency (ms, log scale)",
    title="Retrieval cost per query",
)
axes[1].tick_params(axis="x", rotation=45)
for label in axes[1].get_xticklabels():
    label.set_ha("right")

# Panel 3: paired bootstrap win/tie/loss vs BM25 on NDCG@10.
comparisons = DATA.get("comparisons", {})
comp_names = [n for n in ORDER if n != "bm25" and n in comparisons]
cx = np.arange(len(comp_names))
wins = [comparisons[n]["wins"] for n in comp_names]
ties = [comparisons[n]["ties"] for n in comp_names]
losses = [comparisons[n]["losses"] for n in comp_names]
axes[2].bar(cx, wins, color=COLORS[1], label="Wins")
axes[2].bar(cx, ties, bottom=wins, color=COLORS[4], label="Ties")
axes[2].bar(
    cx,
    losses,
    bottom=np.array(wins, dtype=float) + np.array(ties, dtype=float),
    color=COLORS[2],
    label="Losses",
)
axes[2].set(
    xticks=cx,
    xticklabels=[DISPLAY[n] for n in comp_names],
    ylabel="Number of test queries",
    title="NDCG@10 vs BM25 (paired bootstrap)",
)
axes[2].tick_params(axis="x", rotation=45)
for label in axes[2].get_xticklabels():
    label.set_ha("right")
# Legend lives in the right margin so it never covers the stacked bars.
axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

fig.suptitle("Paper retrieval | real BEIR SciFact benchmark on Kaggle", fontsize=16)
fig.supxlabel(DATA.get("source", ""), fontsize=10)

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(
    ROOT / "image.png",
    dpi=180,
    bbox_inches="tight",
    pad_inches=0.25,
    metadata={"Software": "Matplotlib"},
)
plt.close(fig)
print(f"wrote {ROOT / 'image.png'} from scripts/figure-data.json")
