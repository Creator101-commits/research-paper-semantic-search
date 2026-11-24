#!/usr/bin/env python3
"""Render image.png from results/scifact_benchmark.json in the shared
portfolio style.

Run: uv run --with matplotlib python scripts/render_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D1495B"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

DPI = 150

def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
    rotation: float = 0,
    ha: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Categorical ticks: FixedLocator + FixedFormatter applied AFTER
    style(), so the numeric FuncFormatter cannot clobber the labels.
    Rotation/alignment/fontsize are set on the label Texts directly."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_xticklabels()
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_yticklabels()
    for t in texts:
        if rotation:
            t.set_rotation(rotation)
        if ha:
            t.set_ha(ha)
        if fontsize:
            t.set_fontsize(fontsize)
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def check_annotation_data(fig: plt.Figure) -> list[str]:
    """In-axes annotations vs plotted data."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(ren)
            hits = 0
            for line in ax.lines:
                if line.get_transform() != ax.transData:
                    continue
                pts = ax.transData.transform(line.get_path().vertices)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) == 0 or len(offs) > 5000:
                    continue
                pts = ax.transData.transform(offs)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            if hits:
                out.append(f"'{t.get_text()}' overlaps ~{hits} data pts")
    return out


def check_legend_pixels(fig: plt.Figure, path: Path) -> list[str]:
    """Saturated (colored) data pixels showing through legend boxes."""
    if not path.exists():
        return []
    img = plt.imread(path)
    H, W = img.shape[:2]
    rgb = img[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colored = spread > 0.18
    def _collect(o: object, swatches: list) -> None:
        if type(o).__name__ == "DrawingArea":
            swatches.append(o)
            return
        try:
            children = o.get_visible_children()
        except Exception:
            children = []
        for c in children:
            _collect(c, swatches)

    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = int(bb.x0), min(int(bb.x1) + 1, W)
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(H - bb.y0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = colored[y0:y1, x0:x1].copy()
        # mask every legend handle swatch (any column) so only data-colored
        # bleed through the legend body counts; swatches live in
        # DrawingArea children of the legend's offset box
        swatches: list = []
        box = getattr(leg, "_legend_handle_box", None)
        if box is not None:
            _collect(box, swatches)
        for sw in swatches:
            try:
                hb = sw.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if not all(np.isfinite([hb.x0, hb.x1, hb.y0, hb.y1])):
                continue
            pad = 3
            hx0, hx1 = int(hb.x0) - x0 - pad, int(hb.x1) - x0 + pad
            hy0, hy1 = int(H - hb.y1) - y0 - pad, int(H - hb.y0) - y0 + pad
            rx0, rx1 = max(0, hx0), min(region.shape[1], hx1)
            ry0, ry1 = max(0, hy0), min(region.shape[0], hy1)
            if rx1 > rx0 and ry1 > ry0:
                region[ry0:ry1, rx0:rx1] = False
        frac = region.mean() if region.size else 0.0
        if frac > 0.015:
            out.append(f"legend body ~{frac:.1%} covered by data color")
    return out


def finish(fig: plt.Figure, out: Path) -> None:
    """Verify categorical labels, save, then fail on any check violation."""
    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(out, dpi=DPI, facecolor="white")
    problems: list[str] = []
    problems += check_text_overlap(fig)
    problems += check_legend_data(fig)
    problems += check_annotation_data(fig)
    problems += check_legend_pixels(fig, out)
    plt.close(fig)
    if problems:
        for p in problems[:30]:
            print(f"  CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(f"figure checks failed for {out.name}: {len(problems)} problem(s)")
    print("wrote", out)



ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "image.png"

FROZEN = "hybrid-rrf"
BEST_DENSE = "dense-bge-base"

DISPLAY = {
    "bm25": "BM25",
    "tfidf": "Hashed TF-IDF",
    "dense-minilm": "MiniLM-L6",
    "dense-bge-base": "BGE-base",
    "dense-specter": "SPECTER",
    "hybrid-rrf": "Hybrid RRF",
    "hybrid+lexical-rerank": "Lexical rerank",
    "hybrid+ce-rerank@20": "CE rerank@20",
    "hybrid+ce-rerank@50": "CE rerank@50",
    "hybrid+ce-rerank@100": "CE rerank@100",
}

DELTA_SYSTEMS = [FROZEN, BEST_DENSE]


def main() -> None:
    bench = json.loads(
        (RESULTS / "scifact_benchmark.json").read_text(encoding="utf-8")
    )
    systems = bench["test_systems"]
    comps = bench["comparisons_vs_bm25"]
    deltas = bench["per_query_ndcg_deltas_vs_bm25"]
    order = sorted(
        systems, key=lambda n: -systems[n]["ndcg_at_10"]
    )

    fig = plt.figure(figsize=(19.0, 10.0), dpi=DPI)
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(2, 3)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[0, 2])
    a4 = fig.add_subplot(gs[1, 0])
    a5 = fig.add_subplot(gs[1, 1])
    a6 = fig.add_subplot(gs[1, 2])

    # (a) test ranking quality: nDCG@10 and MRR@10 dots per system.
    ys = np.arange(len(order))[::-1]
    for metric, color, label in (
        ("ndcg_at_10", BLUE, "nDCG@10"),
        ("mrr_at_10", TEAL, "MRR@10"),
    ):
        for y, name in zip(ys, order, strict=True):
            frozen = name == FROZEN
            a1.plot(
                systems[name][metric], y,
                marker="o", ms=7 if frozen else 5.5,
                color=AMBER if frozen else color,
                markeredgecolor=INK if frozen else "none",
                markeredgewidth=1.2 if frozen else 0,
                zorder=3,
            )
        a1.plot([], [], marker="o", ls="none", color=color, label=label)
    style(
        a1, "a", "Test ranking quality (SciFact, n=300)",
        xlabel="score", grid=False,
    )
    a1.set_xlim(0.0, 0.85)
    cat(a1, "y", ys, [DISPLAY[n] for n in order], fontsize=7.5)
    a1.legend(fontsize=7, loc="upper left", ncol=2)
    frozen_y = ys[order.index(FROZEN)]
    a1.annotate(
        "frozen", (systems[FROZEN]["ndcg_at_10"] + 0.02, frozen_y),
        va="center", fontsize=8, color=AMBER,
    )

    # (b) per-query nDCG@10 change vs BM25 for frozen + best dense.
    rng = np.random.default_rng(0)
    for row, name in enumerate(DELTA_SYSTEMS):
        vals = np.array(
            [float(v) for v in deltas[name].values()], dtype=float
        )
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        frozen = name == FROZEN
        a2.scatter(
            vals, row + jitter, s=5.5,
            color=AMBER if frozen else TEAL,
            alpha=0.4 if not frozen else 0.55,
            edgecolors="none", zorder=2,
        )
        mean_d = float(comps[name]["mean_delta"])
        a2.plot(
            mean_d, row, marker="D", ms=8,
            color=AMBER if frozen else INK,
            markeredgecolor=INK, zorder=4,
        )
    a2.axvline(0.0, color=SPINE, lw=1.0)
    style(
        a2, "b",
        "Per-query nDCG@10 change vs BM25 "
        f"(means: {DISPLAY[FROZEN]} "
        f"{comps[FROZEN]['mean_delta']:+.3f}, "
        f"{DISPLAY[BEST_DENSE]} "
        f"{comps[BEST_DENSE]['mean_delta']:+.3f})",
        xlabel="per-query nDCG@10 delta",
    )
    cat(a2, "y", [0, 1], [DISPLAY[BEST_DENSE], DISPLAY[FROZEN]])
    a2.set_ylim(-0.6, 1.6)
    a2.set_yticks([0, 1])

    # (c) mean nDCG@10 delta vs BM25 with paired bootstrap CI.
    comp_order = sorted(
        comps, key=lambda n: -comps[n]["mean_delta"]
    )
    yc = np.arange(len(comp_order))[::-1]
    for y, name in zip(yc, comp_order, strict=True):
        c = comps[name]
        col = AMBER if name == FROZEN else (
            BLUE if c["mean_delta"] > 0 else CORAL
        )
        a3.errorbar(
            c["mean_delta"], y,
            xerr=[
                [c["mean_delta"] - c["ci_lower"]],
                [c["ci_upper"] - c["mean_delta"]],
            ],
            fmt="o", ms=6, color=col, ecolor=col,
            elinewidth=1.4, capsize=4,
        )
    a3.axvline(0.0, color=SPINE, lw=1.0)
    style(
        a3, "c", "Mean nDCG@10 delta vs BM25 (paired bootstrap 95% CI)",
        xlabel="mean delta",
        grid=False,
    )
    cat(a3, "y", yc, [DISPLAY[n] for n in comp_order], fontsize=7)

    # (d) median per-query latency, log scale, horizontal bars.
    yd = np.arange(len(order))[::-1]
    lats = [
        max(float(systems[n]["latency"]["median_ms"]), 1e-3)
        for n in order
    ]
    bar_colors = [AMBER if n == FROZEN else BLUE for n in order]
    a4.barh(yd, lats, color=bar_colors, alpha=0.85, height=0.62)
    a4.set_xscale("log")
    a4.set_xlim(0.7, 2000.0)
    style(
        a4, "d", "Median retrieval cost per query (log ms)",
        xlabel="median ms (log)", grid=False,
    )
    a4.xaxis.set_major_formatter(
        mticker.LogFormatterMathtext(labelOnlyBase=True)
    )
    cat(a4, "y", yd, [DISPLAY[n] for n in order], fontsize=7.5)

    # (e) paired per-query wins/ties/losses vs BM25 on nDCG@10.
    wt_order = [n for n in comp_order]
    yw = np.arange(len(wt_order))
    wins = np.array([comps[n]["wins"] for n in wt_order], dtype=float)
    ties = np.array([comps[n]["ties"] for n in wt_order], dtype=float)
    losses = np.array([comps[n]["losses"] for n in wt_order], dtype=float)
    a5.barh(yw, wins, color=TEAL, alpha=0.85, label="wins")
    a5.barh(
        yw, ties, left=wins, color=GRAY, alpha=0.7, label="ties"
    )
    a5.barh(
        yw, losses, left=wins + ties, color=CORAL, alpha=0.85,
        label="losses",
    )
    style(
        a5, "e", "Per-query nDCG@10 vs BM25 (300 queries)",
        xlabel="queries", grid=False,
    )
    a5.set_xlim(0, 305)
    cat(a5, "y", yw, [DISPLAY[n] for n in wt_order], fontsize=7)
    a5.legend(
        fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
        ncol=1,
    )

    # (f) test recall@100 dot plot.
    yf = np.arange(len(order))[::-1]
    for y, name in zip(yf, order, strict=True):
        frozen = name == FROZEN
        a6.plot(
            systems[name]["recall_at_100"], y,
            marker="o", ms=7 if frozen else 5.5,
            color=AMBER if frozen else BLUE,
            markeredgecolor=INK if frozen else "none",
            markeredgewidth=1.2 if frozen else 0,
            zorder=3,
        )
    style(
        a6, "f", "Test recall@100 (deep retrieval coverage)",
        xlabel="recall@100", grid=False,
    )
    a6.set_xlim(0.80, 1.0)
    cat(a6, "y", yf, [DISPLAY[n] for n in order], fontsize=7.5)
    a6.annotate(
        "frozen", (systems[FROZEN]["recall_at_100"] + 0.006, yf[order.index(FROZEN)]),
        va="center", fontsize=8, color=AMBER,
    )

    finish(fig, OUT)


if __name__ == "__main__":
    main()
