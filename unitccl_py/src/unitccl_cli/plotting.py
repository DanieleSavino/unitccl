"""Rank-sweep plots: time-vs-message-size, time-vs-rank-count, and a
best-algorithm heatmap (size x ranks). Consumes the
`<N>_ranks/<coll>/<coll>_<proto>.csv` layout written by
`unitccl scaling ... ranks=...` (see slurm_utils.submit_rank_sweep).
"""
from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .logging_utils import ok
from .schema import find_protos, load_rank_sweep, records_to_df, size_to_bytes

# Short names used to annotate the best-algo heatmap cells.
ALGO_ABBREV = {
    "BINE-SEND": "BS",
    "BINE-BLOCK_BY_BLOCK": "BB",
    "BINE-DOUBLE_SEND": "BD",
    "BINE-PERMUTATION": "BP",
    "BINE-TREE" : "BT",
    "PAT": "P",
    "TREE": "T",
    "RING": "R",
}
# Fixed colors so an algo keeps its color across every plot.
ALGO_COLORS = {
    "BINE-SEND": "#00d2ff",
    "BINE-BLOCK_BY_BLOCK": "#3a7bd5",
    "BINE-DOUBLE_SEND": "#a8ff78",
    "BINE-PERMUTATION": "#c471ed",
    "BINE-TREE": "#ff3cac",
    "PAT": "#f7971e",
    "TREE": "#f9e45b",
    "RING": "#ff6b6b",
}
_FALLBACK_COLORS = ["#4ecdc4", "#ff9ff3", "#c8d6e5", "#ee5253", "#10ac84"]

# Distinguishes rank counts when several are drawn in the same axes.
_RANK_LINESTYLES = ["-", "--", ":", "-."]


def _algo_colors(algos) -> dict:
    """{algo: color}. Known algos get their fixed color; unknown ones get
    fallback colors assigned in sorted order, so the result is deterministic.
    Pass the full (unfiltered) algo list so colors stay stable across plots."""
    fallback = cycle(_FALLBACK_COLORS)
    return {a: ALGO_COLORS.get(a) or next(fallback) for a in sorted(algos)}


def _abbrev(algo: str) -> str:
    if algo in ALGO_ABBREV:
        return ALGO_ABBREV[algo]
    # Unknown label: initials of its '-'/'_' separated parts.
    return "".join(w[0] for w in algo.replace("-", "_").split("_") if w).upper() or algo


def _apply_dark_theme(ax, xlabel: str, ylabel: str, title: str):
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, color="#333355", linestyle=":", alpha=0.5)
    ax.tick_params(colors="#aaaaaa", which="both")
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
    ax.set_xlabel(xlabel, color="#cccccc", fontsize=11)
    ax.set_ylabel(ylabel, color="#cccccc", fontsize=11)
    ax.set_title(title, color="#e0e0e0", fontsize=13)


def _style_legend(ax):
    legend = ax.legend(loc="upper left", fontsize=10, facecolor="#1a1a2e", edgecolor="#333355")
    if legend:
        for text in legend.get_texts():
            text.set_color("#cccccc")


def _set_size_ticks(ax, df):
    unique_bytes = sorted(df["size_bytes"].unique())
    ax.set_xticks(unique_bytes)
    byte_to_str = dict(zip(df["size_bytes"], df["size_str"]))
    ax.set_xticklabels([byte_to_str[b] for b in unique_bytes])


def plot_size(df: pd.DataFrame, collective: str, proto: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1a1a2e")
    _apply_dark_theme(ax, "Message size", "Latency (µs)", f"{collective} scaling — {proto}")

    has_ranks = df["ranks"].notna().any()
    ranks_list = sorted(df["ranks"].dropna().astype(int).unique()) if has_ranks else [None]
    algos = sorted(df["algo"].unique())
    colors = _algo_colors(algos)

    for ri, ranks in enumerate(ranks_list):
        for algo in algos:
            if ranks is not None:
                sub = df[(df["ranks"] == ranks) & (df["algo"] == algo)].sort_values("size_bytes")
                label = f"{algo} - {ranks} ranks"
            else:
                sub = df[(df["ranks"].isna()) & (df["algo"] == algo)].sort_values("size_bytes")
                label = f"{algo}"

            if sub.empty:
                continue

            ax.plot(
                sub["size_bytes"],
                sub["mean_ns"] / 1e3,
                marker="D",
                markersize=8,
                linewidth=2.5,
                linestyle=_RANK_LINESTYLES[ri % len(_RANK_LINESTYLES)],
                color=colors[algo],
                label=label,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _set_size_ticks(ax, df)
    _style_legend(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="#1a1a2e")
    plt.close(fig)
    ok(f"wrote {out}")


def plot_diff(df: pd.DataFrame, collective: str, proto: str, out: Path, baseline_algo: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1a1a2e")
    _apply_dark_theme(ax, "Message size", f"Δ vs {baseline_algo}", f"{collective} scaling — {proto}  (Δ vs {baseline_algo})")

    has_ranks = df["ranks"].notna().any()
    ranks_list = sorted(df["ranks"].dropna().astype(int).unique()) if has_ranks else [None]
    algos = sorted(df["algo"].unique())
    # Build the color map from the full sorted list BEFORE reordering, so
    # colors match plot_size regardless of draw order.
    colors = _algo_colors(algos)

    # Draw the baseline first (legend order only; color is unaffected).
    if baseline_algo in algos:
        algos.remove(baseline_algo)
        algos.insert(0, baseline_algo)

    for ri, ranks in enumerate(ranks_list):
        if ranks is not None:
            base_df = df[(df["ranks"] == ranks) & (df["algo"] == baseline_algo)]
        else:
            base_df = df[(df["ranks"].isna()) & (df["algo"] == baseline_algo)]

        if base_df.empty:
            continue
        base_map = dict(zip(base_df["size_bytes"], base_df["mean_ns"]))

        for algo in algos:
            if ranks is not None:
                sub = df[(df["ranks"] == ranks) & (df["algo"] == algo)].sort_values("size_bytes")
                label = f"{algo} - {ranks} ranks"
            else:
                sub = df[(df["ranks"].isna()) & (df["algo"] == algo)].sort_values("size_bytes")
                label = f"{algo}"

            if sub.empty:
                continue

            y_diff = []
            for val, sz in zip(sub["mean_ns"], sub["size_bytes"]):
                base_val = base_map.get(sz)
                if base_val and base_val > 0:
                    perc = ((val - base_val) / base_val) * 100.0
                else:
                    perc = 0.0
                y_diff.append(perc)

            ax.plot(
                sub["size_bytes"],
                y_diff,
                marker="D",
                markersize=8,
                linewidth=2.5,
                linestyle=_RANK_LINESTYLES[ri % len(_RANK_LINESTYLES)],
                color=colors[algo],
                label=label,
            )

    ax.set_xscale("log", base=2)
    _set_size_ticks(ax, df)

    # Format Y-axis to show percentages (+300.0%, +0.0%, etc.)
    def perc_formatter(y, _):
        return f"{'+' if y > 0 else ''}{y:.1f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(perc_formatter))

    # Subtle zero-line reference
    ax.axhline(0, color="#ffffff", linewidth=1, linestyle=":", alpha=0.3)
    _style_legend(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="#1a1a2e")
    plt.close(fig)
    ok(f"wrote {out}")


def plot_ranks(
    df: pd.DataFrame, collective: str, proto: str, out: Path, size_filter: Optional[str] = None
) -> None:
    fig_facecolor = "#1a1a2e"

    if size_filter:
        sizes = [size_filter]
        sub_df = df[df["size_str"] == size_filter]
        if sub_df.empty:
            raise SystemExit(f"No rows found with size_str == '{size_filter}'.")
        fig, axes = plt.subplots(1, 1, figsize=(7, 5), facecolor=fig_facecolor)
        axes = [axes]
    else:
        sizes = sorted(df["size_str"].unique(), key=size_to_bytes)
        n = len(sizes)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, facecolor=fig_facecolor)
        axes = axes.flatten()

    algos = sorted(df["algo"].unique())
    colors = _algo_colors(algos)

    for ax, size_str in zip(axes, sizes):
        _apply_dark_theme(ax, "Ranks", "Latency (µs)", size_str)
        sub = df[df["size_str"] == size_str]

        for algo in algos:
            asub = sub[sub["algo"] == algo].sort_values("ranks")
            if asub.empty:
                continue

            ax.plot(
                asub["ranks"],
                asub["mean_ns"] / 1e3,
                marker="D",
                markersize=8,
                linewidth=2.5,
                color=colors[algo],
                label=algo,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted(sub["ranks"].dropna().astype(int).unique()))
        ax.set_xticklabels([int(x) for x in sorted(sub["ranks"].dropna().astype(int).unique())])
        _style_legend(ax)

    for ax in axes[len(sizes) :]:
        ax.axis("off")

    fig.suptitle(f"{collective} ({proto}) — time vs ranks", color="#e0e0e0", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig_facecolor)
    plt.close(fig)
    ok(f"wrote {out}")


def plot_heatmap(df: pd.DataFrame, collective: str, proto: str, out: Path) -> None:
    """Best-algorithm heatmap: x = message size, y = number of ranks, each cell
    labelled with the (abbreviated) algo with the lowest mean latency. Under the
    winner, the closest runners-up are listed as `<abbrev> +<slowdown>%`; as many
    as fit inside the cell (measured from the final layout) are shown."""
    df = df.dropna(subset=["ranks", "mean_ns"])
    if df.empty:
        raise SystemExit(f"No rank-tagged data to build a heatmap for {collective} ({proto}).")

    sizes = sorted(df["size_bytes"].unique())
    ranks_list = sorted(df["ranks"].astype(int).unique())
    byte_to_str = dict(zip(df["size_bytes"], df["size_str"]))

    # All algos present (not just winners) so runners-up are in the legend too.
    algos = sorted(df["algo"].unique())
    algo_to_i = {a: i for i, a in enumerate(algos)}
    color_map = _algo_colors(algos)
    colors = [color_map[a] for a in algos]

    # Per cell: algos ordered best -> worst with slowdown vs the winner (%).
    grid = np.full((len(ranks_list), len(sizes)), np.nan)
    cell_rank: dict = {}
    for (r, sz), g in df.groupby(["ranks", "size_bytes"]):
        g = g.sort_values("mean_ns")
        best_ns = g["mean_ns"].iloc[0]
        yi, xi = ranks_list.index(int(r)), sizes.index(sz)
        grid[yi, xi] = algo_to_i[g["algo"].iloc[0]]
        cell_rank[(yi, xi)] = [
            (a, (ns - best_ns) / best_ns * 100.0 if best_ns > 0 else 0.0)
            for a, ns in zip(g["algo"], g["mean_ns"])
        ]

    fig_facecolor = "#1a1a2e"
    fig, ax = plt.subplots(
        figsize=(max(7, 1.5 * len(sizes) + 3), max(4, 1.0 * len(ranks_list) + 2.5)),
        facecolor=fig_facecolor,
    )
    cmap = ListedColormap(colors)
    cmap.set_bad("#2a2a44")
    ax.imshow(
        np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5, vmax=len(algos) - 0.5,
        aspect="auto", origin="lower",
    )

    _apply_dark_theme(ax, "Message size", "Ranks", f"{collective} ({proto}) — best algorithm")
    ax.grid(False)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([byte_to_str[b] for b in sizes])
    ax.set_yticks(range(len(ranks_list)))
    ax.set_yticklabels(ranks_list)
    ax.set_xticks(np.arange(-0.5, len(sizes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ranks_list), 1), minor=True)
    ax.grid(which="minor", color=fig_facecolor, linewidth=2)
    ax.tick_params(which="minor", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in colors]
    legend = ax.legend(
        handles, [f"{_abbrev(a)} = {a}" for a in algos],
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10,
        facecolor=fig_facecolor, edgecolor="#333355",
    )
    for text in legend.get_texts():
        text.set_color("#cccccc")

    fig.tight_layout()

    # How many runner-up lines fit? Measure the real cell size (points) after
    # layout, leaving a margin so the text never touches the cell border.
    bbox = ax.get_window_extent()
    cell_h_pt = bbox.height / len(ranks_list) * 72.0 / fig.dpi
    cell_w_pt = bbox.width / len(sizes) * 72.0 / fig.dpi
    main_pt, line_pt, pad_pt = 15.0, 9.0, 12.0
    fit = int((cell_h_pt - pad_pt - main_pt) // line_pt)
    # "BB +123.4%" at 7pt needs ~45pt of width; give up the margin if too narrow.
    n_runners = max(0, min(fit, len(algos) - 1)) if cell_w_pt >= 55 else 0

    for (yi, xi), ranked in cell_rank.items():
        runners = ranked[1 : 1 + n_runners]
        total = main_pt + line_pt * len(runners)
        top = total / 2.0
        ax.annotate(
            _abbrev(ranked[0][0]), (xi, yi), xytext=(0, top - main_pt / 2.0),
            textcoords="offset points", ha="center", va="center",
            color="#10101e", fontsize=12, fontweight="bold",
        )
        for k, (a, pct) in enumerate(runners):
            ax.annotate(
                f"{_abbrev(a)} +{pct:.0f}%" if pct >= 10 else f"{_abbrev(a)} +{pct:.1f}%",
                (xi, yi), xytext=(0, top - main_pt - line_pt * (k + 0.5)),
                textcoords="offset points", ha="center", va="center",
                color="#10101e", fontsize=7, alpha=0.8,
            )

    fig.savefig(out, dpi=200, facecolor=fig_facecolor, bbox_inches="tight")
    plt.close(fig)
    ok(f"wrote {out}")


def run_plot_command(
    mode: str,
    root: Path,
    collectives: List[str],
    protos: Optional[List[str]],
    outdir: Path,
    size_filter: Optional[str] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for collective in collectives:
        protos_to_use = protos or find_protos(root, collective)
        if not protos_to_use:
            raise SystemExit(f"No protocols found for collective '{collective}' under {root}")
        for proto in protos_to_use:
            records = load_rank_sweep(root, collective, proto)
            df = records_to_df(records)
            if mode == "size":
                out = outdir / f"{collective}_{proto}_scaling.png"
                plot_size(df, collective, proto, out)
            elif mode == "heatmap":
                out = outdir / f"{collective}_{proto}_best_algo_heatmap.png"
                plot_heatmap(df, collective, proto, out)
            else:
                suffix = f"_{size_filter}" if size_filter else "_by_size"
                out = outdir / f"{collective}_{proto}_vs_ranks{suffix}.png"
                plot_ranks(df, collective, proto, out, size_filter=size_filter)
