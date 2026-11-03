"""Rank-sweep plots: time-vs-message-size, time-vs-rank-count, and a
best-algorithm heatmap (size x ranks); `ranks` and `heatmap` can overlay several
protocols in one figure (see run_plot_command). Consumes the
`<N>_ranks/<coll>/<coll>_<proto>.csv` layout written by
`unitccl scaling ... ranks=...` (see slurm_utils.submit_rank_sweep).
"""
from __future__ import annotations

import re
from fnmatch import fnmatchcase
from itertools import cycle
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from . import config
from .logging_utils import ok, warn
from .schema import find_protos, load_rank_sweep, records_to_df, size_to_bytes

# Short names used to annotate the best-algo heatmap cells.
ALGO_ABBREV = {
    "BINE-TREE": "BT",
    "PAT": "P",
    "TREE": "T",
    "RING": "R",
}
# Fixed colors so an algo keeps its color across every plot.
# Labels are BINE-TREE or BINE-<variant>-<buffman> (variant: BUTTERFLY | PAT).
# Baselines are cool/neutral; each Bine variant owns a hue family (BUTTERFLY =
# reds/oranges, PAT = purples/magentas, TREE = teal) with one shade per buffman
# mode, and each buffman mode also has its own marker (see _BUFFMAN_MARKERS).
ALGO_COLORS = {
    "RING": "#d62828",                          # red
    "TREE": "#495057",                          # slate
    "PAT": "#2f9e44",                           # green (NCCL algo)
    "BINE-TREE": "#f77f00",                     # orange
    "BINE-BUTTERFLY-SEND": "#1c7eff",           # blue
    "BINE-BUTTERFLY-DOUBLE_SEND": "#0c8599",    # teal
    "BINE-BUTTERFLY-BLOCK_BY_BLOCK": "#8c1c13", # maroon
    "BINE-BUTTERFLY-PERMUTATION": "#f2b705",    # amber
    "BINE-PAT-SEND": "#7b2cbf",                 # purple
    "BINE-PAT-DOUBLE_SEND": "#c77dff",          # light violet
    "BINE-PAT-BLOCK_BY_BLOCK": "#3c1a7a",       # deep indigo
    "BINE-PAT-PERMUTATION": "#d63384",          # magenta
}
_FALLBACK_COLORS = [
    "#5f3dc4", "#099268", "#a0522d", "#3b5bdb", "#868e96",
    "#74b816", "#e8590c", "#364fc7", "#c2255c", "#22b8cf",
]
_BUFFMAN_MARKERS = {"SEND": "o", "DOUBLE_SEND": "s", "BLOCK_BY_BLOCK": "^", "PERMUTATION": "D"}

# Light, publication-style theme (white background, thin dotted grid).
_FG = "#000000"          # text, spines, ticks
_GRID = "#b0b0b0"        # grid lines
_BG = "#ffffff"          # figure/axes background
_MISSING = "#eeeeee"     # heatmap cells without data
_LINE_KW = dict(markersize=5, linewidth=1.6)

# Labels are BINE[-<variant>][-<buffman>] (see config.bine_label).
_VARIANT_ABBREV = {"PAT": "Pa", "BUTTERFLY": "Bf", "TREE": "T"}
_BUFFMAN_ABBREV = {"SEND": "S", "DOUBLE_SEND": "D", "BLOCK_BY_BLOCK": "B", "PERMUTATION": "P"}

# Distinguishes rank counts when several are drawn in the same axes.
_RANK_LINESTYLES = ["-", "--", ":", "-."]

# Multi-protocol plots: color = algo, line style + marker = protocol.
# SIMPLE is solid/filled; every other protocol is dashed/dotted with hollow markers.
_PROTO_STYLES = [
    dict(linestyle="-", marker="o", hollow=False),
    dict(linestyle="--", marker="s", hollow=True),
    dict(linestyle=":", marker="^", hollow=True),
    dict(linestyle="-.", marker="D", hollow=True),
]
_KNOWN_PROTO_STYLE = {"SIMPLE": 0, "LL": 1, "LL128": 2}
# Short protocol names for heatmap cell labels (<algo-abbrev>-<proto-abbrev>).
_PROTO_ABBREV = {"SIMPLE": "S", "LL": "LL", "LL128": "L128", "DEFAULT": "DEF"}
# Heatmap fill per protocol: algo color as-is (SIMPLE), lightened (LL), darkened (LL128).
_PROTO_SHADE = {"SIMPLE": 0.0, "LL": 0.5, "LL128": -0.4}


# ── metric / uncertainty band ────────────────────────────────────────────────
#
# `apply_metric` turns a records_to_df frame into one with two extra columns the
# plot functions read: `val_ns` (the plotted line) and `hw_ns` (half-width of the
# shaded band, NaN = no band). Frames that skip it (the `scaling --plot` path)
# are given val_ns = mean_ns and no band by `_ensure_val`, i.e. unchanged output.

METRICS = {"median": ("median_ns", "Median"), "avg": ("iter_avg_ns", "Mean")}
BANDS = ("stddev", "confidence")
_CI_Z = 1.96                 # 95% two-sided normal quantile
_BAND_FLOOR_FRAC = 0.05      # lower edge never below 5% of the line (log axes)
_BAND_ALPHA = 0.18
_BAND_TEXT = {
    "stddev": "band = ±1 stddev over iterations",
    "confidence": "band = 95% CI of the mean",
}


def apply_metric(df: pd.DataFrame, metric: str = "median", band: Optional[str] = None) -> pd.DataFrame:
    """Select the plotted statistic and, optionally, its uncertainty band.

    metric  median -> median_ns (median over the per-iteration slowest-rank times;
                      for several repeats: median of the per-launch medians)
            avg    -> iter_avg_ns (mean of all those per-iteration times)
    band    stddev     -> line ± s
            confidence -> line ± 1.96 * s / sqrt(n)   (95% CI of the mean)
            where n = iter_n and s is the SAMPLE stddev, recovered from the stored
            population stddev as s = sigma * sqrt(n / (n - 1)).
    """
    col, _ = METRICS[metric]
    df = df.copy()
    vals = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)
    if vals.isna().all():
        raise SystemExit(
            f"No '{col}' data in these CSVs (needed for --metric {metric}). "
            "Re-run `unitccl scaling --csv` with the updated build"
            + (" (nccl-tests CSVs have no per-iteration stats)." if metric == "avg" else ".")
        )
    if vals.isna().any():
        warn(f"{int(vals.isna().sum())} row(s) have no '{col}' and are not plotted")
    df["val_ns"] = vals
    df["hw_ns"] = np.nan

    if band:
        sd = pd.to_numeric(df["iter_stddev_ns"], errors="coerce") if "iter_stddev_ns" in df else pd.Series(np.nan, index=df.index)
        n = pd.to_numeric(df["iter_n"], errors="coerce") if "iter_n" in df else pd.Series(np.nan, index=df.index)
        n1 = (n - 1).where(n > 1)                       # NaN unless n >= 2
        s = sd * np.sqrt(n / n1)
        hw = s if band == "stddev" else _CI_Z * s / np.sqrt(n)
        if hw.isna().all():
            raise SystemExit(
                f"No 'iter_stddev_ns'/'iter_n' data in these CSVs (needed for --stddev {band}); "
                "nccl-tests and pre-update CSVs don't have them."
            )
        if hw.isna().any():
            warn(f"{int(hw.isna().sum())} row(s) have no usable stddev/n: drawn without a band")
        if band == "confidence" and metric == "median":
            warn("--stddev confidence is a CI of the MEAN; it is drawn around the median line "
                 "(use --metric avg for a matching line)")
        df["hw_ns"] = hw

    df = df[df["val_ns"].notna()]
    df.attrs["metric"] = metric
    df.attrs["band"] = band
    return df


def _ensure_val(df: pd.DataFrame) -> pd.DataFrame:
    """Frames that never went through apply_metric: plot mean_ns, no band."""
    if "val_ns" in df:
        return df
    df = df.copy()
    df["val_ns"] = df["mean_ns"]
    df["hw_ns"] = np.nan
    return df


def _ylabel(df: pd.DataFrame) -> str:
    metric = df.attrs.get("metric")
    return f"{METRICS[metric][1]} latency (µs)" if metric else "Latency (µs)"


def _note(df: pd.DataFrame) -> str:
    """Title suffix describing line + band (empty for the legacy mean_ns plots)."""
    metric, band = df.attrs.get("metric"), df.attrs.get("band")
    if not metric:
        return ""
    return f"  ({metric}" + (f", {_BAND_TEXT[band]}" if band else "") + ")"


def _draw_band(ax, x, val_ns, hw_ns, color) -> None:
    """Shaded area line ± hw (µs on the y axis). No-op without band data."""
    v, h = np.asarray(val_ns, dtype=float), np.asarray(hw_ns, dtype=float)
    if not np.isfinite(h).any():
        return
    lo = np.maximum(v - h, v * _BAND_FLOOR_FRAC)
    ax.fill_between(
        np.asarray(x, dtype=float), lo / 1e3, (v + h) / 1e3,
        color=color, alpha=_BAND_ALPHA, linewidth=0, zorder=1,
    )


def _algo_colors(algos) -> dict:
    """{algo: color}. Known algos get their fixed color; unknown ones get
    fallback colors assigned in sorted order, so the result is deterministic.
    Pass the full (unfiltered) algo list so colors stay stable across plots."""
    fallback = cycle(_FALLBACK_COLORS)
    return {a: ALGO_COLORS.get(a) or next(fallback) for a in sorted(algos)}


def _proto_styles(protos) -> dict:
    """{proto: style dict}. Known protocols keep a fixed style; unknown ones
    take the remaining styles in order."""
    styles = {p: _PROTO_STYLES[_KNOWN_PROTO_STYLE[p]] for p in protos if p in _KNOWN_PROTO_STYLE}
    used = {_KNOWN_PROTO_STYLE[p] for p in styles}
    spare = cycle([i for i in range(len(_PROTO_STYLES)) if i not in used] or [1])
    for p in protos:
        if p not in styles:
            styles[p] = _PROTO_STYLES[next(spare)]
    return styles


def _shade(hex_color: str, amount: float) -> str:
    """amount > 0 mixes with white, amount < 0 mixes with black."""
    rgb = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    target = 255 if amount > 0 else 0
    rgb = [round(c + (target - c) * abs(amount)) for c in rgb]
    return "#%02x%02x%02x" % tuple(rgb)


def _abbrev(algo: str) -> str:
    if algo in ALGO_ABBREV:
        return ALGO_ABBREV[algo]
    # BINE-<variant>[-<buffman>]: "Pa" / "Pa/S" (kept distinct from BINE-<buffman> like "BS")
    if algo.startswith("BINE-"):
        rest = algo[len("BINE-"):]
        variant, _, buff = rest.partition("-")
        if variant in _VARIANT_ABBREV:
            v = _VARIANT_ABBREV[variant]
            return f"{v}/{_BUFFMAN_ABBREV.get(buff, buff[:1])}" if buff else v
    # Unknown label: initials of its '-'/'_' separated parts.
    return "".join(w[0] for w in algo.replace("-", "_").split("_") if w).upper() or algo


def _algo_marker(algo: str) -> str:
    """Marker keyed on the buffer-management suffix of BINE-<variant>-<buffman>
    labels (so shades within a variant family stay distinguishable in print);
    everything else gets a circle."""
    if algo.startswith("BINE-"):
        _, _, buff = algo[len("BINE-"):].partition("-")
        return _BUFFMAN_MARKERS.get(buff, "o")
    return "o"


def _apply_theme(ax, xlabel: str, ylabel: str, title: str):
    ax.set_facecolor(_BG)
    ax.grid(True, which="major", color=_GRID, linestyle=":", linewidth=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=_GRID, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_FG, which="both")
    for spine in ax.spines.values():
        spine.set_color(_FG)
    ax.set_xlabel(xlabel, color=_FG, fontsize=10)
    ax.set_ylabel(ylabel, color=_FG, fontsize=10)
    ax.set_title(title, color=_FG, fontsize=12)


def _style_legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(loc="upper left", fontsize=8, facecolor=_BG, edgecolor="#cccccc", framealpha=0.9)


def _text_color_for(hex_color: str) -> str:
    """Black or white text, whichever contrasts better with the given fill."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.45 else "#ffffff"


def _set_size_ticks(ax, df):
    unique_bytes = sorted(df["size_bytes"].unique())
    ax.set_xticks(unique_bytes)
    byte_to_str = dict(zip(df["size_bytes"], df["size_str"]))
    ax.set_xticklabels([byte_to_str[b] for b in unique_bytes])


def _draw_size_lines(ax, df, algos, colors, ranks, linestyle="-", label_ranks=False):
    for algo in algos:
        if ranks is not None:
            sub = df[(df["ranks"] == ranks) & (df["algo"] == algo)].sort_values("size_bytes")
        else:
            sub = df[(df["ranks"].isna()) & (df["algo"] == algo)].sort_values("size_bytes")
        if sub.empty:
            continue
        _draw_band(ax, sub["size_bytes"], sub["val_ns"], sub["hw_ns"], colors[algo])
        ax.plot(
            sub["size_bytes"],
            sub["val_ns"] / 1e3,
            **_LINE_KW,
            marker=_algo_marker(algo),
            linestyle=linestyle,
            color=colors[algo],
            label=f"{algo} - {ranks} ranks" if (label_ranks and ranks is not None) else algo,
            zorder=2,
        )


def plot_size(df: pd.DataFrame, collective: str, proto: str, out: Path) -> None:
    df = _ensure_val(df)
    ylabel, note = _ylabel(df), _note(df)
    has_ranks = df["ranks"].notna().any()
    ranks_list = sorted(df["ranks"].dropna().astype(int).unique()) if has_ranks else [None]
    algos = sorted(df["algo"].unique())
    colors = _algo_colors(algos)
    title = f"{collective} scaling — {proto}{note}"

    if len(ranks_list) <= 1:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=_BG)
        _apply_theme(ax, "Message size", ylabel, title)
        _draw_size_lines(ax, df, algos, colors, ranks_list[0], label_ranks=True)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _set_size_ticks(ax, df)
        _style_legend(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=200, facecolor=_BG)
    else:
        # Several rank counts: one panel per rank count + one shared legend,
        # instead of algos x ranks overlaid on a single axis.
        n = len(ranks_list)
        ncols = min(3, n)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), facecolor=_BG, squeeze=False
        )
        flat = list(axes.flat)
        for ax, ranks in zip(flat, ranks_list):
            sub_df = df[df["ranks"] == ranks]
            _apply_theme(ax, "Message size", ylabel, f"{ranks} ranks")
            _draw_size_lines(ax, sub_df, algos, colors, ranks)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _set_size_ticks(ax, sub_df)
        for ax in flat[n:]:
            ax.set_visible(False)
        handles, labels = {}, {}
        for ax in flat[:n]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        order = [a for a in algos if a in handles]
        fig.suptitle(title, color=_FG, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.legend(
            [handles[a] for a in order], order, loc="center left", bbox_to_anchor=(1.0, 0.5),
            fontsize=8, facecolor=_BG, edgecolor="#cccccc", framealpha=0.9,
        )
        fig.savefig(out, dpi=200, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    ok(f"wrote {out}")


def plot_diff(df: pd.DataFrame, collective: str, proto: str, out: Path, baseline_algo: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=_BG)
    _apply_theme(ax, "Message size", f"Δ vs {baseline_algo}", f"{collective} scaling — {proto}  (Δ vs {baseline_algo})")

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
                **_LINE_KW,
                marker=_algo_marker(algo),
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
    ax.axhline(0, color=_FG, linewidth=0.8, linestyle="--", alpha=0.5)
    _style_legend(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=_BG)
    plt.close(fig)
    ok(f"wrote {out}")


def plot_ranks(
    df: pd.DataFrame, collective: str, proto: str, out: Path, size_filter: Optional[str] = None
) -> None:
    fig_facecolor = _BG
    df = _ensure_val(df)
    ylabel, note = _ylabel(df), _note(df)

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
        _apply_theme(ax, "Ranks", ylabel, size_str)
        sub = df[df["size_str"] == size_str]

        for algo in algos:
            asub = sub[sub["algo"] == algo].sort_values("ranks")
            if asub.empty:
                continue

            _draw_band(ax, asub["ranks"], asub["val_ns"], asub["hw_ns"], colors[algo])
            ax.plot(
                asub["ranks"],
                asub["val_ns"] / 1e3,
                **_LINE_KW,
                marker=_algo_marker(algo),
                color=colors[algo],
                label=algo,
                zorder=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted(sub["ranks"].dropna().astype(int).unique()))
        ax.set_xticklabels([int(x) for x in sorted(sub["ranks"].dropna().astype(int).unique())])
        _style_legend(ax)

    for ax in axes[len(sizes) :]:
        ax.axis("off")

    fig.suptitle(f"{collective} ({proto}) — time vs ranks{note}", color=_FG, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig_facecolor)
    plt.close(fig)
    ok(f"wrote {out}")


def plot_ranks_multi(
    df: pd.DataFrame, collective: str, protos: List[str], out: Path, size_filter: Optional[str] = None
) -> None:
    """Time vs ranks with several protocols overlaid: one panel per message
    size, color = algo, line style/marker = protocol, one shared legend.
    Protocols that only exist for some rank counts (e.g. LL128, NVLink-only)
    simply produce shorter lines."""
    df = _ensure_val(df)
    ylabel, note = _ylabel(df), _note(df)
    if size_filter:
        if df[df["size_str"] == size_filter].empty:
            raise SystemExit(f"No rows found with size_str == '{size_filter}'.")
        sizes = [size_filter]
    else:
        sizes = sorted(df["size_str"].unique(), key=size_to_bytes)
    n = len(sizes)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows + 0.9), squeeze=False, facecolor=_BG
    )
    flat = list(axes.flat)

    algos = sorted(df["algo"].unique())
    colors = _algo_colors(algos)
    styles = _proto_styles(protos)

    for ax, size_str in zip(flat, sizes):
        _apply_theme(ax, "Ranks", ylabel, size_str)
        sub = df[df["size_str"] == size_str]
        for proto in protos:
            st = styles[proto]
            for algo in algos:
                asub = sub[(sub["algo"] == algo) & (sub["proto"] == proto)].sort_values("ranks")
                if asub.empty:
                    continue
                _draw_band(ax, asub["ranks"], asub["val_ns"], asub["hw_ns"], colors[algo])
                ax.plot(
                    asub["ranks"], asub["val_ns"] / 1e3,
                    linewidth=1.6, markersize=6, linestyle=st["linestyle"], marker=st["marker"],
                    color=colors[algo],
                    markerfacecolor=_BG if st["hollow"] else colors[algo],
                    markeredgewidth=1.6,
                    label=f"{algo} {proto}",
                    zorder=2,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = sorted(sub["ranks"].dropna().astype(int).unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
    for ax in flat[n:]:
        ax.set_visible(False)

    handles = {}
    for ax in flat[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [f"{a} {p}" for p in protos for a in algos if f"{a} {p}" in handles]

    fig.suptitle(
        f"{collective} — time vs ranks, {' vs '.join(algos)}, {' vs '.join(protos)}{note}",
        color=_FG, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.legend(
        [handles[l] for l in order], order, loc="lower center", ncol=min(len(order), 6),
        fontsize=9, facecolor=_BG, edgecolor="none", frameon=False,
    )
    fig.savefig(out, dpi=200, facecolor=_BG)
    plt.close(fig)
    ok(f"wrote {out}")


def plot_heatmap(df: pd.DataFrame, collective: str, proto: str, out: Path) -> None:
    """Best-algorithm heatmap: x = message size, y = number of ranks, each cell
    labelled with the (abbreviated) algo with the lowest mean latency. Under the
    winner, the closest runners-up are listed as `<abbrev> +<slowdown>%`; as many
    as fit inside the cell (measured from the final layout) are shown."""
    df = _ensure_val(df).dropna(subset=["ranks", "val_ns"])
    note = _note(df)
    has_band = bool(df["hw_ns"].notna().any())
    if df.empty:
        raise SystemExit(f"No rank-tagged data to build a heatmap for {collective} ({proto}).")

    sizes = sorted(df["size_bytes"].unique())
    ranks_list = sorted(df["ranks"].astype(int).unique())
    byte_to_str = dict(zip(df["size_bytes"], df["size_str"]))

    # Several protocols: every (algo, proto) pair is its own option, labelled
    # <algo>-<proto> (RING-LL, RING-SIMPLE, BINE-LL128, ...). Same hue per algo,
    # lighter/darker shade per protocol.
    multi = df["proto"].nunique() > 1
    label_abbrev: dict = {}
    if multi:
        base_colors = _algo_colors(sorted(df["algo"].unique()))
        label_color: dict = {}
        for a, p in sorted(set(zip(df["algo"], df["proto"]))):
            lbl = f"{a}-{p}"
            label_color[lbl] = _shade(base_colors[a], _PROTO_SHADE.get(p, 0.25))
            label_abbrev[lbl] = f"{_abbrev(a)}-{_PROTO_ABBREV.get(p, p)}"
        df = df.assign(algo=df["algo"] + "-" + df["proto"])
    abbr = lambda a: label_abbrev.get(a) or _abbrev(a)  # noqa: E731

    # All algos present (not just winners) so runners-up are in the legend too.
    algos = sorted(df["algo"].unique())
    algo_to_i = {a: i for i, a in enumerate(algos)}
    color_map = label_color if multi else _algo_colors(algos)
    colors = [color_map[a] for a in algos]

    # Per cell: algos ordered best -> worst with slowdown vs the winner (%).
    grid = np.full((len(ranks_list), len(sizes)), np.nan)
    cell_rank: dict = {}
    cell_tie: set = set()   # winner's band overlaps the runner-up's band
    for (r, sz), g in df.groupby(["ranks", "size_bytes"]):
        g = g.sort_values("val_ns")
        best_ns = g["val_ns"].iloc[0]
        yi, xi = ranks_list.index(int(r)), sizes.index(sz)
        grid[yi, xi] = algo_to_i[g["algo"].iloc[0]]
        cell_rank[(yi, xi)] = [
            (a, (ns - best_ns) / best_ns * 100.0 if best_ns > 0 else 0.0)
            for a, ns in zip(g["algo"], g["val_ns"])
        ]
        if len(g) > 1:
            w_hi = g["val_ns"].iloc[0] + g["hw_ns"].iloc[0]
            r_lo = g["val_ns"].iloc[1] - g["hw_ns"].iloc[1]
            if np.isfinite(w_hi) and np.isfinite(r_lo) and w_hi >= r_lo:
                cell_tie.add((yi, xi))

    fig_facecolor = _BG
    fig, ax = plt.subplots(
        figsize=(max(7, (2.1 if multi else 1.5) * len(sizes) + 3), max(4, 1.0 * len(ranks_list) + 2.5)),
        facecolor=fig_facecolor,
    )
    cmap = ListedColormap(colors)
    cmap.set_bad(_MISSING)
    ax.imshow(
        np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5, vmax=len(algos) - 0.5,
        aspect="auto", origin="lower",
    )

    _apply_theme(ax, "Message size", "Ranks", f"{collective} ({proto}) — best algorithm{note}")
    ax.grid(False)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([byte_to_str[b] for b in sizes])
    ax.set_yticks(range(len(ranks_list)))
    ax.set_yticklabels(ranks_list)
    ax.set_xticks(np.arange(-0.5, len(sizes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ranks_list), 1), minor=True)
    ax.grid(which="minor", color=_BG, linestyle="-", linewidth=2, alpha=1.0)
    ax.tick_params(which="minor", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in colors]
    labels = [f"{abbr(a)} = {a}" for a in algos]
    if has_band:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=_BG, edgecolor="#555555",
                                     hatch="////", linewidth=0.5))
        labels.append("hatched: winner's band overlaps\nthe runner-up's (no clear winner)")
    legend = ax.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10,
        facecolor=_BG, edgecolor="#cccccc", framealpha=0.9,
    )
    for text in legend.get_texts():
        text.set_color(_FG)

    fig.tight_layout()

    # How many runner-up lines fit? Measure the real cell size (points) after
    # layout, leaving a margin so the text never touches the cell border.
    bbox = ax.get_window_extent()
    cell_h_pt = bbox.height / len(ranks_list) * 72.0 / fig.dpi
    cell_w_pt = bbox.width / len(sizes) * 72.0 / fig.dpi
    main_pt, line_pt, pad_pt = 15.0, 9.0, 12.0
    fit = int((cell_h_pt - pad_pt - main_pt) // line_pt)
    # "BB +123.4%" at 7pt needs ~45pt of width (more with <algo>-<proto> names);
    # give up the margin if too narrow.
    longest = max(len(abbr(a)) for a in algos) + len(" +123.4%")
    n_runners = max(0, min(fit, len(algos) - 1)) if cell_w_pt >= max(55.0, 4.6 * longest + 8) else 0

    for (yi, xi), ranked in cell_rank.items():
        runners = ranked[1 : 1 + n_runners]
        txt = _text_color_for(color_map[ranked[0][0]])
        if (yi, xi) in cell_tie:
            ax.add_patch(plt.Rectangle(
                (xi - 0.5, yi - 0.5), 1, 1, fill=False, hatch="////",
                edgecolor=txt, linewidth=0, alpha=0.35,
            ))
        total = main_pt + line_pt * len(runners)
        top = total / 2.0
        ax.annotate(
            abbr(ranked[0][0]), (xi, yi), xytext=(0, top - main_pt / 2.0),
            textcoords="offset points", ha="center", va="center",
            color=txt, fontsize=12, fontweight="bold",
        )
        for k, (a, pct) in enumerate(runners):
            ax.annotate(
                f"{abbr(a)} +{pct:.0f}%" if pct >= 10 else f"{abbr(a)} +{pct:.1f}%",
                (xi, yi), xytext=(0, top - main_pt - line_pt * (k + 0.5)),
                textcoords="offset points", ha="center", va="center",
                color=txt, fontsize=7, alpha=0.9,
            )

    fig.savefig(out, dpi=200, facecolor=fig_facecolor, bbox_inches="tight")
    plt.close(fig)
    ok(f"wrote {out}")


def _algo_matches(label: str, pattern: str) -> bool:
    """Case-insensitive. A pattern matches a label if it is the label itself, a
    '-'-delimited prefix of it (BINE -> every BINE-*, BINE-PAT -> BINE-PAT-*),
    or a glob (BINE-*-SEND). TREE never matches BINE-TREE."""
    label, pattern = label.upper(), pattern.strip().upper()
    if not pattern:
        return False
    if any(c in pattern for c in "*?["):
        return fnmatchcase(label, pattern)
    return label == pattern or label.startswith(pattern + "-")


def drop_unsupported_ll(df: pd.DataFrame, collective: str) -> pd.DataFrame:
    """Backstop for CSVs written before `unitccl scaling` learned to skip them:
    drop BINE-TREE / BINE-PAT-* rows measured with LL or LL128 on AllGather,
    ReduceScatter and AllReduce (see config.bine_supported). Plain BINE,
    BINE-<buffman> and BINE-BUTTERFLY-* labels and every non-Bine algo stay."""
    if collective not in config.LL_BUTTERFLY_ONLY_COLLS or df.empty:
        return df
    variant = df["algo"].str.upper().str.split("-").str[:2].str[-1]
    bad = (
        df["proto"].isin(config.LL_PROTOS)
        & df["algo"].str.upper().str.startswith("BINE-")
        & variant.isin(config.LL_UNSUPPORTED_BINE_VARIANTS)
    )
    return df[~bad]


def filter_algos(df: pd.DataFrame, patterns: Optional[List[str]]) -> pd.DataFrame:
    """Keep only rows whose algo label matches one of `patterns` (None = keep all)."""
    if not patterns:
        return df
    present = sorted(df["algo"].unique())
    for pat in patterns:
        if not any(_algo_matches(a, pat) for a in present):
            warn(f"--algo '{pat}' matches nothing (available: {', '.join(present)})")
    return df[df["algo"].apply(lambda a: any(_algo_matches(a, p) for p in patterns))]


def _dedup(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in (i.strip() for i in items):
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _proto_tag(protos: List[str]) -> str:
    return "+".join(protos)


def run_plot_command(
    mode: str,
    root: Path,
    collectives: List[str],
    protos: Optional[List[str]],
    outdir: Path,
    size_filter: Optional[str] = None,
    algos: Optional[List[str]] = None,
    metric: str = "median",
    band: Optional[str] = None,
) -> None:
    """metric: 'median' | 'avg' (see apply_metric); band: None | 'stddev' | 'confidence'.

    With an explicit multi-protocol `--proto A,B,...`:
      ranks   -> one combined figure per collective (color = algo, style = proto)
      heatmap -> one combined figure per collective, options <algo>-<proto>
      size    -> error (not supported)
    Without `--proto` (autodetect) or with a single one, each protocol gets its
    own figure as before."""
    protos = _dedup(protos) if protos else None
    combined = bool(protos) and len(protos) > 1
    if combined and mode == "size":
        raise SystemExit(
            f"`plot size` does not support multiple protocols (got --proto {','.join(protos)}); "
            "pass a single protocol, or run it once per protocol."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    tag = "_" + "+".join(re.sub(r"[^A-Za-z0-9_.-]", "x", a.strip()) for a in algos) if algos else ""
    # default (median, no band) keeps the historical file names
    tag += (f"_{metric}" if metric != "median" else "") + (f"_{band}" if band else "")

    for collective in collectives:
        if combined:
            frames = []
            for proto in protos:
                try:
                    frames.append(records_to_df(load_rank_sweep(root, collective, proto)))
                except SystemExit as e:
                    warn(f"{collective}/{proto}: {e} -- skipping this protocol")
            if not frames:
                raise SystemExit(f"No data found for {collective} with any of: {', '.join(protos)}")
            loaded = [p for p in protos if any(f["proto"].iloc[0] == p for f in frames)]
            if len(loaded) > 1:
                df = drop_unsupported_ll(filter_algos(pd.concat(frames, ignore_index=True), algos), collective)
                if df.empty:
                    warn(f"{collective}: no data left after filtering, skipping")
                    continue
                df = apply_metric(df, metric, band)
                ptag = _proto_tag(loaded)
                if mode == "heatmap":
                    out = outdir / f"{collective}_{ptag}_best_algo_heatmap{tag}.png"
                    plot_heatmap(df, collective, ", ".join(loaded), out)
                else:
                    suffix = f"_{size_filter}" if size_filter else "_by_size"
                    out = outdir / f"{collective}_{ptag}_vs_ranks{suffix}{tag}.png"
                    plot_ranks_multi(df, collective, loaded, out, size_filter=size_filter)
                continue
            protos_to_use = loaded  # only one protocol had data: plain single-proto plot
        else:
            protos_to_use = protos or find_protos(root, collective)
        if not protos_to_use:
            raise SystemExit(f"No protocols found for collective '{collective}' under {root}")
        for proto in protos_to_use:
            records = load_rank_sweep(root, collective, proto)
            df = drop_unsupported_ll(filter_algos(records_to_df(records), algos), collective)
            if df.empty:
                warn(f"{collective}/{proto}: no data left after filtering, skipping")
                continue
            df = apply_metric(df, metric, band)
            if mode == "size":
                out = outdir / f"{collective}_{proto}_scaling{tag}.png"
                plot_size(df, collective, proto, out)
            elif mode == "heatmap":
                out = outdir / f"{collective}_{proto}_best_algo_heatmap{tag}.png"
                plot_heatmap(df, collective, proto, out)
            else:
                suffix = f"_{size_filter}" if size_filter else "_by_size"
                out = outdir / f"{collective}_{proto}_vs_ranks{suffix}{tag}.png"
                plot_ranks(df, collective, proto, out, size_filter=size_filter)
