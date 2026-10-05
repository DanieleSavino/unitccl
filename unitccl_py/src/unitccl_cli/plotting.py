"""Rank-sweep plots: time-vs-message-size and time-vs-rank-count, BINE vs
RING. Consumes the `<N>_ranks/<coll>/<coll>_<proto>.csv` layout written by
`unitccl scaling ... ranks=...` (see slurm_utils.submit_rank_sweep).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from .logging_utils import ok
from .schema import find_protos, load_rank_sweep, records_to_df, size_to_bytes

PLOT_COLORS = ["#00d2ff", "#ff6b6b", "#a8ff78", "#f7971e", "#c471ed"]


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

    color_idx = 0

    for ranks in ranks_list:
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
                color=PLOT_COLORS[color_idx % len(PLOT_COLORS)],
                label=label,
            )
            color_idx += 1

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

    # Ensure the baseline is plotted first so it receives the primary color (like cyan in the old plot)
    if baseline_algo in algos:
        algos.remove(baseline_algo)
        algos.insert(0, baseline_algo)

    color_idx = 0

    for ranks in ranks_list:
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
                color=PLOT_COLORS[color_idx % len(PLOT_COLORS)],
                label=label,
            )
            color_idx += 1

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

    for ax, size_str in zip(axes, sizes):
        _apply_dark_theme(ax, "Ranks", "Latency (µs)", size_str)
        sub = df[df["size_str"] == size_str]

        for i, algo in enumerate(algos):
            asub = sub[sub["algo"] == algo].sort_values("ranks")
            if asub.empty:
                continue

            ax.plot(
                asub["ranks"],
                asub["mean_ns"] / 1e3,
                marker="D",
                markersize=8,
                linewidth=2.5,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
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
            else:
                suffix = f"_{size_filter}" if size_filter else "_by_size"
                out = outdir / f"{collective}_{proto}_vs_ranks{suffix}.png"
                plot_ranks(df, collective, proto, out, size_filter=size_filter)
