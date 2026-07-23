"""Rank-sweep plots: time-vs-message-size and time-vs-rank-count, BINE vs
RING. Consumes the `<N>_ranks/<coll>/<coll>_<proto>.csv` layout written by
`unitccl scaling ... ranks=<r1,r2,...>` (see slurm_utils.submit_rank_sweep).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .logging_utils import ok
from .schema import find_protos, load_rank_sweep, records_to_df, size_to_bytes


def plot_size(df: pd.DataFrame, collective: str, proto: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ranks_list = sorted(df["ranks"].unique())
    algos = sorted(df["algo"].unique())

    cmap = plt.get_cmap("viridis")
    rank_colors = {r: cmap(i / max(1, len(ranks_list) - 1)) for i, r in enumerate(ranks_list)}
    algo_styles = {"RING": "-", "BINE": "--"}
    algo_markers = {"RING": "o", "BINE": "s"}

    for ranks in ranks_list:
        for algo in algos:
            sub = df[(df["ranks"] == ranks) & (df["algo"] == algo)].sort_values("size_bytes")
            if sub.empty:
                continue
            ax.plot(
                sub["size_bytes"],
                sub["mean_ns"] / 1e3,
                marker=algo_markers.get(algo, "x"),
                linestyle=algo_styles.get(algo, ":"),
                color=rank_colors[ranks],
                label=f"{algo} - {ranks} ranks",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Mean time (µs)")
    ax.set_title(f"{collective} ({proto}) — BINE vs RING scaling")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    ok(f"wrote {out}")


def plot_ranks(
    df: pd.DataFrame, collective: str, proto: str, out: Path, size_filter: Optional[str] = None
) -> None:
    algos = sorted(df["algo"].unique())
    algo_colors = {"RING": "tab:blue", "BINE": "tab:orange"}
    cmap = plt.get_cmap("tab10")
    for i, a in enumerate(algos):
        algo_colors.setdefault(a, cmap(i % 10))

    if size_filter:
        sizes = [size_filter]
        sub_df = df[df["size_str"] == size_filter]
        if sub_df.empty:
            raise SystemExit(
                f"No rows found with size_str == '{size_filter}'. "
                f"Available sizes: {sorted(df['size_str'].unique(), key=size_to_bytes)}"
            )
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    else:
        sizes = sorted(df["size_str"].unique(), key=size_to_bytes)
        n = len(sizes)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()

    for ax, size_str in zip(axes, sizes):
        sub = df[df["size_str"] == size_str]
        for algo in algos:
            asub = sub[sub["algo"] == algo].sort_values("ranks")
            if asub.empty:
                continue
            ax.plot(
                asub["ranks"],
                asub["mean_ns"] / 1e3,
                marker="o",
                linestyle="-",
                color=algo_colors[algo],
                label=algo,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Ranks")
        ax.set_ylabel("Mean time (µs)")
        ax.set_title(size_str)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(sub["ranks"].unique()))
        ax.set_xticklabels(sorted(sub["ranks"].unique()))

    for ax in axes[len(sizes) :]:
        ax.axis("off")

    fig.suptitle(f"{collective} ({proto}) — time vs ranks, BINE vs RING")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
