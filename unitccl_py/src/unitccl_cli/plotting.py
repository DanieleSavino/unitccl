"""Scaling plots: time-vs-message-size and time-vs-rank-count, one line per
algorithm variant (RING, TREE, ..., and BINE -- once per Bine buffer
management when several are present).

Consumes the layout described in `schema.py`, written by
`unitccl scaling ... ranks=<r1,r2,...>`; `unitccl scaling ... plot ranks=...`
calls `run_plot_command` itself once the sweep finishes, and `unitccl plot`
re-runs it on whatever is on disk.

Output goes to `<outdir>/<Coll>/<Coll>_<PROTO>[__<BUFF>]_{scaling,vs_ranks_*}.png`
(`outdir` defaults to `<root>/scaling`). When the data holds several BINE buffer
managements they share one figure (a figure per protocol), unless `buffs`
narrows the selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .logging_utils import ok, warn
from .schema import (
    NoDataError,
    find_collectives,
    find_protos,
    load_rank_sweep,
    records_to_df,
    size_to_bytes,
    variant_order,
)

_LINE_STYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
_TAB10 = [plt.get_cmap("tab10")(i) for i in range(10)]
_FIXED_COLORS = {"RING": _TAB10[0]}  # BINE's first variant takes _TAB10[1] below


def _variant_colors(variants: List[str]) -> Dict[str, tuple]:
    """RING always blue, first BINE variant always orange (as before buffer
    management existed); everything else gets the remaining tab10 colours."""
    colors: Dict[str, tuple] = {}
    spare = [c for i, c in enumerate(_TAB10) if i not in (0, 1)]
    bine_seen = False
    for v in variants:
        if v in _FIXED_COLORS:
            colors[v] = _FIXED_COLORS[v]
        elif v.startswith("BINE") and not bine_seen:
            colors[v] = _TAB10[1]
            bine_seen = True
        else:
            colors[v] = spare[len(colors) % len(spare)]
    return colors


def _variant_styles(variants: List[str]) -> Dict[str, tuple]:
    """(linestyle, marker) per variant, for plots where colour is taken by ranks."""
    styles = {}
    for i, v in enumerate(variants):
        styles[v] = (_LINE_STYLES[i % len(_LINE_STYLES)], _MARKERS[i % len(_MARKERS)])
    return styles


def _subtitle(variants: List[str]) -> str:
    return " vs ".join(variants)


def plot_size(df: pd.DataFrame, collective: str, proto: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ranks_list = sorted(df["ranks"].unique())
    variants = variant_order(df)
    styles = _variant_styles(variants)

    cmap = plt.get_cmap("viridis")
    rank_colors = {r: cmap(i / max(1, len(ranks_list) - 1)) for i, r in enumerate(ranks_list)}

    for ranks in ranks_list:
        for variant in variants:
            sub = df[(df["ranks"] == ranks) & (df["variant"] == variant)].sort_values("size_bytes")
            if sub.empty:
                continue
            linestyle, marker = styles[variant]
            ax.plot(
                sub["size_bytes"],
                sub["mean_ns"] / 1e3,
                marker=marker,
                linestyle=linestyle,
                color=rank_colors[ranks],
                label=f"{variant} - {ranks} ranks",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Mean time (µs)")
    ax.set_title(f"{collective} ({proto}) — {_subtitle(variants)}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    ok(f"wrote {out}")


def plot_ranks(
    df: pd.DataFrame, collective: str, proto: str, out: Path, size_filter: Optional[str] = None
) -> None:
    variants = variant_order(df)
    colors = _variant_colors(variants)

    if size_filter:
        sizes = [size_filter]
        sub_df = df[df["size_str"] == size_filter]
        if sub_df.empty:
            raise NoDataError(
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
        for variant in variants:
            vsub = sub[sub["variant"] == variant].sort_values("ranks")
            if vsub.empty:
                continue
            ax.plot(
                vsub["ranks"],
                vsub["mean_ns"] / 1e3,
                marker="o",
                linestyle="-",
                color=colors[variant],
                label=variant,
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

    fig.suptitle(f"{collective} ({proto}) — time vs ranks, {_subtitle(variants)}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    ok(f"wrote {out}")


def run_plot_command(
    mode: str,
    root: Path,
    collectives: Optional[List[str]],
    protos: Optional[List[str]],
    outdir: Path,
    size_filter: Optional[str] = None,
    buffs: Optional[List[str]] = None,
) -> List[Path]:
    """Plot every (collective, protocol) found under `root`.

    mode: 'size' | 'ranks' | 'all'. `collectives`/`protos` None = autodetect.
    `buffs` None (or ['ALL']) = every BINE buffer management found, otherwise
    only those. Combinations with no data are skipped with a warning; a
    NoDataError is raised only if nothing at all could be plotted.
    """
    root, outdir = Path(root), Path(outdir)
    if buffs == ["ALL"]:
        buffs = None
    modes = ["size", "ranks"] if mode == "all" else [mode]

    collectives = collectives or find_collectives(root)
    if not collectives:
        raise NoDataError(f"No <N>_ranks/<collective>/ data found under {root}")

    written: List[Path] = []
    for collective in collectives:
        protos_to_use = protos or find_protos(root, collective)
        if not protos_to_use:
            warn(f"no protocols found for collective '{collective}' under {root}")
            continue
        cdir = outdir / collective
        for proto in protos_to_use:
            try:
                df = records_to_df(load_rank_sweep(root, collective, proto, buffs))
            except NoDataError as e:
                warn(str(e))
                continue
            cdir.mkdir(parents=True, exist_ok=True)
            for m in modes:
                if m == "size":
                    out = cdir / f"{collective}_{proto}_scaling.png"
                    plot_size(df, collective, proto, out)
                else:
                    suffix = f"_{size_filter}" if size_filter else "_by_size"
                    out = cdir / f"{collective}_{proto}_vs_ranks{suffix}.png"
                    try:
                        plot_ranks(df, collective, proto, out, size_filter=size_filter)
                    except NoDataError as e:
                        warn(str(e))
                        continue
                written.append(out)

    if not written:
        raise NoDataError(f"Nothing to plot under {root} for the requested collective/proto/buff filters")
    return written
