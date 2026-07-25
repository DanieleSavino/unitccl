"""Nsight Systems profiling + stats extraction + BINE-vs-RING analysis plots.

Replaces, respectively:
  - `meluxina_nsys.sbatch`'s `run_profile()` loop  -> run_profiles()
  - `generate_stats.sh`                            -> generate_stats()
  - `02_analyze_and_plot.py`                        -> collect_data() / analyze()
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .logging_utils import error, info, ok, section

NSYS_FLAGS = [
    "--trace=cuda,nvtx,osrt,mpi",
    "--cuda-memory-usage=true",
    "--sample=none",
    "--cpuctxsw=none",
    "--force-overwrite=true",
]

REPORTS = [
    "nvtx_sum",
    "osrt_sum",
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
]

# OS-runtime calls treated as "communication / synchronization wait" proxies
# -- calls NCCL's network/proxy/socket threads block on while ranks talk to
# each other (poll/epoll on sockets, semaphores for CUDA IPC handshakes).
NETWORK_WAIT_CALLS = {"poll", "epoll_wait", "sem_wait", "sem_timedwait"}

FNAME_RE = re.compile(
    r"^(?P<collective>[A-Za-z0-9]+)_(?P<algo>bine|ring)_nsys_rank(?P<rank>\d+)_(?P<report>.+)\.csv$"
)


# ── 1. profile capture (mpirun + nsys profile) ──────────────────────────────


def run_profiles(
    outdir: Path,
    colls: List[str],
    algos: List[str],
    size: int,
    nranks: int,
    proto: str = "SIMPLE",
    bench_bin: str = "./build/unitccl_bench",
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
    check: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    if shutil.which("nsys") is None:
        raise RuntimeError("'nsys' not found on PATH. Load/activate your Nsight Systems install first.")

    outdir.mkdir(parents=True, exist_ok=True)

    env: Dict[str, str] = {"NCCL_PROTO": proto, config.CHECK_ENV: "1" if check else "0"}
    if warmup is not None:
        env[config.WARMUP_ENV] = str(warmup)
    if iters is not None:
        env[config.ITERS_ENV] = str(iters)
    env.update(env_overrides or {})
    full_env = {**os.environ, **env}

    for coll in colls:
        for algo in algos:
            name = f"{coll.lower()}_{algo.lower()}_nsys"
            info(f"profiling {coll} with {algo}")
            full_env["NCCL_ALGO"] = algo
            cmd = [
                "mpirun",
                "-n",
                str(nranks),
                "nsys",
                "profile",
                *NSYS_FLAGS,
                "-o",
                str(outdir / f"{name}_rank%q{{OMPI_COMM_WORLD_RANK}}"),
                bench_bin,
                coll,
                str(size),
            ]
            subprocess.run(cmd, env=full_env, check=True)
    ok(f"profiles written to {outdir}")


# ── 2. `nsys stats` CSV export (port of generate_stats.sh) ──────────────────


def generate_stats(nsys_dir: Path, out_subdir: str = "stats") -> Path:
    if shutil.which("nsys") is None:
        raise RuntimeError("'nsys' not found on PATH. Load/activate your Nsight Systems install first.")
    if not nsys_dir.is_dir():
        raise RuntimeError(f"directory '{nsys_dir}' does not exist.")

    out_dir = nsys_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(nsys_dir.glob("*.nsys-rep"))
    if not files:
        raise RuntimeError(f"no .nsys-rep files found in '{nsys_dir}'.")

    report_arg = ",".join(REPORTS)
    failed = False
    for f in files:
        base = f.stem
        out_prefix = out_dir / base
        if Path(f"{out_prefix}_nvtx_sum.csv").exists():
            info(f"[skip, already exists] {base}")
            continue

        info(f"processing {base}")
        log_path = Path(f"{out_prefix}.log")
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--report",
                    report_arg,
                    "--format",
                    "csv",
                    "--force-export=true",
                    "--output",
                    str(out_prefix),
                    str(f),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            error(f"FAILED on {base} -- see {log_path}")
            failed = True

    if failed:
        raise RuntimeError(f"some files failed to process; check .log files in {out_dir}")

    ok(f"CSV reports written to {out_dir}")
    return out_dir


# ── 3. parse stats CSVs + extract metrics (port of 02_analyze_and_plot.py) ──


def _find_report_suffix(report_field: str) -> Optional[str]:
    for suffix in sorted(REPORTS, key=len, reverse=True):
        if report_field == suffix:
            return suffix
    return None


def _parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    if _find_report_suffix(d["report"]) is None:
        return None
    d["rank"] = int(d["rank"])
    return d


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        error(f"could not read {path}: {e}")
        return None


def _extract_metrics(files_by_report: Dict[str, Path]) -> dict:
    """Given the set of report CSVs for one (collective, algo, rank), pull out
    the metrics we care about. All *_ns fields are nanoseconds."""
    metrics = {
        "nccl_init_ns": None,
        "nccl_op_ns": None,
        "gpu_kernel_ns": None,
        "gpu_kernel_name": None,
        "gpu_memcpy_ns": None,
        "gpu_memset_ns": None,
        "gpu_mem_bytes": None,
        "network_wait_ns": 0.0,
        "cuda_api_total_ns": None,
    }

    # --- nvtx_sum: NCCL:ncclCommInitRank (init time) + NCCL:nccl<Collective> (op time)
    if "nvtx_sum" in files_by_report:
        df = _safe_read_csv(files_by_report["nvtx_sum"])
        if df is not None and "Range" in df.columns:
            for _, row in df.iterrows():
                rng = str(row["Range"])
                metric_val = row.get("Med (ns)", row.get("Avg (ns)"))
                if rng == "NCCL:ncclCommInitRank":
                    metrics["nccl_init_ns"] = metric_val
                elif rng.startswith("NCCL:") and rng != "NCCL:ncclCommInitRank":
                    metrics["nccl_op_ns"] = metric_val

    # --- cuda_gpu_kern_sum: actual GPU compute kernel time
    if "cuda_gpu_kern_sum" in files_by_report:
        df = _safe_read_csv(files_by_report["cuda_gpu_kern_sum"])
        if df is not None and len(df):
            val_col = "Med (ns)" if "Med (ns)" in df.columns else "Avg (ns)"
            if val_col in df.columns:
                metrics["gpu_kernel_ns"] = df[val_col].median()
            if "Name" in df.columns and "Total Time (ns)" in df.columns:
                metrics["gpu_kernel_name"] = df.loc[df["Total Time (ns)"].idxmax(), "Name"]

    # --- cuda_gpu_mem_time_sum: memcpy / memset time on the GPU
    if "cuda_gpu_mem_time_sum" in files_by_report:
        df = _safe_read_csv(files_by_report["cuda_gpu_mem_time_sum"])
        if df is not None and "Operation" in df.columns:
            val_col = "Med (ns)" if "Med (ns)" in df.columns else "Avg (ns)"
            if val_col in df.columns:
                memcpy_mask = df["Operation"].str.contains("memcpy", case=False, na=False)
                memset_mask = df["Operation"].str.contains("memset", case=False, na=False)
                metrics["gpu_memcpy_ns"] = df.loc[memcpy_mask, val_col].median()
                metrics["gpu_memset_ns"] = df.loc[memset_mask, val_col].median()

    # --- cuda_gpu_mem_size_sum: bytes moved per operation instance
    if "cuda_gpu_mem_size_sum" in files_by_report:
        df = _safe_read_csv(files_by_report["cuda_gpu_mem_size_sum"])
        if df is not None and len(df):
            size_col = "Med (MB)" if "Med (MB)" in df.columns else "Avg (MB)"
            if size_col not in df.columns and "Total (MB)" in df.columns and "Instances" in df.columns:
                df[size_col] = df["Total (MB)"] / df["Instances"]
            if size_col in df.columns:
                metrics["gpu_mem_bytes"] = df[size_col].median() * 1024 * 1024

    # --- osrt_sum: proxy thread network waiting (poll/epoll/sem_wait)
    if "osrt_sum" in files_by_report:
        df = _safe_read_csv(files_by_report["osrt_sum"])
        if df is not None and "Name" in df.columns:
            val_col = "Med (ns)" if "Med (ns)" in df.columns else "Avg (ns)"
            if val_col in df.columns:
                mask = df["Name"].isin(NETWORK_WAIT_CALLS)
                metrics["network_wait_ns"] = df.loc[mask, val_col].median() if mask.any() else 0.0

    # --- cuda_api_sum: total CUDA driver/runtime API time (setup overhead)
    if "cuda_api_sum" in files_by_report:
        df = _safe_read_csv(files_by_report["cuda_api_sum"])
        if df is not None and len(df):
            val_col = "Med (ns)" if "Med (ns)" in df.columns else "Avg (ns)"
            if val_col in df.columns:
                metrics["cuda_api_total_ns"] = df[val_col].median()

    return metrics


def collect_data(stats_dir: Path) -> pd.DataFrame:
    grouped: Dict[tuple, Dict[str, Path]] = defaultdict(dict)

    for csv_path in stats_dir.glob("*.csv"):
        parsed = _parse_filename(csv_path)
        if parsed is None:
            continue
        key = (parsed["collective"], parsed["algo"], parsed["rank"])
        grouped[key][parsed["report"]] = csv_path

    if not grouped:
        raise RuntimeError(
            f"No matching CSV files found in {stats_dir}.\n"
            "Expected names like: bcast_bine_nsys_rank0_nvtx_sum.csv"
        )

    rows = []
    for (collective, algo, rank), files_by_report in grouped.items():
        m = _extract_metrics(files_by_report)
        m.update({"collective": collective, "algo": algo, "rank": rank})
        rows.append(m)

    return pd.DataFrame(rows).sort_values(["collective", "algo", "rank"]).reset_index(drop=True)


def _ns_to_ms(x):
    return x / 1e6 if pd.notna(x) else x


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    ms_cols = [
        "nccl_init_ns",
        "nccl_op_ns",
        "gpu_kernel_ns",
        "gpu_memcpy_ns",
        "gpu_memset_ns",
        "network_wait_ns",
        "cuda_api_total_ns",
    ]
    df_ms = df.copy()
    for c in ms_cols:
        df_ms[c.replace("_ns", "_ms")] = df_ms[c].apply(_ns_to_ms)

    agg_cols = [c.replace("_ns", "_ms") for c in ms_cols] + ["gpu_mem_bytes"]
    return df_ms.groupby(["collective", "algo"])[agg_cols].agg(["median", "std", "min", "max", "count"])


# ── 4. comparison plots ──────────────────────────────────────────────────────


def plot_metric_comparison(
    df: pd.DataFrame, metric_ns_col: str, ylabel: str, title: str, out_path: Path
) -> None:
    """Grouped bar chart: x = collective, bars = algo (ring/bine), value =
    median(metric) in ms, error bars = std across ranks (clipped at 0)."""
    d = df.copy()
    d[metric_ns_col] = d[metric_ns_col].apply(_ns_to_ms)
    d = d.dropna(subset=[metric_ns_col])
    if d.empty:
        info(f"skipping plot '{title}': no data")
        return

    stats = d.groupby(["collective", "algo"])[metric_ns_col].agg(["median", "std"]).reset_index()
    collectives = sorted(stats["collective"].unique())

    algos = ["ring", "bine"]
    algos = [a for a in algos if a in stats["algo"].unique()] or sorted(stats["algo"].unique())

    x = range(len(collectives))
    width = 0.8 / len(algos)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, algo in enumerate(algos):
        medians, stds = [], []
        for c in collectives:
            row = stats[(stats["collective"] == c) & (stats["algo"] == algo)]
            if row.empty:
                medians.append(0)
                stds.append(0)
            else:
                medians.append(row["median"].values[0])
                stds.append(row["std"].values[0] if pd.notna(row["std"].values[0]) else 0)

        offsets = [xi + (i - (len(algos) - 1) / 2) * width for xi in x]
        lower_err = [min(m, s) for m, s in zip(medians, stds)]
        upper_err = stds
        ax.bar(offsets, medians, width=width, yerr=[lower_err, upper_err], capsize=4, label=algo)

    ax.set_xticks(list(x))
    ax.set_xticklabels(collectives)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    ok(f"wrote {out_path}")


def plot_per_rank(
    df: pd.DataFrame, collective: str, metric_ns_col: str, ylabel: str, title: str, out_path: Path
) -> None:
    d = df[df["collective"] == collective].copy()
    d[metric_ns_col] = d[metric_ns_col].apply(_ns_to_ms)
    d = d.dropna(subset=[metric_ns_col])
    if d.empty:
        return

    target_order = ["ring", "bine"]
    available_algos = [a for a in target_order if a in d["algo"].unique()]

    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in available_algos:
        sub = d[d["algo"] == algo].sort_values("rank")
        ax.plot(sub["rank"], sub[metric_ns_col], marker="o", label=algo)

    ax.set_xlabel("rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="algorithm")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    ok(f"wrote {out_path}")


METRIC_PLOTS = [
    ("nccl_init_ns", "NCCL init time (ms)", "NCCL ncclCommInitRank time: bine vs ring", "nccl_init_time.png"),
    (
        "nccl_op_ns",
        "Collective op wall time (ms)",
        "NCCL collective call time: bine vs ring",
        "nccl_collective_op_time.png",
    ),
    ("gpu_kernel_ns", "GPU kernel time (ms)", "GPU compute kernel time: bine vs ring", "gpu_kernel_time.png"),
    (
        "gpu_memcpy_ns",
        "GPU memcpy time (ms)",
        "GPU Host<->Device memcpy time: bine vs ring",
        "gpu_memcpy_time.png",
    ),
    (
        "network_wait_ns",
        "Network/sync wait time (ms)",
        "OS-level network+sync wait (poll/epoll/sem_wait): bine vs ring",
        "network_wait_time.png",
    ),
    (
        "cuda_api_total_ns",
        "CUDA API total time (ms)",
        "Total CUDA driver/runtime API time: bine vs ring",
        "cuda_api_total_time.png",
    ),
]


def analyze(nsys_dir: Path) -> pd.DataFrame:
    """Full analysis pipeline: parse stats/*.csv -> tables + comparison plots."""
    stats_dir = nsys_dir / "stats"
    plots_dir = nsys_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not stats_dir.exists():
        raise RuntimeError(f"{stats_dir} does not exist. Run stats generation first.")

    section(f"reading CSVs from {stats_dir}")
    df = collect_data(stats_dir)

    raw_out = plots_dir / "per_rank_metrics.csv"
    df.to_csv(raw_out, index=False)
    ok(f"wrote {raw_out}")

    summary = make_summary(df)
    summary_out = plots_dir / "summary_by_algo_collective.csv"
    summary.to_csv(summary_out)
    ok(f"wrote {summary_out}")

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.xs("median", axis=1, level=1))

    section("generating comparison plots")
    for col, ylabel, title, fname in METRIC_PLOTS:
        plot_metric_comparison(df, col, ylabel, title, plots_dir / fname)

    section("generating per-rank breakdowns")
    for collective in sorted(df["collective"].unique()):
        plot_per_rank(
            df,
            collective,
            "nccl_op_ns",
            "Collective op wall time (ms)",
            f"{collective}: per-rank op time, bine vs ring",
            plots_dir / f"{collective}_per_rank_op_time.png",
        )

    ok(f"all done, plots + tables in {plots_dir}")
    return df
