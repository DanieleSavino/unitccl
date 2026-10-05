"""Thin wrapper around the `fastest` python package and the project's
compiled pybind11 backend extension (built by that project's
`build_wheel.sh`, e.g. `import unitccl` for the current BINE project).

The backend module name is configurable (`config.get("backend_module")`) so
this same interface can drive other Bine-collective projects later.
"""
from __future__ import annotations

import importlib
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set

import fastest

from . import config
from .logging_utils import color, section

PLOT_DIR = "plots"


def load_backend():
    """Import the project's compiled test extension and register it with
    `fastest` as the active backend."""
    mod_name = config.get("backend_module", "unitccl")
    backend = importlib.import_module(mod_name)
    fastest.default_runner.set_backend(backend)
    return backend


def apply_env(
    check: bool, iters: Optional[int], warmup: Optional[int], overrides: Dict[str, str]
) -> None:
    os.environ[config.CHECK_ENV] = "1" if check else "0"
    if iters is not None:
        os.environ[config.ITERS_ENV] = str(iters)
    if warmup is not None:
        os.environ[config.WARMUP_ENV] = str(warmup)
    for k, v in overrides.items():
        os.environ[k] = v


def run_standalone(
    env_overrides: Optional[Dict[str, str]] = None, 
    buffmans: Optional[List[str]] = None
) -> None:
    backend = load_backend()
    active_buffmans = buffmans if (buffmans and len(buffmans) > 0) else [None]
    
    for buff in active_buffmans:
        curr_env = dict(env_overrides or {})
        if buff:
            section(f"standalone correctness — BINE BUFFER MAN: {buff}")
            curr_env[config.BINE_BUFFER_MANAGEMENT_ENV] = buff
        else:
            section("standalone correctness")
            if config.BINE_BUFFER_MANAGEMENT_ENV in os.environ:
                del os.environ[config.BINE_BUFFER_MANAGEMENT_ENV]
            
        apply_env(check=True, iters=None, warmup=None, overrides=curr_env)
        for test in backend.get_subtests("standalone"):
            fastest.run_log(test["test_name"])


def run_scaling(
    colls: Optional[Set[str]] = None,
    algos: Optional[Set[str]] = None,
    protos: Optional[Set[str]] = None,
    do_plot: bool = False,
    do_csv: bool = False,
    check: bool = False,
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
    n_repeats: int = 1,
    plot_dir: str = PLOT_DIR,
    env_overrides: Optional[Dict[str, str]] = None,
    buffmans: Optional[Set[str]] = None,
    tick_labels=None,
) -> List[str]:
    """Port of the current `tests.py` scaling block. Returns csv paths written.

    `plot_dir` is reused both for regular runs (defaults to "plots") and for
    rank-sweep runs, where `slurm_utils.submit_rank_sweep` passes
    `<N>_ranks` so csvs land in the layout `plot.py`'s loaders expect.
    """
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    if global_rank != 0:
        return []

    load_backend()
    written: List[str] = []

    section("scaling comparison")
    active_colls = config.active(config.DEFAULT_COLLS, colls)
    active_protos = config.active(config.DEFAULT_PROTOS, protos)
    active_buffmans = sorted(list(buffmans)) if buffmans else [None]

    for coll in active_colls:
        coll_algos = config.active(config.DEFAULT_ALGOS.get(coll, {}), algos)
        if not coll_algos:
            print(color(f"   [skip] {coll}: no active algos", "dim"))
            continue

        file_dir = f"{plot_dir}/{coll}"
        if do_csv or do_plot:
            os.makedirs(file_dir, exist_ok=True)
            
        baseline_algo = coll_algos[0]

        for proto in active_protos:
            os.environ["NCCL_PROTO"] = proto
            proto_dfs = []

            for algo in coll_algos:
                buffs = active_buffmans if algo == "BINE" else [None]
                
                for buffman in buffs:
                    curr_env = dict(env_overrides or {})
                    if buffman:
                        curr_env[config.BINE_BUFFER_MANAGEMENT_ENV] = buffman
                        algo_label = f"{algo}-{buffman}"
                    else:
                        algo_label = algo
                        if config.BINE_BUFFER_MANAGEMENT_ENV in os.environ:
                            del os.environ[config.BINE_BUFFER_MANAGEMENT_ENV]

                    apply_env(check, iters, warmup, curr_env)
                    print(color(f"\n  {coll}  proto={proto}  algo={algo_label}", "blue"))

                    pool = fastest.pool_from_prefix(f"scaling/1kB_64MB/{algo}_{coll}")
                    cmp = fastest.compare(pool, n_repeats=n_repeats)
                    cmp.report()

                    tmp_csv = f"{file_dir}/.tmp_{proto}_{algo_label}.csv"
                    cmp.save_csv(tmp_csv)
                    df = pd.read_csv(tmp_csv)
                    df['test'] = df['test'].str.replace(f"/{algo}_{coll}/", f"/{algo_label}_{coll}/")
                    proto_dfs.append(df)
                    os.remove(tmp_csv)

            if proto_dfs:
                final_df = pd.concat(proto_dfs, ignore_index=True)
                csv_path = f"{file_dir}/{coll}_{proto}.csv"
                
                if do_csv or do_plot:
                    final_df.to_csv(csv_path, index=False)
                    written.append(csv_path)
                    if do_csv:
                        print(color(f"   saved → {csv_path}", "dim"))

                if do_plot:
                    from .schema import load_fastest_csv, records_to_df
                    from .plotting import plot_size, plot_diff
                    recs = load_fastest_csv(Path(csv_path), coll, proto, ranks=None)
                    plot_df = records_to_df(recs)
                    
                    png_path = f"{file_dir}/{coll}_{proto}.png"
                    plot_size(plot_df, coll, proto, Path(png_path))
                    print(color(f"   saved → {png_path}", "dim"))
                    
                    diff_path = f"{file_dir}/{coll}_{proto}_diff.png"
                    plot_diff(plot_df, coll, proto, Path(diff_path), baseline_algo=baseline_algo)
                    print(color(f"   saved → {diff_path}", "dim"))

    return written
