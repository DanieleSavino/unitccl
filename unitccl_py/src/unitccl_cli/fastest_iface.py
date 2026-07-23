"""Thin wrapper around the `fastest` python package and the project's
compiled pybind11 backend extension (built by that project's
`build_wheel.sh`, e.g. `import unitccl` for the current BINE project).

The backend module name is configurable (`config.get("backend_module")`) so
this same interface can drive other Bine-collective projects later.
"""
from __future__ import annotations

import importlib
import os
from typing import Dict, List, Optional, Set

import fastest
from fastest.plotting import (
    LegendLocation,
    LineStyle,
    MarkerStyle,
    PlotMode,
    Plotter,
    PlotTransform,
)

from . import config
from .logging_utils import color, section

PLOT_DIR = "plots"
PLOT_COLORS = ["#00d2ff", "#ff6b6b", "#a8ff78", "#f7971e", "#c471ed"]


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


def run_standalone() -> None:
    backend = load_backend()
    section("standalone correctness")
    for test in backend.get_subtests("standalone"):
        fastest.run_log(test["test_name"])


def _make_plotter(coll: str, proto: str, tick_labels) -> Plotter:
    p = (
        Plotter()
        .set_title(f"{coll} scaling — {proto}")
        .set_x_label("Message size")
        .set_y_label("Latency")
        .set_x_tick_labels(tick_labels)
        .set_bg_color("#1a1a2e")
        .set_title_color("#e0e0e0")
        .set_label_color("#cccccc")
        .set_tick_color("#aaaaaa")
        .set_line_width(2.5)
        .set_marker(MarkerStyle.DIAMOND, size=10)
        .set_legend(LegendLocation.UPPER_LEFT, fontsize=11)
        .set_grid(True, color="#333355", style=LineStyle.DOTTED, alpha=0.5)
        .show_info(False)
        .set_dpi(200)
    )
    for i, c in enumerate(PLOT_COLORS):
        p.set_pool_color(i, c)
    return p


def _make_diff_plotter(coll: str, proto: str, baseline_algo: str, tick_labels) -> Plotter:
    p = _make_plotter(coll, proto, tick_labels)
    p.set_title(f"{coll} scaling — {proto}  (Δ vs {baseline_algo})")
    p.set_y_label(f"Δ vs {baseline_algo}")
    return p


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
    tick_labels=None,
) -> List[str]:
    """Port of the current `tests.py` scaling block. Returns csv paths written.

    `plot_dir` is reused both for regular runs (defaults to "plots") and for
    rank-sweep runs, where `slurm_utils.submit_rank_sweep` passes
    `<N>_ranks` so csvs land in the layout `plot.py`'s loaders expect.
    """
    load_backend()
    apply_env(check, iters, warmup, env_overrides or {})

    tick_labels = tick_labels or Plotter.SIZES_1KB_64MB
    written: List[str] = []

    section("scaling comparison")
    active_colls = config.active(config.DEFAULT_COLLS, colls)
    active_protos = config.active(config.DEFAULT_PROTOS, protos)

    for coll in active_colls:
        coll_algos = config.active(config.DEFAULT_ALGOS.get(coll, {}), algos)
        if not coll_algos:
            print(color(f"   [skip] {coll}: no active algos", "dim"))
            continue

        pools = {a: fastest.pool_from_prefix(f"scaling/1kB_64MB/{a}_{coll}") for a in coll_algos}
        baseline_algo = coll_algos[0]

        for proto in active_protos:
            os.environ["NCCL_PROTO"] = proto
            print(color(f"\n  {coll}  proto={proto}  algos={coll_algos}", "blue"))

            cmp = fastest.compare(*pools.values(), n_repeats=n_repeats)
            cmp.report()

            file_dir = f"{plot_dir}/{coll}"
            if do_csv:
                os.makedirs(file_dir, exist_ok=True)
                csv_path = f"{file_dir}/{coll}_{proto}.csv"
                cmp.save_csv(csv_path)
                written.append(csv_path)
                print(color(f"   saved → {csv_path}", "dim"))

            if do_plot:
                os.makedirs(file_dir, exist_ok=True)

                png_path = f"{file_dir}/{coll}_{proto}.png"
                _make_plotter(coll, proto, tick_labels).plot(cmp, png_path, PlotMode.MEDIAN)
                print(color(f"   saved → {png_path}", "dim"))

                diff_path = f"{file_dir}/{coll}_{proto}_diff.png"
                _make_diff_plotter(coll, proto, baseline_algo, tick_labels).plot(
                    cmp, diff_path, PlotMode.MEDIAN, PlotTransform.DIFF
                )
                print(color(f"   saved → {diff_path}", "dim"))

    return written
