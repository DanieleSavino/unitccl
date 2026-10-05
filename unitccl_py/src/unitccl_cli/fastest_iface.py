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

from . import config, schema
from .logging_utils import color, section, warn

PLOT_DIR = config.DEFAULT_PLOT_ROOT
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


def set_buff_env(buff: Optional[str]) -> None:
    """Select the Bine buffer management for comms created from now on.
    `None` clears it (NCCL falls back to its own default). Read by NCCL when a
    communicator is built, so call this *before* running a test/compare."""
    if buff:
        os.environ[config.BUFF_ENV] = buff
    else:
        os.environ.pop(config.BUFF_ENV, None)


def _test_collective(test_name: str) -> Optional[str]:
    """Best-effort: which collective a standalone subtest name refers to
    (longest registry name contained in it, so AllGatherV wins over AllGather)."""
    low = test_name.lower()
    for coll in sorted(config.DEFAULT_COLLS, key=len, reverse=True):
        if coll.lower() in low:
            return coll
    return None


def run_standalone(buffs: Optional[List[str]] = None) -> None:
    """Correctness tests. With `buffs`, the whole suite runs once per Bine
    buffer management; subtests that clearly belong to a collective which
    doesn't implement that mode are skipped rather than run against an
    unimplemented path."""
    backend = load_backend()
    modes = buffs or [None]
    if modes == ["ALL"]:
        modes = list(config.BUFF_ALL)
    for mode in modes:
        suffix = f" [buff={mode}]" if mode else ""
        section(f"standalone correctness{suffix}")
        set_buff_env(mode)
        for test in backend.get_subtests("standalone"):
            name = test["test_name"]
            coll = _test_collective(name)
            if mode and coll and mode not in config.BUFF_SUPPORT.get(coll, [config.DEFAULT_BUFF]):
                print(color(f"   [skip] {name}: {coll} has no {mode}", "dim"))
                continue
            fastest.run_log(name)


def _variant_tag(proto: str, buff: Optional[str]) -> str:
    return proto if not buff or buff == config.DEFAULT_BUFF else f"{proto}, BINE={buff}"


def _make_plotter(coll: str, proto: str, tick_labels, buff: Optional[str] = None) -> Plotter:
    p = (
        Plotter()
        .set_title(f"{coll} scaling — {_variant_tag(proto, buff)}")
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


def _make_diff_plotter(
    coll: str, proto: str, baseline_algo: str, tick_labels, buff: Optional[str] = None
) -> Plotter:
    p = _make_plotter(coll, proto, tick_labels, buff)
    p.set_title(f"{coll} scaling — {_variant_tag(proto, buff)}  (Δ vs {baseline_algo})")
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
    buffs: Optional[List[str]] = None,
) -> List[str]:
    """Port of the current `tests.py` scaling block. Returns csv paths written.

    `plot_dir` is the directory holding `<coll>/` folders: "plots" for a local
    run, `plots/<NNN>_ranks` inside a rank-sweep job (see
    `slurm_utils.submit_rank_sweep`). File names come from `schema.csv_stem`,
    which is also what `plotting` parses back.

    `buffs` selects the Bine buffer management(s) (see `config.parse_buffs`):
    None -> BLOCK_BY_BLOCK only (NCCL's default), ['ALL'] -> every mode the
    collective implements. Each mode is a separate compare (the mode is fixed
    per communicator), so a non-baseline algo like RING is re-measured in each
    one; the loaders average those duplicates. Modes a collective doesn't
    implement are dropped, and the collective is skipped if that leaves BINE
    with nothing to run.
    """
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    if global_rank != 0:
        return []

    load_backend()
    apply_env(check, iters, warmup, env_overrides or {})

    tick_labels = tick_labels or ["1kB", "16kB", "256kB", "1MB", "4MB", "64MB"]
    written: List[str] = []

    section("scaling comparison")
    active_colls = config.active(config.DEFAULT_COLLS, colls)
    active_protos = config.active(config.DEFAULT_PROTOS, protos)

    for coll in active_colls:
        coll_algos = config.active(config.DEFAULT_ALGOS.get(coll, {}), algos)
        if not coll_algos:
            print(color(f"   [skip] {coll}: no active algos", "dim"))
            continue

        # Buffer-management variants only exist for BINE; every other algo
        # runs once, with the env var cleared.
        if "BINE" in coll_algos:
            coll_buffs: List[Optional[str]] = list(config.buffs_for(coll, buffs))
            if not coll_buffs:
                warn(
                    f"{coll}: BINE has no {', '.join(buffs or [])} -- skipping "
                    f"(supported: {', '.join(config.BUFF_SUPPORT.get(coll, [config.DEFAULT_BUFF]))})"
                )
                continue
            if buffs and buffs != ["ALL"]:
                dropped = [b for b in buffs if b not in coll_buffs]
                if dropped:
                    warn(f"{coll}: BINE does not implement {', '.join(dropped)} -- skipped")
        else:
            coll_buffs = [None]

        pools = {a: fastest.pool_from_prefix(f"scaling/1kB_64MB/{a}_{coll}") for a in coll_algos}
        baseline_algo = next((a for a in coll_algos if a != "BINE"), coll_algos[0])

        for proto in active_protos:
            os.environ[config.PROTO_ENV] = proto
            for buff in coll_buffs:
                set_buff_env(buff)
                tag = f"  buff={buff}" if buff else ""
                print(color(f"\n  {coll}  proto={proto}{tag}  algos={coll_algos}", "blue"))

                cmp = fastest.compare(*pools.values(), n_repeats=n_repeats)
                cmp.report()

                file_dir = f"{plot_dir}/{coll}"
                stem = f"{file_dir}/{schema.csv_stem(coll, proto, buff)}"
                if do_csv:
                    os.makedirs(file_dir, exist_ok=True)
                    cmp.save_csv(f"{stem}.csv")
                    written.append(f"{stem}.csv")
                    print(color(f"   saved → {stem}.csv", "dim"))

                if do_plot:
                    os.makedirs(file_dir, exist_ok=True)

                    _make_plotter(coll, proto, tick_labels, buff).plot(cmp, f"{stem}.png", PlotMode.MEDIAN)
                    print(color(f"   saved → {stem}.png", "dim"))

                    _make_diff_plotter(coll, proto, baseline_algo, tick_labels, buff).plot(
                        cmp, f"{stem}_diff.png", PlotMode.MEDIAN, PlotTransform.DIFF
                    )
                    print(color(f"   saved → {stem}_diff.png", "dim"))

    set_buff_env(None)
    return written
