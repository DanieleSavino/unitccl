#!/usr/bin/env python
import sys, os
import fastest
import unitccl
from fastest.plotting import Plotter, PlotMode, PlotTransform, LegendLocation, LineStyle, MarkerStyle

# ── configuration ────────────────────────────────────────────────────────────

COLLS = {
    "AllGather":     True,
    "AllGatherV":    True,
    "AllReduce":     True,
    "Bcast":         True,
    "Reduce":        True,
    "ReduceScatter": True,
}

ALGOS = {
    "AllGather": {
        "RING":           True,
        "COLLNET_DIRECT": False,
        "NVLS":           False,
        "PAT":            False,
        "BINE":           True,
    },
    "AllGatherV": {
        "RING": False,
        "BINE": False,
    },
    "AllReduce": {
        "TREE":           True,
        "RING":           True,
        "COLLNET_DIRECT": False,
        "COLLNET_CHAIN":  False,
        "NVLS":           False,
        "NVLS_TREE":      False,
        "BINE":           True,
    },
    "Bcast": {
        "RING": True,
        "BINE": True,
    },
    "Reduce": {
        "RING": True,
        "BINE": True,
    },
    "ReduceScatter": {
        "RING":           True,
        "COLLNET_DIRECT": False,
        "NVLS":           False,
        "PAT":            False,
        "BINE":           True,
    },
}

PROTOS = {
    "SIMPLE": True,
    "LL":     True,
    "LL128":  True,
}

ENV = {
    "UNITCCL_CHECK_BINE": "1",  # set to "1" or "ON" to enable bine correctness checks
    # "NCCL_DEBUG": "INFO",
}

N_REPEATS   = 1
PLOT_MODE   = PlotMode.MEDIAN
PLOT_DIR    = "plots"
PLOT_COLORS = ["#00d2ff", "#ff6b6b", "#a8ff78", "#f7971e", "#c471ed"]

# X-axis tick labels matching DEFINE_TEST_1KB_64MB expansion order.
# If you add more sizes in bench_test.h, extend this list to match.
TICK_LABELS = Plotter.SIZES_1KB_64MB   # ["1kB", "16kB", "256kB", "4MB", "64MB"]

# ── args ─────────────────────────────────────────────────────────────────────
#
# Usage:
#   ./tests.py                                  # everything on
#   ./tests.py standalone                       # correctness only
#   ./tests.py scaling                          # scaling, no plot
#   ./tests.py scaling plot                     # scaling + plots
#   ./tests.py coll=Bcast                       # only bcast collective
#   ./tests.py algo=BINE                        # only BINE algo (all colls)
#   ./tests.py proto=SIMPLE,LL                  # only those protocols
#   ./tests.py env=UNITCCL_CHECK_BINE=1         # override an env var
#   ./tests.py scaling coll=Bcast,AllReduce algo=BINE,RING proto=SIMPLE
#
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv):
    flags   = set()
    filters = {"coll": None, "algo": None, "proto": None}
    env_overrides = {}

    for arg in argv:
        if "=" in arg:
            key, _, val = arg.partition("=")
            if key == "env":
                ekey, _, eval_ = val.partition("=")
                env_overrides[ekey] = eval_
            elif key in filters:
                filters[key] = set(val.split(","))
            else:
                print(f"[warn] unknown key arg: {arg}")
        else:
            flags.add(arg)

    suite_flags = flags & {"standalone", "scaling", "plot"}
    run_all     = not suite_flags

    return {
        "standalone":    run_all or "standalone" in flags,
        "scaling":       run_all or "scaling"    in flags,
        "plot":          run_all or "plot"       in flags,
        "coll_filter":   filters["coll"],
        "algo_filter":   filters["algo"],
        "proto_filter":  filters["proto"],
        "env_overrides": env_overrides,
    }

# ── helpers ───────────────────────────────────────────────────────────────────

color     = fastest.logging.color
sep_color = "blue"

def active(d: dict, filter_set=None) -> list:
    return [k for k, v in d.items() if v and (filter_set is None or k in filter_set)]

def make_plotter(coll: str, proto: str) -> Plotter:
    p = (
        Plotter()
        .set_title(f"{coll} scaling — {proto}")
        .set_x_label("Message size")
        .set_y_label("Latency")           # unit appended automatically by formatter
        .set_x_tick_labels(TICK_LABELS)   # "1kB", "16kB", "256kB", "4MB", "64MB"
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

def make_diff_plotter(coll: str, proto: str, baseline_algo: str) -> Plotter:
    """Like make_plotter but pre-titled for DIFF transform output."""
    p = make_plotter(coll, proto)
    p.set_title(f"{coll} scaling — {proto}  (Δ vs {baseline_algo})")
    p.set_y_label(f"Δ vs {baseline_algo}")   # % suffix added by formatter
    return p

# ── main ──────────────────────────────────────────────────────────────────────

opts = parse_args(sys.argv[1:])

for k, v in {**ENV, **opts["env_overrides"]}.items():
    os.environ[k] = v

# ── standalone ────────────────────────────────────────────────────────────────

if opts["standalone"]:
    print(color("\n── standalone correctness ──────────────────────────────────────────", sep_color))
    for test in unitccl.get_subtests("standalone"):
        fastest.run_log(test["test_name"])

# ── scaling ───────────────────────────────────────────────────────────────────

if opts["scaling"]:
    print(color("\n── scaling comparison ──────────────────────────────────────────────", sep_color))

    active_colls  = active(COLLS,  opts["coll_filter"])
    active_protos = active(PROTOS, opts["proto_filter"])

    for coll in active_colls:
        coll_algos = active(ALGOS.get(coll, {}), opts["algo_filter"])
        if not coll_algos:
            print(color(f"   [skip] {coll}: no active algos", "dim"))
            continue

        pools = {
            algo: fastest.pool_from_prefix(f"scaling/1kB_64MB/{algo}_{coll}")
            for algo in coll_algos
        }

        # First active algo is the diff baseline (pool 0 in compare()).
        baseline_algo = coll_algos[0]

        for proto in active_protos:
            os.environ["NCCL_PROTO"] = proto
            print(color(f"\n  {coll}  proto={proto}  algos={coll_algos}", sep_color))

            cmp = fastest.compare(*pools.values(), n_repeats=N_REPEATS)
            cmp.report()

            if opts["plot"]:
                file_dir = f"{PLOT_DIR}/{coll}"
                os.makedirs(file_dir, exist_ok=True)

                file_path = f"{file_dir}/{coll}_{proto}.png"
                make_plotter(coll, proto).plot(cmp, file_path, PLOT_MODE)
                print(color(f"   saved → {file_path}", "dim"))

                file_path = f"{file_dir}/{coll}_{proto}_diff.png"
                make_diff_plotter(coll, proto, baseline_algo).plot(
                    cmp, file_path, PLOT_MODE, PlotTransform.DIFF
                )
                print(color(f"   saved → {file_path}", "dim"))
