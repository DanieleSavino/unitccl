#!/usr/bin/env python
import sys, os
import fastest
import unitccl
from fastest.plotting import Plotter, PlotMode, LegendLocation, LineStyle, MarkerStyle

color = fastest.logging.color
sep_color = "blue"

args = set(sys.argv[1:])
run_standalone = not args or "standalone" in args
run_scaling    = not args or "scaling"    in args
run_plot       = not args or "plot"       in args

NCCL_PROTOS = ["SIMPLE", 'LL', 'LL128']


if run_standalone:
    print(color("\n── standalone correctness ──────────────────────────────────────────", sep_color))
    for test in unitccl.get_subtests("standalone"):
        fastest.run_log(test["test_name"])

if run_scaling:
    print(color("\n── scaling comparison ──────────────────────────────────────────────", sep_color))
    bine_bcast = fastest.pool_from_prefix("scaling/BINE_bcast")
    ring_bcast  = fastest.pool_from_prefix("scaling/RING_bcast")


    if run_plot:
        print(color("\n── plotting ────────────────────────────────────────────────────────", sep_color))

        file_name = "plots/bcast_scaling"

        cmp_plotter = (
            Plotter()
            .set_title("Broadcast scaling comparison")
            .set_x_label("Input size: (1024 * 16^n) bytes")
            .set_y_label("Execution time")
            .set_bg_color("#1a1a2e")
            .set_title_color("#e0e0e0")
            .set_label_color("#cccccc")
            .set_tick_color("#aaaaaa")
            .set_pool_color(0, "#00d2ff")
            .set_pool_color(1, "#ff6b6b")
            .set_line_width(2.5)
            .set_marker(MarkerStyle.DIAMOND, size=10)
            .set_legend(LegendLocation.UPPER_LEFT, fontsize=11)
            .set_grid(True, color="#333355", style=LineStyle.DOTTED, alpha=0.5)
            .show_info(False)
            .set_dpi(200)
        )

        for proto in NCCL_PROTOS:
            os.environ["NCCL_PROTO"] = proto

            cmp = fastest.compare(bine_bcast, ring_bcast, n_repeats=5)
            cmp.report()
            file_path = f"{file_name}_{proto}.png"
            cmp_plotter.plot(cmp, file_path, PlotMode.MEDIAN)
            print(color(f"   saved → {file_path}", "dim"))

