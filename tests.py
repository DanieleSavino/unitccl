#!/usr/bin/env python
import sys
import fastest
import unitccl
from fastest.plotting import Plotter, PlotMode, LegendLocation, LineStyle, MarkerStyle

color = fastest.logging.color
sep_color = "blue"

args = set(sys.argv[1:])
run_standalone = not args or "standalone" in args
run_scaling    = not args or "scaling"    in args
run_plot       = not args or "plot"       in args

if run_standalone:
    print(color("\n── standalone correctness ──────────────────────────────────────────", sep_color))
    for test in unitccl.get_subtests("standalone"):
        fastest.run_log(test["test_name"])

if run_scaling:
    print(color("\n── scaling comparison ──────────────────────────────────────────────", sep_color))
    bine_bcast = fastest.pool_from_prefix("scaling/bine_bcast")
    ring_bcast  = fastest.pool_from_prefix("scaling/ring_bcast")
    cmp = fastest.compare(bine_bcast, ring_bcast, n_repeats=5)
    cmp.report()

    if run_plot:
        print(color("\n── plotting ────────────────────────────────────────────────────────", sep_color))
        path = "plots/bcast_scaling.png"
        (Plotter()
            .set_title("Broadcast scaling comparison")
            .set_x_label("Array index × step (1ms)")
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
            .plot(cmp, path, PlotMode.MEDIAN))
        print(color(f"   saved → {path}", "dim"))
