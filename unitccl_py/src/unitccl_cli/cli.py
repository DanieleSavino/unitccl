"""unitccl command-line entrypoint.

    unitccl standalone <preload> <submit> [--buffman SEND,DOUBLE_SEND]
    unitccl scaling --coll Bcast,AllReduce --algo BINE,RING --proto SIMPLE --plot --csv --check --warmup 10 --iters 40 --buffman SEND,BLOCK_BY_BLOCK
    unitccl scaling --coll Bcast --proto SIMPLE --csv --ranks 4,8,16,32,64,128
    unitccl nsys --outdir nsys --coll Bcast,Reduce --algo BINE,RING --proto SIMPLE --size 16777216 --nranks 8 --warmup 10 --iters 40 --check --buffman DOUBLE_SEND
    unitccl plot ranks --root . --collective Bcast,AllReduce --proto SIMPLE,LL [--size 4MB]
    unitccl plot size  --root . --collective Bcast --proto SIMPLE
    unitccl set account <value>
    unitccl set partition <value>
    unitccl set qos <value>
    unitccl set nccl_lib </path/to/nccl/lib>
    unitccl preload add <module> [<module> ...]
    unitccl preload rm  <module> [<module> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import config
from .logging_utils import error, ok

# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_build(args) -> None:
    from . import build_utils

    if args.submit:
        from . import slurm_utils
        jobs = slurm_utils.submit_build(args.target, args.clean)
        slurm_utils.wait_for(jobs)
        return

    if args.preload:
        from . import slurm_utils
        cfg = config.load()
        slurm_utils.apply_preload_modules(cfg.get("preload_modules") or [])

    build_utils.run_build(args.target, args.clean)


def cmd_standalone(args) -> None:
    from . import fastest_iface

    if args.mode == "submit":
        from . import slurm_utils
        jobs = slurm_utils.submit_standalone(
            buffmans=args.buffman.split(",") if args.buffman else None
        )
        slurm_utils.wait_for(jobs)
        return

    if args.mode == "preload":
        from . import slurm_utils
        cfg = config.load()
        slurm_utils.apply_preload_modules(cfg.get("preload_modules") or [])
        
    buffmans = args.buffman.split(",") if args.buffman else None
    fastest_iface.run_standalone(buffmans=buffmans)


def cmd_preload_add(args) -> None:
    cfg = None
    for module in args.modules:
        cfg = config.add_preload_module(module)
    ok(f"preload modules: {cfg['preload_modules']} ({config.CONFIG_FILE})")


def cmd_preload_rm(args) -> None:
    cfg = None
    for module in args.modules:
        cfg = config.remove_preload_module(module)
    ok(f"preload modules: {cfg['preload_modules']} ({config.CONFIG_FILE})")


def cmd_scaling(args) -> None:
    from . import fastest_iface, slurm_utils
    
    scaling_kwargs = dict(
        colls=set(args.coll.split(",")) if args.coll else None,
        algos=set(args.algo.split(",")) if args.algo else None,
        protos=set(args.proto.split(",")) if args.proto else None,
        do_plot=args.plot,
        do_csv=args.csv,
        check=args.check,
        warmup=args.warmup,
        iters=args.iters,
        buffmans=set(args.buffman.split(",")) if args.buffman else None,
    )

    if args.ranks:
        ranks_list = [int(r) for r in args.ranks.split(",")]
        jobs = slurm_utils.submit_rank_sweep(ranks_list, scaling_kwargs)
        slurm_utils.wait_for(jobs)
    else:
        fastest_iface.run_scaling(**scaling_kwargs)


def cmd_nsys(args) -> None:
    from . import slurm_utils

    colls = sorted(args.coll.split(",")) if args.coll else ["Bcast", "Reduce"]
    algos = sorted(args.algo.split(",")) if args.algo else ["BINE", "RING"]
    proto = args.proto.split(",")[0] if args.proto else "SIMPLE"
    sizes = [int(s) for s in args.size.split(",")]
    nranks_list = [int(r) for r in args.nranks.split(",")]
    buffmans = args.buffman.split(",") if args.buffman else None
    
    jobs = slurm_utils.submit_nsys_sweep(
        args.outdir, colls, algos, sizes=sizes, ranks_list=nranks_list, proto=proto,
        warmup=args.warmup,
        iters=args.iters,
        check=args.check,
        buffmans=buffmans
    )
    slurm_utils.wait_for(jobs)


def cmd_plot(args) -> None:
    from . import plotting

    root = Path(args.root)
    collectives = args.collective.split(",")
    protos = args.proto.split(",") if args.proto else None
    outdir = Path(args.outdir)
    plotting.run_plot_command(args.mode, root, collectives, protos, outdir, size_filter=args.size)


def cmd_set(args) -> None:
    key_map = {"account": "slurm_account", "partition": "slurm_partition", "qos": "slurm_qos", "nccl_lib": "nccl_lib"}
    cfg_key = key_map[args.what]
    config.set_value(cfg_key, args.value)
    ok(f"{args.what} set to '{args.value}' ({config.CONFIG_FILE})")


# ── argparse wiring ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unitccl", description="Run, profile, and plot Bine-tree NCCL collective benchmarks."
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("build", help="Build nccl / fastest / unitccl components.")
    sp.add_argument("target", choices=["nccl", "fastest", "unitccl", "all"], help="Component to build.")
    sp.add_argument("--clean", action="store_true", help="Clean before building")
    sp.add_argument("--submit", action="store_true", help="Run in a Slurm job")
    sp.add_argument("--preload", action="store_true", help="Load configured modules first")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("standalone", help="Run correctness tests (no Slurm).")
    sp.add_argument(
        "mode", nargs="?", choices=["preload", "submit"], default=None,
        help="preload: apply preload_modules then run locally. submit: run on 1 allocated node, no GPU.",
    )
    sp.add_argument("--buffman", help="Comma-separated buffer management modes (BLOCK_BY_BLOCK, SEND, DOUBLE_SEND, PERMUTATION)")
    sp.set_defaults(func=cmd_standalone)

    sp = sub.add_parser("scaling", help="Run scaling comparisons via fastest pools.")
    sp.add_argument("--coll", help="Comma-separated collectives")
    sp.add_argument("--algo", help="Comma-separated algorithms")
    sp.add_argument("--proto", help="Comma-separated protocols")
    sp.add_argument("--plot", action="store_true", help="Generate PNG plots")
    sp.add_argument("--csv", action="store_true", help="Save CSV outputs")
    sp.add_argument("--check", action="store_true", help="Enable correctness verification")
    sp.add_argument("--warmup", type=int, help="Number of warmup iterations")
    sp.add_argument("--iters", type=int, help="Number of measured iterations")
    sp.add_argument("--ranks", help="Comma-separated rank counts (triggers Slurm sweep)")
    sp.add_argument("--buffman", help="Comma-separated buffer management modes")
    sp.set_defaults(func=cmd_scaling)

    sp = sub.add_parser("nsys", help="Capture nsys profiles, generate stats, and analyze.")
    sp.add_argument("--outdir", type=Path, default=Path("nsys"), help="Output directory")
    sp.add_argument("--coll", help="Comma-separated collectives")
    sp.add_argument("--algo", help="Comma-separated algorithms")
    sp.add_argument("--proto", help="Comma-separated protocols")
    sp.add_argument("--size", default="16777216", help="Comma-separated message sizes")
    sp.add_argument("--nranks", default="8", help="Comma-separated rank counts")
    sp.add_argument("--warmup", type=int, help="Number of warmup iterations")
    sp.add_argument("--iters", type=int, help="Number of measured iterations")
    sp.add_argument("--check", action="store_true", help="Enable correctness verification")
    sp.add_argument("--buffman", help="Comma-separated buffer management modes")
    sp.set_defaults(func=cmd_nsys)

    sp = sub.add_parser("plot", help="Plot rank-sweep data (time vs size, or time vs ranks).")
    sp.add_argument("mode", choices=["ranks", "size"])
    sp.add_argument("--root", default=".", help="Root dir containing <N>_ranks folders")
    sp.add_argument("--collective", "--coll", dest="collective", required=True, help="Comma-separated collectives")
    sp.add_argument("--proto", default=None, help="Comma-separated protocols (default: autodetect)")
    sp.add_argument("--size", default=None, help="(mode=ranks) restrict to one message size, e.g. 4MB")
    sp.add_argument("--outdir", default="plots", help="Output directory for PNGs")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("set", help="Persist a Slurm default (account/partition/qos).")
    sp.add_argument("what", choices=["account", "partition", "qos", "nccl_lib"])
    sp.add_argument("value")
    sp.set_defaults(func=cmd_set)

    sp = sub.add_parser("preload", help="Manage `module load` entries applied before every Slurm job.")
    preload_sub = sp.add_subparsers(dest="preload_command", required=True)
    psp = preload_sub.add_parser("add", help="Add module(s) to the preload list.")
    psp.add_argument("modules", nargs="+", help="Module name(s), e.g. cuda/12.4")
    psp.set_defaults(func=cmd_preload_add)
    psp = preload_sub.add_parser("rm", help="Remove module(s) from the preload list.")
    psp.add_argument("modules", nargs="+", help="Module name(s), e.g. cuda/12.4")
    psp.set_defaults(func=cmd_preload_rm)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
