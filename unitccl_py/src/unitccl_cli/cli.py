"""unitccl command-line entrypoint.

    unitccl standalone
    unitccl scaling coll=Bcast,AllReduce algo=BINE,RING proto=SIMPLE plot csv check warmup=10 iters=40
    unitccl scaling coll=Bcast proto=SIMPLE csv ranks=4,8,16,32,64,128     # submitit sweep, one alloc/rank-count
    unitccl nsys [outdir] coll=Bcast,Reduce algo=BINE,RING proto=SIMPLE size=16777216 nranks=8 warmup=10 iters=40 check
    unitccl plot ranks --root . --collective Bcast,AllReduce --proto SIMPLE,LL [--size 4MB]
    unitccl plot size  --root . --collective Bcast --proto SIMPLE
    unitccl set account <value>
    unitccl set partition <value>
    unitccl set qos <value>
    unitccl preload add <module> [<module> ...]   # module load before every Slurm job
    unitccl preload rm  <module> [<module> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from . import config
from .logging_utils import error, ok

# ── shared "key=value / bare-flag" parsing for standalone/scaling/nsys ──────
#
# Mirrors the current tests.py argv convention:
#   coll=Bcast,AllReduce   -> filter
#   algo=BINE,RING         -> filter
#   proto=SIMPLE,LL        -> filter
#   plot / csv / check     -> bare flags
#   warmup=10 / iters=40   -> key=value settings
#   ranks=4,8,16           -> key=value setting (scaling: triggers a sweep)


def _parse_kv_tokens(argv: List[str]) -> Dict:
    flags: Set[str] = set()
    filters = {"coll": None, "algo": None, "proto": None}
    kv: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            key, _, val = arg.partition("=")
            if key in filters:
                filters[key] = set(val.split(","))
            else:
                kv[key] = val
        else:
            flags.add(arg)
    return {"flags": flags, "filters": filters, "kv": kv}


def _int_or_none(kv: dict, key: str) -> Optional[int]:
    return int(kv[key]) if key in kv else None


# ── subcommands ──────────────────────────────────────────────────────────────


def cmd_standalone(args) -> None:
    from . import fastest_iface

    if args.mode == "submit":
        from . import slurm_utils

        jobs = slurm_utils.submit_standalone()
        slurm_utils.wait_for(jobs)
        return

    if args.mode == "preload":
        from . import slurm_utils

        cfg = config.load()
        slurm_utils.apply_preload_modules(cfg.get("preload_modules") or [])

    fastest_iface.run_standalone()


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

    parsed = _parse_kv_tokens(args.rest)
    flags, filters, kv = parsed["flags"], parsed["filters"], parsed["kv"]

    scaling_kwargs = dict(
        colls=filters["coll"],
        algos=filters["algo"],
        protos=filters["proto"],
        do_plot="plot" in flags,
        do_csv="csv" in flags,
        check="check" in flags,
        warmup=_int_or_none(kv, "warmup"),
        iters=_int_or_none(kv, "iters"),
    )

    if "ranks" in kv:
        ranks_list = [int(r) for r in kv["ranks"].split(",")]
        jobs = slurm_utils.submit_rank_sweep(ranks_list, scaling_kwargs)
        slurm_utils.wait_for(jobs)
    else:
        fastest_iface.run_scaling(**scaling_kwargs)


def cmd_nsys(args) -> None:
    from . import nsys_utils

    positional = [a for a in args.rest if "=" not in a and a not in {"check"}]
    parsed = _parse_kv_tokens(args.rest)
    flags, filters, kv = parsed["flags"], parsed["filters"], parsed["kv"]

    outdir = Path(positional[0]) if positional else Path("nsys")
    colls = sorted(filters["coll"]) if filters["coll"] else ["Bcast", "Reduce"]
    algos = sorted(filters["algo"]) if filters["algo"] else ["BINE", "RING"]
    proto = sorted(filters["proto"])[0] if filters["proto"] else "SIMPLE"
    size = int(kv.get("size", 16777216))
    nranks = int(kv.get("nranks", 8))
    check = "check" in flags or kv.get("check", "").lower() in ("1", "true", "on")

    nsys_utils.run_profiles(
        outdir,
        colls,
        algos,
        size=size,
        nranks=nranks,
        proto=proto,
        warmup=_int_or_none(kv, "warmup"),
        iters=_int_or_none(kv, "iters"),
        check=check,
    )
    nsys_utils.generate_stats(outdir)
    nsys_utils.analyze(outdir)


def cmd_plot(args) -> None:
    from . import plotting

    root = Path(args.root)
    collectives = args.collective.split(",")
    protos = args.proto.split(",") if args.proto else None
    outdir = Path(args.outdir)
    plotting.run_plot_command(args.mode, root, collectives, protos, outdir, size_filter=args.size)


def cmd_set(args) -> None:
    key_map = {"account": "slurm_account", "partition": "slurm_partition", "qos": "slurm_qos"}
    cfg_key = key_map[args.what]
    config.set_value(cfg_key, args.value)
    ok(f"{args.what} set to '{args.value}' ({config.CONFIG_FILE})")


# ── argparse wiring ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unitccl", description="Run, profile, and plot Bine-tree NCCL collective benchmarks."
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("standalone", help="Run correctness tests (no Slurm).")
    sp.add_argument(
        "mode",
        nargs="?",
        choices=["preload", "submit"],
        default=None,
        help="preload: apply preload_modules then run locally. submit: run on 1 allocated node, no GPU.",
    )
    sp.set_defaults(func=cmd_standalone)

    sp = sub.add_parser("scaling", help="Run scaling comparisons via fastest pools.")
    sp.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="coll=.. algo=.. proto=.. plot csv check warmup=N iters=N [ranks=4,8,16,...]",
    )
    sp.set_defaults(func=cmd_scaling)

    sp = sub.add_parser("nsys", help="Capture nsys profiles, generate stats, and analyze.")
    sp.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[outdir] coll=.. algo=.. proto=.. size=N nranks=N warmup=N iters=N check",
    )
    sp.set_defaults(func=cmd_nsys)

    sp = sub.add_parser("plot", help="Plot rank-sweep data (time vs size, or time vs ranks).")
    sp.add_argument("mode", choices=["ranks", "size"])
    sp.add_argument("--root", default=".", help="Root dir containing <N>_ranks folders")
    sp.add_argument(
        "--collective", "--coll", dest="collective", required=True, help="Comma-separated collectives"
    )
    sp.add_argument("--proto", default=None, help="Comma-separated protocols (default: autodetect)")
    sp.add_argument("--size", default=None, help="(mode=ranks) restrict to one message size, e.g. 4MB")
    sp.add_argument("--outdir", default="plots", help="Output directory for PNGs")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("set", help="Persist a Slurm default (account/partition/qos).")
    sp.add_argument("what", choices=["account", "partition", "qos"])
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
    except Exception as e:  # surface cluster/tool errors cleanly, no traceback spam
        error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
