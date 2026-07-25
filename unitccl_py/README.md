# unitccl_py

CLI + library that consolidates `tests.py`, `scaling.py` (the `<N>_ranks`
plotting script), `nsys.py` (`02_analyze_and_plot.py`), `generate_stats.sh`,
and the hand-written `meluxina*.sbatch` / `leonardo*.sbatch` submission
scripts into one tool. This is a full rewrite from scratch -- none of the
original scripts are wrapped or shelled out to internally; they're fully
replaced.

## Install

```bash
pip install -e .                 # core (standalone, scaling, nsys, plot, Slurm submission)
pip install -e '.[tui]'          # + rich, for the live job dashboard

# Build the `fastest` orchestration package and the compiled backend --
# replaces the old manual `pip install vendor/fastest/fastest_py` +
# `./build_wheel.sh` steps.
unitccl build nccl          # builds the NCCL fork
unitccl build fastest
unitccl build unitccl
# or just: unitccl build all
```

## Dependencies

Core, installed with plain `pip install -e .`:

- `pandas>=1.5`
- `matplotlib>=3.6`
- `submitit>=1.5` -- Slurm job submission (`ranks=...` sweeps, `build ...
  submit`, `standalone submit`)

Optional, via `pip install -e '.[tui]'`:

- `rich>=13.0` -- powers `tui_utils.watch_jobs`'s live table/log view.
  Without it (or when stdout isn't a TTY), Slurm job watching falls back
  automatically to the plain `[job_id:stream] line` polling loop -- no
  functionality is lost, you just don't get the dashboard/cancel keys.
  `requirements.txt` lists it unconditionally; `pyproject.toml` keeps it
  as its own extra since it's a dashboard nicety, not required for
  Slurm submission itself to work.

Not on PyPI at all -- install separately into the same venv:

- `fastest_py` (`vendor/fastest/fastest_py`) and your compiled pybind11
  backend (e.g. `unitccl`) -- both handled by `unitccl build fastest` /
  `unitccl build unitccl` above, no manual `pip install`/`build_wheel.sh`
  needed anymore.

## Layout

```
src/unitccl_cli/
├── build_utils.py      # nccl/fastest/unitccl build orchestration (make/nvcc invocations)
├── cli.py              # `unitccl` entrypoint (argparse subcommands)
├── config.py           # collective/algo/proto registries, env-var names,
│                       # persisted Slurm defaults + backend module name
├── fastest_iface.py    # wraps `fastest` + dynamic backend import;
│                       # standalone tests + scaling comparisons/plots
├── logging_utils.py    # shared ANSI logging helpers ([info]/[ok]/[error])
├── nsys_utils.py       # nsys profile capture, `nsys stats` CSV export,
│                       # and BINE-vs-RING analysis plots
├── plotting.py         # rank-sweep plots (time vs size, time vs ranks)
├── schema.py           # BenchRecord + size parsing; unifies fastest's own
│                       # CSVs and the <N>_ranks/ sweep CSVs into one loader
├── slurm_utils.py       # submitit: one right-sized alloc per rank count
└── tui_utils.py          # live rich-based dashboard for watching/cancelling
                          # submitted Slurm jobs (falls back to plain text)
```

## Commands

```bash
# Correctness tests, no Slurm involved.
unitccl standalone
unitccl standalone preload      # apply preload_modules, then run locally
unitccl standalone submit       # run on 1 allocated node, no GPU

# Build nccl / fastest / unitccl / all.
unitccl build unitccl
unitccl build all clean submit  # clean build, submitted as a Slurm job
unitccl build nccl preload      # apply preload_modules, then build locally

# Scaling comparison via fastest pools (matches the original tests.py behavior).
unitccl scaling coll=Bcast,AllReduce algo=BINE,RING proto=SIMPLE plot csv check warmup=10 iters=40

# Same, but swept across rank counts: submits one independently-sized
# submitit/Slurm job per rank count (no shared oversized allocation), then
# writes csvs to <N>_ranks/<coll>/<coll>_<proto>.csv for `unitccl plot` to
# read. Submitting drops you into a live TUI dashboard (see below) that
# blocks until every job finishes.
unitccl scaling coll=Bcast proto=SIMPLE csv ranks=4,8,16,32,64,128

# Capture nsys profiles for BINE vs RING, export nsys-stats CSVs, and
# generate the bine-vs-ring comparison plots -- all in one call.
unitccl nsys nsys_out coll=Bcast,Reduce algo=BINE,RING proto=SIMPLE size=16777216 nranks=8 warmup=10 iters=40 check

# Plot a rank sweep already on disk.
unitccl plot ranks --root . --collective Bcast,AllReduce --proto SIMPLE,LL
unitccl plot size  --root . --collective Bcast --proto SIMPLE

# Persist Slurm defaults used by `ranks=...` sweeps and (optionally) nsys jobs.
unitccl set account p201236
unitccl set partition boost_usr_prod
unitccl set qos default
unitccl set nccl_lib /path/to/nccl/lib

# Manage `module load` entries applied before every Slurm job (and before
# `preload`-mode local runs).
unitccl preload add cuda/12.4 openmpi/4.1
unitccl preload rm  cuda/12.4
```

Config is stored at `~/.config/unitccl/config.json` (override the directory
with `UNITCCL_CONFIG_DIR`). To point the CLI at a different project's
backend: `unitccl set backend_module <name>` isn't wired into `set` (only
account/partition/qos/nccl_lib are, per the spec) -- for now, set it via
`UNITCCL_BACKEND_MODULE=<name>` in the environment, or edit
`~/.config/unitccl/config.json` directly.

`default_confs/` (next to `unitccl_py/`) ships ready-made per-cluster values
-- `meluxina.json` and `leonardo.json` -- with the account/partition/qos/
module settings for each HPC system this has been run on. Nothing loads
these automatically; copy the relevant fields into your own
`~/.config/unitccl/config.json` (or run the `unitccl set`/`preload add`
commands above with those values) when switching clusters.

## Live job dashboard

Submitting anything through Slurm (`ranks=...` sweeps, `build ... submit`,
`standalone submit`) hands off to `tui_utils.watch_jobs`, which -- if `rich`
is installed and stdout is a real terminal -- renders a live table (job id,
state, rank count, elapsed time, a throttled `squeue --start` ETA for
pending jobs, and a done marker) plus a scrolling tail of every job's
stdout/stderr, all refreshed in place. Keyboard controls:

| Key | Action |
|---|---|
| `↑`/`k`, `↓`/`j` | move the selection |
| `c` / `x` | cancel the selected job (`scancel`) |
| `a` | cancel **all** jobs (press again, or `y`, within 3s to confirm) |
| `q` | detach -- stop watching; jobs keep running on the cluster |

A job that ends up `CANCELLED` (via these keys, an external `scancel`, or
ctrl-C) is treated as a clean stop, not a crash -- only a genuine
`FAILED`/`TIMEOUT` raises and surfaces as an error. Without `rich`, or when
stdout isn't a TTY (e.g. piped into a log file or run in CI), this falls
back automatically to the original plain `[job_id:stream] line` polling
loop, with no keyboard controls.

## Output layout

A `ranks=... plot` sweep, followed by `unitccl plot ranks` and/or `unitccl
nsys`, produces (paths relative to wherever you ran `unitccl` from, or
`--outdir` for `plot`/`nsys`):

```
plots/
├── 004_ranks/                          # one folder per `ranks=` value, %03d-padded
│   ├── Bcast/
│   │   ├── Bcast_SIMPLE.csv            # written directly by the sweep job
│   │   ├── Bcast_SIMPLE.png            # written by `unitccl plot`
│   │   ├── Bcast_SIMPLE_diff.png       # BINE vs RING difference plot
│   │   ├── Bcast_LL.csv / .png / _diff.png
│   │   └── Bcast_LL128.csv / .png / _diff.png
│   └── Reduce/                         # same per-protocol layout
├── 008_ranks/ … 128_ranks/
├── scaling/                            # `unitccl plot ranks`: time vs ranks
│   ├── Bcast/
│   │   ├── Bcast_SIMPLE_scaling.png
│   │   └── Bcast_SIMPLE_vs_ranks_by_size.png
│   └── Reduce/
└── <nsys outdir>/
    └── <run_label>/<PROTO>/
        ├── per_rank_metrics.csv
        ├── summary_by_algo_collective.csv
        ├── nccl_collective_op_time.png
        ├── gpu_kernel_time.png
        ├── gpu_memcpy_time.png
        ├── cuda_api_total_time.png
        ├── network_wait_time.png
        ├── nccl_init_time.png
        └── {bcast,reduce}_per_rank_op_time.png
```

`<N>_ranks/<Coll>/<Coll>_<PROTO>.csv` is what the submitit sweep jobs write
directly; `plotting.py` reads those csvs back for both `plot ranks` and
`plot size`, and `nsys_utils.py` owns the `nsys/`-shaped subtree above.
