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
- `submitit>=1.5` -- Slurm job submission (`--ranks=...` sweeps, `build ...
  --submit`, `standalone submit`)

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
├── nccltests_utils.py  # nccl-tests backend: same sweep matrix as `scaling`, plot-compatible CSVs
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
unitccl standalone --buffman SEND,DOUBLE_SEND  # override NCCL_BINE_BUFFER_MANAGEMENT

# Build nccl / fastest / unitccl / all.
unitccl build unitccl
unitccl build all --clean --submit  # clean build, submitted as a Slurm job
unitccl build nccl --preload        # apply preload_modules, then build locally

# Scaling comparison via fastest pools (matches the original tests.py behavior).
unitccl scaling --coll Bcast,AllReduce --algo BINE,RING --proto SIMPLE --plot --csv --check --warmup 10 --iters 40 --buffman SEND,BLOCK_BY_BLOCK

# Same, but swept across rank counts: submits one independently-sized
# submitit/Slurm job per rank count (no shared oversized allocation), then
# writes csvs to <N>_ranks/<coll>/<coll>_<proto>.csv for `unitccl plot` to
# read. Submitting drops you into a live TUI dashboard (see below) that
# blocks until every job finishes.
unitccl scaling --coll Bcast --proto SIMPLE --csv --ranks 4,8,16,32,64,128

# Same matrix measured with nccl-tests instead of fastest/unitccl_bench. Writes
# plots/<NNN>_ranks/<Coll>/<Coll>_<PROTO>.csv in the exact scaling schema, so
# `unitccl plot ...` and the Δ-vs-baseline plots just work. --repeats R = R
# nccl-tests cycles (-N) = R samples per row (mean/stddev/min/max/median).
unitccl nccltests --coll AllGather --algo BINE,RING --proto SIMPLE --repeats 5 --plot --ranks 4,8,16,32,64,128
unitccl nccltests --coll AllGather --buffman SEND,BLOCK_BY_BLOCK --nranks 8 --outdir plots/008_ranks   # local, inside an allocation
unitccl nccltests --coll AllGather --sizes 1kB,64kB,1MB,16MB --ranks 8,16   # custom per-rank sizes

# Capture nsys profiles for BINE vs RING, export nsys-stats CSVs, and
# generate the bine-vs-ring comparison plots -- all in one call.
unitccl nsys --outdir nsys_out --coll Bcast,Reduce --algo BINE,RING --proto SIMPLE --size 16777216 --nranks 8 --warmup 10 --iters 40 --check --buffman SEND,DOUBLE_SEND

# Plot a rank sweep already on disk.
unitccl plot ranks --root . --collective Bcast,AllReduce --proto SIMPLE,LL
unitccl plot size  --root . --collective Bcast --proto SIMPLE

# Persist Slurm defaults used by `--ranks=...` sweeps and (optionally) nsys jobs.
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

## nccl-tests backend

`unitccl nccltests` runs `all_gather_perf` / `all_reduce_perf` / `broadcast_perf` /
`reduce_perf` / `reduce_scatter_perf` (AllGatherV has no nccl-tests binary and is
skipped) under `mpirun`, one rank per GPU, with the same `NCCL_ALGO` /
`NCCL_PROTO` / `NCCL_BINE_BUFFER_MANAGEMENT` matrix as `scaling`. Rank sweeps use
one right-sized Slurm job per rank count and launch `mpirun` from task 0, the same
pattern as the nsys jobs.

Build it once with `unitccl build nccl && unitccl build nccltests` (needs `mpicc`
and `nvcc` on PATH, or `MPI_HOME` / `CUDA_HOME` set; the clone needs internet, so
do it on a login node). To use an existing checkout instead:
`unitccl set nccltests_dir /path/to/nccl-tests` or `UNITCCL_NCCLTESTS_DIR`.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--coll` | all | Comma-separated: AllGather, AllReduce, Bcast, Reduce, ReduceScatter |
| `--algo` | per-collective defaults | `NCCL_ALGO` values; the first one is the baseline of the `_diff` plot |
| `--proto` | SIMPLE,LL,LL128 | `NCCL_PROTO` values |
| `--buffman` | none | BINE buffer management modes, e.g. `SEND,BLOCK_BY_BLOCK`; BINE rows are labelled `BINE-<mode>` |
| `--sizes` | `1kB,16kB,256kB,1MB,4MB,64MB` | **Per-rank** message sizes, see below |
| `--warmup` | 10 | nccl-tests `-w` |
| `--iters` | 40 | nccl-tests `-n`, timed back to back |
| `--repeats` | 1 | nccl-tests `-N`: cycles per size; each cycle average is one sample in the CSV |
| `--avg` | `max` | Cross-rank reduction (nccl-tests `-a`): `max`, `avg`, `min`, `rank0` |
| `--inplace` | off | Record the in-place column instead of out-of-place |
| `--per-size` | off | One nccl-tests process (fresh communicator) per size instead of one sweep |
| `--check` | off | nccl-tests `-c 1`; wrong values are reported as failures |
| `--extra` | none | Extra nccl-tests arguments, e.g. `--extra '-z 1'` |
| `--timeout` | 900 | Seconds per nccl-tests process before it is killed |
| `--strict` | off | Raise (Slurm job FAILED) if any size failed, timed out or was wrong |
| `--plot` | off | Write `<Coll>_<PROTO>.png` and `_diff.png` next to each CSV |
| `--ranks` | none | Comma-separated rank counts: submit one Slurm job per count |
| `--nranks` | `SLURM_NTASKS` or `gpus_per_node` | Local runs only (no `--ranks`) |
| `--time` | 60 | Slurm time limit per job, in minutes |
| `--outdir` | `plots` | Root for `<NNN>_ranks/<Coll>/` (local runs: written to `<outdir>/<Coll>/`) |

### Choosing message sizes (`--sizes`)

`--sizes` takes a comma-separated list such as `--sizes 1kB,64kB,1MB,16MB`.

- Sizes are **per rank**: the byte count of the `count` argument given to the
  collective, the same as the `1kB ... 64MB` labels in `scaling` and `unitccl_bench`'s
  `vec_size * 4`. For AllGather the gathered buffer is `size * ranks`, for
  ReduceScatter the input is. You do not convert anything: nccl-tests' `-b/-e`
  are totals for those two, so the tool multiplies by the rank count itself.
- Accepted units are `B`, `kB`, `MB`, `GB` (case-insensitive, powers of 1024),
  e.g. `4096B`, `256kB`, `1.5MB`. Every size must be a multiple of 4 bytes
  (floats). Duplicates are dropped and the list is sorted.
- If **all** sizes are powers of two (the default set is), they run as a single
  `-b <min> -e <max> -f 2` sweep in one communicator. The intermediate sizes the
  sweep also visits (2kB, 4kB, ...) are discarded, not written to the CSV.
- If any size is not a power of two (e.g. `1.5MB`, `3000B`), or you pass
  `--per-size`, each size runs in its own nccl-tests process with `-b = -e`. That
  costs one communicator init per size and every size gets a fresh communicator.
- Labels in the CSV are normalised (`1048576B` becomes `1MB`), and the pool name
  uses the smallest and largest size (`scaling/<min>_<max>/<Algo>_<Coll>`). A custom
  `--sizes` therefore produces CSVs that `unitccl plot` reads like any other.
- Mixing runs with different `--sizes` under one `<NNN>_ranks/` directory is fine
  for `plot size`, but `plot ranks` draws one panel per size found, so sizes
  missing at some rank counts give panels with fewer points.
- The exact range nccl-tests accepts is limited by GPU memory: for AllGather the
  receive buffer is `max size * ranks` per GPU.

### Semantics vs `unitccl_bench`

- **Timing** is `-n` iterations back to back with no per-iteration barrier. One `-N`
  cycle average is one sample; `mean_ns`, `stddev_ns`, `min_ns`, `max_ns`,
  `median_ns` and `samples_ns` are computed over those samples (a single repeat
  gives `stddev_ns = 0`, like the existing CSVs). `-N` needs a reasonably recent
  nccl-tests.
- `--avg max` reports the slowest rank's average, which is the closest analogue of
  the straggler metric but not identical to a per-iteration max.
- nccl-tests always runs out-of-place and in-place; only one column is recorded.
- A size that times out, crashes, or reports wrong values is listed at the end and
  left out of the CSV (not zero-filled, so log-scale plots keep working).
- Output defaults to the same `plots/` root as `scaling`, so it overwrites those
  CSVs (a warning is printed). Use `--outdir plots_nt` and
  `unitccl plot ranks --root plots_nt` to keep both.
- The first run prints which `libnccl` the binary resolves, and `NCCL_DEBUG=VERSION`
  prints the NCCL version, to confirm the fork is the one being measured.

## Live job dashboard

Submitting anything through Slurm (`--ranks=...` sweeps, `build ... --submit`,
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

A `--ranks=...` plot sweep, followed by `unitccl plot ranks` and/or `unitccl
nsys`, produces (paths relative to wherever you ran `unitccl` from, or
`--outdir` for `plot`/`nsys`):

```
plots/
├── 004_ranks/                          # one folder per `--ranks=` value, %03d-padded
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
