# unitccl

CLI + library that consolidates `tests.py`, `scaling.py` (the `<N>_ranks`
plotting script), `nsys.py` (`02_analyze_and_plot.py`), `generate_stats.sh`,
and the two `meluxina*.sbatch` scripts into one tool.

> **Naming note:** the *distribution/command* is `unitccl` (`pip install
> unitccl` gives you the `unitccl` shell command), but the *import package*
> is `unitccl_cli`. This is deliberate: your project's compiled `fastest`
> pybind11 backend extension is conventionally also called `unitccl` (built
> by `build_wheel.sh`). If this library's import package were also named
> `unitccl`, one would shadow the other on `sys.path`. `unitccl_cli` imports
> the backend dynamically by name (`config.backend_module`, default
> `"unitccl"`), so the two never collide.

## Install

```bash
pip install -e .                 # core (standalone, scaling, nsys, plot)
pip install -e '.[slurm]'        # + submitit, for `ranks=...` sweeps
```

Separately, in the same venv, install the project-specific pieces this
depends on but doesn't vendor:

```bash
pip install -e vendor/fastest/fastest_py     # the `fastest` orchestration package
./build_wheel.sh && pip install vendor/fastest/<project>_backend_wheel  # your compiled backend
```

## Layout

```
src/unitccl_cli/
├── config.py          # collective/algo/proto registries, env-var names,
│                       # persisted Slurm defaults + backend module name
├── logging_utils.py    # shared ANSI logging helpers
├── schema.py            # BenchRecord + size parsing; unifies fastest's own
│                       # CSVs and the <N>_ranks/ sweep CSVs into one loader
├── fastest_iface.py     # wraps `fastest` + dynamic backend import;
│                       # standalone tests + scaling comparisons/plots
├── nsys_utils.py         # nsys profile capture, `nsys stats` CSV export,
│                       # and BINE-vs-RING analysis plots
├── plotting.py           # rank-sweep plots (time vs size, time vs ranks)
├── slurm_utils.py        # submitit: one right-sized alloc per rank count
└── cli.py                 # `unitccl` entrypoint
```

## Commands

```bash
# Correctness tests, no Slurm involved.
unitccl standalone

# Scaling comparison via fastest pools (matches current tests.py behavior).
unitccl scaling coll=Bcast,AllReduce algo=BINE,RING proto=SIMPLE plot csv check warmup=10 iters=40

# Same, but swept across rank counts: submits one independently-sized
# submitit/Slurm job per rank count (no shared oversized allocation), then
# writes csvs to <N>_ranks/<coll>/<coll>_<proto>.csv for `unitccl plot` to read.
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
```

Config is stored at `~/.config/unitccl/config.json` (override the directory
with `UNITCCL_CONFIG_DIR`). To point the CLI at a different project's
backend: `unitccl set backend_module <name>` isn't wired into `set` (only
account/partition/qos are, per the spec) -- for now, set it via
`UNITCCL_BACKEND_MODULE=<name>` in the environment, or edit
`~/.config/unitccl/config.json` directly.

## What changed vs. the original scripts

- **One schema** (`schema.BenchRecord`) replaces three separate regex/parsing
  schemes (fastest's own `test` column, the `<N>_ranks` directory walk, and
  size-string handling) that used to live independently in `tests.py` and
  `scaling.py`.
- **`ranks=...` sweeps are submitit-backed**, one job per rank count, each
  sized to exactly that rank count -- this replaces hand-maintained
  `meluxina.sbatch`-style scripts for the scaling sweep.
- **`unitccl nsys`** replaces the `run_profile()` loop in
  `meluxina_nsys.sbatch`, `generate_stats.sh`, and `02_analyze_and_plot.py`'s
  `main()`, as one pipeline: capture -> stats -> analyze/plot.
- `check`/`warmup`/`iters` are now opt-in flags on `scaling`/`nsys` rather
  than hardcoded globals -- `UNITCCL_CHECK` defaults to `"0"` unless you pass
  `check`, where the original `tests.py` hardcoded it to `"1"`. Worth
  double-checking this matches what you want before relying on it.
