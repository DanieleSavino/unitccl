"""nccl-tests backend for the scaling sweep.

Runs NVIDIA's nccl-tests binaries (all_gather_perf, all_reduce_perf, ...)
under the same collective / algo / proto / buffman / rank-count matrix as
`unitccl scaling`, and writes CSVs with the *same schema and directory
layout* (`<outdir>/<NNN>_ranks/<Coll>/<Coll>_<PROTO>.csv`, columns
`pool,test,mean_ns,stddev_ns,min_ns,max_ns,median_ns,samples_ns`), so
`unitccl plot ranks|size` and the Δ-vs-baseline plots work unchanged.

Semantics worth knowing when comparing against `unitccl scaling`:

* Message size.  The size label (1kB ... 64MB) is the *per-rank* `count`
  handed to the collective, exactly like unitccl_bench's vec_size*4.  nccl-tests
  `-b/-e` are totals for AllGather/ReduceScatter, so those are multiplied by
  the rank count here.  The row is matched back via nccl-tests' `count`
  column (which is per-rank for every collective), not the `size` column.
* Timing.  nccl-tests launches `-n` iterations back to back (no barrier
  between them) and prints one average per size per cycle.  One cycle = one
  sample.  `--repeats R` runs R cycles (`-N R`) in the same communicator;
  mean/stddev/min/max/median are taken over those R cycle averages.
* Cross-rank reduction.  `-a max` (default) reports the slowest rank's
  average, the closest analogue of unitccl_bench's per-iteration straggler.
* Out-of-place (default) vs in-place: nccl-tests always runs both; `inplace`
  picks which column is recorded.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import statistics
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from . import config
from .logging_utils import color, error, info, ok, section, warn
from .schema import load_fastest_csv, records_to_df, size_to_bytes

NT_BINARIES = {
    "AllGather": "all_gather_perf",
    "AllReduce": "all_reduce_perf",
    "Bcast": "broadcast_perf",
    "Reduce": "reduce_perf",
    "ReduceScatter": "reduce_scatter_perf",
}
# nccl-tests -b/-e are totals for these; per-rank count * nranks.
SCALES_WITH_RANKS = {"AllGather", "ReduceScatter"}
REDUCING = {"AllReduce", "Reduce", "ReduceScatter"}
ROOTED = {"Bcast", "Reduce"}
AVG_MODES = {"rank0": 0, "avg": 1, "min": 2, "max": 3}
TYPE_BYTES = 4  # `-d float`, same as unitccl_bench

CSV_COLUMNS = ["pool", "test", "mean_ns", "stddev_ns", "min_ns", "max_ns", "median_ns", "samples_ns"]


# ── helpers ──────────────────────────────────────────────────────────────────


def format_size(nbytes: int) -> str:
    """4194304 -> '4MB', 1024 -> '1kB' (the labels fastest's CSVs use)."""
    for unit, factor in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if nbytes % factor == 0:
            return f"{nbytes // factor}{unit}"
    return f"{nbytes}B"


def parse_sizes(sizes: Optional[Sequence[str]]) -> List[int]:
    out = sorted({int(size_to_bytes(s)) for s in (sizes or config.DEFAULT_SIZES)})
    bad = [format_size(b) for b in out if b % TYPE_BYTES]
    if bad:
        raise ValueError(f"sizes must be multiples of {TYPE_BYTES} bytes: {bad}")
    return out


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def plan_invocations(
    sizes: Sequence[int], mult: int, per_size: bool
) -> List[Tuple[int, int, int]]:
    """[(min_total, max_total, step_factor)] nccl-tests invocations to cover `sizes`.

    When every size is a power of two, one `-b min -e max -f 2` sweep covers
    them all in a single communicator (extra intermediate sizes are simply
    ignored when parsing). Otherwise, or with per_size, one invocation each.
    """
    if not per_size and len(sizes) > 1 and all(_is_pow2(s) for s in sizes):
        return [(min(sizes) * mult, max(sizes) * mult, 2)]
    return [(s * mult, s * mult, 2) for s in sizes]


def parse_rows(text: str, inplace: bool = False) -> List[Tuple[int, float, Optional[int]]]:
    """Parse nccl-tests' table -> [(per_rank_count, time_us, n_wrong|None)].

    Row layout (both 2.1x layouts agree):
      size count type redop root | time algbw busbw #wrong (out-of-place)
                                 | time algbw busbw #wrong (in-place)
    """
    t_col = 9 if inplace else 5
    rows = []
    for line in text.splitlines():
        tok = line.split()
        if len(tok) < 13 or line.lstrip().startswith("#"):
            continue
        if not (tok[0].isdigit() and tok[1].isdigit() and tok[2].isalpha()):
            continue
        try:
            time_us = float(tok[t_col])
        except ValueError:
            continue
        wrong_s = tok[t_col + 3]
        rows.append((int(tok[1]), time_us, int(wrong_s) if wrong_s.isdigit() else None))
    return rows


def summarize(samples_ns: Sequence[float]) -> Dict[str, object]:
    n = len(samples_ns)
    return {
        "mean_ns": statistics.fmean(samples_ns),
        "stddev_ns": statistics.stdev(samples_ns) if n > 1 else 0.0,
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "median_ns": statistics.median(samples_ns),
        "samples_ns": "[" + ", ".join(str(int(round(x))) for x in samples_ns) + "]",
    }


def resolve_bin_dir() -> Path:
    bin_dir = config.nccltests_dir() / "build"
    if not bin_dir.is_dir():
        raise RuntimeError(
            f"nccl-tests binaries not found in {bin_dir}. Run `unitccl build nccltests`, or "
            f"point at a checkout with `unitccl set nccltests_dir <path>` / UNITCCL_NCCLTESTS_DIR."
        )
    return bin_dir


# ── process running ──────────────────────────────────────────────────────────


def _run_streaming(
    cmd: List[str], env: Dict[str, str], timeout: float, drain_grace: float = 10.0
) -> Tuple[int, str, bool]:
    """Run `cmd`, echoing output live. Kills the whole process group on timeout.

    Completion is decided by the *process* exiting, not by the output pipe
    hitting EOF: a leftover child (orted, ssh/srun helper, ...) that inherited
    the pipe would otherwise block us forever after mpirun is long gone.
    """
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    timed_out = threading.Event()
    lines: List[str] = []

    def _signal(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def _read():
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()

    def _on_timeout():
        timed_out.set()
        _signal(signal.SIGTERM)

    timer = threading.Timer(timeout, _on_timeout)
    killer = threading.Timer(timeout + 15, lambda: _signal(signal.SIGKILL))
    timer.start()
    killer.start()
    try:
        proc.wait()
        reader.join(drain_grace)
        if reader.is_alive():
            warn(
                f"{cmd[0]} exited (rc={proc.returncode}) but a leftover process still holds its "
                f"output pipe after {drain_grace:.0f}s; killing its process group"
            )
            _signal(signal.SIGKILL)
            reader.join(5)
    finally:
        timer.cancel()
        killer.cancel()
    return proc.returncode, "".join(lines), timed_out.is_set()


def _job_env(
    algo: str, proto: str, buffman: Optional[str], extra_env: Optional[Dict[str, str]]
) -> Tuple[Dict[str, str], List[str]]:
    """Environment for one run and the names `mpirun -x` must forward."""
    env = dict(os.environ)
    env.update(extra_env or {})
    env["NCCL_ALGO"] = algo
    env["NCCL_PROTO"] = proto
    env.setdefault("NCCL_DEBUG", "VERSION")  # prints the NCCL version actually loaded
    if buffman:
        env[config.BINE_BUFFER_MANAGEMENT_ENV] = buffman
    else:
        env.pop(config.BINE_BUFFER_MANAGEMENT_ENV, None)
    nccl_lib = config.get("nccl_lib")
    if nccl_lib:
        env["LD_LIBRARY_PATH"] = f"{nccl_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    forward = sorted(
        k for k in env if k.startswith("NCCL_") or k.startswith("UNITCCL_") or k == "LD_LIBRARY_PATH"
    )
    return env, forward


def _build_cmd(
    binary: Path, coll: str, nranks: int, b_total: int, e_total: int, factor: int,
    warmup: int, iters: int, cycles: int, check: bool, avg_mode: str,
    forward: Sequence[str], extra: Sequence[str],
) -> List[str]:
    cmd = ["mpirun", "-n", str(nranks)]
    for name in forward:
        cmd += ["-x", name]
    if shutil.which("stdbuf"):
        # stdout is a pipe, so libc would block-buffer: the live log lags and
        # looks stalled until the process exits
        cmd += ["stdbuf", "-oL", "-eL"]
    cmd += [
        str(binary),
        "-b", str(b_total), "-e", str(e_total), "-f", str(factor),
        "-g", "1", "-d", "float",
        "-w", str(warmup), "-n", str(iters), "-N", str(cycles),
        "-c", "1" if check else "0",
        "-a", str(AVG_MODES[avg_mode]),
    ]
    if coll in REDUCING:
        cmd += ["-o", "sum"]
    if coll in ROOTED:
        cmd += ["-r", "0"]
    return cmd + list(extra)


def _report_fork(binary: Path, env: Dict[str, str]) -> None:
    """Best effort: show which libnccl the binary will resolve (fork vs system)."""
    if shutil.which("ldd") is None:
        return
    out = subprocess.run(["ldd", str(binary)], env=env, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "libnccl" in line:
            info(f"{binary.name} -> {line.strip()}")


def run_one(
    coll: str, algo: str, proto: str, buffman: Optional[str], sizes: Sequence[int],
    nranks: int, bin_dir: Path, warmup: int, iters: int, repeats: int, check: bool,
    avg_mode: str, inplace: bool, per_size: bool, timeout: float,
    extra: Sequence[str], extra_env: Optional[Dict[str, str]],
) -> Tuple[Dict[int, List[float]], List[str]]:
    """One (coll, algo, proto, buffman) cell. Returns ({per_rank_bytes: [sample_ns]}, failures)."""
    binary = bin_dir / NT_BINARIES[coll]
    if not binary.exists():
        raise RuntimeError(f"{binary} not found; run `unitccl build nccltests`")
    env, forward = _job_env(algo, proto, buffman, extra_env)
    mult = nranks if coll in SCALES_WITH_RANKS else 1
    wanted = set(sizes)
    samples: Dict[int, List[float]] = {}
    failures: List[str] = []

    for b_total, e_total, factor in plan_invocations(sizes, mult, per_size):
        cmd = _build_cmd(
            binary, coll, nranks, b_total, e_total, factor, warmup, iters, repeats,
            check, avg_mode, forward, extra,
        )
        info("$ " + " ".join(cmd))
        t0 = time.monotonic()
        rc, out, timed_out = _run_streaming(cmd, env, timeout)
        span = f"{format_size(b_total // mult)}..{format_size(e_total // mult)}/rank"
        info(f"process exited rc={rc} after {time.monotonic() - t0:.0f}s ({span})")
        if timed_out:
            failures.append(f"{coll} {algo} {proto} {span}: timed out after {timeout:.0f}s")
        elif rc != 0:
            failures.append(f"{coll} {algo} {proto} {span}: exit code {rc}")
        for count, time_us, wrong in parse_rows(out, inplace=inplace):
            nbytes = count * TYPE_BYTES
            if nbytes not in wanted:
                continue
            if wrong:
                failures.append(f"{coll} {algo} {proto} {format_size(nbytes)}: {wrong} wrong values")
            samples.setdefault(nbytes, []).append(time_us * 1e3)
    for nbytes in sizes:
        if nbytes not in samples:
            failures.append(f"{coll} {algo} {proto} {format_size(nbytes)}: no result")
    return samples, failures


# ── scaling driver (mirror of fastest_iface.run_scaling) ────────────────────


def run_scaling(
    colls: Optional[Set[str]] = None,
    algos: Optional[Set[str]] = None,
    protos: Optional[Set[str]] = None,
    do_plot: bool = False,
    check: bool = False,
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
    repeats: int = 1,
    nranks: int = 4,
    sizes: Optional[Sequence[str]] = None,
    buffmans: Optional[Set[str]] = None,
    avg_mode: str = "max",
    inplace: bool = False,
    per_size: bool = False,
    timeout: float = 900.0,
    extra: Optional[Sequence[str]] = None,
    bin_dir: Optional[str] = None,
    plot_dir: str = "plots",
    env_overrides: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> List[str]:
    """Run the matrix and write `<plot_dir>/<coll>/<coll>_<proto>.csv`. Returns csv paths."""
    warmup = config.DEFAULT_WARMUP if warmup is None else warmup
    iters = config.DEFAULT_ITERS if iters is None else iters
    size_list = parse_sizes(sizes)
    lo, hi = format_size(size_list[0]), format_size(size_list[-1])
    bins = Path(bin_dir) if bin_dir else resolve_bin_dir()
    if avg_mode not in AVG_MODES:
        raise ValueError(f"avg_mode must be one of {sorted(AVG_MODES)}")

    section(f"nccl-tests scaling — {nranks} ranks")
    written: List[str] = []
    all_failures: List[str] = []
    active_buffmans = sorted(buffmans) if buffmans else [None]
    checked_links = False

    for coll in config.active(config.DEFAULT_COLLS, colls):
        if coll not in NT_BINARIES:
            print(color(f"   [skip] {coll}: no nccl-tests binary", "dim"))
            continue
        coll_algos = config.active(config.DEFAULT_ALGOS.get(coll, {}), algos)
        if not coll_algos:
            print(color(f"   [skip] {coll}: no active algos", "dim"))
            continue
        file_dir = f"{plot_dir}/{coll}"
        os.makedirs(file_dir, exist_ok=True)
        baseline_algo = coll_algos[0]

        for proto in config.active(config.DEFAULT_PROTOS, protos):
            frames = []
            for algo in coll_algos:
                for buffman in (active_buffmans if algo == "BINE" else [None]):
                    label = f"{algo}-{buffman}" if buffman else algo
                    print(color(f"\n  {coll}  proto={proto}  algo={label}  (nccl-tests)", "blue"))
                    if not checked_links:
                        env, _ = _job_env(algo, proto, buffman, env_overrides)
                        _report_fork(bins / NT_BINARIES[coll], env)
                        checked_links = True
                    samples, failures = run_one(
                        coll, algo, proto, buffman, size_list, nranks, bins, warmup, iters,
                        repeats, check, avg_mode, inplace, per_size, timeout, extra or [],
                        env_overrides,
                    )
                    all_failures += failures
                    frames.append(_to_frame(coll, algo, label, lo, hi, size_list, samples))

            df = pd.concat(frames, ignore_index=True)
            csv_path = f"{file_dir}/{coll}_{proto}.csv"
            if os.path.exists(csv_path):
                warn(f"overwriting {csv_path}")
            df.to_csv(csv_path, index=False)
            written.append(csv_path)
            print(color(f"   saved → {csv_path}", "dim"), flush=True)

            if do_plot and not df.empty:
                from .plotting import plot_diff, plot_size

                plot_df = records_to_df(load_fastest_csv(Path(csv_path), coll, proto, ranks=None))
                plot_size(plot_df, coll, proto, Path(f"{file_dir}/{coll}_{proto}.png"))
                plot_diff(plot_df, coll, proto, Path(f"{file_dir}/{coll}_{proto}_diff.png"),
                          baseline_algo=baseline_algo)

    if all_failures:
        error(f"{len(all_failures)} problem(s):")
        for f in all_failures:
            error(f"  {f}")
        if strict:
            raise RuntimeError(f"{len(all_failures)} nccl-tests failure(s), see above")
    else:
        ok("all nccl-tests runs completed")
    info("run_scaling returning")
    return written


def _to_frame(
    coll: str, algo: str, label: str, lo: str, hi: str,
    sizes: Sequence[int], samples: Dict[int, List[float]],
) -> pd.DataFrame:
    """Rows in fastest's CSV shape. Sizes with no data are omitted (not zero-filled,
    which would wreck the log-scale plots)."""
    rows = []
    for nbytes in sizes:
        if nbytes not in samples:
            continue
        rows.append({
            "pool": f"scaling/{lo}_{hi}/{algo}_{coll}",
            "test": f"scaling/{lo}_{hi}/{label}_{coll}/{format_size(nbytes)}",
            **summarize(samples[nbytes]),
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)
