"""Submitit-based Slurm helpers.

Each rank count in a sweep gets its own, independently-sized `sbatch`
submission -- there is never one oversized allocation held for the whole
sweep. This directly replaces hand-written per-rank-count sbatch scripts
(e.g. `meluxina.sbatch`) with `unitccl scaling ... ranks=4,8,16,...`.
"""
from __future__ import annotations

from typing import List, Optional

from . import config
from .logging_utils import info, ok

import os
import subprocess

try:
    import submitit
except ImportError:  # pragma: no cover - optional dependency
    submitit = None


def _require_submitit() -> None:
    if submitit is None:
        raise RuntimeError(
            "submitit is not installed. Install the 'slurm' extra: pip install unitccl[slurm]"
        )


def _run_build_job(target: str, clean: bool, root):
    """Runs *inside* the submitted Slurm job."""
    from . import build_utils

    build_utils.run_build(target, clean, root=root)


def apply_preload_modules(modules: List[str]) -> None:
    """Load modules in a login shell and merge the resulting env into this
    process. Lets local (non-Slurm) runs pick up the same modules a
    submitted job would get via slurm_setup."""
    if not modules:
        return
    load_cmd = " && ".join(f"module load {m}" for m in modules)
    proc = subprocess.run(
        ["bash", "-l", "-c", f"{load_cmd} && env -0"],
        capture_output=True, check=True, text=True,
    )
    for entry in proc.stdout.split("\0"):
        if "=" in entry:
            k, _, v = entry.partition("=")
            os.environ[k] = v


def _executor(job_name, nodes, tasks_per_node, gpus, timeout_min, log_dir):
    _require_submitit()
    executor = submitit.AutoExecutor(folder=f"{log_dir}/%j")
    params = dict(
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        timeout_min=timeout_min,
        name=job_name,
    )
    if gpus > 0:
        params["slurm_gres"] = f"gpu:{gpus}"

    slurm_partition = config.get("slurm_partition")
    if slurm_partition:
        params["slurm_partition"] = slurm_partition

    slurm_account = config.get("slurm_account")
    if slurm_account:
        params["slurm_account"] = slurm_account

    slurm_qos = config.get("slurm_qos")
    if slurm_qos:
        params["slurm_qos"] = slurm_qos

    modules = config.get("preload_modules") or []
    setup = ["source /etc/profile"]
    setup += [f"module load {m}" for m in modules]

    # ensure the fork's NCCL is found before any module-provided system NCCL
    nccl_lib = config.get("nccl_lib")
    if nccl_lib:
        setup.append(f"export LD_LIBRARY_PATH={nccl_lib}:$LD_LIBRARY_PATH")

    params["slurm_setup"] = setup
    executor.update_parameters(**params)
    return executor


def _run_scaling_for_ranks(ranks: int, scaling_kwargs: dict):
    """Runs *inside* the submitted Slurm job: one rank count, writes csvs to
    `<ranks>_ranks/<coll>/<coll>_<proto>.csv` (the layout `plotting.py`'s
    loaders expect)."""
    from . import fastest_iface  # imported here: only needed inside the job

    kwargs = dict(scaling_kwargs)
    kwargs["plot_dir"] = f"plots/{ranks:03d}_ranks"
    kwargs["do_csv"] = True
    overrides = dict(kwargs.get("env_overrides") or {})
    overrides[config.NRANKS_ENV] = str(ranks)
    kwargs["env_overrides"] = overrides
    return fastest_iface.run_scaling(**kwargs)


def submit_rank_sweep(
    ranks_list: List[int],
    scaling_kwargs: dict,
    gpus_per_node: Optional[int] = None,
    timeout_min: int = 30,
    log_dir: str = "logs/sweep",
) -> List:
    """Fire off one independently-sized submitit job per rank count."""
    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    jobs = []
    for ranks in ranks_list:
        nodes = max(1, -(-ranks // gpus_per_node))  # ceil division
        tasks_per_node = min(ranks, gpus_per_node)
        gpus = min(ranks, gpus_per_node)
        executor = _executor(f"unitccl-scaling-{ranks}", nodes, tasks_per_node, gpus, timeout_min, log_dir)
        info(f"submitting ranks={ranks} nodes={nodes} gpus/node={gpus}")
        job = executor.submit(_run_scaling_for_ranks, ranks, scaling_kwargs)
        jobs.append(job)
    ok(f"submitted {len(jobs)} jobs (one right-sized alloc per rank count)")
    return jobs


def submit_build(
    target: str,
    clean: bool = False,
    timeout_min: int = 60,
    log_dir: str = "logs/build",
    gpus: int = 1,
) -> List:
    from pathlib import Path
    """Submit a build (nccl/fastest/unitccl/all) on 1 GPU node.

    Captures the current working directory as `root` at submit time
    (same idea as `$SLURM_SUBMIT_DIR` in the hand-written sbatch script)
    so the job builds the same checkout you're calling `unitccl` from.
    """
    root = Path.cwd()
    executor = _executor(f"unitccl-build-{target}", nodes=1, tasks_per_node=1, gpus=gpus, timeout_min=timeout_min, log_dir=log_dir)
    info(f"submitting build job (target={target} clean={clean})")
    job = executor.submit(_run_build_job, target, clean, root)
    ok("submitted 1 job")
    return [job]


def _run_standalone_job():
    """Runs *inside* the submitted Slurm job."""
    from . import fastest_iface  # imported here: only needed inside the job

    fastest_iface.run_standalone()


def submit_standalone(timeout_min: int = 30, log_dir: str = "logs/standalone") -> List:
    """Submit `run_standalone` on a single CPU-only node/allocation."""
    executor = _executor("unitccl-standalone", nodes=1, tasks_per_node=1, gpus=0, timeout_min=timeout_min, log_dir=log_dir)
    info("submitting standalone job (1 node, no gpu)")
    job = executor.submit(_run_standalone_job)
    ok("submitted 1 job")
    return [job]


def wait_for(jobs: List, poll_interval: float = 2.0, tui: bool = True) -> None:
    """Block until all jobs finish, raising if any failed.

    By default renders a live dashboard (job table + scrolling log tail)
    via `tui_utils.watch_jobs`. Pass `tui=False` for the old plain
    `[job_id:stream] line`-per-line printing (e.g. when piping to a file
    or running in a non-interactive CI shell).

    Either way, a job that ends up CANCELLED (via the TUI's cancel keys,
    an external `scancel`, ctrl-C, etc.) is treated as a clean stop rather
    than a hard failure -- only a genuine FAILED/TIMEOUT/exception raises.
    """
    if tui:
        from .tui_utils import watch_jobs

        watch_jobs(jobs, poll_interval=poll_interval)
        return

    from .tui_utils import raise_on_failure
    from pathlib import Path
    import time

    offsets = {job.job_id: {"stdout": 0, "stderr": 0} for job in jobs}
    last_state = {job.job_id: None for job in jobs}

    def _drain(job, stream: str) -> None:
        path = Path(job.paths.stdout if stream == "stdout" else job.paths.stderr)
        if not path.exists():
            return
        with open(path, "r") as f:
            f.seek(offsets[job.job_id][stream])
            chunk = f.read()
            offsets[job.job_id][stream] = f.tell()
        if chunk:
            prefix = f"[{job.job_id}:{stream}] "
            for line in chunk.splitlines():
                print(prefix + line)

    while not all(job.done() for job in jobs):
        for job in jobs:
            state = job.state
            if state != last_state[job.job_id]:
                info(f"job {job.job_id} → {state}")
                last_state[job.job_id] = state
            _drain(job, "stdout")
            _drain(job, "stderr")
        time.sleep(poll_interval)

    for job in jobs:
        state = job.state
        if state != last_state[job.job_id]:
            info(f"job {job.job_id} → {state}")
        _drain(job, "stdout")
        _drain(job, "stderr")

    raise_on_failure(jobs)
    ok("all sweep jobs finished")
