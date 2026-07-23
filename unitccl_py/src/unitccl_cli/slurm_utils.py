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


def _executor(job_name: str, nodes: int, tasks_per_node: int, gpus: int, timeout_min: int, log_dir: str):
    _require_submitit()
    cfg = config.load()
    executor = submitit.AutoExecutor(folder=f"{log_dir}/%j")
    params = dict(
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        timeout_min=timeout_min,
        name=job_name,
    )
    if gpus > 0:
        params["slurm_gres"] = f"gpu:{gpus}"

    if cfg.get("slurm_partition"):
        params["slurm_partition"] = cfg["slurm_partition"]
    if cfg.get("slurm_account"):
        params["slurm_account"] = cfg["slurm_account"]
    if cfg.get("slurm_qos"):
        params["slurm_qos"] = cfg["slurm_qos"]
    modules = cfg.get("preload_modules") or []
    if modules:
        params["slurm_setup"] = [
            "source /etc/profile",
        ] + [f"module load {m}" for m in modules]
    executor.update_parameters(**params)
    return executor


def _run_scaling_for_ranks(ranks: int, scaling_kwargs: dict):
    """Runs *inside* the submitted Slurm job: one rank count, writes csvs to
    `<ranks>_ranks/<coll>/<coll>_<proto>.csv` (the layout `plotting.py`'s
    loaders expect)."""
    from . import fastest_iface  # imported here: only needed inside the job

    kwargs = dict(scaling_kwargs)
    kwargs["plot_dir"] = f"{ranks:03d}_ranks"
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


def wait_for(jobs: List) -> None:
    """Blocks until every job finishes, prints each job's stdout/stderr,
    then raises if any job failed."""
    failures = []
    for job in jobs:
        try:
            result = job.result()
        except Exception as e:  # job raised inside the Slurm process
            failures.append((job, e))
            result = None

        stdout = job.stdout() or ""
        stderr = job.stderr() or ""
        info(f"--- job {job.job_id} stdout ---")
        if stdout.strip():
            print(stdout)
        if stderr.strip():
            info(f"--- job {job.job_id} stderr ---")
            print(stderr)
        if result is not None:
            info(f"--- job {job.job_id} return value ---")
            print(result)

    if failures:
        job, e = failures[0]
        raise RuntimeError(f"job {job.job_id} failed: {e}") from e

    ok("all sweep jobs finished")
