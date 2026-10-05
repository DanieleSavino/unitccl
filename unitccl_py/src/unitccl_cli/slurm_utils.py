"""Submitit-based Slurm helpers Submitit-based Slurm helpers.
Each rank count in a sweep gets its own, independently-sized `sbatch`
submission -- there is never one oversized allocation held for the whole
sweep. This directly replaces hand-written per-rank-count sbatch scripts
(e.g. `meluxina.sbatch`) with `unitccl scaling ... ranks=4,8,16,...`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

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
    `<ranks>_ranks/<coll>/<coll>_<proto>.csv` (the layout `plotting.py`
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
    timeout_min: int = 90,
    log_dir: str = "logs/build",
    gpus: int = 1,
) -> List:
    from pathlib import Path
    """Submit a build (nccl/fastest/unitccl/all) on 1 GPU node."""
    root = Path.cwd()
    executor = _executor(f"unitccl-build-{target}", nodes=1, tasks_per_node=1, gpus=gpus, timeout_min=timeout_min, log_dir=log_dir)
    info(f"submitting build job (target={target} clean={clean})")
    job = executor.submit(_run_build_job, target, clean, root)
    ok("submitted 1 job")
    return [job]


def _run_nsys_job(outdir_str, colls, algos, sizes, nranks, proto, warmup, iters, check, buffmans=None, env_overrides=None):
    from pathlib import Path
    from . import nsys_utils

    import os
    rank = int(os.environ.get("SLURM_PROCID", 0))
    if rank != 0:
        return

    gpus_per_node = config.get("gpus_per_node", 4)
    job_env_overrides = dict(os.environ)
    if env_overrides:
        job_env_overrides.update(env_overrides)
    job_env_overrides[config.NRANKS_ENV] = str(nranks)
    job_env_overrides["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus_per_node))

    base_outdir = Path(outdir_str)
    active_buffmans = buffmans if buffmans else [""]

    for size in sizes:
        for buffman in active_buffmans:
            job_env_copy = dict(job_env_overrides)
            if buffman:
                job_env_copy[config.BINE_BUFFER_MANAGEMENT_ENV] = buffman
                outdir = base_outdir / _human_size(size) / buffman
            else:
                outdir = base_outdir / _human_size(size)
                if config.BINE_BUFFER_MANAGEMENT_ENV in job_env_copy:
                    del job_env_copy[config.BINE_BUFFER_MANAGEMENT_ENV]
                    
            nsys_utils.run_profiles(
                outdir, colls, algos,
                size=size, nranks=nranks, proto=proto,
                warmup=warmup, iters=iters, check=check, env_overrides=job_env_copy,
            )
            nsys_utils.generate_stats(outdir)
            nsys_utils.analyze(outdir)


def submit_nsys(
    outdir,
    colls: List[str],
    algos: List[str],
    sizes: List[int],
    nranks: int,
    proto: str = "SIMPLE",
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
    check: bool = False,
    buffmans: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    gpus_per_node: Optional[int] = None,
    timeout_min: int = 60,
    log_dir: str = "logs/nsys",
) -> List:
    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    nodes = max(1, -(-nranks // gpus_per_node))
    executor = _executor(
        f"unitccl-nsys-{nranks}", nodes, tasks_per_node=gpus_per_node, gpus=gpus_per_node,
        timeout_min=timeout_min, log_dir=log_dir,
    )
    info(f"submitting nsys job nranks={nranks} nodes={nodes} gpus/node={gpus_per_node} sizes={sizes} (1 task/node)")
    job = executor.submit(
        _run_nsys_job, str(outdir), colls, algos, sizes, nranks, proto, warmup, iters, check, buffmans, env_overrides
    )
    ok("submitted 1 job")
    return [job]


def _human_size(n: int) -> str:
    n *= 4
    for unit, factor in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if n % factor == 0:
            return f"{n // factor}{unit}"
    return f"{n}B"


def submit_nsys_sweep(
    outdir,
    colls: List[str],
    algos: List[str],
    sizes: List[int],
    ranks_list: List[int],
    proto: str = "SIMPLE",
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
    check: bool = False,
    buffmans: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    gpus_per_node: Optional[int] = None,
    timeout_min: int = 60,
    log_dir: str = "logs/nsys",
) -> List:
    from pathlib import Path

    jobs = []
    for nranks in ranks_list:
        sub_outdir = Path(outdir) / f"{nranks:03d}_ranks"
        jobs.extend(
            submit_nsys(
                sub_outdir, colls, algos, sizes=sizes, nranks=nranks, proto=proto,
                warmup=warmup, iters=iters, check=check, buffmans=buffmans, env_overrides=env_overrides,
                gpus_per_node=gpus_per_node, timeout_min=timeout_min, log_dir=log_dir,
            )
        )
    ok(f"submitted {len(jobs)} nsys jobs ({len(ranks_list)} rank counts, {len(sizes)} sizes each, one alloc per rank count)")
    return jobs


def _run_standalone_job(buffmans: Optional[List[str]] = None, env_overrides: Optional[Dict[str, str]] = None):
    from . import fastest_iface
    fastest_iface.run_standalone(env_overrides=env_overrides, buffmans=buffmans)


def submit_standalone(
    buffmans: Optional[List[str]] = None, 
    timeout_min: int = 30, 
    log_dir: str = "logs/standalone", 
    env_overrides: Optional[Dict[str, str]] = None
) -> List:
    executor = _executor("unitccl-standalone", nodes=1, tasks_per_node=1, gpus=0, timeout_min=timeout_min, log_dir=log_dir)
    info("submitting standalone job (1 node, no gpu)")
    job = executor.submit(_run_standalone_job, buffmans, env_overrides)
    ok("submitted 1 job")
    return [job]


def wait_for(jobs: List, poll_interval: float = 2.0, tui: bool = True) -> None:
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
