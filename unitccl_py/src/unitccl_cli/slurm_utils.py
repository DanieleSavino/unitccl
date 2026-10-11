"""Submitit-based Slurm helpers.

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
import sys

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


def _after(job) -> Optional[Dict[str, str]]:
    """sbatch dependency that starts after `job` ends (any state), or None."""
    return {"dependency": f"afterany:{job.job_id}"} if job is not None else None


def _executor(job_name, nodes, tasks_per_node, gpus, timeout_min, log_dir, extra_slurm=None):
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
    if extra_slurm:
        params["slurm_additional_parameters"] = dict(extra_slurm)

    slurm_partition = config.get("slurm_partition")
    if slurm_partition:
        params["slurm_partition"] = slurm_partition

    slurm_account = config.get("slurm_account")
    if slurm_account:
        params["slurm_account"] = slurm_account

    slurm_qos = config.get("slurm_qos")
    if slurm_qos:
        params["slurm_qos"] = slurm_qos

    # CPUs per task (= per GPU rank). Without this Slurm uses its default
    # (usually 1), which can leave a rank and NCCL's proxy/helper threads
    # competing for one core. Set with `unitccl set cpus_per_task <n>`.
    cpus_per_task = config.get("cpus_per_task")
    if cpus_per_task:
        params["cpus_per_task"] = int(cpus_per_task)

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
    """Fire off one independently-sized submitit job per rank count.

    If a single-GPU algo (NCCL PAT) is selected, each rank count gets a second
    job for it with 1 GPU per node (nodes == ranks). It is chained after the
    standard job (`afterany`) so the two never write the CSV at the same time;
    it merges its rows into the standard job's CSV.
    """
    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    groups = config.layout_groups(scaling_kwargs.get("colls"), scaling_kwargs.get("algos"))
    split = len(groups) > 1
    jobs = []
    for ranks in ranks_list:
        prev = None
        for group_algos, single in groups:
            gpn = 1 if single else gpus_per_node
            nodes = max(1, -(-ranks // gpn))  # ceil division
            tasks_per_node = min(ranks, gpn)
            gpus = min(ranks, gpn)
            kwargs = dict(scaling_kwargs, algos=group_algos)
            if split:
                kwargs["baseline_algos"] = scaling_kwargs.get("algos")
                kwargs["merge_csv"] = single  # std job writes first, PAT merges after
            suffix = "-pat" if single else ""
            executor = _executor(
                f"unitccl-scaling-{ranks}{suffix}", nodes, tasks_per_node, gpus, timeout_min, log_dir,
                extra_slurm=_after(prev) if single else None,
            )
            info(f"submitting ranks={ranks} nodes={nodes} gpus/node={gpus}"
                 + (f" (PAT, 1 GPU/node, after job {prev.job_id})" if single and prev else " (PAT, 1 GPU/node)" if single else ""))
            job = executor.submit(_run_scaling_for_ranks, ranks, kwargs)
            jobs.append(job)
            prev = job
    ok(f"submitted {len(jobs)} jobs (one right-sized alloc per rank count"
       + (" and layout)" if split else ")"))
    return jobs


def _run_nccltests_for_ranks(ranks: int, outdir: str, nt_kwargs: dict):
    """Runs *inside* the submitted Slurm job: one rank count, writes csvs to
    `<outdir>/<ranks:03d>_ranks/<coll>/<coll>_<proto>.csv`, the same layout as
    `_run_scaling_for_ranks`. Only task 0 works; it launches mpirun itself (same
    pattern as the nsys jobs)."""
    if int(os.environ.get("SLURM_PROCID", 0)) != 0:
        return []
    from . import nccltests_utils

    kwargs = dict(nt_kwargs)
    kwargs["nranks"] = ranks
    kwargs["plot_dir"] = f"{outdir}/{ranks:03d}_ranks"
    overrides = dict(kwargs.get("env_overrides") or {})
    overrides[config.NRANKS_ENV] = str(ranks)
    kwargs["env_overrides"] = overrides
    written = nccltests_utils.run_scaling(**kwargs)
    info(f"ranks={ranks}: job function done, {len(written)} csv(s) written, exiting")
    sys.stdout.flush()
    return written


def submit_nccltests_sweep(
    ranks_list: List[int],
    nt_kwargs: dict,
    outdir: str = "plots",
    gpus_per_node: Optional[int] = None,
    timeout_min: int = 60,
    log_dir: str = "logs/nccltests",
) -> List:
    """One independently-sized submitit job per rank count, running nccl-tests.
    PAT gets its own 1-GPU-per-node job per rank count (see `submit_rank_sweep`)."""
    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    groups = config.layout_groups(nt_kwargs.get("colls"), nt_kwargs.get("algos"))
    split = len(groups) > 1
    jobs = []
    for ranks in ranks_list:
        prev = None
        for group_algos, single in groups:
            gpn = 1 if single else gpus_per_node
            nodes = max(1, -(-ranks // gpn))
            per_node = min(ranks, gpn)
            kwargs = dict(nt_kwargs, algos=group_algos)
            if split:
                kwargs["baseline_algos"] = nt_kwargs.get("algos")
                kwargs["merge_csv"] = single
            suffix = "-pat" if single else ""
            executor = _executor(
                f"unitccl-nccltests-{ranks}{suffix}", nodes, per_node, per_node, timeout_min, log_dir,
                extra_slurm=_after(prev) if single else None,
            )
            info(f"submitting nccl-tests ranks={ranks} nodes={nodes} gpus/node={per_node}"
                 + (" (PAT)" if single else ""))
            job = executor.submit(_run_nccltests_for_ranks, ranks, outdir, kwargs)
            jobs.append(job)
            prev = job
    ok(f"submitted {len(jobs)} nccl-tests jobs (one right-sized alloc per rank count"
       + (" and layout)" if split else ")"))
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


def _run_nsys_job(outdir_str, colls, algos, sizes, nranks, proto, warmup, iters, check, buffmans=None, env_overrides=None, binevars=None, gpus_per_node=None):
    from pathlib import Path
    from . import nsys_utils

    import os
    rank = int(os.environ.get("SLURM_PROCID", 0))
    if rank != 0:
        return

    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    job_env_overrides = dict(os.environ)
    if env_overrides:
        job_env_overrides.update(env_overrides)
    job_env_overrides[config.NRANKS_ENV] = str(nranks)
    job_env_overrides["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus_per_node))

    base_outdir = Path(outdir_str)
    
    # We pass buffmans down to nsys_utils.run_profiles so it plots all variations together
    for size in sizes:
        outdir = base_outdir / _human_size(size)
        nsys_utils.run_profiles(
            outdir, colls, algos,
            size=size, nranks=nranks, proto=proto,
            warmup=warmup, iters=iters, check=check, env_overrides=job_env_overrides, buffmans=buffmans,
            binevars=binevars,
        )
        nsys_utils.generate_stats(outdir)
        nsys_utils.analyze(outdir)
        # rank -> node map next to per_rank_metrics.csv and the plots
        from . import nodes_utils
        for single in (False, True):
            if any((a in config.SINGLE_GPU_ALGOS) == single for a in algos):
                nodes_utils.export_nodes([outdir / "plots"], nranks, single_gpu=single)


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
    binevars: Optional[List[str]] = None,
    after=None,
    name_suffix: str = "",
) -> List:
    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    nodes = max(1, -(-nranks // gpus_per_node))
    executor = _executor(
        f"unitccl-nsys-{nranks}{name_suffix}", nodes, tasks_per_node=gpus_per_node, gpus=gpus_per_node,
        timeout_min=timeout_min, log_dir=log_dir, extra_slurm=_after(after),
    )
    info(f"submitting nsys job nranks={nranks} nodes={nodes} gpus/node={gpus_per_node} sizes={sizes} (1 task/node)")
    job = executor.submit(
        _run_nsys_job, str(outdir), colls, algos, sizes, nranks, proto, warmup, iters, check, buffmans,
        env_overrides, binevars, gpus_per_node,
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
    binevars: Optional[List[str]] = None,
) -> List:
    from pathlib import Path

    gpus_per_node = gpus_per_node or config.get("gpus_per_node", 4)
    # PAT (NCCL algo) needs 1 GPU/node -> its own job per rank count, chained
    # after the standard one since both write into the same outdir.
    std_algos = [a for a in algos if a not in config.SINGLE_GPU_ALGOS]
    single_algos = [a for a in algos if a in config.SINGLE_GPU_ALGOS]
    groups = ([(std_algos, gpus_per_node, "")] if std_algos else []) + \
             ([(single_algos, 1, "-pat")] if single_algos else [])

    jobs = []
    for nranks in ranks_list:
        sub_outdir = Path(outdir) / f"{nranks:03d}_ranks"
        prev = None
        for group_algos, gpn, suffix in groups:
            new = submit_nsys(
                sub_outdir, colls, group_algos, sizes=sizes, nranks=nranks, proto=proto,
                warmup=warmup, iters=iters, check=check, buffmans=buffmans, env_overrides=env_overrides,
                gpus_per_node=gpn, timeout_min=timeout_min, log_dir=log_dir,
                binevars=binevars, after=prev if suffix else None, name_suffix=suffix,
            )
            jobs.extend(new)
            prev = new[-1]
    ok(f"submitted {len(jobs)} nsys jobs ({len(ranks_list)} rank counts, {len(sizes)} sizes each, one alloc per rank count"
       + (" and layout)" if len(groups) > 1 else ")"))
    return jobs


def _run_standalone_job(
    buffmans: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    binevars: Optional[List[str]] = None,
):
    from . import fastest_iface
    fastest_iface.run_standalone(env_overrides=env_overrides, buffmans=buffmans, binevars=binevars)


def submit_standalone(
    buffmans: Optional[List[str]] = None,
    timeout_min: int = 30,
    log_dir: str = "logs/standalone",
    env_overrides: Optional[Dict[str, str]] = None,
    binevars: Optional[List[str]] = None,
) -> List:
    executor = _executor("unitccl-standalone", nodes=1, tasks_per_node=1, gpus=0, timeout_min=timeout_min, log_dir=log_dir)
    info("submitting standalone job (1 node, no gpu)")
    job = executor.submit(_run_standalone_job, buffmans, env_overrides, binevars)
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
