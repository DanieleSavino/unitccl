"""Local build helpers for nccl / fastest / unitccl.

Mirrors the manual build steps in the sbatch build script and
scripts/build_wheel.sh, so `unitccl build ...` can run them either
locally (venv already activated) or inside a submitted Slurm job.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .logging_utils import info, ok

NVCC_GENCODE = "-gencode arch=compute_80,code=sm_80"


def _run(cmd: str, cwd: Path) -> None:
    info(f"$ {cmd}  (cwd={cwd})")
    subprocess.run(cmd, shell=True, cwd=str(cwd), check=True)


def build_nccl(root: Path, clean: bool) -> None:
    nccl_dir = root / "nccl"
    if clean:
        _run("make clean", nccl_dir)
    _run(f'make -j src.build NVCC_GENCODE="{NVCC_GENCODE}"', nccl_dir)


def build_fastest(root: Path, clean: bool) -> None:
    fastest_dir = root / "vendor" / "fastest"
    build_dir = fastest_dir / "build"
    if clean and build_dir.exists():
        _run(f"rm -rf {build_dir}", fastest_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    _run("cmake ..", build_dir)
    _run("make -j", build_dir)
    _run("pip install .", fastest_dir / "fastest_py")


def build_unitccl(root: Path, clean: bool) -> None:
    if clean:
        _run("make clean", root)
    _run(f'make -j NVCC_GENCODE="{NVCC_GENCODE}"', root)
    _run("./scripts/build_wheel.sh", root)


_BUILDERS = {
    "nccl": build_nccl,
    "fastest": build_fastest,
    "unitccl": build_unitccl,
}

_ORDER = ("nccl", "fastest", "unitccl")  # dependency order for target=all


def run_build(target: str, clean: bool, root: Optional[Path] = None) -> None:
    root = root or Path.cwd()
    if target == "all":
        for name in _ORDER:
            info(f"── build: {name} ──")
            _BUILDERS[name](root, clean)
    else:
        _BUILDERS[target](root, clean)
    ok(f"build '{target}' complete (clean={clean})")
