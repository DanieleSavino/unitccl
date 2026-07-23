"""Central configuration: collective/algo/proto registries, env-var names
read by the C benchmark binary, and a small persisted config file for Slurm
defaults (account/partition/qos) and the backend module name.

The "backend module" is the compiled `fastest` pybind11 extension for a given
project (built by that project's `build_wheel.sh`). It's configurable so this
same CLI can drive other Bine-collective projects later, not just the current
`unitccl` C++/CUDA project -- e.g. `unitccl set backend_module collalgo`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# ── env vars read by the C benchmark binary (unitccl_bench) ────────────────
CHECK_ENV = "UNITCCL_CHECK"
ITERS_ENV = "UNITCCL_ITERS"
WARMUP_ENV = "UNITCCL_WARMUP"
NRANKS_ENV = "UNITCCL_NRANKS"  # informational tag set during rank sweeps

# ── default registries (mirrors the current tests.py) ───────────────────────
DEFAULT_COLLS: Dict[str, bool] = {
    "AllGather": True,
    "AllGatherV": True,
    "AllReduce": True,
    "Bcast": True,
    "Reduce": True,
    "ReduceScatter": True,
}

DEFAULT_ALGOS: Dict[str, Dict[str, bool]] = {
    "AllGather": {
        "RING": True,
        "COLLNET_DIRECT": False,
        "NVLS": False,
        "PAT": False,
        "BINE": True,
    },
    "AllGatherV": {
        "RING": False,
        "BINE": False,
    },
    "AllReduce": {
        "TREE": True,
        "RING": True,
        "COLLNET_DIRECT": False,
        "COLLNET_CHAIN": False,
        "NVLS": False,
        "NVLS_TREE": False,
        "BINE": True,
    },
    "Bcast": {"RING": True, "BINE": True},
    "Reduce": {"RING": True, "BINE": True},
    "ReduceScatter": {
        "RING": True,
        "COLLNET_DIRECT": False,
        "NVLS": False,
        "PAT": False,
        "BINE": True,
    },
}

DEFAULT_PROTOS: Dict[str, bool] = {"SIMPLE": True, "LL": True, "LL128": True}

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 40

# ── persisted config (~/.config/unitccl/config.json) ────────────────────────
CONFIG_DIR = Path(os.environ.get("UNITCCL_CONFIG_DIR", Path.home() / ".config" / "unitccl"))
CONFIG_FILE = CONFIG_DIR / "config.json"

_DEFAULTS = {
    "backend_module": os.environ.get("UNITCCL_BACKEND_MODULE", "unitccl"),
    "slurm_account": None,
    "slurm_partition": None,
    "slurm_qos": None,
    "gpus_per_node": 4,
    "preload_modules": [],
}


def load() -> dict:
    cfg = dict(_DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def set_value(key: str, value) -> dict:
    cfg = load()
    cfg[key] = value
    save(cfg)
    return cfg


def get(key: str, default=None):
    return load().get(key, default)


def active(d: Dict[str, bool], filter_set: Optional[Set[str]] = None) -> List[str]:
    """Names whose flag is True, optionally restricted to `filter_set`."""
    return [k for k, v in d.items() if v and (filter_set is None or k in filter_set)]

def add_preload_module(module: str) -> dict:
    """Append a module to the preload list, preserving insertion order, no dupes."""
    cfg = load()
    modules = list(cfg.get("preload_modules") or [])
    if module not in modules:
        modules.append(module)
    cfg["preload_modules"] = modules
    save(cfg)
    return cfg


def remove_preload_module(module: str) -> dict:
    cfg = load()
    modules = [m for m in (cfg.get("preload_modules") or []) if m != module]
    cfg["preload_modules"] = modules
    save(cfg)
    return cfg
