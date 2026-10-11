"""Export which node each MPI rank ran on (`rank,node` CSV).

The mapping is measured, not predicted: one throwaway `mpirun` with the same
`--map-by` flags as the real benchmark launch prints `$OMPI_COMM_WORLD_RANK`
and the hostname from every rank. It runs once per (nranks, ppn) per process.
Failures only warn; they never fail a benchmark job.
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .logging_utils import info, warn

_LINE_RE = re.compile(r"^(\d+),(\S+)$")
_cache: Dict[Tuple[int, Optional[int]], Optional[List[Tuple[int, str]]]] = {}


def nodes_filename(single_gpu: bool) -> str:
    """`nodes.csv` for the standard layout, `nodes_pat.csv` for the 1-GPU-per-node
    (NCCL PAT) job, so split jobs never overwrite each other. Neither name matches
    the `<Coll>_<proto>.csv` pattern the plot loaders glob for."""
    return "nodes_pat.csv" if single_gpu else "nodes.csv"


def probe_nodes(nranks: int, ppn: Optional[int] = None, timeout: float = 120.0) -> Optional[List[Tuple[int, str]]]:
    """[(rank, node)] sorted by rank, or None on failure. `ppn` adds
    `--map-by ppr:<ppn>:node` exactly like the real launch."""
    key = (nranks, ppn)
    if key in _cache:
        return _cache[key]

    cmd = ["mpirun", "-np", str(nranks)]
    if ppn:
        cmd += ["--map-by", f"ppr:{ppn}:node"]
    cmd += ["bash", "-c", 'echo "$OMPI_COMM_WORLD_RANK,$(hostname -s)"']

    rows: Optional[List[Tuple[int, str]]] = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        found = {}
        for line in proc.stdout.splitlines():
            m = _LINE_RE.match(line.strip())
            if m:
                found[int(m.group(1))] = m.group(2)
        if proc.returncode != 0 or sorted(found) != list(range(nranks)):
            warn(f"node probe `{' '.join(cmd[:6])} ...` returned rc={proc.returncode} with "
                 f"{len(found)}/{nranks} ranks; nodes csv skipped")
        else:
            rows = sorted(found.items())
    except (OSError, subprocess.TimeoutExpired) as e:
        warn(f"node probe failed ({e}); nodes csv skipped")

    _cache[key] = rows
    return rows


def export_nodes(
    out_dirs: Iterable[str], nranks: int, single_gpu: bool = False, ppn: Optional[int] = None
) -> List[str]:
    """Write `rank,node` into each of `out_dirs` (next to the main csv / plots).
    Returns the paths written (empty if the probe failed or nranks < 1)."""
    if nranks < 1:
        return []
    rows = probe_nodes(nranks, 1 if single_gpu else ppn)
    if not rows:
        return []
    written: List[str] = []
    for d in out_dirs:
        path = Path(d) / nodes_filename(single_gpu)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "node"])
            w.writerows(rows)
        written.append(str(path))
    if written:
        info(f"wrote {len({n for _, n in rows})} node(s) for {nranks} ranks -> {written[0]}"
             + (f" (+{len(written) - 1} more)" if len(written) > 1 else ""))
    return written
