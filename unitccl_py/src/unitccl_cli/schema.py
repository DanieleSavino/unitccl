"""Canonical benchmark-record schema + size parsing, shared by every scaling
data source: fastest's own `CompareResult.save_csv()` output, and the
per-rank-sweep CSVs that `unitccl scaling ... ranks=...` produces (which are
just the former, written into a `<N>_ranks/<coll>/` directory tree).

nsys stats CSVs have a different, wide multi-metric shape and are handled
separately in `nsys_utils.py` rather than being forced into this schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

SIZE_RE = re.compile(r"^([\d.]+)\s*([KMGT]?B)$", re.IGNORECASE)
SIZE_MULT = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def size_to_bytes(s: str) -> float:
    """Convert a human size like '1kB', '256kB', '4MB' to bytes."""
    m = SIZE_RE.match(s.strip())
    if not m:
        raise ValueError(f"Could not parse size '{s}'")
    val, unit = m.groups()
    return float(val) * SIZE_MULT[unit.upper()]


@dataclass
class BenchRecord:
    collective: str
    algo: str
    proto: str
    size_str: str
    mean_ns: float
    ranks: Optional[int] = None
    stddev_ns: Optional[float] = None
    min_ns: Optional[float] = None
    max_ns: Optional[float] = None
    median_ns: Optional[float] = None

    @property
    def size_bytes(self) -> float:
        return size_to_bytes(self.size_str)


def _parse_test_column(test: str):
    """'scaling/1kB_64MB/RING_Bcast/1kB' -> (algo, size_str)."""
    parts = test.split("/")
    size_str = parts[-1]
    algo = parts[-2].split("_")[0]
    return algo, size_str


def load_fastest_csv(
    path: Path, collective: str, proto: str, ranks: Optional[int] = None
) -> List[BenchRecord]:
    """Parse a CSV written by `fastest.CompareResult.save_csv()`."""
    df = pd.read_csv(path)
    out: List[BenchRecord] = []
    for _, row in df.iterrows():
        algo, size_str = _parse_test_column(row["test"])
        out.append(
            BenchRecord(
                collective=collective,
                algo=algo,
                proto=proto,
                size_str=size_str,
                mean_ns=row.get("mean_ns"),
                ranks=ranks,
                stddev_ns=row.get("stddev_ns"),
                min_ns=row.get("min_ns"),
                max_ns=row.get("max_ns"),
                median_ns=row.get("median_ns"),
            )
        )
    return out


def find_rank_dirs(root: Path):
    rank_dirs = []
    for p in sorted(root.glob("*_ranks")):
        if p.is_dir():
            m = re.match(r"(\d+)_ranks", p.name)
            if m:
                rank_dirs.append((int(m.group(1)), p))
    rank_dirs.sort(key=lambda x: x[0])
    return rank_dirs


def find_protos(root: Path, collective: str) -> List[str]:
    protos = set()
    for _, rdir in find_rank_dirs(root):
        cdir = rdir / collective
        if not cdir.is_dir():
            continue
        for csv in cdir.glob(f"{collective}_*.csv"):
            protos.add(csv.stem[len(collective) + 1 :])
    return sorted(protos)


def load_rank_sweep(root: Path, collective: str, proto: str) -> List[BenchRecord]:
    """Load the `<N>_ranks/<coll>/<coll>_<proto>.csv` layout produced by
    `unitccl scaling ... ranks=...` (see slurm_utils.submit_rank_sweep)."""
    records: List[BenchRecord] = []
    for ranks, rdir in find_rank_dirs(root):
        csv_path = rdir / collective / f"{collective}_{proto}.csv"
        if not csv_path.exists():
            continue
        records.extend(load_fastest_csv(csv_path, collective, proto, ranks=ranks))
    if not records:
        raise SystemExit(
            f"No data found for collective='{collective}', proto='{proto}' under {root}"
        )
    return records


def records_to_df(records: List[BenchRecord]) -> pd.DataFrame:
    rows = [
        {
            "ranks": r.ranks,
            "algo": r.algo,
            "collective": r.collective,
            "proto": r.proto,
            "size_str": r.size_str,
            "size_bytes": r.size_bytes,
            "mean_ns": r.mean_ns,
        }
        for r in records
    ]
    return pd.DataFrame(rows)
