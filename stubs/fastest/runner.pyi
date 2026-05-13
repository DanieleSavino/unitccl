from __future__ import annotations
from datetime import datetime
from datetime import timezone
from fastest import flags
from fastest.logging import color
import importlib as importlib
import json as json
import math as math
import os as os
import typing
from typing import Any
__all__: list[str] = ['Any', 'CompareResult', 'Pool', 'Runner', 'Stats', 'color', 'datetime', 'flags', 'importlib', 'json', 'math', 'os', 'timezone']
class CompareResult:
    def __init__(self, pools: list[Pool], n_repeats: int, data: dict[str, dict[str, Stats]]):
        ...
    def _build_comparisons(self) -> list[dict[str, typing.Any]]:
        ...
    def report(self) -> None:
        ...
    def save(self, path: str) -> None:
        ...
    def to_dict(self) -> dict[str, typing.Any]:
        ...
class Pool:
    """
    A named collection of test names to run together.
    """
    def __init__(self, name: str, tests: list[str]):
        ...
    def __repr__(self) -> str:
        ...
class Runner:
    def __init__(self, backend = None):
        ...
    def _auto_detect_backend(self) -> None:
        ...
    def compare(self, *pools: Pool, n_repeats: int = 5) -> CompareResult:
        """
        Run every test in every pool n_repeats times and collect timing.
        """
    def get_test(self, name: str) -> dict:
        ...
    def get_tests(self) -> list[dict]:
        ...
    def pool(self, name: str, *test_names: str) -> Pool:
        """
        Create a pool from an explicit list of test names.
        """
    def pool_from_prefix(self, prefix: str) -> Pool:
        """
        Create a pool from all tests whose names start with `prefix/`.
        """
    def run_all(self) -> None:
        ...
    def run_log(self, name: str) -> dict:
        ...
    def run_log_all(self, tests = None) -> None:
        ...
    def run_test(self, name: str) -> None:
        ...
    def set_backend(self, module) -> None:
        ...
    @property
    def backend(self):
        ...
    @property
    def is_ready(self) -> bool:
        ...
class Stats:
    """
    Descriptive statistics over a list of nanosecond samples.
    """
    def __init__(self, samples: list[int], exit_statuses: list[int] | None = None):
        ...
    def to_dict(self) -> dict[str, typing.Any]:
        ...
def _bar(value: float, max_value: float, width: int = 20) -> str:
    ...
def _fmt_ns(ns: float) -> str:
    ...
def _report(result: CompareResult) -> None:
    ...
_NS_THRESHOLDS: list = [(1000000000, 's', 1000000000), (1000000, 'ms', 1000000), (1000, 'µs', 1000), (1, 'ns', 1)]
