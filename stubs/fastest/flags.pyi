"""

fastest.flags
─────────────
Python mirror of fastest/tests.h bitmask constants, plus
strerror-style helpers for exit_status and test_flags.
"""
from __future__ import annotations
from fastest.logging import color
import typing
__all__: list[str] = ['ASSERT_EQ', 'ASSERT_GE', 'ASSERT_GT', 'ASSERT_LE', 'ASSERT_LT', 'ASSERT_NEQ', 'DEFAULT_LOG', 'ERROR_ASSERT', 'ERROR_COLLISION', 'ERROR_CUDA', 'ERROR_EXCEPTION', 'ERROR_INTERNAL', 'ERROR_MEMORY', 'ERROR_MPI', 'ERROR_NOT_FOUND', 'ERROR_OMP', 'ERROR_RESOURCE', 'ERROR_TIMEOUT', 'ERROR_UNEXPECTED', 'ERROR_UNKNOWN', 'FAIL_ERROR', 'FAIL_LOG', 'FAIL_WARNING', 'INCOMPLETE', 'MEM_TRACK', 'SKIPPED', 'SUCCESS', 'TIME_MS', 'TIME_NS', 'TIME_S', 'TIME_US', 'color', 'passed', 'strassert', 'strexit', 'strleak', 'strtest', 'strtime', 'symbol']
def passed(exit_status: int) -> bool:
    ...
def strassert(flags: int) -> str:
    ...
def strexit(exit_status: int, flags: int = 0) -> str:
    ...
def strleak(allocation: int, deallocation: int) -> str | None:
    ...
def strtest(test: dict) -> str:
    ...
def strtime(flags: int) -> str:
    ...
def symbol(exit_status: int) -> str:
    ...
ASSERT_EQ: int = 1
ASSERT_GE: int = 8
ASSERT_GT: int = 4
ASSERT_LE: int = 32
ASSERT_LT: int = 16
ASSERT_NEQ: int = 2
DEFAULT_LOG: int = 2097152
ERROR_ASSERT: int = 256
ERROR_COLLISION: int = 524288
ERROR_CUDA: int = 65536
ERROR_EXCEPTION: int = 1024
ERROR_INTERNAL: int = 131072
ERROR_MEMORY: int = 2048
ERROR_MPI: int = 16384
ERROR_NOT_FOUND: int = 1048576
ERROR_OMP: int = 32768
ERROR_RESOURCE: int = 8192
ERROR_TIMEOUT: int = 4096
ERROR_UNEXPECTED: int = 512
ERROR_UNKNOWN: int = 262144
FAIL_ERROR: int = 64
FAIL_LOG: int = 256
FAIL_WARNING: int = 128
INCOMPLETE: int = 4
MEM_TRACK: int = 8192
SKIPPED: int = 2
SUCCESS: int = 1
TIME_MS: int = 1024
TIME_NS: int = 4096
TIME_S: int = 512
TIME_US: int = 2048
