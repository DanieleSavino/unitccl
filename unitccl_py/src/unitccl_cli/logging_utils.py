"""Small ANSI logging helpers shared across the unitccl CLI modules."""
from __future__ import annotations

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def color(text: str, name: str) -> str:
    code = _CODES.get(name)
    if not code:
        return text
    return f"{code}{text}{_CODES['reset']}"


def section(title: str, name: str = "blue") -> None:
    print(color(f"\n── {title} " + "─" * max(0, 60 - len(title)), name))


def info(msg: str) -> None:
    print(color(f"[info] {msg}", "cyan"))


def warn(msg: str) -> None:
    print(color(f"[warn] {msg}", "yellow"))


def error(msg: str) -> None:
    print(color(f"[error] {msg}", "red"))


def ok(msg: str) -> None:
    print(color(f"[ok] {msg}", "green"))
