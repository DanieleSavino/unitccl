from __future__ import annotations
__all__: list[str] = ['color']
def color(text: str, colour: str) -> str:
    """
    Wrap text in ANSI colour codes (no-op if colour unknown).
    """
_ANSI: dict = {'reset': '\x1b[0m', 'green': '\x1b[32m', 'red': '\x1b[31m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m', 'cyan': '\x1b[36m', 'dim': '\x1b[2m', 'bold': '\x1b[1m'}
