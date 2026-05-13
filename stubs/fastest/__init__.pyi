from __future__ import annotations
from fastest.runner import CompareResult
from fastest.runner import Pool
from fastest.runner import Runner
from fastest.runner import Stats
from . import flags
from . import logging
from . import runner
__all__: list = ['Runner', 'Pool', 'Stats', 'CompareResult', 'default_runner', 'PlotMode', 'Plotter']
def __dir__() -> list[str]:
    """
    Enhance tab-completion with runner's public names.
    """
def __getattr__(name: str):
    """
    
    Delegate unknown module attributes to `default_runner`.
    Special-cases `tests` to return the backend's test constants.
    """
PlotMode = None
default_runner: runner.Runner  # value = <fastest.runner.Runner object>
plotter = None
