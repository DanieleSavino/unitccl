"""Live TUI for watching submitit/Slurm jobs.

Drop-in replacement for `slurm_utils.wait_for()`'s print-based polling loop.
Renders a table of jobs (id, state, ranks, elapsed, node) plus a scrolling
tail of the most recent stdout/stderr lines across all jobs, refreshed in
place. Supports keyboard-driven job cancellation (no mouse — see notes in
the module docstring below).

Usage (from slurm_utils.wait_for):

    from .tui_utils import watch_jobs
    watch_jobs(jobs, poll_interval=2.0)

Falls back gracefully to the old plain-text behavior if `rich` isn't
installed, so it's safe to import unconditionally.

Keyboard controls (only active when stdin is a real TTY):
    ↑ / k       move selection up
    ↓ / j       move selection down
    c / x       cancel the selected job (scancel)
    a           cancel ALL jobs (press again / 'y' within 3s to confirm)
    q           detach: stop watching (jobs are left running, NOT cancelled)

Note: this is a keyboard-only UI. `rich.Live` doesn't give you clickable
widgets/mouse hit-testing — that would require `textual` instead. If you
want an actual clickable "cancel" button, that's a different framework.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _HAVE_RICH = True
except ImportError:  # pragma: no cover - optional dependency
    _HAVE_RICH = False

try:
    import termios
    import tty
    import select as _select

    _HAVE_TTY = True
except ImportError:  # pragma: no cover - e.g. on Windows
    _HAVE_TTY = False

from .logging_utils import info, ok


_STATE_STYLE = {
    "PENDING": "yellow",
    "RUNNING": "cyan",
    "COMPLETED": "green",
    "COMPLETING": "cyan",
    "FAILED": "bold red",
    "TIMEOUT": "bold red",
    "CANCELLED": "red",
    "UNKNOWN": "dim",
}

_LOG_BUFFER_LINES = 2000  # scrollback kept in memory; visible portion is sized to fit the panel at render time


def _state_of(job) -> str:
    try:
        state = job.state or "UNKNOWN"
    except Exception:
        return "UNKNOWN"
    # Slurm/submitit sometimes appends detail after the keyword, e.g.
    # "CANCELLED by 104045" instead of plain "CANCELLED". Normalize to the
    # leading word so state comparisons (and the style/color lookup below)
    # aren't silently thrown off by the suffix.
    return state.split()[0]


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _ranks_of(job) -> str:
    """Best-effort guess at the job's rank/task count.

    NOTE: this doesn't know your actual Job class, so it tries a handful of
    common submitit/slurm attribute names and falls back to '?'. If none of
    these match, tell me the real attribute and I'll wire it up exactly.
    """
    for attr in ("num_tasks", "ntasks", "n_tasks", "world_size"):
        val = getattr(job, attr, None)
        if val:
            return str(val)

    tasks = getattr(job, "tasks", None)
    if tasks is not None:
        try:
            return str(len(tasks))
        except TypeError:
            pass

    params = getattr(job, "parameters", None) or getattr(job, "_job_params", None)
    if isinstance(params, dict):
        for key in ("tasks_per_node", "ntasks_per_node", "ntasks", "nodes"):
            if key in params:
                return str(params[key])

    return "?"


# ── ETA (squeue --start) for still-pending jobs ──────────────────────────────
#
# squeue's scheduler estimate can be stale/expensive to poll every refresh,
# so we throttle actual `squeue` calls per job and reuse the cached value
# in between.

_ETA_REFRESH_SECONDS = 15.0
_eta_cache: Dict[str, Tuple[float, str]] = {}


def _humanize_start(raw: str) -> str:
    """squeue `%S` gives e.g. '2026-07-24T15:30:00', or 'N/A' if unknown."""
    try:
        dt = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return raw
    delta = (dt - datetime.now()).total_seconds()
    if delta <= 0:
        return "starting"
    return f"~{_fmt_elapsed(delta)}"


def _fetch_eta(job_id: str) -> str:
    """Best-effort start-time estimate via `squeue --start`. Returns '' if
    squeue isn't available, the job isn't found, or no estimate exists yet."""
    now = time.time()
    cached = _eta_cache.get(job_id)
    if cached and now - cached[0] < _ETA_REFRESH_SECONDS:
        return cached[1]

    eta = ""
    try:
        proc = subprocess.run(
            ["squeue", "--start", "-j", str(job_id), "--noheader", "-o", "%S"],
            capture_output=True, text=True, timeout=5,
        )
        lines = proc.stdout.strip().splitlines()
        if lines:
            raw = lines[0].strip()
            if raw and raw.upper() not in ("N/A", "UNKNOWN"):
                eta = _humanize_start(raw)
    except Exception:
        eta = ""

    _eta_cache[job_id] = (now, eta)
    return eta


def _cancel_job(job) -> None:
    """Best-effort `scancel` for a single job. Failures are swallowed —
    the job table will just keep showing whatever state squeue reports."""
    try:
        subprocess.run(["scancel", str(job.job_id)], capture_output=True, timeout=5)
    except Exception:
        pass


# ── keyboard input (raw-mode stdin reader, POSIX only) ───────────────────────


class _KeyReader:
    """Reads single keypresses (including arrow-key escape sequences) from a
    background thread without blocking the main poll loop. No-ops entirely
    if stdin isn't a real TTY or the `termios`/`tty` modules aren't available
    (e.g. Windows, or output piped to a file)."""

    def __init__(self) -> None:
        import queue
        import threading

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional["threading.Thread"] = None
        self._old_settings = None
        self._active = _HAVE_TTY and sys.stdin.isatty()

    def start(self) -> None:
        if not self._active:
            return
        import threading

        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop_event.is_set():
            ready, _, _ = _select.select([fd], [], [], 0.1)
            if not ready:
                continue
            # Read straight off the fd (not sys.stdin) so select()'s view of
            # what's available and what we actually consume stay in sync.
            # Mixing select(fd) with the buffered sys.stdin.read() caused
            # arrow-key escape sequences (ESC '[' 'A'/'B') to get split: the
            # first read would silently slurp all 3 bytes into Python's
            # internal buffer but only return the ESC, then the follow-up
            # select() would see nothing left on the fd and never fetch the
            # rest, dropping the arrow entirely. A single os.read grabs
            # whatever's currently buffered (up to 3 bytes covers a plain
            # key or a full arrow sequence) in one shot.
            data = os.read(fd, 3)
            if not data:
                continue
            ch = data.decode(errors="ignore")
            if ch:
                self._queue.put(ch)

    def get_nowait(self) -> Optional[str]:
        import queue

        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        if not self._active:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


_UP_KEYS = {"k", "\x1b[A"}
_DOWN_KEYS = {"j", "\x1b[B"}


def _build_table(jobs: List, start_times: Dict[str, float], selected: int,
                  cancelling: set, status_line: str) -> "Table":
    table = Table(title="unitccl · job status", expand=True)
    table.add_column("", width=1)  # selection marker
    table.add_column("job id", style="bold")
    table.add_column("state")
    table.add_column("ranks", justify="right")
    table.add_column("elapsed", justify="right")
    table.add_column("eta (squeue --start)", justify="right")
    table.add_column("done", justify="center")

    for i, job in enumerate(jobs):
        state = _state_of(job)
        if job.job_id in cancelling and state in ("PENDING", "RUNNING", "COMPLETING"):
            state_display, style = "CANCELLING…", "bold red"
        else:
            state_display, style = state, _STATE_STYLE.get(state, "white")
        elapsed = _fmt_elapsed(time.time() - start_times[job.job_id])
        eta = _fetch_eta(job.job_id) if state == "PENDING" else ""
        done_mark = "✓" if job.done() else ""
        marker = Text("▶", style="bold red") if i == selected else Text("")
        row_style = "on grey19" if i == selected else None
        table.add_row(
            marker,
            str(job.job_id),
            Text(state_display, style=style),
            _ranks_of(job),
            elapsed,
            Text(eta, style="yellow" if eta else "dim"),
            Text(done_mark, style="green"),
            style=row_style,
        )
    if status_line:
        table.caption = status_line
        table.caption_style = "bold red"
    else:
        table.caption = "↑/↓ or j/k select · c cancel selected · a cancel all · q detach"
        table.caption_style = "dim"
    return table


def _build_log_panel(tail: List[str]) -> "Panel":
    body = "\n".join(tail) if tail else Text("waiting for output...", style="dim")
    return Panel(body, title="recent log output", border_style="grey50", expand=True, padding=(1, 2))


def _drain_new_lines(job, stream: str, offsets: Dict[Tuple[str, str], int]) -> List[str]:
    path = Path(job.paths.stdout if stream == "stdout" else job.paths.stderr)
    if not path.exists():
        return []
    key = (job.job_id, stream)
    with open(path, "r") as f:
        f.seek(offsets.get(key, 0))
        chunk = f.read()
        offsets[key] = f.tell()
    if not chunk:
        return []
    prefix = f"[{job.job_id}:{stream}]"
    return [f"{prefix} {line}" for line in chunk.splitlines()]


def watch_jobs(jobs: List, poll_interval: float = 2.0) -> None:
    """Live-updating dashboard: job table on top, scrolling log tail below.
    Blocks until every job is done, then raises if any of them failed
    (same contract as the plain-text `wait_for`).

    Interactive keyboard controls (cancel a job, cancel all, detach) are
    available when stdin is a TTY — see module docstring."""
    if not _HAVE_RICH:
        return _watch_jobs_plain(jobs, poll_interval)

    offsets: Dict[Tuple[str, str], int] = {}
    start_times = {job.job_id: time.time() for job in jobs}
    tail: Deque[str] = deque(maxlen=_LOG_BUFFER_LINES)
    console = Console()

    selected = 0
    cancelling: set = set()
    confirm_cancel_all_until = 0.0
    status_line = ""
    last_forced_check = 0.0
    _FORCE_CHECK_INTERVAL = 1.0  # cap how often we force sacct/squeue calls

    # table title + top border + header row + header sep + 1 row/job +
    # bottom border + caption line
    table_height = len(jobs) + 6

    def render() -> "Layout":
        layout = Layout()
        layout.split_column(
            Layout(name="table", size=table_height),
            Layout(name="spacer", size=1),
            Layout(name="log", ratio=1),
        )
        layout["table"].update(_build_table(jobs, start_times, selected, cancelling, status_line))
        layout["spacer"].update("")
        avail = max(console.size.height - table_height - 1 - 5, 3)
        layout["log"].update(_build_log_panel(list(tail)[-avail:]))
        return layout

    def handle_key(key: str) -> bool:
        """Returns False if the user asked to detach (stop watching)."""
        nonlocal selected, confirm_cancel_all_until, status_line

        now = time.time()
        if key in _UP_KEYS:
            selected = (selected - 1) % len(jobs)
        elif key in _DOWN_KEYS:
            selected = (selected + 1) % len(jobs)
        elif key in ("c", "x"):
            job = jobs[selected]
            if not job.done():
                _cancel_job(job)
                cancelling.add(job.job_id)
                status_line = f"sent scancel for job {job.job_id}"
        elif key == "a":
            if now < confirm_cancel_all_until:
                for job in jobs:
                    if not job.done():
                        _cancel_job(job)
                        cancelling.add(job.job_id)
                status_line = "sent scancel for all jobs"
                confirm_cancel_all_until = 0.0
            else:
                confirm_cancel_all_until = now + 3.0
                status_line = "press 'a' or 'y' again within 3s to cancel ALL jobs"
        elif key == "y" and now < confirm_cancel_all_until:
            for job in jobs:
                if not job.done():
                    _cancel_job(job)
                    cancelling.add(job.job_id)
            status_line = "sent scancel for all jobs"
            confirm_cancel_all_until = 0.0
        elif key == "q":
            return False
        return True

    key_reader = _KeyReader()
    key_reader.start()

    # screen=True switches to the terminal's alternate buffer: the dashboard
    # owns the full screen and redraws in place on every refresh, instead of
    # printing a new frame below the last one (which is what caused the
    # stacked "unitccl · job status" panels / stale-frame artifacts on
    # resize). The original screen content is restored when this exits.
    detached = False
    tick = min(poll_interval, 0.25) if poll_interval > 0 else 0.25
    last_poll = 0.0

    try:
        with Live(render(), console=console, refresh_per_second=4, screen=True) as live:
            while not all(job.done() for job in jobs):
                key = key_reader.get_nowait()
                needs_redraw = False
                if key is not None:
                    if not handle_key(key):
                        detached = True
                        break
                    needs_redraw = True

                # clear the "confirm cancel-all" prompt once it expires
                if status_line and confirm_cancel_all_until and time.time() >= confirm_cancel_all_until:
                    status_line = ""
                    needs_redraw = True

                now = time.time()

                # job.done()/job.state normally poll at most every 2-60s
                # (submitit's own exponential backoff) so a job we just
                # cancelled can take a while to be reflected -- force one
                # fresh sacct/squeue call (throttled to ~1/s) for jobs we
                # know we just cancelled so the table/exit reflect it fast.
                pending_cancel = {jid for jid in cancelling
                                   if not any(j.job_id == jid and j.done() for j in jobs)}
                if pending_cancel and now - last_forced_check >= _FORCE_CHECK_INTERVAL:
                    for job in jobs:
                        if job.job_id in pending_cancel:
                            job.done(force_check=True)
                    last_forced_check = now
                    needs_redraw = True

                if now - last_poll >= poll_interval:
                    for job in jobs:
                        for stream in ("stdout", "stderr"):
                            tail.extend(_drain_new_lines(job, stream, offsets))
                    last_poll = now
                    needs_redraw = True

                if needs_redraw:
                    live.update(render())
                time.sleep(tick)

            if not detached:
                # final drain after completion
                for job in jobs:
                    for stream in ("stdout", "stderr"):
                        tail.extend(_drain_new_lines(job, stream, offsets))
                live.update(render())
    finally:
        key_reader.stop()

    if detached:
        return
    raise_on_failure(jobs, cancelling)


def _watch_jobs_plain(jobs: List, poll_interval: float) -> None:
    """Fallback identical to the original slurm_utils.wait_for loop."""
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


def raise_on_failure(jobs: List, cancelled_ids: Optional[set] = None) -> None:
    """Check job outcomes and raise if anything genuinely failed.

    Jobs left in the CANCELLED state (whether cancelled through this TUI's
    'c'/'a' keys or via an external `scancel`/timeout-cancel) are reported
    as a short, clean message rather than the raw submitit "no output
    produced" pickle-path dump -- cancellation is an expected, deliberate
    outcome, not a crash. Any other real failure (FAILED/TIMEOUT/exception)
    still raises with its message so genuine errors aren't hidden.
    """
    cancelled_ids = cancelled_ids or set()
    cancelled, failed = [], []

    for job in jobs:
        try:
            job.results()
        except Exception as e:
            if _state_of(job) == "CANCELLED":
                cancelled.append(job)
            else:
                failed.append((job, e))

    if cancelled:
        ids = ", ".join(str(j.job_id) for j in cancelled)
        info(f"job(s) {ids} were cancelled — stopping cleanly")

    if failed:
        job, e = failed[0]
        raise RuntimeError(f"job {job.job_id} failed: {e}") from e
