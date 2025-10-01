"""Run a command with automatic reload when project files change.

This helper is designed for Docker development workflows where code on the host
codex/fix-missing-watchfiles-module-error-dq2o5o
is bind-mounted into the container (see ``docker-compose.yml``). Without the
bind mount, Docker will continue to run the version of the code baked into the
image which makes file changes from an IDE invisible to the reloader.  Change
detection defaults to :mod:`watchgod` which works out-of-the-box inside the
provided Docker image, while :mod:`watchfiles` is also installed for
compatibility with other tooling that prefers its CLI.

codex/fix-missing-watchfiles-module-error-cc0ljm
is bind-mounted into the container (see ``docker-compose.yml``). Without the
bind mount, Docker will continue to run the version of the code baked into the
image which makes file changes from an IDE invisible to the reloader.  The
script intentionally avoids the optional ``watchfiles`` dependency which has
caused installation issues in some environments.  Instead we rely solely on
:mod:`watchgod`, a pure-Python watcher that is already part of our dependency
set and works out-of-the-box inside the provided Docker image.

is bind-mounted into the container.  It intentionally avoids the optional
``watchfiles`` dependency which has caused installation issues in some
environments.  Instead we rely solely on :mod:`watchgod`, a pure-Python watcher
that is already part of our dependency set and works out-of-the-box inside the
provided Docker image.
main
main

Example usage from the repository root::

    python scripts/run_with_reloader.py --watch /app --ignore /app/logs \
        --ignore /app/state -- python auto_trader.py

When a watched file changes we terminate the running command and start it again.
The script exits cleanly when it receives ``SIGINT``/``SIGTERM``.
"""
from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Set, Tuple

DEFAULT_IGNORES = {'.git', '__pycache__', '.pytest_cache', 'logs', 'state'}

from watchgod.watcher import DefaultDirWatcher

LOGGER = logging.getLogger("reloader")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch",
        "-w",
        action="append",
        dest="watch",
        default=[],
        help="Path to watch for changes (default: current working directory).",
    )
    parser.add_argument(
        "--ignore",
        "-i",
        action="append",
        dest="ignore",
        default=[],
        help="Paths to ignore when watching for changes.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds when falling back to watchgod.",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.2,
        help="Minimum seconds between restarts after a change is detected.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (defaults to `python auto_trader.py`).",
    )
    return parser.parse_args(argv)


class _ProcessRunner:
    """Helper that manages the lifecycle of the child process."""

    def __init__(self, command: Sequence[str]):
        self._command = list(command)
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        LOGGER.info("Starting command: %s", " ".join(self._command))
        self._process = subprocess.Popen(self._command)

    def stop(self, *, force: bool = False) -> None:
        if not self._process or self._process.poll() is not None:
            return
        LOGGER.info("Stopping command (force=%s)...", force)
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if force:
                self._process.kill()
                self._process.wait()
            else:
                LOGGER.warning("Graceful shutdown timed out; killing process.")
                self._process.kill()
                self._process.wait()
        finally:
            self._process = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def close(self) -> None:
        self.stop(force=True)


def _normalise_paths(paths: Iterable[str]) -> List[Path]:
    result = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        result.append(path)
    return result


def _watch_changes(
    watch_paths: Sequence[Path],
    ignored: Set[str],
    stop_event: threading.Event,
    interval: float,
) -> Iterator[Set[Tuple[object, str]]]:
    """Yield file changes as they occur using :mod:`watchgod` polling."""

    LOGGER.info("Using watchgod for change detection.")
    watchers = [DefaultDirWatcher(str(path), ignored_paths=set(ignored)) for path in watch_paths]

    while not stop_event.is_set():
        changes: Set[Tuple[object, str]] = set()
        for watcher in watchers:
            changes |= watcher.check()
        if changes:
            yield changes
        stop_event.wait(interval)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    command = args.command or [sys.executable, "auto_trader.py"]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [sys.executable, "auto_trader.py"]

    watch_paths = _normalise_paths(args.watch) or [Path.cwd()]
    ignored_paths = {str(path) for path in _normalise_paths(DEFAULT_IGNORES)}
    ignored_paths.update(str(path) for path in _normalise_paths(args.ignore))

    runner = _ProcessRunner(command)
    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object | None) -> None:
        LOGGER.info("Received signal %s, shutting down...", signum)
        stop_event.set()
        runner.close()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    runner.start()
    last_restart = 0.0
    try:
        for changes in _watch_changes(watch_paths, ignored_paths, stop_event, args.interval):
            if stop_event.is_set():
                break
            now = time.monotonic()
            if now - last_restart < args.debounce:
                continue
            last_restart = now
            formatted = ", ".join(path for _change, path in sorted(changes, key=lambda item: item[1]))
            LOGGER.info("Detected changes: %s", formatted)
            runner.restart()
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
