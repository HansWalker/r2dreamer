"""Throttled worker progress and a single, thread-safe terminal display."""

import contextlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

PREFIX = "Progress | "


def duration(seconds):
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Progress:
    def __init__(self, phase, total, initial=0, interval=30):
        self.phase, self.total, self.initial = phase, int(total), int(initial)
        self.interval = interval
        self.started = time.monotonic()
        self.last = float("-inf")

    def due(self):
        return time.monotonic() - self.last >= self.interval

    def update(self, current, detail="", *, force=False):
        if not force and not self.due():
            return
        now = time.monotonic()
        done = int(current) - self.initial
        seconds = (now - self.started) / done if done > 0 else None
        payload = {
            "phase": self.phase, "current": int(current), "total": self.total,
            "eta": None if seconds is None else max(0, self.total - int(current)) * seconds,
            "detail": detail,
        }
        # Workers are pipes under main.py; standalone runs remain readable too.
        if os.environ.get("DMC_PROGRESS_PIPE") == "1":
            print(PREFIX + json.dumps(payload, allow_nan=False), flush=True)
        else:
            print(f"{self.phase} | {current}/{self.total} | eta={duration(payload['eta'])} | {detail}", flush=True)
        self.last = now


class Console:
    def __init__(self):
        self.lock = threading.RLock()
        self.tasks = {}
        self.rows = 0
        self.live = False
        self.stream = None
        self.log = None

    @contextlib.contextmanager
    def capture(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = sys.stdout
        self.live = self.stream.isatty() and os.environ.get("TERM", "") != "dumb"
        stop = threading.Event()

        def refresh():
            while not stop.wait(1):
                with self.lock:
                    self._clear()
                    self._render()

        with Path(path).open("a", encoding="utf-8", buffering=1) as log:
            self.log = log
            thread = threading.Thread(target=refresh, daemon=True) if self.live else None
            if thread:
                thread.start()
            try:
                yield self
            finally:
                stop.set()
                if thread:
                    thread.join()
                with self.lock:
                    self._clear()
                    self.stream.flush()
                    self.tasks.clear()
                    self.log = None
                    self.live = False
                    self.stream = None

    def _clear(self):
        if self.rows:
            self.stream.write("\x1b[1A\r\x1b[2K" * self.rows)
            self.rows = 0

    def _render(self):
        if not self.live:
            return
        width, height = shutil.get_terminal_size((100, 24))
        lines = []
        for label, task in self.tasks.items():
            name = label.split(" | config=", 1)[0]
            lines.append(f"{name} | elapsed={duration(time.monotonic() - task['started'])}")
            event = task.get("progress")
            if event:
                fraction = min(1, max(0, event['current'] / max(1, event['total'])))
                bar = "#" * int(fraction * 12)
                lines.append(
                    f"  {event['phase']} [{bar:<12}] {fraction:4.0%} "
                    f"{event['current']}/{event['total']} | eta={duration(event['eta'])}"
                )
                if event['detail']:
                    lines.append("  " + event['detail'])
            else:
                lines.append("  " + task.get("message", "starting"))
        # Never wrap a row or scroll the dashboard out of a short terminal.
        lines = [line[:max(1, width - 1)] for line in lines[:max(1, height - 3)]]
        if lines:
            self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self.rows = len(lines)

    def message(self, message="", *, flush=True):
        with self.lock:
            self._clear()
            print(message, file=self.stream or sys.stdout, flush=flush)
            if self.log:
                self.log.write(str(message) + "\n")
            self._render()

    def start(self, label):
        with self.lock:
            self.tasks[label] = {"started": time.monotonic()}
            self.message(f"START | {label}")

    def update(self, label, message):
        with self.lock:
            if message.startswith(("Expert | update=", "Online | env_step=")):
                if self.log:
                    self.log.write(f"[{label}] {message}\n")
                return
            task = self.tasks.get(label)
            if message.startswith(PREFIX):
                event = json.loads(message[len(PREFIX):])
                if task is not None:
                    task['progress'] = event
                text = (
                    f"{event['phase']} | {event['current']}/{event['total']} | "
                    f"eta={duration(event['eta'])} | {event['detail']}"
                )
                if self.live:
                    if self.log:
                        self.log.write(f"[{label}] {text}\n")
                    self._clear()
                    self._render()
                    return
            else:
                text = message
                if task is not None:
                    task['message'] = message
                    # Stage transitions must not leave an old completed bar visible.
                    if message.startswith(("Evaluation |", "Checkpoint |", "Online | steps=", "Expert | updates=")):
                        task.pop('progress', None)
            self.message(f"  [{label}] {text}")

    def finish(self, label, status):
        with self.lock:
            task = self.tasks.pop(label, None)
            elapsed = duration(time.monotonic() - task['started']) if task else "-"
            self.message(f"{status} | {label} | elapsed={elapsed}")


console = Console()
