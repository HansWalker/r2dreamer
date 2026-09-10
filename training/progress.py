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


def short_label(label):
    label = label.split(" | config=", 1)[0]
    stage, separator, name = label.partition(" | ")
    return f"{stage.split()[0]} {name.replace('/seed_', '/s')}" if separator else label


def worker_status(message):
    """Startup details belong in the live row, not permanent terminal history."""
    parts = message.split(" | ")
    fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
    if parts[0] == "Run":
        return "Starting"
    if parts[0] == "Model":
        count = fields.get("parameters")
        return f"Model ready | parameters={count}" if count else "Building model"
    if parts[0] == "Data":
        return f"Data ready | episodes={fields.get('episodes', '-')}"
    if parts[0] == "Checkpoint":
        for action in ("saved", "loaded"):
            if action in fields:
                return f"Checkpoint {action} | {Path(fields[action]).name}"
    if parts[0] == "Expert":
        if "batch_size" in fields:
            return f"Expert | batch={fields['batch_size']}x{fields.get('sequence_length', '-')}"
        return " | ".join(parts[:2])
    if parts[0] == "Online":
        if "steps" in fields:
            return f"Online | steps={fields['steps']} | updates={fields.get('model_updates', '-')}"
        return "Starting environments" if "environment_seed" in fields else " | ".join(parts[:2])
    if parts[0] == "Evaluation":
        if fields.get("state_prediction") == "running":
            return "Predicting held-out trajectories"
        if "running" in parts or fields.get("policy_rollout") == "running":
            return "Evaluating policy"
    return None


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
            name = short_label(label)
            lines.append(f"{name} | elapsed={duration(time.monotonic() - task['started'])}")
            event = task.get("progress")
            if event:
                fraction = min(1, max(0, event['current'] / max(1, event['total'])))
                bar = "#" * int(fraction * 12)
                line = (
                    f"  {event['phase']} [{bar:<12}] {fraction:5.1%} "
                    f"{event['current']}/{event['total']} | eta={duration(event['eta'])}"
                )
                detail = event['detail']
                if detail and len(line) + len(detail) + 3 < width:
                    lines.append(f"{line} | {detail}")
                else:
                    lines.append(line)
                if detail and len(line) + len(detail) + 3 >= width:
                    lines.append("  " + event['detail'])
            else:
                lines.append("  " + task.get("message", "starting"))
        # Never wrap a row or scroll the dashboard out of a short terminal.
        lines = [line[:max(1, width - 1)] for line in lines[:max(1, height - 3)]]
        if lines:
            self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self.rows = len(lines)

    def _record(self, message):
        if self.log:
            self.log.write(str(message) + "\n")

    def message(self, message="", *, flush=True, log_message=None):
        with self.lock:
            self._clear()
            print(message, file=self.stream or sys.stdout, flush=flush)
            self._record(message if log_message is None else log_message)
            self._render()

    def start(self, label):
        with self.lock:
            self.tasks[label] = {"started": time.monotonic()}
            if self.live:
                self._record(f"START | {label}")
                self._clear()
                self._render()
            else:
                self.message(f"START | {short_label(label)}", log_message=f"START | {label}")

    def update(self, label, message):
        with self.lock:
            if message.startswith(("Expert | update=", "Online | env_step=")):
                self._record(f"[{label}] {message}")
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
                    self._record(f"[{label}] {text}")
                    self._clear()
                    self._render()
                    return
            else:
                text = message
                status = worker_status(message)
                if task is not None:
                    task['message'] = status or message
                    # Stage transitions must not leave an old completed bar visible.
                    if status is not None or message.startswith("Evaluation |"):
                        task.pop('progress', None)
                if status is not None:
                    self._record(f"[{label}] {message}")
                    self._clear()
                    self._render()
                    return
            self.message(f"  [{short_label(label)}] {text}", log_message=f"[{label}] {text}")

    def finish(self, label, status):
        with self.lock:
            task = self.tasks.pop(label, None)
            elapsed = duration(time.monotonic() - task['started']) if task else "-"
            self.message(
                f"{status} | {short_label(label)} | elapsed={elapsed}",
                log_message=f"{status} | {label} | elapsed={elapsed}",
            )


console = Console()
