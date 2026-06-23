"""Helpers for launching one offline DMC expert training job."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


CORE_LOG_KEYS = {
    "train/opt/loss",
    "train/loss/dyn",
    "train/loss/rep",
    "train/loss/barlow",
    "train/loss/rew",
    "train/loss/con",
    "train/loss/policy",
    "train/loss/value",
    "train/ret_replay_mean",
    "episode/eval_score",
    "fps/fps",
}


def read_new_log_lines(log_path, offset):
    if not Path(log_path).exists():
        return offset, []
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()
    return offset, lines


def fmt_metric(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.3g}"
    except Exception:
        return str(value)


def compact_log_line(line):
    match = re.match(r"^(\[\d+\])\s+(.*)$", line)
    if not match or " / " not in line:
        return line
    step, body = match.groups()
    kept = []
    for item in body.split(" / "):
        parts = item.rsplit(" ", 1)
        if len(parts) == 2 and parts[0] in CORE_LOG_KEYS:
            kept.append(item)
    return f"{step} " + " / ".join(kept) if kept else line


def read_metric_summary(logdir):
    metrics_path = Path(logdir) / "metrics.jsonl"
    update = None
    opt_loss = None
    eval_score = None
    if not metrics_path.exists():
        return "-", "-", "-"
    try:
        with metrics_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                update = row.get("train/opt/updates", update)
                opt_loss = row.get("train/opt/loss", opt_loss)
                eval_score = row.get("episode/eval_score", eval_score)
    except Exception as exc:
        return "read_err", type(exc).__name__, "-"
    return fmt_metric(update), fmt_metric(opt_loss), fmt_metric(eval_score)


def start_training(
    *,
    r2dreamer_dir,
    config,
    data,
    task,
    logdir,
    resume=False,
    extra=None,
):
    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(f"Missing training dataset: {data}")

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    stdout_log = logdir / "notebook_stdout.log"
    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--config-name",
        config,
        f"offline.data_path={data}",
        f"env.task={task}",
        f"offline.resume={str(bool(resume)).lower()}",
        f"logdir={logdir}",
    ]
    cmd.extend(extra or [])

    log_file = stdout_log.open("w", buffering=1, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=Path(r2dreamer_dir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, log_file, stdout_log
