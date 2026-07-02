"""Helpers for launching one DMC expert training job."""

import json
import os
import subprocess
import sys
from pathlib import Path


def fmt_metric(value):
    if value is None:
        return "-"
    return f"{float(value):.3g}"


def read_metric_summary(logdir):
    metrics_path = Path(logdir) / "metrics.jsonl"
    update = None
    opt_loss = None
    eval_score = None
    if not metrics_path.exists():
        return "-", "-", "-"
    with metrics_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            update = row.get("train/opt/updates", update)
            opt_loss = row.get("train/opt/loss", opt_loss)
            eval_score = row.get("episode/eval_score", eval_score)
    return fmt_metric(update), fmt_metric(opt_loss), fmt_metric(eval_score)


def start_training(
    *,
    r2dreamer_dir,
    config,
    data,
    task,
    logdir,
    extra=None,
):
    data = Path(data)
    if not all((data / name).exists() for name in ("metadata.json", "data.hdf5")):
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
        f"expert_data.data_path={data}",
        f"env.task={task}",
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
