"""Single-run notebook helpers for offline DMC expert training."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
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


DMC_EXPERT_SCENARIOS = [
    {
        "name": "cartpole",
        "env_task": "dmc_cartpole_swingup",
        "data_rel": "cartpole_swingup/cartpole_swingup.zarr",
    },
    {
        "name": "walker",
        "env_task": "dmc_walker_walk",
        "data_rel": "walker_walk/walker_walk.zarr",
    },
    {
        "name": "humanoid",
        "env_task": "dmc_humanoid_run",
        "data_rel": "humanoid_run/humanoid_run.zarr",
    },
]

DMC_EXPERT_MODELS = [
    {"name": "gru", "config_name": "offline_dmc_expert_gru"},
    {"name": "mamba3", "config_name": "offline_dmc_expert_mamba3"},
]


def make_training_run(workdir: Path, *, scenario: dict, model: dict, logdir_name: str):
    data_dir = Path(workdir) / "data" / "dmc_expert"
    return {
        "name": f"{scenario['name']}_{model['name']}",
        "scenario": scenario["name"],
        "model": model["name"],
        "config_name": model["config_name"],
        "data_path": data_dir / scenario["data_rel"],
        "env_task": scenario["env_task"],
        "logdir": Path(workdir) / "runs" / logdir_name,
    }


def default_training_runs(workdir: Path, run_tag: str | None = None):
    run_tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
    runs = []
    for scenario in DMC_EXPERT_SCENARIOS:
        for model in DMC_EXPERT_MODELS:
            runs.append(
                make_training_run(
                    workdir,
                    scenario=scenario,
                    model=model,
                    logdir_name=f"offline_{scenario['name']}_{model['name']}_fresh_{run_tag}",
                )
            )
    return runs


def read_new_log_lines(log_path: Path, offset: int):
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


def compact_log_line(line: str) -> str:
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


def read_metric_summary(logdir: Path):
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


def start_training_run(
    run: dict,
    *,
    r2dreamer_dir: Path,
    train_store: Path | None = None,
    env_task: str | None = None,
    resume: bool = False,
    extra_overrides=None,
):
    train_store = train_store if train_store is not None else run.get("data_path")
    env_task = env_task if env_task is not None else run.get("env_task")

    logdir = Path(run["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    stdout_log = logdir / "notebook_stdout.log"
    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--config-name",
        run["config_name"],
        f"offline.resume={str(bool(resume)).lower()}",
        f"logdir={logdir}",
    ]
    if train_store is not None:
        train_store = Path(train_store)
        if not train_store.exists():
            raise FileNotFoundError(f"Missing training dataset: {train_store}")
        cmd.append(f"offline.data_path={train_store}")
    if env_task is not None:
        cmd.append(f"env.task={env_task}")
    cmd.extend(extra_overrides or [])

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
