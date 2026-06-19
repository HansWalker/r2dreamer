"""Single-run notebook helpers for TD-MPC2 DMC data collection."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import zarr


def expand_value(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    return value


def load_collection_config(path: Path, *, tdmpc2_dir: Path | None = None, data_dir: Path | None = None):
    import yaml

    path = Path(path)
    defaults = {
        "tdmpc2_root": str(tdmpc2_dir) if tdmpc2_dir is not None else "${TDMPC2_DIR}",
        "output_dir": str(data_dir) if data_dir is not None else "${DMC_EXPERT_DATA_DIR}",
        "local_output_dir": "/content/dmc_expert",
        "num_episodes": 2000,
        "checkpoint_seed": 1,
        "seed": 1,
        "action_repeat": 2,
        "max_episode_steps": 500,
        "save_images": False,
        "image_size": 64,
        "refresh_seconds": 10,
        "recent_log_lines": 25,
        "progress_every": 25,
        "start_from_scratch": True,
        "tasks": [],
        "expert": {
            "mpc": False,
        },
    }
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = {**defaults, **loaded}
    cfg["expert"] = {**defaults["expert"], **(loaded.get("expert") or {})}
    cfg = expand_value(cfg)
    cfg["config_path"] = path
    cfg["tdmpc2_root"] = Path(cfg["tdmpc2_root"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["local_output_dir"] = Path(cfg["local_output_dir"]) if cfg.get("local_output_dir") else None
    return cfg


def scenario_name(task: str) -> str:
    return task.replace("/", "_").replace("-", "_").replace("__", "_")


def task_store_name(task: str) -> str:
    if task == "ball_in_cup/catch":
        return "cup_catch.zarr"
    return scenario_name(task) + ".zarr"


def read_progress(out_dir: Path, task: str):
    store_path = Path(out_dir) / task_store_name(task)
    if not store_path.exists():
        return 0, 0, "not created"
    try:
        root = zarr.open(str(store_path), mode="r")
        episodes = int(root["episode_length"].shape[0]) if "episode_length" in root else 0
        rows = int(root["obs"].shape[0]) if "obs" in root else 0
        return episodes, rows, str(store_path)
    except Exception as exc:
        return 0, 0, f"read pending: {type(exc).__name__}"


def read_new_log_lines(log_path: Path, offset: int):
    if not Path(log_path).exists():
        return offset, []
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()
    return offset, lines


def prepare_collection_run(config: dict, task_item: dict | str, *, r2dreamer_dir: Path):
    task = task_item["task"] if isinstance(task_item, dict) else str(task_item)
    difficulty = task_item.get("difficulty", "-") if isinstance(task_item, dict) else "-"
    run_root = Path(config.get("local_output_dir") or config["output_dir"])
    out_dir = run_root / scenario_name(task)
    final_dir = Path(config["output_dir"]) / scenario_name(task)

    if bool(config.get("start_from_scratch", False)):
        for path in (out_dir, final_dir):
            if path.exists():
                shutil.rmtree(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "collect_config.yaml"
    log_path = out_dir / "collector.log"
    config_path.write_text(
        "\n".join(
            [
                f"tdmpc2_root: {Path(config['tdmpc2_root'])}",
                f"output_dir: {out_dir}",
                "",
                "tasks:",
                f"  - {task}",
                "",
                f"num_episodes: {int(config['num_episodes'])}",
                f"checkpoint_seed: {int(config['checkpoint_seed'])}",
                f"seed: {int(config['seed'])}",
                "",
                f"action_repeat: {int(config['action_repeat'])}",
                f"max_episode_steps: {int(config['max_episode_steps'])}",
                "",
                f"save_images: {str(bool(config['save_images'])).lower()}",
                f"image_size: {int(config['image_size'])}",
                "",
                "resume: false",
                f"progress_every: {int(config['progress_every'])}",
                "",
                "expert:",
                f"  mpc: {str(bool(config.get('expert', {}).get('mpc', False))).lower()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "task": task,
        "difficulty": difficulty,
        "out_dir": out_dir,
        "final_dir": final_dir,
        "config_path": config_path,
        "log_path": log_path,
        "collector": Path(r2dreamer_dir) / "scripts" / "collect_dmc_expert_data.py",
    }


def start_collection_run(run: dict):
    log_file = Path(run["log_path"]).open("w", buffering=1, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(run["collector"]), "--config", str(run["config_path"])],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, log_file


def sync_collection_run(run: dict):
    out_dir = Path(run["out_dir"])
    final_dir = Path(run["final_dir"])
    if out_dir.resolve() == final_dir.resolve():
        return final_dir
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_dir, final_dir)
    return final_dir
