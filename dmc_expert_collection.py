"""Helpers for launching one TD-MPC2 DMC collection job."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def expand_value(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    return value


def load_collection_config(path, tdmpc2_dir=None, data_dir=None):
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
        "resume": False,
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


def scenario_name(task):
    return task.replace("/", "_").replace("-", "_").replace("__", "_")


def task_store_name(task):
    if task == "ball_in_cup/catch":
        return "cup_catch.zarr"
    return scenario_name(task) + ".zarr"


def read_progress(out_dir, task):
    import zarr

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


def read_new_log_lines(log_path, offset):
    if not Path(log_path).exists():
        return offset, []
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()
    return offset, lines


def make_collect_config(base, task, output_dir, path):
    output_dir = Path(output_dir)
    path = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"tdmpc2_root: {Path(base['tdmpc2_root'])}",
                f"output_dir: {output_dir}",
                "",
                "tasks:",
                f"  - {task}",
                "",
                f"num_episodes: {int(base['num_episodes'])}",
                f"checkpoint_seed: {int(base['checkpoint_seed'])}",
                f"seed: {int(base['seed'])}",
                "",
                f"action_repeat: {int(base['action_repeat'])}",
                f"max_episode_steps: {int(base['max_episode_steps'])}",
                "",
                f"save_images: {str(bool(base['save_images'])).lower()}",
                f"image_size: {int(base['image_size'])}",
                "",
                f"resume: {str(bool(base.get('resume', False))).lower()}",
                f"progress_every: {int(base['progress_every'])}",
                "",
                "expert:",
                f"  mpc: {str(bool(base.get('expert', {}).get('mpc', False))).lower()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def start_collector(collector, config, log):
    log_file = Path(log).open("w", buffering=1, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(collector), "--config", str(config)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, log_file


def sync_dataset(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.resolve() == dst.resolve():
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst
