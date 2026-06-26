"""Helpers for launching one TD-MPC2 DMC collection job."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def expand_value(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    return value


def load_collection_config(path, tdmpc2_dir=None, data_dir=None):
    path = Path(path)
    defaults = {
        "tdmpc2_root": str(tdmpc2_dir) if tdmpc2_dir is not None else "${TDMPC2_DIR}",
        "output_dir": str(data_dir) if data_dir is not None else "${DMC_EXPERT_DATA_DIR}",
        "num_episodes": 2000,
        "checkpoint_seed": 1,
        "seed": 1,
        "action_repeat": 2,
        "max_episode_steps": 500,
        "save_images": False,
        "image_size": 64,
        "refresh_seconds": 10,
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
    return cfg


def read_progress(out_dir, _task=None):
    store_path = Path(out_dir)
    progress_path = store_path / "progress.json"
    if progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        return int(payload.get("episodes", 0)), int(payload.get("rows", 0)), str(store_path)

    import h5py

    data_path = store_path / "data.hdf5"
    if not data_path.exists():
        return 0, 0, "not created"

    rows = 0
    episodes = 0
    with h5py.File(data_path, "r") as root:
        complete = root["complete"][:].astype(bool)
        lengths = root["lengths"][:].astype("int64")
        episodes = int(complete.sum())
        rows = int(lengths[complete].sum())
    return episodes, rows, str(store_path)


def make_collect_config(base, task, output_dir, path):
    output_dir = Path(output_dir)
    path = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "tdmpc2_root": str(Path(base["tdmpc2_root"])),
        "output_dir": str(output_dir),
        "tasks": [task],
        "num_episodes": int(base["num_episodes"]),
        "checkpoint_seed": int(base["checkpoint_seed"]),
        "seed": int(base["seed"]),
        "action_repeat": int(base["action_repeat"]),
        "max_episode_steps": int(base["max_episode_steps"]),
        "save_images": bool(base["save_images"]),
        "image_size": int(base["image_size"]),
        "resume": bool(base.get("resume", False)),
        "progress_every": int(base["progress_every"]),
        "expert": {
            "mpc": bool(base.get("expert", {}).get("mpc", False)),
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
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
