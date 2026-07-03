"""Preflight checks for STORM-native DMC comparison runs."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_storm_dmc import ExpertReplay, build_agent, build_world_model


TASKS = (
    ("cartpole", "cartpole_swingup", "dmc_cartpole_swingup"),
    ("walker", "walker_walk", "dmc_walker_walk"),
    ("humanoid", "humanoid_run", "dmc_humanoid_run"),
)

CONFIGS = (
    ("transformer", "storm_dmc_transformer"),
    ("mamba", "storm_dmc_mamba"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DMC_EXPERT_DATA_DIR", "/home/ubuntu/DMC/data/dmc_expert_hdf5_mpc_5k"),
    )
    parser.add_argument("--expert-epochs", type=float, default=16)
    parser.add_argument("--online-steps", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--env-num", type=int, default=16)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--wm-updates", type=int, default=4)
    parser.add_argument("--ac-updates", type=int, default=4)
    parser.add_argument("--max-size-ratio", type=float, default=1.20)
    return parser.parse_args()


def check_data(data_root: Path):
    missing = []
    for _, data, _ in TASKS:
        for filename in ("data.hdf5", "metadata.json"):
            path = data_root / data / filename
            if not path.exists():
                missing.append(path)
    if missing:
        raise SystemExit("Missing expert data:\n" + "\n".join(str(path) for path in missing))


def count_params(config_dir: Path, args):
    data_root = Path(args.data_root)
    common = [
        f"training.expert_epochs={args.expert_epochs}",
        f"training.online_steps={args.online_steps}",
        f"batch_size={args.batch_size}",
        f"batch_length={args.batch_length}",
        f"env.env_num={args.env_num}",
        f"env.action_repeat={args.action_repeat}",
        f"storm_train.world_model_updates={args.wm_updates}",
        f"storm_train.actor_critic_updates={args.ac_updates}",
        "eval_episode_num=2",
        "env.time_limit=1000",
    ]
    counts_by_task = {}
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for task_name, data, task in TASKS:
            counts_by_task[task_name] = {}
            for core, config_name in CONFIGS:
                config = compose(
                    config_name=config_name,
                    overrides=[
                        f"expert_data.data_path={data_root / data}",
                        f"env.task={task}",
                        *common,
                    ],
                )
                replay = ExpertReplay(config.expert, config.storm_train.batch_length)
                world_model = build_world_model(config, replay.obs_space(), replay.act_space())
                agent = build_agent(config, world_model.feat_size, replay.act_space())
                wm_params = sum(p.numel() for p in world_model.parameters())
                ac_params = sum(p.numel() for p in agent.parameters())
                total = wm_params + ac_params
                counts_by_task[task_name][core] = total
                print(
                    f"{task_name:8} {core:11} "
                    f"wm={wm_params:,} ac={ac_params:,} total={total:,}"
                )
                replay.close()
                del world_model, agent, replay
                torch.cuda.empty_cache()
                gc.collect()
            ratio = max(counts_by_task[task_name].values()) / min(counts_by_task[task_name].values())
            print(f"{task_name:8} size_ratio={ratio:.3f}")
            if ratio > args.max_size_ratio:
                raise SystemExit(
                    f"Model sizes are not comparable enough for {task_name}: "
                    f"ratio={ratio:.3f} > {args.max_size_ratio:.3f}."
                )
    return counts_by_task


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    check_data(data_root)
    expected_updates = math.ceil(args.online_steps / (args.env_num * args.action_repeat)) * args.wm_updates
    print(
        "settings "
        f"expert_epochs={args.expert_epochs:g} online_steps={args.online_steps} "
        f"batch={args.batch_size}x{args.batch_length} env_num={args.env_num} "
        f"action_repeat={args.action_repeat} wm_updates={args.wm_updates} "
        f"ac_updates={args.ac_updates} approx_online_updates={expected_updates}"
    )
    count_params(ROOT / "configs", args)
    print("Preflight passed.")


if __name__ == "__main__":
    main()
