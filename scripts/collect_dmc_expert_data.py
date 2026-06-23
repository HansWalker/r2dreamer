"""Collect expert DeepMind Control Suite rollouts with TD-MPC2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_REPO = "nicklashansen/tdmpc2"
DATA_FORMAT = "dmc_expert_interleaved_episodes_v1"


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    slug: str

    @property
    def dmc_name(self) -> str:
        return f"{self.domain}/{self.task}"

    @property
    def zarr_name(self) -> str:
        return self.slug.replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DMC expert rollouts with TD-MPC2 checkpoints."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dmc_expert.yaml"),
        help="YAML config file for collection settings.",
    )
    return parser.parse_args()


def load_config(path: Path) -> argparse.Namespace:
    import yaml

    defaults = {
        "tdmpc2_root": None,
        "output_dir": "data/dmc_expert",
        "num_episodes": 500,
        "checkpoint_seed": 1,
        "tasks": ["all"],
        "seed": 1,
        "action_repeat": 2,
        "max_episode_steps": 500,
        "image_size": 64,
        "save_images": True,
        "resume": False,
        "progress_every": 25,
        "expert": {
            "mpc": False,
        },
    }
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config = {**defaults, **config}
    config["expert"] = {**defaults["expert"], **(config.get("expert") or {})}
    config = {key: expand_config_value(value) for key, value in config.items()}

    if config["tdmpc2_root"] is None:
        raise ValueError(f"{path} must define tdmpc2_root")
    if isinstance(config["tasks"], str):
        config["tasks"] = [config["tasks"]]

    config["config"] = path
    config["tdmpc2_root"] = Path(config["tdmpc2_root"])
    config["output_dir"] = Path(config["output_dir"])
    return argparse.Namespace(**config)


def expand_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config_value(item) for key, item in value.items()}
    return value


def tdmpc2_slug(domain: str, task: str) -> str:
    if domain == "ball_in_cup" and task == "catch":
        return "cup-catch"
    return f"{domain}/{task}".replace("/", "-").replace("_", "-")


def discover_tasks() -> list[TaskSpec]:
    from dm_control import suite

    return [
        TaskSpec(domain, task, tdmpc2_slug(domain, task))
        for domain, task in sorted(tuple(suite.ALL_TASKS))
    ]


def select_tasks(all_tasks: list[TaskSpec], requested: list[str]) -> list[TaskSpec]:
    if requested == ["all"]:
        return all_tasks

    by_name = {task.dmc_name: task for task in all_tasks}
    by_short = {f"{task.domain}_{task.task}": task for task in all_tasks}
    by_slug = {task.slug: task for task in all_tasks}

    selected = []
    for item in requested:
        task = by_name.get(item) or by_short.get(item) or by_slug.get(item)
        if task is None:
            raise ValueError(f"Unknown DMC task: {item}")
        selected.append(task)
    return selected


def checkpoint_name(task: TaskSpec, seed: int) -> str:
    return f"dmcontrol/{task.slug}-{seed}.pt"


def list_checkpoints() -> set[str]:
    from huggingface_hub import HfApi

    return set(HfApi().list_repo_files(CHECKPOINT_REPO, repo_type="model"))


def download_checkpoint(task: TaskSpec, seed: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            repo_type="model",
            filename=checkpoint_name(task, seed),
        )
    )


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in obs.values()],
        dtype=np.float32,
    )


def add_tdmpc2_to_path(root: Path) -> Path:
    package_root = root.expanduser().resolve() / "tdmpc2"
    config_path = package_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing TD-MPC2 config: {config_path}")
    sys.path.insert(0, str(package_root))
    return config_path


def make_tdmpc2_cfg(
    config_path: Path,
    task: TaskSpec,
    obs_dim: int,
    action_dim: int,
    seed: int,
    max_episode_steps: int,
    expert: dict[str, Any] | None = None,
):
    from common import MODEL_SIZE, TASK_SET
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    cfg.task = task.slug
    cfg.obs = "state"
    cfg.model_size = 5
    cfg.compile = False
    cfg.seed = seed
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.checkpoint = ""
    cfg.work_dir = ""
    cfg.exp_name = "expert_collect"

    for key, value in MODEL_SIZE[cfg.model_size].items():
        cfg[key] = value

    cfg.multitask = cfg.task in TASK_SET
    cfg.task_dim = 0
    cfg.tasks = [cfg.task]
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.obs_shape = {"state": (obs_dim,)}
    cfg.action_dim = action_dim
    cfg.episode_length = max_episode_steps
    cfg.obs_shapes = None
    cfg.action_dims = None
    cfg.episode_lengths = None

    expert = expert or {}
    if "mpc" in expert:
        cfg.mpc = bool(expert["mpc"])
    return cfg_to_dataclass(cfg)


def move_state_key(
    state_dict: dict[str, Any],
    target_state_dict: dict[str, Any],
    old_key: str,
    new_key: str,
):
    if old_key in state_dict and new_key in target_state_dict:
        state_dict[new_key] = state_dict.pop(old_key)


def convert_old_flat_mlp_state_dict(
    target_state_dict: dict[str, Any],
    source_state_dict: dict[str, Any],
) -> dict[str, Any]:
    converted = dict(source_state_dict)

    if "_encoder.state.1.weight" in converted and "_encoder.state.1.ln.weight" not in converted:
        for idx in range(16):
            old_linear = 1 + 3 * idx
            old_norm = 2 + 3 * idx
            new_prefix = f"_encoder.state.{idx}"
            move_state_key(converted, target_state_dict, f"_encoder.state.{old_linear}.weight", f"{new_prefix}.weight")
            move_state_key(converted, target_state_dict, f"_encoder.state.{old_linear}.bias", f"{new_prefix}.bias")
            move_state_key(converted, target_state_dict, f"_encoder.state.{old_norm}.weight", f"{new_prefix}.ln.weight")
            move_state_key(converted, target_state_dict, f"_encoder.state.{old_norm}.bias", f"{new_prefix}.ln.bias")

    if "_dynamics.0.0.weight" in converted:
        final_norm_weight = converted.pop("_dynamics.1.weight", None)
        final_norm_bias = converted.pop("_dynamics.1.bias", None)
        for idx in range(16):
            new_prefix = f"_dynamics.{idx}"
            if f"{new_prefix}.weight" not in target_state_dict:
                continue
            old_linear = 3 * idx
            old_norm = 3 * idx + 1
            old_norm_weight = f"_dynamics.0.{old_norm}.weight"
            has_old_norm = old_norm_weight in converted
            move_state_key(converted, target_state_dict, f"_dynamics.0.{old_linear}.weight", f"{new_prefix}.weight")
            move_state_key(converted, target_state_dict, f"_dynamics.0.{old_linear}.bias", f"{new_prefix}.bias")
            move_state_key(converted, target_state_dict, old_norm_weight, f"{new_prefix}.ln.weight")
            move_state_key(converted, target_state_dict, f"_dynamics.0.{old_norm}.bias", f"{new_prefix}.ln.bias")
            if not has_old_norm and f"{new_prefix}.ln.weight" in target_state_dict:
                if final_norm_weight is not None:
                    converted[f"{new_prefix}.ln.weight"] = final_norm_weight
                if final_norm_bias is not None:
                    converted[f"{new_prefix}.ln.bias"] = final_norm_bias

    for prefix in ("_reward", "_pi", "_termination"):
        if f"{prefix}.1.weight" not in converted or f"{prefix}.0.ln.weight" in converted:
            continue
        for idx in range(16):
            new_prefix = f"{prefix}.{idx}"
            if f"{new_prefix}.weight" not in target_state_dict:
                continue
            old_linear = 3 * idx
            old_norm = 3 * idx + 1
            move_state_key(converted, target_state_dict, f"{prefix}.{old_linear}.weight", f"{new_prefix}.weight")
            move_state_key(converted, target_state_dict, f"{prefix}.{old_linear}.bias", f"{new_prefix}.bias")
            move_state_key(converted, target_state_dict, f"{prefix}.{old_norm}.weight", f"{new_prefix}.ln.weight")
            move_state_key(converted, target_state_dict, f"{prefix}.{old_norm}.bias", f"{new_prefix}.ln.bias")

    return converted


def convert_tdmpc2_state_dict(target_state_dict: dict[str, Any], source_state_dict: dict[str, Any]) -> dict[str, Any]:
    source_state_dict = convert_old_flat_mlp_state_dict(target_state_dict, source_state_dict)
    if "_detach_Qs_params.0.weight" not in source_state_dict:
        name_map = ["weight", "bias", "ln.weight", "ln.bias"]
        converted = dict(source_state_dict)
        for key, value in list(source_state_dict.items()):
            if key.startswith("_Qs.params."):
                num = int(key[len("_Qs.params."):])
                new_key = f"{num // 4}.{name_map[num % 4]}"
                converted.pop(key, None)
                converted[f"_Qs.params.{new_key}"] = value
                converted[f"_detach_Qs_params.{new_key}"] = value
            elif key.startswith("_target_Qs.params."):
                num = int(key[len("_target_Qs.params."):])
                new_key = f"{num // 4}.{name_map[num % 4]}"
                converted.pop(key, None)
                converted[f"_target_Qs_params.{new_key}"] = value

        for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
            for key in ("__batch_size", "__device"):
                meta_key = f"{prefix}params.{key}"
                if meta_key in target_state_dict:
                    converted[meta_key] = target_state_dict[meta_key]

        for key in ("log_std_min", "log_std_dif", "_action_masks"):
            if key in target_state_dict:
                converted[key] = target_state_dict[key]
        source_state_dict = converted

    return {
        key: value
        for key, value in source_state_dict.items()
        if key in target_state_dict
    }


def load_tdmpc2_checkpoint(agent, checkpoint_path: Path):
    import torch

    state_dict = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=False,
    )
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    target_state_dict = agent.model.state_dict()
    state_dict = convert_tdmpc2_state_dict(target_state_dict, state_dict)
    incompatible = agent.model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith((".__batch_size", ".__device"))
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Could not load TD-MPC2 checkpoint cleanly. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )


def load_agent(
    tdmpc2_root: Path,
    checkpoint_path: Path,
    task: TaskSpec,
    obs_dim: int,
    action_dim: int,
    seed: int,
    max_episode_steps: int,
    expert: dict[str, Any] | None = None,
):
    config_path = add_tdmpc2_to_path(tdmpc2_root)
    cfg = make_tdmpc2_cfg(
        config_path,
        task,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seed=seed,
        max_episode_steps=max_episode_steps,
        expert=expert,
    )

    from tdmpc2 import TDMPC2

    agent = TDMPC2(cfg)
    load_tdmpc2_checkpoint(agent, checkpoint_path)
    agent.eval()
    return agent


def make_env(task: TaskSpec, seed: int):
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    raw_env = suite.load(
        task.domain,
        task.task,
        task_kwargs={"random": seed},
        visualize_reward=False,
    )
    raw_action_spec = raw_env.action_spec()
    env = action_scale.Wrapper(raw_env, minimum=-1.0, maximum=1.0)
    return raw_env, env, raw_action_spec, env.action_spec()


def render(raw_env, domain: str, image_size: int) -> np.ndarray:
    camera_id = 2 if domain == "quadruped" else 0
    return raw_env.physics.render(image_size, image_size, camera_id=camera_id)


def zarr_attrs(
    task: TaskSpec,
    checkpoint_path: Path,
    checkpoint_seed: int,
    obs: dict[str, np.ndarray],
    obs_dim: int,
    action_spec,
    raw_action_spec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": DATA_FORMAT,
        "env_type": "dmc",
        "domain_name": task.domain,
        "task_name": task.task,
        "task_slug": task.slug,
        "policy": "tdmpc2",
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_path": checkpoint_name(task, checkpoint_seed),
        "checkpoint_local_path": str(checkpoint_path),
        "checkpoint_seed": checkpoint_seed,
        "seed": args.seed,
        "episode_seed_rule": "seed + episode_index",
        "obs_dim": obs_dim,
        "action_dim": int(np.prod(action_spec.shape)),
        "observation_keys": list(obs.keys()),
        "observation_shapes": {
            key: list(np.asarray(value).shape) for key, value in obs.items()
        },
        "action_min": np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1).tolist(),
        "action_max": np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1).tolist(),
        "raw_action_min": np.asarray(raw_action_spec.minimum, dtype=np.float32).reshape(-1).tolist(),
        "raw_action_max": np.asarray(raw_action_spec.maximum, dtype=np.float32).reshape(-1).tolist(),
        "action_repeat": args.action_repeat,
        "max_episode_steps": args.max_episode_steps,
        "image_size": args.image_size if args.save_images else None,
        "row_layout": "obs_t/image_t/action_t plus reward/discount/done from s_t -> s_{t+1}",
    }


def ensure_array(root, name: str, values: np.ndarray):
    if name not in root:
        row_shape = values.shape[1:]
        chunks = (min(max(values.shape[0], 1), 1024),) + row_shape
        root.zeros(
            name=name,
            shape=(0,) + row_shape,
            chunks=chunks,
            dtype=str(values.dtype),
        )
    return root[name]


def open_store(path: Path, attrs: dict[str, Any], resume: bool):
    import zarr

    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists. Pass --resume or delete it.")

    root = zarr.open(str(path), mode="a" if resume and path.exists() else "w")
    root.attrs.update(attrs)
    if "episode_start" not in root:
        root.zeros(name="episode_start", shape=(0,), chunks=(512,), dtype="int64")
    if "episode_length" not in root:
        root.zeros(name="episode_length", shape=(0,), chunks=(512,), dtype="int32")
    return root


def append_episode(root, episode: dict[str, np.ndarray]):
    start = int(root["obs"].shape[0]) if "obs" in root else 0
    length = int(episode["obs"].shape[0])
    for key, values in episode.items():
        values = np.asarray(values)
        ensure_array(root, key, values).append(values, axis=0)
    root["episode_start"].append(np.array([start], dtype=np.int64), axis=0)
    root["episode_length"].append(np.array([length], dtype=np.int32), axis=0)


def collect_episode(
    raw_env,
    env,
    agent,
    task: TaskSpec,
    save_images: bool,
    image_size: int,
    action_repeat: int,
    max_episode_steps: int,
) -> tuple[dict[str, np.ndarray], float]:
    import torch

    time_step = env.reset()
    rows: dict[str, list[np.ndarray]] = {
        "obs": [],
        "action": [],
        "reward": [],
        "discount": [],
        "is_last": [],
        "terminated": [],
        "timeout": [],
    }
    if save_images:
        rows["image"] = []

    episode_return = 0.0
    for step in range(max_episode_steps):
        obs_vec = flatten_obs(time_step.observation)
        action = agent.act(
            torch.from_numpy(obs_vec),
            t0=(step == 0),
            eval_mode=True,
        )
        action = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        rows["obs"].append(obs_vec)
        rows["action"].append(action)
        if save_images:
            rows["image"].append(render(raw_env, task.domain, image_size))

        reward = 0.0
        discount = 1.0
        dmc_last = False
        for _ in range(action_repeat):
            time_step = env.step(action)
            reward += float(time_step.reward or 0.0)
            discount = float(1.0 if time_step.discount is None else time_step.discount)
            dmc_last = bool(time_step.last())
            if dmc_last:
                break

        timeout = (step + 1) >= max_episode_steps and not dmc_last
        is_last = dmc_last or timeout
        terminated = dmc_last and discount == 0.0

        episode_return += reward
        rows["reward"].append(np.array([reward], dtype=np.float32))
        rows["discount"].append(np.array([discount], dtype=np.float32))
        rows["is_last"].append(np.array([is_last], dtype=np.uint8))
        rows["terminated"].append(np.array([terminated], dtype=np.uint8))
        rows["timeout"].append(np.array([timeout], dtype=np.uint8))
        if is_last:
            break

    return {key: np.stack(values, axis=0) for key, values in rows.items()}, episode_return


def collect_task(args: argparse.Namespace, task: TaskSpec, checkpoint_path: Path) -> dict[str, Any]:
    _schema_raw_env, schema_env, raw_action_spec, action_spec = make_env(task, args.seed)
    first_step = schema_env.reset()
    first_obs = dict(first_step.observation)
    obs_dim = int(flatten_obs(first_obs).shape[0])
    action_dim = int(np.prod(action_spec.shape))

    agent = load_agent(
        args.tdmpc2_root,
        checkpoint_path,
        task,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        expert=args.expert,
    )

    store_path = args.output_dir / f"{task.zarr_name}.zarr"
    root = open_store(
        store_path,
        zarr_attrs(
            task,
            checkpoint_path,
            args.checkpoint_seed,
            first_obs,
            obs_dim,
            action_spec,
            raw_action_spec,
            args,
        ),
        resume=args.resume,
    )

    completed = int(root["episode_length"].shape[0])
    returns = []
    print(
        f"Collecting {task.dmc_name}: {completed}/{args.num_episodes} episodes already present, "
        f"writing to {store_path}"
    )
    for episode_idx in range(completed, args.num_episodes):
        episode_seed = int(args.seed) + int(episode_idx)
        raw_env, env, _, _ = make_env(task, episode_seed)
        episode, episode_return = collect_episode(
            raw_env,
            env,
            agent,
            task,
            save_images=args.save_images,
            image_size=args.image_size,
            action_repeat=args.action_repeat,
            max_episode_steps=args.max_episode_steps,
        )
        append_episode(root, episode)
        returns.append(float(episode_return))
        episode_num = episode_idx + 1
        progress_every = max(int(args.progress_every), 1)
        should_log = (
            episode_num == completed + 1
            or episode_num == args.num_episodes
            or episode_num % progress_every == 0
        )
        if should_log:
            recent = returns[-progress_every:]
            recent_mean = float(np.mean(recent)) if recent else float("nan")
            print(
                f"{task.dmc_name} {episode_num}/{args.num_episodes}: "
                f"last_return={episode_return:.3f}, recent_mean={recent_mean:.3f}, "
                f"rows={int(root['obs'].shape[0])}"
            )

    return {
        "domain_name": task.domain,
        "task_name": task.task,
        "task_slug": task.slug,
        "zarr_path": str(store_path),
        "checkpoint_path": str(checkpoint_path),
        "episodes": int(root["episode_length"].shape[0]),
        "rows": int(root["obs"].shape[0]) if "obs" in root else 0,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "mean_new_return": float(np.mean(returns)) if returns else None,
    }


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    collected: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
):
    config = vars(args).copy()
    config["tdmpc2_root"] = str(args.tdmpc2_root)
    config["output_dir"] = str(args.output_dir)
    config["config"] = str(args.config)
    manifest = {
        "format": "dmc_expert_collection_manifest_v1",
        "checkpoint_repo": CHECKPOINT_REPO,
        "config": config,
        "collected": collected,
        "skipped": skipped,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = discover_tasks()
    selected_tasks = select_tasks(all_tasks, cfg.tasks)
    checkpoint_files = list_checkpoints()

    collected = []
    skipped = []
    for task in selected_tasks:
        expected = checkpoint_name(task, cfg.checkpoint_seed)
        if expected not in checkpoint_files:
            skipped.append(
                {
                    "domain_name": task.domain,
                    "task_name": task.task,
                    "task_slug": task.slug,
                    "reason": f"missing checkpoint {expected}",
                }
            )
            print(f"Skipping {task.dmc_name}: missing checkpoint {expected}")
            continue

        checkpoint_path = download_checkpoint(task, cfg.checkpoint_seed)
        collected.append(collect_task(cfg, task, checkpoint_path))
        write_manifest(cfg.output_dir, cfg, collected, skipped)

    write_manifest(cfg.output_dir, cfg, collected, skipped)


if __name__ == "__main__":
    main()
