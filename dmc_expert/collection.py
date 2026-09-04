"""DMC expert collection orchestration."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .storage import (
    DATA_FORMAT,
    append_episode,
    completed_episodes,
    open_dataset,
    write_progress,
)
from .tdmpc2 import (
    CHECKPOINT_REPO,
    TaskSpec,
    checkpoint_name,
    discover_tasks,
    load_agent,
    resolve_checkpoint,
    select_tasks,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_config(name="dmc_expert_collection", overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=Path(name).stem, overrides=list(overrides))
    OmegaConf.resolve(config)
    return config


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in obs.values()],
        dtype=np.float32,
    )


def make_env(task: TaskSpec, seed: int, time_limit: float | None = None):
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    task_kwargs = {"random": seed}
    if time_limit is not None:
        task_kwargs["time_limit"] = float(time_limit)
    raw_env = suite.load(
        task.domain,
        task.task,
        task_kwargs=task_kwargs,
        visualize_reward=False,
    )
    raw_action_spec = raw_env.action_spec()
    env = action_scale.Wrapper(raw_env, minimum=-1.0, maximum=1.0)
    return raw_env, env, raw_action_spec, env.action_spec()


def _render(raw_env, domain: str, image_size: int) -> np.ndarray:
    camera_id = 2 if domain == "quadruped" else 0
    return raw_env.physics.render(image_size, image_size, camera_id=camera_id)


def _goal_relation_spec(physics, task: TaskSpec):
    if task.dmc_name == "cartpole/balance_sparse":
        tolerance = [0.25, float(np.arccos(0.995))]
        name, geometry = "cart_and_pole_to_balance", "box"
    elif task.domain == "point_mass":
        tolerance = [float(physics.named.model.geom_size["target", 0])]
        name, geometry = "mass_to_target", "radial"
    elif task.domain == "reacher":
        tolerance = [float(physics.named.model.geom_size["finger", 0] + physics.named.model.geom_size["target", 0])]
        name, geometry = "finger_to_target", "radial"
    elif task.domain == "ball_in_cup":
        target = np.asarray(physics.named.model.site_size["target"], dtype=np.float32)[[0, 2]]
        ball = float(physics.named.model.geom_size["ball", 0])
        tolerance = (target - ball).tolist()
        name, geometry = "ball_to_target", "box"
    else:
        return None
    return {"name": name, "geometry": geometry, "shape": [2], "tolerance": tolerance}


def _goal_relation(physics, task: TaskSpec):
    if task.dmc_name == "cartpole/balance_sparse":
        angle = float(physics.named.data.qpos["hinge_1"])
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        return np.asarray([physics.cart_position(), angle], dtype=np.float32)

    method = {
        "point_mass": "mass_to_target",
        "reacher": "finger_to_target",
        "ball_in_cup": "ball_to_target",
    }.get(task.domain)
    if method is None:
        return None
    relation = np.asarray(getattr(physics, method)(), dtype=np.float32).reshape(-1)
    if task.domain == "point_mass":
        relation = relation[:2]
    if relation.size != 2:
        raise RuntimeError(f"{task.dmc_name} returned a {relation.size}D goal relation; expected 2D.")
    return relation


def _dataset_metadata(
    task: TaskSpec,
    checkpoint_path: Path,
    obs: dict[str, np.ndarray],
    action_spec,
    raw_action_spec,
    goal_relation,
    config,
) -> dict[str, Any]:
    custom_checkpoint = task.dmc_name in config.checkpoints
    train_episodes = int(config.episodes.train)
    heldout_episodes = int(config.episodes.heldout)
    total_episodes = train_episodes + heldout_episodes
    return {
        "format": DATA_FORMAT,
        "env_type": "dmc",
        "domain_name": task.domain,
        "task_name": task.task,
        "task_slug": task.slug,
        "policy": "tdmpc2",
        "policy_mode": "mpc" if config.expert["mpc"] else "actor",
        "expert": OmegaConf.to_container(config.expert, resolve=True),
        "checkpoint_repo": None if custom_checkpoint else CHECKPOINT_REPO,
        "checkpoint_path": (
            str(checkpoint_path.resolve()) if custom_checkpoint else checkpoint_name(task, config.checkpoint_seed)
        ),
        "checkpoint_local_path": str(checkpoint_path),
        "checkpoint_seed": config.checkpoint_seed,
        "seed": config.seed,
        "episode_seed_rule": "seed + episode_index",
        "num_episodes": total_episodes,
        "episode_splits": {
            "train": [0, train_episodes],
            "heldout": [train_episodes, total_episodes],
        },
        "obs_dim": int(flatten_obs(obs).shape[0]),
        "action_dim": int(np.prod(action_spec.shape)),
        "observation_keys": list(obs.keys()),
        "observation_shapes": {key: list(np.asarray(value).shape or (1,)) for key, value in obs.items()},
        "action_min": np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1).tolist(),
        "action_max": np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1).tolist(),
        "raw_action_min": np.asarray(raw_action_spec.minimum, dtype=np.float32).reshape(-1).tolist(),
        "raw_action_max": np.asarray(raw_action_spec.maximum, dtype=np.float32).reshape(-1).tolist(),
        "action_repeat": config.action_repeat,
        "max_episode_steps": config.max_episode_steps,
        "time_limit": config.time_limit,
        "image_size": config.image_size,
        "goal_relation": goal_relation,
        "layout": (
            "episode-major dense arrays; observations/images/goal_relations[e, t], "
            "actions[e, t] -> observations/images/goal_relations[e, t + 1]"
        ),
        "data_file": "data.hdf5",
    }


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def seed_episode(seed: int):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_episode(raw_env, env, agent, task: TaskSpec, config):
    import torch

    time_step = env.reset()
    rows: dict[str, list[np.ndarray]] = {
        "observations": [flatten_obs(time_step.observation)],
        "images": [_render(raw_env, task.domain, config.image_size)],
        "actions": [],
        "rewards": [],
        "discounts": [],
        "terminations": [],
        "truncations": [],
    }
    relation = _goal_relation(raw_env.physics, task)
    if relation is not None:
        rows["goal_relations"] = [relation]
    episode_return = 0.0
    for step in range(config.max_episode_steps):
        action = agent.act(
            torch.from_numpy(rows["observations"][-1]),
            t0=(step == 0),
            eval_mode=True,
        )
        action = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        reward = 0.0
        discount = 1.0
        dmc_last = False
        for _ in range(config.action_repeat):
            time_step = env.step(action)
            reward += float(time_step.reward or 0.0)
            discount = float(1.0 if time_step.discount is None else time_step.discount)
            dmc_last = bool(time_step.last())
            if dmc_last:
                break

        terminated = dmc_last and discount == 0.0
        truncated = not terminated and (dmc_last or (step + 1) >= config.max_episode_steps)
        episode_return += reward
        rows["actions"].append(action)
        rows["rewards"].append(np.array([reward], dtype=np.float32))
        rows["discounts"].append(np.array([discount], dtype=np.float32))
        rows["terminations"].append(np.array([terminated], dtype=np.uint8))
        rows["truncations"].append(np.array([truncated], dtype=np.uint8))
        rows["observations"].append(flatten_obs(time_step.observation))
        rows["images"].append(_render(raw_env, task.domain, config.image_size))
        if relation is not None:
            rows["goal_relations"].append(_goal_relation(raw_env.physics, task))
        if dmc_last or truncated:
            break

    return {key: np.stack(values, axis=0) for key, values in rows.items()}, episode_return


def collect_task(config, task: TaskSpec, checkpoint_path: Path):
    schema_raw_env, schema_env, raw_action_spec, action_spec = make_env(task, config.seed, config.time_limit)
    try:
        first_obs = dict(schema_env.reset().observation)
        obs_dim = int(flatten_obs(first_obs).shape[0])
        action_dim = int(np.prod(action_spec.shape))
        goal_relation = _goal_relation_spec(schema_raw_env.physics, task)
    finally:
        schema_raw_env.close()

    print(f"Collection | task={task.dmc_name} | loading_checkpoint={checkpoint_path.name}")
    agent = load_agent(
        config,
        checkpoint_path,
        task,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    store_path = Path(config.output_dir).expanduser() / task.store_name
    metadata = _dataset_metadata(
        task,
        checkpoint_path,
        first_obs,
        action_spec,
        raw_action_spec,
        goal_relation,
        config,
    )
    target_episodes = int(metadata["num_episodes"])
    h5, metadata_path, data_path = open_dataset(
        store_path,
        metadata,
        resume=config.resume,
    )

    completed = completed_episodes(h5)
    final_episodes = completed
    final_rows = int(np.asarray(h5["lengths"][:completed], dtype=np.int64).sum())
    returns = []
    started = time.perf_counter()
    progress_every = max(int(config.progress_every), 1)
    print(
        f"Collection | task={task.dmc_name} | episodes={completed}/{target_episodes} | "
        f"output={store_path}"
    )
    write_progress(store_path, final_episodes, final_rows, target_episodes)
    try:
        for episode_idx in range(completed, target_episodes):
            episode_seed = int(config.seed) + episode_idx
            seed_episode(episode_seed)
            raw_env, env, _, _ = make_env(task, episode_seed, config.time_limit)
            try:
                episode, episode_return = collect_episode(raw_env, env, agent, task, config)
            finally:
                raw_env.close()

            append_episode(h5, episode_idx, episode, episode_return)
            h5.flush()
            final_episodes = episode_idx + 1
            final_rows += int(episode["actions"].shape[0])
            write_progress(store_path, final_episodes, final_rows, target_episodes)
            returns.append(float(episode_return))

            should_log = (
                final_episodes == completed + 1
                or final_episodes == target_episodes
                or final_episodes % progress_every == 0
            )
            if should_log:
                elapsed = time.perf_counter() - started
                sec_per_episode = elapsed / max(final_episodes - completed, 1)
                eta = (target_episodes - final_episodes) * sec_per_episode
                print(
                    f"Collection | task={task.dmc_name} | episodes={final_episodes}/{target_episodes} "
                    f"({100 * final_episodes / target_episodes:.0f}%) | "
                    f"return={episode_return:.2f} | recent_return={float(np.mean(returns[-progress_every:])):.2f} | "
                    f"transitions={final_rows} | speed={sec_per_episode:.2f}s/episode | "
                    f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
                )
    finally:
        h5.close()

    return {
        "domain_name": task.domain,
        "task_name": task.task,
        "task_slug": task.slug,
        "data_path": str(store_path),
        "hdf5_path": str(data_path),
        "metadata_path": str(metadata_path),
        "checkpoint_path": str(checkpoint_path),
        "episodes": final_episodes,
        "rows": final_rows,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "mean_new_return": float(np.mean(returns)) if returns else None,
    }


def collect(config):
    Path(config.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(discover_tasks(), config.tasks)
    return [
        collect_task(config, task, resolve_checkpoint(task, config.checkpoint_seed, config.checkpoints))
        for task in tasks
    ]
