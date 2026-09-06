from typing import ClassVar

import gymnasium as gym
import numpy as np

from .parallel import ParallelEnv

GOAL_RELATION_KEY = "goal_relation"


def goal_relation_spec(physics, domain, task):
    name = f"{domain}/{task}"
    if name == "cartpole/balance_sparse":
        tolerance = [0.25, float(np.arccos(0.995))]
        relation_name, geometry = "cart_and_pole_to_balance", "box"
    elif domain == "point_mass":
        tolerance = [float(physics.named.model.geom_size["target", 0])]
        relation_name, geometry = "mass_to_target", "radial"
    elif domain == "reacher":
        tolerance = [float(physics.named.model.geom_size["finger", 0] + physics.named.model.geom_size["target", 0])]
        relation_name, geometry = "finger_to_target", "radial"
    elif domain == "ball_in_cup":
        target = np.asarray(physics.named.model.site_size["target"], dtype=np.float32)[[0, 2]]
        ball = float(physics.named.model.geom_size["ball", 0])
        tolerance = (target - ball).tolist()
        relation_name, geometry = "ball_to_target", "box"
    else:
        return None
    return {"name": relation_name, "geometry": geometry, "shape": [2], "tolerance": tolerance}


def goal_relation(physics, domain, task):
    name = f"{domain}/{task}"
    if name == "cartpole/balance_sparse":
        angle = float(physics.named.data.qpos["hinge_1"])
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        return np.asarray([physics.cart_position(), angle], dtype=np.float32)

    method = {
        "point_mass": "mass_to_target",
        "reacher": "finger_to_target",
        "ball_in_cup": "ball_to_target",
    }.get(domain)
    if method is None:
        return None
    relation = np.asarray(getattr(physics, method)(), dtype=np.float32).reshape(-1)
    if domain == "point_mass":
        relation = relation[:2]
    if relation.size != 2:
        raise RuntimeError(f"{name} returned a {relation.size}D goal relation; expected 2D.")
    return relation


class DeepMindControl(gym.Env):
    metadata: ClassVar[dict] = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        max_steps=None,
        proprio=None,
        include_goal_relation=False,
    ):
        from dm_control import suite

        if int(action_repeat) < 1:
            raise ValueError("action_repeat must be positive.")
        domain, task = {f"{domain}_{task}": (domain, task) for domain, task in suite.ALL_TASKS}[name]
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._domain = domain
        self._task = task
        self._action_repeat = int(action_repeat)
        self._size = size
        self._max_steps = max_steps
        self._episode_step = None
        self._proprio = {str(key): np.asarray(indices, dtype=np.int64) for key, indices in (proprio or {}).items()}
        self._include_goal_relation = bool(include_goal_relation)
        if camera is None:
            camera = {"quadruped": 2, "fish": 3}.get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

        spec = self._env.action_spec()
        minimum = np.asarray(spec.minimum, dtype=np.float32)
        maximum = np.asarray(spec.maximum, dtype=np.float32)
        self._action_mask = np.isfinite(minimum) & np.isfinite(maximum)
        self._action_low = np.where(self._action_mask, minimum, -1)
        self._action_high = np.where(self._action_mask, maximum, 1)
        low = np.where(self._action_mask, -1, self._action_low)
        high = np.where(self._action_mask, 1, self._action_high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def observation_space(self):
        spaces = {"image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)}
        if self._proprio:
            size = sum(len(indices) for indices in self._proprio.values())
            spaces["proprio"] = gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        if self._include_goal_relation:
            spaces[GOAL_RELATION_KEY] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        if self._episode_step is None:
            raise RuntimeError("Must reset environment before stepping it.")
        normalized = np.clip(action, self.action_space.low, self.action_space.high)
        action = (normalized + 1) / 2 * (self._action_high - self._action_low) + self._action_low
        action = np.where(self._action_mask, action, normalized)
        if not np.isfinite(action).all():
            raise ValueError(f"Environment received a non-finite action: {action}")

        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break

        info = {"discount": np.array(time_step.discount, np.float32)}
        obs = self._observation(time_step)
        self._episode_step += 1
        timed_out = self._max_steps is not None and self._episode_step >= self._max_steps
        done = bool(time_step.last() or timed_out)
        if timed_out:
            obs["is_last"] = np.asarray(True)
        if done:
            self._episode_step = None
        return obs, np.float32(reward), done, info

    def reset(self, **kwargs):
        self._episode_step = 0
        return self._observation(self._env.reset())

    def _observation(self, time_step):
        observation = {
            "image": np.asarray(self.render(), dtype=np.uint8),
            "is_terminal": np.asarray(not time_step.first() and time_step.discount == 0),
            "is_first": np.asarray(time_step.first()),
            "is_last": np.asarray(time_step.last()),
        }
        if self._proprio:
            observation["proprio"] = np.concatenate([
                np.asarray(time_step.observation[key], dtype=np.float32).reshape(-1)[indices]
                for key, indices in self._proprio.items()
            ])
        if self._include_goal_relation:
            observation[GOAL_RELATION_KEY] = goal_relation(self._env.physics, self._domain, self._task)
        return observation

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


def make_env(config, seed, include_goal_relation=False):
    proprio = config.get("proprio")
    return DeepMindControl(
        str(config.task).removeprefix("dmc_"),
        config.action_repeat,
        config.size,
        seed=seed,
        max_steps=config.time_limit // config.action_repeat,
        proprio=proprio,
        include_goal_relation=include_goal_relation,
    )


def _make_parallel_envs(config, env_num, seed, include_goal_relation=False):
    def constructor(index):
        return lambda: make_env(config, int(seed) + index, include_goal_relation)

    return ParallelEnv(constructor, env_num, pin_memory=str(config.device).startswith("cuda"))


def make_envs(config, seed=None):
    seed = int(config.seed) if seed is None else int(seed)
    return _make_parallel_envs(config, config.env_num, seed, bool(config.get("goal_relation", False)))


def make_eval_envs(config):
    if int(config.eval_episode_num) <= 0:
        return None
    return _make_parallel_envs(config, config.eval_episode_num, config.eval_seed)


def close_envs(envs):
    if envs is not None:
        envs.close()
