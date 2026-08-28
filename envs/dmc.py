from typing import ClassVar

import gymnasium as gym
import numpy as np

from .parallel import ParallelEnv


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
    ):
        from dm_control import suite

        if int(action_repeat) < 1:
            raise ValueError("action_repeat must be positive.")
        domain, task = {f"{domain}_{task}": (domain, task) for domain, task in suite.ALL_TASKS}[name]
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._action_repeat = int(action_repeat)
        self._size = size
        self._max_steps = max_steps
        self._episode_step = None
        self._proprio = {str(key): np.asarray(indices, dtype=np.int64) for key, indices in (proprio or {}).items()}
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
        return observation

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


def make_env(config, seed):
    proprio = config.get("proprio")
    return DeepMindControl(
        str(config.task).removeprefix("dmc_"),
        config.action_repeat,
        config.size,
        seed=seed,
        max_steps=config.time_limit // config.action_repeat,
        proprio=proprio,
    )


def _make_parallel_envs(config, env_num, seed):
    def constructor(index):
        return lambda: make_env(config, int(seed) + index)

    return ParallelEnv(constructor, env_num, pin_memory=str(config.device).startswith("cuda"))


def make_envs(config):
    return _make_parallel_envs(config, config.env_num, config.seed)


def make_eval_envs(config):
    if int(config.eval_episode_num) <= 0:
        return None
    return _make_parallel_envs(config, config.eval_episode_num, config.eval_seed)


def close_envs(envs):
    if envs is not None:
        envs.close()
