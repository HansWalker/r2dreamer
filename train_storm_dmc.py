"""STORM-native training for continuous DMC control."""

from __future__ import annotations

import atexit
import math
import pathlib
import sys
import time
import warnings
from collections import deque

import hydra
import numpy as np
import torch
from tensordict import TensorDict

import tools
from envs import make_envs
from models.storm.native import ActorCriticAgent, WorldModel
from offline_replay import _DMCExpertDataset

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


def close_envs(envs):
    if envs is None:
        return
    close = getattr(envs, "close", None)
    if callable(close):
        close()
        return
    for env in getattr(envs, "envs", []):
        close = getattr(env, "close", None)
        if callable(close):
            close()


def stack_obs(rows, device="cpu"):
    tensors = {key: torch.as_tensor(np.stack([row[key] for row in rows]), device="cpu") for key in rows[0].keys()}
    td = TensorDict(tensors, batch_size=(len(rows),), device="cpu")
    for key in td.keys():
        if td[key].ndim == 1:
            td[key] = td[key].unsqueeze(-1)
    if str(device).startswith("cuda"):
        return td.pin_memory().to(device, non_blocking=True)
    return td.to(device)


def env_reset_all(envs, device):
    obs = [promise() for promise in [env.reset() for env in envs.envs]]
    return stack_obs(obs, device=device)


def env_reset_indices(envs, current_obs, done, device):
    done_cpu = tools.to_np(done).astype(bool)
    if not done_cpu.any():
        return current_obs
    rows = []
    promises = {idx: envs.envs[idx].reset() for idx, flag in enumerate(done_cpu) if flag}
    for idx, flag in enumerate(done_cpu):
        if flag:
            rows.append(promises[idx]())
        else:
            rows.append({key: tools.to_np(current_obs[key][idx]) for key in current_obs.keys()})
    return stack_obs(rows, device=device)


def env_step_all(envs, action, device):
    action_np = tools.to_np(action)
    promises = [env.step(act) for env, act in zip(envs.envs, action_np)]
    obs, rew, done = [], [], []
    for promise in promises:
        next_obs, reward, is_done, _ = promise()
        obs.append(next_obs)
        rew.append(reward)
        done.append(is_done)
    td = stack_obs(obs, device=device)
    reward = torch.as_tensor(rew, dtype=torch.float32, device=device).reshape(-1, 1)
    done = torch.as_tensor(done, dtype=torch.bool, device=device)
    return td, reward, done


def obs_to_cpu_dict(obs, idx):
    return {key: tools.to_np(obs[key][idx]).copy() for key in obs.keys()}


class ReplayBuffer:
    def __init__(self, batch_size, batch_length, device, seed=0):
        self.batch_size = int(batch_size)
        self.batch_length = int(batch_length)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(seed))
        self.completed = []
        self.current = None

    def start(self, env_num):
        self.current = [[] for _ in range(int(env_num))]

    def append(self, obs, action, reward, terminal):
        for idx in range(action.shape[0]):
            self.current[idx].append(
                {
                    "obs": obs_to_cpu_dict(obs, idx),
                    "action": tools.to_np(action[idx]).astype(np.float32).copy(),
                    "reward": np.asarray([float(reward[idx].item())], dtype=np.float32),
                    "terminal": np.asarray([bool(terminal[idx].item())], dtype=bool),
                }
            )
            if bool(terminal[idx].item()):
                self.completed.append(self.current[idx])
                self.current[idx] = []

    def episodes(self):
        eps = list(self.completed)
        if self.current is not None:
            eps.extend(ep for ep in self.current if len(ep) >= self.batch_length)
        return [ep for ep in eps if len(ep) >= self.batch_length]

    def ready(self):
        return len(self.episodes()) > 0

    def count(self):
        total = sum(len(ep) for ep in self.completed)
        if self.current is not None:
            total += sum(len(ep) for ep in self.current)
        return total

    def sample(self, batch_size=None, batch_length=None):
        batch_size = int(batch_size or self.batch_size)
        batch_length = int(batch_length or self.batch_length)
        episodes = [ep for ep in self.episodes() if len(ep) >= batch_length]
        if not episodes:
            raise RuntimeError(f"No replay episode has length >= {batch_length}.")
        rows = []
        for _ in range(batch_size):
            ep = episodes[int(self.rng.integers(0, len(episodes)))]
            start = int(self.rng.integers(0, len(ep) - batch_length + 1))
            rows.append(ep[start : start + batch_length])
        return self._stack(rows)

    def _stack(self, rows):
        obs = {}
        for key in rows[0][0]["obs"]:
            obs[key] = torch.as_tensor(np.stack([[step["obs"][key] for step in row] for row in rows]), device=self.device)
        action = torch.as_tensor(np.stack([[step["action"] for step in row] for row in rows]), device=self.device)
        reward = torch.as_tensor(np.stack([[step["reward"] for step in row] for row in rows]), device=self.device)
        terminal = torch.as_tensor(np.stack([[step["terminal"] for step in row] for row in rows]), device=self.device)
        return obs, action.float(), reward.float(), terminal.float()


class ExpertReplay(_DMCExpertDataset):
    def __init__(self, config, batch_length):
        super().__init__(config)
        self.batch_length = int(batch_length)
        self.episodes = self.complete_episodes
        self.num_episodes = int(len(self.episodes))
        self._episode_order = np.array([], dtype=np.int64)
        self._episode_pos = 0

    def _next_episode_indices(self):
        rows = []
        while len(rows) < self.batch_size:
            if self._episode_pos >= len(self._episode_order):
                self._episode_order = self.rng.permutation(self.episodes) if self.shuffle else self.episodes.copy()
                self._episode_pos = 0
            take = min(self.batch_size - len(rows), len(self._episode_order) - self._episode_pos)
            rows.extend(self._episode_order[self._episode_pos : self._episode_pos + take])
            self._episode_pos += take
        return np.asarray(rows, dtype=np.int64)

    def sample_episode_batch(self):
        indices = self._next_episode_indices()
        lengths = self.lengths[indices]
        length = int(lengths.min())
        rows = [self._make_window(ep_idx, 0, length) for ep_idx in indices]
        return self._stack_native(rows)

    def _make_window(self, ep_idx, start, length):
        ep_idx = int(ep_idx)
        start = int(start)
        length = int(length)
        end = start + length
        obs = np.asarray(self.observations[ep_idx, start:end], dtype=np.float32)
        actions = np.asarray(self.actions[ep_idx, start:end], dtype=np.float32)
        rewards = np.asarray(self.rewards[ep_idx, start:end], dtype=np.float32)
        terminations = np.asarray(self.terminations[ep_idx, start:end], dtype=bool)
        data = self._split_obs(obs)
        data.update(
            {
                "action": actions,
                "reward": rewards.reshape(length, 1),
                "terminal": terminations.reshape(length, 1),
            }
        )
        return data

    def _stack_native(self, rows):
        data = {}
        for key in rows[0]:
            data[key] = torch.as_tensor(np.stack([row[key] for row in rows], axis=0))
        obs = {key: data[key] for key in self.obs_keys}
        return obs, data["action"].float(), data["reward"].float(), data["terminal"].float()


def expert_updates(config, replay):
    requested = math.ceil(float(config.training.expert_epochs) * replay.num_episodes / replay.batch_size)
    minimum_epochs = float(getattr(config.expert, "min_epochs", 2.0))
    minimum = math.ceil(minimum_epochs * replay.num_episodes / replay.batch_size)
    return max(requested, minimum)


def monte_carlo_returns(reward, terminal, gamma):
    reward = reward.squeeze(-1)
    terminal = terminal.squeeze(-1)
    returns = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[0], dtype=reward.dtype, device=reward.device)
    for idx in reversed(range(reward.shape[1])):
        running = reward[:, idx] + float(gamma) * (1.0 - terminal[:, idx]) * running
        returns[:, idx] = running
    return returns.unsqueeze(-1)


def save_checkpoint(path, world_model, agent, **metadata):
    payload = {
        "world_model": world_model.state_dict(),
        "actor_critic": agent.state_dict(),
        "wm_optimizer": world_model.optimizer.state_dict(),
        "ac_optimizer": agent.optimizer.state_dict(),
        "wm_scaler": world_model.scaler.state_dict(),
        "ac_scaler": agent.scaler.state_dict(),
        **metadata,
    }
    torch.save(payload, path)


def format_expert_console(update, total, epoch, metrics, sec_per_update):
    eta = (total - update) * sec_per_update
    return (
        f"phase=expert | update={update}/{total} ({tools.format_percent(update, total)}) | "
        f"epoch={tools.format_scalar(epoch, 1)} | speed={tools.format_scalar(sec_per_update, 2)}s/update | "
        f"eta={tools.format_eta(eta)} | wm={tools.format_scalar(metrics.get('wm/loss'), 2)} | "
        f"bc={tools.format_scalar(metrics.get('expert/bc'), 2)} | value={tools.format_scalar(metrics.get('expert/value'), 2)}"
    )


def format_online_console(step, total, updates, fps, metrics, eval_score, best):
    eta = None if not fps else (total - step) / max(float(fps), 1e-6)
    return (
        f"phase=online | env_step={step}/{total} ({tools.format_percent(step, total)}) | "
        f"updates={updates} | speed={tools.format_scalar(fps, 0)}fps | eta={tools.format_eta(eta)} | "
        f"wm={tools.format_scalar(metrics.get('wm/loss'), 2)} | ac={tools.format_scalar(metrics.get('ac/loss'), 2)} | "
        f"eval={tools.format_scalar(eval_score, 1)} | best={tools.format_scalar(best, 1)}"
    )


def build_world_model(config, obs_space, act_space):
    obs_shapes = {key: tuple(space.shape) for key, space in obs_space.spaces.items()}
    action_dim = int(np.prod(act_space.shape))
    return WorldModel(obs_shapes, action_dim, config.storm_model).to(config.device)


def build_agent(config, feat_dim, act_space):
    action_dim = int(np.prod(act_space.shape))
    return ActorCriticAgent(feat_dim, action_dim, config.actor_critic).to(config.device)


def train_world_model_step(replay_buffer, world_model, batch_size, batch_length):
    return world_model.update(*replay_buffer.sample(batch_size=batch_size, batch_length=batch_length))


@torch.no_grad()
def world_model_imagine_data(
    replay_buffer,
    world_model,
    agent,
    imagine_batch_size,
    imagine_context_length,
    imagine_batch_length,
):
    world_model.eval()
    agent.eval()
    obs, action, _, _ = replay_buffer.sample(batch_size=imagine_batch_size, batch_length=imagine_context_length)
    obs = {key: value.to(world_model.device, non_blocking=True) for key, value in obs.items()}
    action = action.to(world_model.device, non_blocking=True)
    return world_model.imagine_data(
        agent,
        obs,
        action,
        imagine_batch_length=imagine_batch_length,
    )


def train_agent_step(replay_buffer, world_model, agent, config):
    latent, action, reward, termination = world_model_imagine_data(
        replay_buffer,
        world_model,
        agent,
        imagine_batch_size=int(config.storm_train.imagine_batch_size),
        imagine_context_length=int(config.storm_train.imagine_context_length),
        imagine_batch_length=int(config.storm_train.imagine_horizon),
    )
    return agent.update(latent, action, reward=reward, termination=termination)


def train_expert_step(expert_replay, world_model, agent):
    batch = expert_replay.sample_episode_batch()
    wm_metrics, state, (_, action, reward, terminal) = world_model.update(*batch)
    returns = monte_carlo_returns(reward, terminal, agent.gamma)
    ac_metrics = agent.update_expert(state["feat"], action.to(agent.device), returns)
    return {**wm_metrics, **ac_metrics}


def pretrain_world_model_agent(config, logger, logdir, expert_replay, world_model, agent):
    updates = expert_updates(config, expert_replay)
    start = time.perf_counter()
    metrics = {}
    for update in range(1, updates + 1):
        metrics = train_expert_step(expert_replay, world_model, agent)
        if update % int(config.expert.log_every) == 0 or update == updates:
            sec = (time.perf_counter() - start) / update
            epoch = update * expert_replay.batch_size / expert_replay.num_episodes
            for key, value in metrics.items():
                logger.scalar(f"train/{key}", tools.scalar_float(value))
            logger.scalar("train/expert/update", update)
            logger.scalar("train/expert/epoch", epoch)
            logger.write(update, console_message=format_expert_console(update, updates, epoch, metrics, sec))
        if update % int(config.expert.save_every) == 0:
            save_checkpoint(logdir / "pretrained_latest.pt", world_model, agent, update=update)
    save_checkpoint(logdir / "pretrained.pt", world_model, agent, update=updates)


def context_obs_item(obs, idx):
    return {key: obs[key][idx : idx + 1].unsqueeze(1) for key in obs.keys()}


def stack_context_obs(context_obs, device):
    keys = list(context_obs[0].keys())
    return {
        key: torch.cat([item[key] for item in context_obs], dim=1).to(device)
        for key in keys
    }


@torch.no_grad()
def act_from_context(world_model, agent, context_obs, context_action, obs, deterministic=False):
    if not context_action:
        return torch.empty(obs.batch_size[0], agent.action_dim, device=world_model.device).uniform_(-1, 1)
    obs_batch = stack_context_obs(context_obs, world_model.device)
    action_batch = torch.cat(list(context_action), dim=1).to(world_model.device)
    feat, _, _ = world_model.context_feature(obs_batch, action_batch)
    action, _ = agent.sample(feat, deterministic=deterministic)
    return action


def evaluate_policy(config, logger, world_model, agent, eval_envs, step, last_best):
    if eval_envs is None or eval_envs.env_num == 0:
        return None, last_best, False
    print("Evaluating the policy...")
    world_model.eval()
    agent.eval()
    obs = env_reset_all(eval_envs, world_model.device)
    done_once = torch.zeros(eval_envs.env_num, dtype=torch.bool, device=world_model.device)
    returns = torch.zeros(eval_envs.env_num, dtype=torch.float32, device=world_model.device)
    lengths = torch.zeros(eval_envs.env_num, dtype=torch.int32, device=world_model.device)
    context_obs = [deque(maxlen=int(config.storm_train.context_length)) for _ in range(eval_envs.env_num)]
    context_action = [deque(maxlen=int(config.storm_train.context_length)) for _ in range(eval_envs.env_num)]
    while not done_once.all():
        actions = []
        for idx in range(eval_envs.env_num):
            actions.append(
                act_from_context(
                    world_model,
                    agent,
                    context_obs[idx],
                    context_action[idx],
                    obs[idx : idx + 1],
                    deterministic=True,
                )
            )
        action = torch.cat(actions, dim=0)
        next_obs, reward, done = env_step_all(eval_envs, action, world_model.device)
        active = ~done_once
        returns += reward[:, 0] * active
        lengths += active
        for idx in range(eval_envs.env_num):
            if not bool(done_once[idx].item()):
                context_obs[idx].append(context_obs_item(obs, idx))
                context_action[idx].append(action[idx : idx + 1].unsqueeze(1))
        done_once |= done
        if done.any():
            for idx, flag in enumerate(done):
                if bool(flag.item()):
                    context_obs[idx].clear()
                    context_action[idx].clear()
            obs = env_reset_indices(eval_envs, next_obs, done, world_model.device)
        else:
            obs = next_obs

    score = float(returns.mean())
    length = float(lengths.to(torch.float32).mean())
    improved = last_best is None or score > last_best
    best = score if improved else last_best
    logger.scalar("episode/eval_score", score)
    logger.scalar("episode/eval_length", length)
    logger.write(
        step,
        console_message=(
            f"phase=eval | env_step={step} | score={tools.format_scalar(score, 1)} | "
            f"length={tools.format_scalar(length, 0)} | best={tools.format_scalar(best, 1)}"
        ),
    )
    return score, best, improved


def joint_train_world_model_agent(config, logger, logdir, expert_replay, world_model, agent):
    train_envs, eval_envs, _, _ = make_envs(config.env)
    replay_buffer = ReplayBuffer(
        config.storm_train.batch_size,
        config.storm_train.batch_length,
        world_model.device,
        seed=config.seed,
    )
    replay_buffer.start(train_envs.env_num)
    try:
        obs = env_reset_all(train_envs, world_model.device)
        context_obs = [deque(maxlen=int(config.storm_train.context_length)) for _ in range(train_envs.env_num)]
        context_action = [deque(maxlen=int(config.storm_train.context_length)) for _ in range(train_envs.env_num)]
        returns = torch.zeros(train_envs.env_num, dtype=torch.float32, device=world_model.device)
        lengths = torch.zeros(train_envs.env_num, dtype=torch.int32, device=world_model.device)
        step = 0
        updates = 0
        last_log_step = 0
        last_log_time = time.perf_counter()
        last_eval_score = None
        best_eval_score = None
        metrics = {}

        while step < int(config.training.online_steps):
            actions = []
            for idx in range(train_envs.env_num):
                actions.append(
                    act_from_context(
                        world_model,
                        agent,
                        context_obs[idx],
                        context_action[idx],
                        obs[idx : idx + 1],
                        deterministic=False,
                    )
                )
            action = torch.cat(actions, dim=0)
            next_obs, reward, done = env_step_all(train_envs, action, world_model.device)
            replay_buffer.append(obs, action, reward, done)
            for idx in range(train_envs.env_num):
                context_obs[idx].append(context_obs_item(obs, idx))
                context_action[idx].append(action[idx : idx + 1].unsqueeze(1))
            returns += reward[:, 0]
            lengths += 1
            step += train_envs.env_num * int(config.env.action_repeat)

            if done.any():
                for idx, flag in enumerate(done):
                    if bool(flag.item()):
                        logger.scalar("episode/score", returns[idx])
                        logger.scalar("episode/length", lengths[idx])
                        logger.write(step)
                        returns[idx] = 0
                        lengths[idx] = 0
                        context_obs[idx].clear()
                        context_action[idx].clear()
                obs = env_reset_indices(train_envs, next_obs, done, world_model.device)
            else:
                obs = next_obs

            if replay_buffer.ready() and replay_buffer.count() >= int(config.storm_train.warmup_steps):
                for _ in range(int(config.storm_train.world_model_updates)):
                    wm_metrics, _, _ = train_world_model_step(
                        replay_buffer,
                        world_model,
                        batch_size=int(config.storm_train.batch_size),
                        batch_length=int(config.storm_train.batch_length),
                    )
                    metrics.update(wm_metrics)
                    updates += 1
                for _ in range(int(config.storm_train.actor_critic_updates)):
                    metrics.update(train_agent_step(replay_buffer, world_model, agent, config))

            eval_every = int(config.storm_train.eval_every)
            if eval_every and step % eval_every < train_envs.env_num * int(config.env.action_repeat):
                last_eval_score, best_eval_score, improved = evaluate_policy(
                    config, logger, world_model, agent, eval_envs, step, best_eval_score
                )
                save_checkpoint(logdir / "latest.pt", world_model, agent, update=updates, online_step=step)
                if improved:
                    save_checkpoint(logdir / "best.pt", world_model, agent, update=updates, online_step=step)

            log_every = int(config.storm_train.log_every)
            if log_every and step - last_log_step >= log_every:
                now = time.perf_counter()
                fps = (step - last_log_step) / max(now - last_log_time, 1e-6)
                last_log_step = step
                last_log_time = now
                for key, value in metrics.items():
                    logger.scalar(f"train/{key}", tools.scalar_float(value))
                logger.scalar("train/opt/updates", updates)
                logger.scalar("replay/size", replay_buffer.count())
                logger.write(
                    step,
                    console_message=format_online_console(
                        step,
                        int(config.training.online_steps),
                        updates,
                        fps,
                        metrics,
                        last_eval_score,
                        best_eval_score,
                    ),
                )

        save_checkpoint(logdir / "latest.pt", world_model, agent, update=updates, online_step=step)
    finally:
        close_envs(train_envs)
        close_envs(eval_envs)


def run(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())
    print("Logdir", logdir)
    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    expert_replay = ExpertReplay(config.expert, config.storm_train.batch_length)
    try:
        world_model = build_world_model(config, expert_replay.obs_space(), expert_replay.act_space())
        agent = build_agent(config, world_model.feat_size, expert_replay.act_space())
        print(f"World model params: {sum(p.numel() for p in world_model.parameters()):,}")
        print(f"Actor critic params: {sum(p.numel() for p in agent.parameters()):,}")
        if bool(config.expert.enabled):
            pretrain_world_model_agent(config, logger, logdir, expert_replay, world_model, agent)
        if int(config.training.online_steps) > 0:
            joint_train_world_model_agent(config, logger, logdir, expert_replay, world_model, agent)
    finally:
        close = getattr(expert_replay, "close", None)
        if callable(close):
            close()


@hydra.main(version_base=None, config_path="configs", config_name="storm_dmc_transformer")
def main(config):
    run(config)


if __name__ == "__main__":
    main()
