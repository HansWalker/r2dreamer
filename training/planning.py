"""Shared rollout and replay plumbing for latent planning model families."""

from collections import deque

import torch

from buffer import SequenceBuffer
from dmc_expert.replay import DMCExpertTransitionReplay

ExpertReplay = DMCExpertTransitionReplay


def checkpoint(model):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer_state_dict(),
    }


def load_checkpoint(model, payload, training=True):
    model.load_state_dict(payload["model_state_dict"])
    if training:
        model.load_optimizer_state_dict(payload["optimizer_state_dict"])


def expert_update(model, batch):
    return model.update(batch)


def build_context(obs, obs_history, action_history, history_size, action_dim):
    sample = next(iter(obs.values()))
    rows = []
    actions = []
    for index in range(sample.shape[0]):
        current = {key: obs[key][index] for key in obs.keys()}
        states = [*obs_history[index], current][-history_size:]
        states = [states[0]] * (history_size - len(states)) + states
        rows.append({key: torch.stack([state[key] for state in states]) for key in current})

        action_count = max(history_size - 1, 0)
        past = list(action_history[index])[-action_count:] if action_count else []
        zero = torch.zeros(action_dim, device=sample.device)
        past = [zero] * (action_count - len(past)) + past
        actions.append(torch.stack(past) if past else zero.new_empty((0, action_dim)))
    return (
        {key: torch.stack([row[key] for row in rows]) for key in rows[0]},
        torch.stack(actions),
    )


@torch.no_grad()
def evaluate(config, model, envs):
    was_training = model.training
    model.eval()
    try:
        obs = envs.reset().to(model.device, non_blocking=True)
        finished = torch.zeros(envs.env_num, dtype=torch.bool, device=model.device)
        returns = torch.zeros(envs.env_num, device=model.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=model.device)
        successes = torch.zeros(envs.env_num, dtype=torch.bool, device=model.device)
        success_threshold = float(config.evaluation.success_threshold)
        action_repeat = int(config.env.action_repeat)
        obs_history = [deque(maxlen=max(model.history_size - 1, 1)) for _ in range(envs.env_num)]
        action_history = [deque(maxlen=max(model.history_size - 1, 1)) for _ in range(envs.env_num)]
        first = torch.ones(envs.env_num, dtype=torch.bool, device=model.device)

        while not finished.all():
            history, past_action = build_context(
                obs,
                obs_history,
                action_history,
                model.history_size,
                model.action_dim,
            )
            action = model.act(history, past_action, deterministic=True, first=first)
            if not torch.isfinite(action).all():
                raise RuntimeError("Evaluation policy produced a non-finite action.")
            next_obs, reward, done = envs.step(action, reset_mask=finished)
            reward = reward.to(model.device, non_blocking=True)
            model_done = done.to(model.device, non_blocking=True)
            active = ~finished
            returns += reward[:, 0] * active
            lengths += active
            successes |= (reward[:, 0] / action_repeat >= success_threshold) & active
            for index in range(envs.env_num):
                if not finished[index]:
                    obs_history[index].append({key: obs[key][index].detach() for key in obs.keys()})
                    action_history[index].append(action[index].detach())
            finished |= model_done
            for index, flag in enumerate(model_done):
                if flag:
                    obs_history[index].clear()
                    action_history[index].clear()
            obs = (envs.reset_done(next_obs, done) if done.any() else next_obs).to(model.device, non_blocking=True)
            first = finished
        return_std = returns.std(unbiased=returns.numel() > 1)
        success = successes.float().mean()
        return (
            float(returns.mean()),
            float(lengths.float().mean()),
            {
                "success": success,
                "return_std": return_std,
                "return_stderr": return_std / returns.numel() ** 0.5,
            },
        )
    finally:
        model.train(was_training)


class OnlineSession:
    def __init__(self, config, model, envs):
        self.model = model
        self.envs = envs
        self.replay = SequenceBuffer(config.replay)
        self.action_repeat = int(config.env.action_repeat)
        self.warmup_steps = int(config.planning_train.warmup_steps)
        self.random_steps = int(config.planning_train.random_steps)
        self.updates_per_collect = int(config.planning_train.updates_per_collect)
        self.initial_updates = int(config.planning_train.initial_updates)

    def start(self, resumed=False, env_steps=0, world_model_updates=0):
        self.replay.start(self.envs.env_num)
        self.obs = self.envs.reset().to(self.model.device, non_blocking=True)
        size = max(self.model.history_size - 1, 1)
        self.obs_history = [deque(maxlen=size) for _ in range(self.envs.env_num)]
        self.action_history = [deque(maxlen=size) for _ in range(self.envs.env_num)]
        self.first = torch.ones(self.envs.env_num, dtype=torch.bool, device=self.model.device)
        self.returns = torch.zeros(self.envs.env_num, device=self.model.device)
        self.lengths = torch.zeros(self.envs.env_num, dtype=torch.int32, device=self.model.device)
        self.initial_updates_pending = not world_model_updates

    def collect(self):
        history, past_action = build_context(
            self.obs,
            self.obs_history,
            self.action_history,
            self.model.history_size,
            self.model.action_dim,
        )
        if self.replay.count() < self.random_steps:
            action = torch.empty(self.envs.env_num, self.model.action_dim, device=self.model.device).uniform_(-1, 1)
        else:
            action = self.model.act(history, past_action, first=self.first)
        next_obs, reward, done = self.envs.step(action)
        model_obs = next_obs.to(self.model.device, non_blocking=True)
        reward = reward.to(self.model.device, non_blocking=True)
        model_done = done.to(self.model.device, non_blocking=True)
        self.replay.append(self.obs, action, reward, model_obs["is_terminal"].reshape(-1), model_done)
        for index in range(self.envs.env_num):
            self.obs_history[index].append({key: self.obs[key][index].detach() for key in self.obs.keys()})
            self.action_history[index].append(action[index].detach())
        self.returns += reward[:, 0]
        self.lengths += 1

        episodes = []
        for index, flag in enumerate(model_done):
            if flag:
                episodes.append((self.returns[index], self.lengths[index]))
                self.returns[index] = self.lengths[index] = 0
                self.obs_history[index].clear()
                self.action_history[index].clear()
        self.obs = (self.envs.reset_done(next_obs, done) if done.any() else next_obs).to(
            self.model.device, non_blocking=True
        )
        self.first = model_done
        return self.envs.env_num * self.action_repeat, episodes

    def update(self, step):
        if self.replay.count() < self.warmup_steps or not self.replay.ready():
            return {}, 0
        update_count = self.initial_updates if self.initial_updates_pending else self.updates_per_collect
        self.initial_updates_pending = False
        metrics = {}
        for _ in range(update_count):
            obs, action, reward, terminal = self.replay.sample(sequence_length=self.model.sequence_length)
            metrics = self.model.update((obs, action[:, :-1], reward[:, :-1], terminal[:, :-1]))
        return metrics, update_count
