"""Shared rollout and replay plumbing for latent planning model families."""

from collections import deque

import torch

from buffer import SequenceBuffer
from dmc_expert.replay import DMCExpertTransitionReplay
from envs.dmc import GOAL_RELATION_KEY

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
    planner_state = {
        name: getattr(model, name)
        for name in ("_previous_mean", "_cem_mean", "_gradient_actions")
        if hasattr(model, name)
    }
    model.eval()
    try:
        obs = envs.reset().to(model.device, non_blocking=True)
        finished = torch.zeros(envs.env_num, dtype=torch.bool, device=model.device)
        returns = torch.zeros(envs.env_num, device=model.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=model.device)
        successes = torch.zeros(envs.env_num, dtype=torch.bool, device=model.device)
        sustained_successes = torch.zeros_like(successes)
        success_streak = torch.zeros(envs.env_num, dtype=torch.int32, device=model.device)
        success_threshold = float(config.evaluation.success_threshold)
        sustained_steps = int(config.evaluation.sustained_success_steps)
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
            qualifies = (reward[:, 0] / action_repeat >= success_threshold) & active
            successes |= qualifies
            success_streak = torch.where(qualifies, success_streak + 1, torch.where(active, 0, success_streak))
            sustained_successes |= success_streak >= sustained_steps
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
                "sustained_success": sustained_successes.float().mean(),
                "return_std": return_std,
                "return_stderr": return_std / returns.numel() ** 0.5,
            },
        )
    finally:
        for name, value in planner_state.items():
            setattr(model, name, value)
        model.train(was_training)


class OnlineSession:
    def __init__(self, config, model, envs):
        self.model = model
        self.envs = envs
        self.replay = SequenceBuffer(config.replay)
        self.action_repeat = int(config.env.action_repeat)

    @staticmethod
    def _split_goal_relation(obs):
        if GOAL_RELATION_KEY not in obs.keys():
            return obs, None
        return obs.exclude(GOAL_RELATION_KEY), obs[GOAL_RELATION_KEY]

    def start(self):
        self.replay.start(self.envs.env_num)
        obs, relation = self._split_goal_relation(self.envs.reset())
        if getattr(self.model, "goal_conditioned", False) and relation is None:
            raise ValueError("Goal-conditioned online training requires DMC goal-relation labels.")
        self.obs = obs.to(self.model.device, non_blocking=True)
        self.goal_relation = relation.to(self.model.device, non_blocking=True) if relation is not None else None
        size = max(self.model.history_size - 1, 1)
        self.obs_history = [deque(maxlen=size) for _ in range(self.envs.env_num)]
        self.action_history = [deque(maxlen=size) for _ in range(self.envs.env_num)]
        self.first = torch.ones(self.envs.env_num, dtype=torch.bool, device=self.model.device)
        self.returns = torch.zeros(self.envs.env_num, device=self.model.device)
        self.lengths = torch.zeros(self.envs.env_num, dtype=torch.int32, device=self.model.device)

    def collect(self):
        history, past_action = build_context(
            self.obs,
            self.obs_history,
            self.action_history,
            self.model.history_size,
            self.model.action_dim,
        )
        action = self.model.act(history, past_action, first=self.first)
        next_obs, reward, done = self.envs.step(action)
        terminal = next_obs["is_terminal"].to(self.model.device, non_blocking=True).reshape(-1)
        reward = reward.to(self.model.device, non_blocking=True)
        model_done = done.to(self.model.device, non_blocking=True)
        self.replay.append(
            self.model.replay_observation(history),
            action,
            reward,
            terminal,
            model_done,
            self.goal_relation,
        )
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
        if done.any():
            next_obs = self.envs.reset_done(next_obs, done)
        next_obs, relation = self._split_goal_relation(next_obs)
        self.obs = next_obs.to(self.model.device, non_blocking=True)
        self.goal_relation = relation.to(self.model.device, non_blocking=True) if relation is not None else None
        self.first = model_done
        return self.envs.env_num * self.action_repeat, episodes

    def update(self, update_count):
        metrics = {}
        for _ in range(update_count):
            obs, action, reward, terminal, *extra = self.replay.sample(sequence_length=self.model.sequence_length)
            metrics = self.model.update((obs, action[:, :-1], reward[:, :-1], terminal[:, :-1], *extra))
        return metrics
