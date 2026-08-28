"""STORM-specific operations used by the shared trainer."""

from collections import deque

import torch

from buffer import SequenceBuffer
from dmc_expert.replay import DMCExpertSequenceReplay
from models.storm import StormModel

ExpertReplay = DMCExpertSequenceReplay
EXPERT_METRICS = {"wm": "wm/loss", "bc": "expert/bc", "value": "expert/value"}
ONLINE_METRICS = {"wm": "wm/loss", "ac": "ac/loss"}

__all__ = [
    "EXPERT_METRICS",
    "ONLINE_METRICS",
    "ExpertReplay",
    "OnlineSession",
    "build_model",
    "checkpoint",
    "evaluate",
    "expert_update",
    "load_checkpoint",
]


def build_model(config):
    model = StormModel(config, config.model_io).to(config.device)
    print(f"World model params: {sum(parameter.numel() for parameter in model.world_model.parameters()):,}")
    print(f"Actor critic params: {sum(parameter.numel() for parameter in model.actor_critic.parameters()):,}")
    return model


def checkpoint(model):
    world_model = model.world_model
    agent = model.actor_critic
    return {
        "world_model": world_model.state_dict(),
        "actor_critic": agent.state_dict(),
        "wm_optimizer": world_model.optimizer.state_dict(),
        "ac_optimizer": agent.optimizer.state_dict(),
        "wm_scaler": world_model.scaler.state_dict(),
        "ac_scaler": agent.scaler.state_dict(),
        "agent_training_state": agent.training_state_dict(),
    }


def load_checkpoint(model, payload, training=True):
    world_model = model.world_model
    agent = model.actor_critic
    world_model.load_state_dict(payload["world_model"])
    agent.load_state_dict(payload["actor_critic"])
    if training:
        world_model.optimizer.load_state_dict(payload["wm_optimizer"])
        agent.optimizer.load_state_dict(payload["ac_optimizer"])
        if "wm_scaler" in payload:
            world_model.scaler.load_state_dict(payload["wm_scaler"])
        if "ac_scaler" in payload:
            agent.scaler.load_state_dict(payload["ac_scaler"])
        agent.load_training_state_dict(payload.get("agent_training_state"))


def expert_update(model, batch):
    world_model = model.world_model
    agent = model.actor_critic
    obs, action, reward, terminal, returns = batch
    wm_metrics, state, _ = world_model.update(obs, action, reward, terminal)
    feature = world_model.next_policy_features(state)
    ac_metrics = agent.update_expert(
        feature,
        action[:, 1:].to(agent.device),
        returns[:, 1:].to(agent.device),
    )
    return {**wm_metrics, **ac_metrics}


@torch.no_grad()
def act_from_context(model, context_obs, context_action, deterministic=False):
    world_model = model.world_model
    agent = model.actor_critic
    world_model.eval()
    agent.eval()
    action = torch.zeros(len(context_action), agent.action_dim, device=world_model.device)
    groups = {}
    for index, history in enumerate(context_action):
        groups.setdefault(len(history), []).append(index)

    for length, indices in groups.items():
        index = torch.as_tensor(indices, device=world_model.device)
        if not length:
            if not deterministic:
                action[index] = torch.empty(len(indices), agent.action_dim, device=world_model.device).uniform_(-1, 1)
            continue
        obs_batch = {
            key: torch.cat(
                [torch.cat([item[key] for item in context_obs[i]], dim=1) for i in indices],
                dim=0,
            ).to(world_model.device)
            for key in context_obs[indices[0]][0]
        }
        action_batch = torch.cat([torch.cat(list(context_action[i]), dim=1) for i in indices], dim=0).to(
            world_model.device
        )
        feature = world_model.context_feature(obs_batch, action_batch)
        action[index], _ = agent.sample(feature, deterministic=deterministic)
    return action


class StormPolicyContext:
    """Fixed history for Transformer, persistent state for streaming cores."""

    def __init__(self, model, batch_size, context_length):
        self.model = model
        self.batch_size = int(batch_size)
        world_model = model.world_model
        self.streaming = bool(getattr(world_model.sequence_core, "streaming", False))
        if self.streaming:
            self.feature = torch.zeros(self.batch_size, world_model.feat_size, device=world_model.device)
            self.ready = torch.zeros(self.batch_size, dtype=torch.bool, device=world_model.device)
            self.cache = None
        else:
            self.obs = [deque(maxlen=int(context_length)) for _ in range(self.batch_size)]
            self.action = [deque(maxlen=int(context_length)) for _ in range(self.batch_size)]

    @torch.no_grad()
    def act(self, deterministic=False):
        world_model = self.model.world_model
        agent = self.model.actor_critic
        world_model.eval()
        agent.eval()
        if not self.streaming:
            return act_from_context(self.model, self.obs, self.action, deterministic=deterministic)

        action = torch.zeros(self.batch_size, agent.action_dim, device=world_model.device)
        if self.ready.any():
            index = self.ready.nonzero(as_tuple=False).flatten()
            action[index], _ = agent.sample(self.feature[index], deterministic=deterministic)
        if not deterministic and (~self.ready).any():
            action[~self.ready].uniform_(-1, 1)
        return action

    @torch.no_grad()
    def advance(self, obs, action, active=None, reset=None):
        device = self.model.world_model.device
        active = torch.ones(self.batch_size, dtype=torch.bool, device=device) if active is None else active.to(device)
        reset = torch.zeros_like(active) if reset is None else reset.to(device)
        if not self.streaming:
            for index in range(self.batch_size):
                if active[index]:
                    self.obs[index].append({key: obs[key][index : index + 1].unsqueeze(1) for key in obs.keys()})
                    self.action[index].append(action[index : index + 1].unsqueeze(1))
                if reset[index]:
                    self.obs[index].clear()
                    self.action[index].clear()
            return

        self.feature, self.cache = self.model.world_model.context_step(obs, action, self.cache)
        clear = ~active | reset
        self.feature = torch.where(clear.unsqueeze(-1), torch.zeros_like(self.feature), self.feature)
        self.cache = self.model.world_model.sequence_core.reset_cache(self.cache, clear)
        self.ready = active & ~reset


@torch.no_grad()
def evaluate(config, model, envs):
    world_model = model.world_model
    agent = model.actor_critic
    wm_training, agent_training = world_model.training, agent.training
    world_model.eval()
    agent.eval()
    try:
        obs = envs.reset().to(world_model.device, non_blocking=True)
        finished = torch.zeros(envs.env_num, dtype=torch.bool, device=world_model.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=world_model.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=world_model.device)
        successes = torch.zeros(envs.env_num, dtype=torch.bool, device=world_model.device)
        success_threshold = float(config.evaluation.success_threshold)
        action_repeat = int(config.env.action_repeat)
        policy = StormPolicyContext(model, envs.env_num, config.storm_train.context_length)

        while not finished.all():
            action = policy.act(deterministic=True)
            if not torch.isfinite(action).all():
                raise RuntimeError("Evaluation policy produced a non-finite action.")
            next_obs, reward, done = envs.step(action, reset_mask=finished)
            reward = reward.to(world_model.device, non_blocking=True)
            model_done = done.to(world_model.device, non_blocking=True)
            active = ~finished
            returns += reward[:, 0] * active
            lengths += active
            successes |= (reward[:, 0] / action_repeat >= success_threshold) & active
            policy.advance(obs, action, active=active, reset=model_done)
            finished |= model_done
            obs = (envs.reset_done(next_obs, done) if done.any() else next_obs).to(
                world_model.device, non_blocking=True
            )

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
        world_model.train(wm_training)
        agent.train(agent_training)


class OnlineSession:
    """STORM's observation-action sequence replay and context policy."""

    def __init__(self, config, model, envs):
        self.model = model
        self.envs = envs
        self.replay = SequenceBuffer(config.replay)
        settings = config.storm_train
        self.settings = settings
        self.action_repeat = int(config.env.action_repeat)
        self.context_length = int(settings.context_length)
        self.warmup_steps = int(settings.warmup_steps)
        self.world_model_updates = int(settings.world_model_updates)
        self.actor_critic_updates = int(settings.actor_critic_updates)

    def start(self, resumed=False):
        self.replay.start(self.envs.env_num)
        self.obs = self.envs.reset().to(self.model.world_model.device, non_blocking=True)
        self.policy = StormPolicyContext(self.model, self.envs.env_num, self.context_length)
        self.returns = torch.zeros(self.envs.env_num, dtype=torch.float32, device=self.model.world_model.device)
        self.lengths = torch.zeros(self.envs.env_num, dtype=torch.int32, device=self.model.world_model.device)

    def collect(self):
        action = self.policy.act(deterministic=False)
        next_obs, reward, done = self.envs.step(action)
        model_obs = next_obs.to(self.model.world_model.device, non_blocking=True)
        reward = reward.to(self.model.world_model.device, non_blocking=True)
        model_done = done.to(self.model.world_model.device, non_blocking=True)
        self.replay.append(self.obs, action, reward, model_obs["is_terminal"].reshape(-1), model_done)
        self.policy.advance(self.obs, action, reset=model_done)
        self.returns += reward[:, 0]
        self.lengths += 1
        episodes = []
        if model_done.any():
            for index, flag in enumerate(model_done):
                if flag:
                    episodes.append((self.returns[index], self.lengths[index]))
                    self.returns[index] = self.lengths[index] = 0
            self.obs = self.envs.reset_done(next_obs, done).to(self.model.world_model.device, non_blocking=True)
        else:
            self.obs = model_obs
        return self.envs.env_num * self.action_repeat, episodes

    def update(self, step):
        if self.replay.count() < self.warmup_steps or not self.replay.ready():
            return {}, 0
        world_model = self.model.world_model
        metrics = {}
        for _ in range(self.world_model_updates):
            wm_metrics, _, _ = world_model.update(*self.replay.sample())
            metrics.update(wm_metrics)
        for _ in range(self.actor_critic_updates):
            metrics.update(self._update_actor_critic())
        return metrics, self.world_model_updates

    def _update_actor_critic(self):
        settings = self.settings
        world_model = self.model.world_model
        agent = self.model.actor_critic
        obs, action, _, _ = self.replay.sample(
            batch_size=int(settings.imagine_batch_size),
            sequence_length=int(settings.imagine_context_length),
        )
        imagined = world_model.imagine(
            agent,
            {key: value.to(world_model.device, non_blocking=True) for key, value in obs.items()},
            action.to(world_model.device, non_blocking=True),
            horizon=int(settings.imagine_horizon),
        )
        return agent.update(
            imagined["feat"],
            imagined["action"],
            reward=imagined["reward"],
            termination=imagined["terminal"],
        )
