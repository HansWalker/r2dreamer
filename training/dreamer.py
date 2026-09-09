"""Dreamer-specific operations used by the shared trainer."""

import torch

from buffer import Buffer
from dmc_expert.replay import DMCExpertEpisodeReplay
from models.dreamer import Dreamer
from models.shared.physical_state import STATE_KEY
from training.evaluation import EpisodeMetrics

build_replay = DMCExpertEpisodeReplay
EXPERT_METRICS = {"loss": "opt/loss", "bc": "loss/bc", "state": "state/loss"}
ONLINE_METRICS = {"loss": "opt/loss", "state": "state/loss"}


def build_model(config):
    return Dreamer(config.model, config.model_io).to(config.device)


def checkpoint(model):
    return {
        "agent_state_dict": model.state_dict(),
        "optims_state_dict": {
            "_optimizer": model._optimizer.state_dict(),
            "state_head": model.state_head.optimizer.state_dict(),
        },
        "agent_training_state": model.training_state_dict(),
    }


def load_checkpoint(model, payload, training=True):
    model.load_state_dict(payload["agent_state_dict"])
    if training:
        model._optimizer.load_state_dict(payload["optims_state_dict"]["_optimizer"])
        model.state_head.optimizer.load_state_dict(payload["optims_state_dict"]["state_head"])
        model.load_training_state_dict(payload.get("agent_training_state"))


def expert_update(model, batch):
    if isinstance(batch, tuple):
        contexts, batch = batch
        contexts = [(context.to(model.device, non_blocking=True), starts) for context, starts in contexts]
    else:
        contexts = None
    return model.update_expert_pretrain(batch.to(model.device, non_blocking=True), contexts)


@torch.no_grad()
def evaluate(config, model, envs):
    was_training = model.training
    model.eval()
    try:
        done = torch.ones(envs.env_num, dtype=torch.bool, device=model.device)
        finished = torch.zeros_like(done)
        metrics = EpisodeMetrics(envs.env_num, model.device, config)
        state = model.get_initial_state(envs.env_num)
        action = state["prev_action"].clone()

        while not finished.all():
            stepped = ~done & ~finished
            transition, reward, next_done = envs.step(action.detach(), reset_mask=done | finished)
            transition["reward"] = reward
            transition = transition.to(model.device, non_blocking=True)
            done = next_done.to(model.device)
            action, state = model.act(transition.exclude(STATE_KEY), state, eval=True)
            if not torch.isfinite(action).all():
                raise RuntimeError("Evaluation policy produced a non-finite action.")
            metrics.update(transition["reward"], stepped)
            finished |= done

        return metrics.result()
    finally:
        model.train(was_training)


class OnlineSession:
    """Dreamer's recurrent actor and previous-action replay convention."""

    def __init__(self, config, model, envs):
        self.model = model
        self.envs = envs
        self.replay = Buffer(config.replay)
        self.action_repeat = int(config.env.action_repeat)

    def start(self):
        self.done = torch.ones(self.envs.env_num, dtype=torch.bool, device=self.model.device)
        self.returns = torch.zeros(self.envs.env_num, dtype=torch.float32, device=self.model.device)
        self.lengths = torch.zeros(self.envs.env_num, dtype=torch.int32, device=self.model.device)
        self.agent_state = self.model.get_initial_state(self.envs.env_num)
        self.action = self.agent_state["prev_action"].clone()

    def collect(self):
        step_delta = int((~self.done).sum()) * self.action_repeat
        self.lengths += ~self.done
        transition, reward, next_done = self.envs.step(self.action.detach(), reset_mask=self.done)
        transition["reward"] = reward
        replay_transition = transition
        transition = transition.to(self.model.device, non_blocking=True)
        self.done = next_done.to(self.model.device)
        self.action, self.agent_state = self.model.act(transition.exclude(STATE_KEY), self.agent_state, eval=False)
        replay_transition["action"] = (self.action * ~self.done.unsqueeze(-1)).to(replay_transition.device)
        self.replay.add_transition(replay_transition)
        self.returns += transition["reward"][:, 0]
        episodes = []
        for index, done in enumerate(self.done):
            if done:
                episodes.append((self.returns[index].item(), self.lengths[index].item()))
                self.returns[index] = self.lengths[index] = 0
        return step_delta, episodes

    def update(self, update_count):
        metrics = {}
        for _ in range(update_count):
            metrics = self.model.update(self.replay)
        return metrics
