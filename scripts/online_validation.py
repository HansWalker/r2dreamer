"""Fixed, disjoint simulator trajectories and head-only calibration for online diagnostics."""

import copy
import hashlib
from collections import deque
from dataclasses import asdict

import torch

import tools
from envs import close_envs, make_envs
from models.shared.physical_state import STATE_KEY, readout_mode
from training.evaluation import Window
from training.planning import build_context


@tools.preserve_rng_state
def collect_episode(config, model, seed, mode):
    """A complete diagnostic episode, never inserted into a training replay."""
    if mode not in {"policy", "zero", "random"}:
        raise ValueError(f"Unknown diagnostic policy: {mode}")
    settings = copy.deepcopy(config.env)
    settings.env_num = 1
    envs = None
    planner_state = {key: getattr(model, key) for key in ("_cem_mean", "_gradient_actions")}
    generator = torch.Generator().manual_seed(seed)
    tools.configure_randomness(seed, bool(config.deterministic_run))
    images, states, actions, rewards = [], [], [], []
    try:
        envs = make_envs(settings, seed=seed)
        with readout_mode(model):
            obs = envs.reset().to(model.device)
            history, past = [deque(maxlen=model.history_size - 1)], [deque(maxlen=model.history_size - 1)]
            for step in range(int(settings.time_limit) // int(settings.action_repeat)):
                images.append(obs["image"][0].cpu().clone())
                states.append(obs[STATE_KEY][0].cpu().clone())
                if mode == "policy":
                    context, previous = build_context({"image": obs["image"]}, history, past,
                                                      model.history_size, model.action_dim)
                    action = model.act(context, previous, deterministic=True,
                                       first=torch.tensor([step == 0], device=model.device))
                elif mode == "zero":
                    action = torch.zeros(1, model.action_dim, device=model.device)
                else:
                    action = (2 * torch.rand(1, model.action_dim, generator=generator) - 1).to(model.device)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Diagnostic policy produced non-finite actions.")
                next_obs, reward, done = envs.step(action)
                history[0].append({"image": obs["image"][0]})
                past[0].append(action[0].detach())
                actions.append(action[0].detach().cpu().clone())
                rewards.append(float(reward.sum()))
                obs = next_obs.to(model.device)
                if done.all():
                    break
            else:
                raise RuntimeError("Diagnostic episode did not terminate at its configured time limit.")
            images.append(obs["image"][0].cpu().clone())
            states.append(obs[STATE_KEY][0].cpu().clone())
    finally:
        close_envs(envs)
        for key, value in planner_state.items():
            setattr(model, key, value)
    episode = {"image": torch.stack(images), "state": torch.stack(states), "action": torch.stack(actions),
               "seed": seed, "policy": mode, "return": sum(rewards), "agent_steps": len(actions)}
    digest = hashlib.sha256()
    for key in ("image", "state", "action"):
        digest.update(episode[key].numpy().tobytes())
    episode["sha256"] = digest.hexdigest()
    return episode


def episode_metadata(episodes):
    return [{key: value for key, value in episode.items() if not isinstance(value, torch.Tensor)} for episode in episodes]


def sample_trajectory_windows(episodes, count, length, seed):
    generator = torch.Generator().manual_seed(seed)
    windows = []
    for index, episode in enumerate(episodes):
        choices = len(episode["image"]) - length + 1
        if choices < 1:
            raise ValueError("Diagnostic episode is shorter than the requested context and forecast.")
        starts = torch.randperm(choices, generator=generator)[:min(count, choices)].tolist()
        windows.extend(Window(index, start, episode["policy"], 0.0) for start in starts)
    return windows


class TrajectoryDataset:
    def __init__(self, episodes, *, forbidden_seeds=()):
        self.episodes = episodes
        seeds = [episode["seed"] for episode in episodes]
        if len(set(seeds)) != len(seeds) or set(seeds) & set(forbidden_seeds):
            raise ValueError("Diagnostic training and validation trajectories must have disjoint episode seeds.")

    def sample_windows(self, count, length, seed):
        return sample_trajectory_windows(self.episodes, count, length, seed)

    def read_batch(self, windows, length):
        def read(key, size):
            return torch.stack([self.episodes[window.episode][key][window.start:window.start + size]
                                for window in windows])
        return {"image": read("image", length)}, read("action", length - 1), read("state", length)


def validation_metadata(dataset, windows):
    return {"episodes": episode_metadata(dataset.episodes), "windows": [asdict(window) for window in windows],
            "role": "fixed held-out simulator episodes; never fitted or inserted into online replay"}


@tools.preserve_rng_state
def calibrate_readout(model, episodes, updates, seed):
    """Candidate only: detached frozen-native features, fixed physical conditioning, no validation fitting."""
    head = model.state_head
    generator = torch.Generator().manual_seed(seed)
    cached = []
    with readout_mode(model):
        for episode in episodes:
            features = torch.cat([model.encode({"image": chunk[None].to(model.device)})[0].cpu()
                                  for chunk in episode["image"].split(64)])
            cached.append((features, episode["state"]))
    labels = torch.cat([labels for _, labels in cached])
    # Fixed for the entire trial, estimated on calibration TRAINING states only.
    # Metre-valued positions have a 0.1 m floor; angular features are dimensionless.
    floors = torch.full_like(head.std.cpu(), .1)
    floors[head.targets.velocities] = 1.0
    floors[head.targets.trigonometric] = 1.0
    scales = torch.maximum(labels.std(0, unbiased=False), floors)
    scales[head.targets.trigonometric] = 1.0
    feature = cached[0][0][:head.history + 1][None].to(model.device)
    with torch.no_grad():
        before = head(feature).clone()
        head.set_conditioning(scales)
        difference = (head(feature) - before).abs().max().item()
        torch.testing.assert_close(head(feature), before, rtol=2e-5, atol=1e-6)
    initial_updates = int(head.updates)
    length = model.sequence_length
    rows = (head.samples_per_update + length - head.history) // (length - head.history + 1)
    for _ in range(updates):
        features, labels = [], []
        for index in range(rows):
            latent, state = cached[index % len(cached)]
            start = int(torch.randint(len(latent) - length + 1, (), generator=generator))
            features.append(latent[start:start + length])
            labels.append(state[start:start + length])
        head.fit(torch.stack(features).to(model.device), torch.stack(labels).to(model.device))
    # Calibration has its own budget. Start the subsequent online LR schedule anew.
    head.configure_online(head._expert_source, resumed=False)
    return {"updates": int(head.updates) - initial_updates, "native_updates": 0,
            "training_episodes": episode_metadata(episodes), "output_scale": head.output_scale.cpu().tolist(),
            "loss_scale": head.loss_scale.cpu().tolist(), "evaluation_std": head.std.cpu().tolist(),
            "affine_max_error": difference, "expert_fraction": head.expert_fraction,
            "production_default": False}
