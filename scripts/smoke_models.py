#!/usr/bin/env python3
"""Run one synthetic update, checkpoint, policy, and rollout through every model config."""

import math
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from training import load_model_family
from training.evaluation import latent_rollout

ROOT = Path(__file__).resolve().parents[1]


def model_configs():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        benchmark = compose(config_name="dmc_benchmark")
        scenarios = tuple(benchmark.scenarios)
        entries = {
            (entry.config if not isinstance(entry, str) else entry): family
            for family, variants in benchmark.models.items()
            for entry in variants.values()
        }
        configs = {}
        for name, expected_family in entries.items():
            for scenario in scenarios:
                config = compose(config_name=name, overrides=[f"scenario={scenario}"])
                OmegaConf.resolve(config)
                if str(config.model_family) != expected_family:
                    raise RuntimeError(f"{name} selects {config.model_family}, expected {expected_family}.")
                configs.setdefault(name, config)
    return configs


def synthetic_batch(config, model, batch_size=2, length=4):
    action_dim = math.prod(map(int, config.model_io.action.shape))
    observation = {}
    for key, shape in config.model_io.observations.items():
        shape = tuple(map(int, shape))
        observation[str(key)] = (
            torch.randint(256, (batch_size, length, *shape), dtype=torch.uint8)
            if len(shape) == 3
            else torch.randn(batch_size, length, *shape)
        )
    action = torch.empty(batch_size, length, action_dim).uniform_(-1, 1)
    reward = torch.rand(batch_size, length, 1)
    terminal = torch.zeros(batch_size, length, 1)
    family = str(config.model_family)
    if family == "dreamer":
        action[:, 0] = reward[:, 0] = 0
        first = torch.zeros(batch_size, length, 1, dtype=torch.bool)
        last = torch.zeros_like(first)
        first[:, 0] = True
        last[:, -1] = True
        batch = TensorDict(
            {
                **observation,
                "action": action,
                "reward": reward,
                "is_first": first,
                "is_last": last,
                "is_terminal": terminal.bool(),
            },
            batch_size=(batch_size, length),
        )
    elif family == "storm":
        returns = reward.flip(1).cumsum(1).flip(1)
        batch = (observation, action, reward, terminal, returns)
    else:
        training_observation = model.stack_sequence(observation) if family == "tdmpc2" else observation
        batch = (training_observation, action[:, :-1], reward[:, :-1], terminal[:, :-1])
        if family in {"leworldmodel", "temporal_straightening"}:
            tolerance = torch.as_tensor(list(config.jepa_model.goal.tolerance)).reshape(1, 1, -1)
            relation = torch.randn(batch_size, length, 2) * tolerance
            batch = (*batch, relation)
    return batch, observation, action


@torch.no_grad()
def policy_action(config, model, observation, action):
    batch_size = next(iter(observation.values())).shape[0]
    device = next(model.parameters()).device
    family = str(config.model_family)
    if family == "dreamer":
        state = model.get_initial_state(batch_size)
        transition = TensorDict(
            {
                **{key: value[:, 0].to(device) for key, value in observation.items()},
                "is_first": torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            },
            batch_size=(batch_size,),
        )
        return model.act(transition, state, eval=True)[0]
    if family == "storm":
        world_model = model.world_model
        if world_model.sequence_core.streaming:
            feature, _ = world_model.context_step(
                {key: value[:, 0].to(device) for key, value in observation.items()},
                action[:, 0].to(device),
            )
        else:
            feature = world_model.context_feature(
                {key: value[:, :1].to(device) for key, value in observation.items()},
                action[:, :1].to(device),
            )
        return model.actor_critic.sample(feature, deterministic=True)[0]

    model.planner.horizon = 2
    model.planner.samples = 4
    model.planner.iterations = 2
    if hasattr(model.planner, "elites"):
        model.planner.elites = 2
    history = {key: value[:, : model.history_size].to(device) for key, value in observation.items()}
    past_action = action[:, : max(model.history_size - 1, 0)].to(device)
    kwargs = {"deterministic": True}
    if family == "tdmpc2":
        kwargs["first"] = torch.ones(batch_size, dtype=torch.bool, device=device)
    return model.act(history, past_action, **kwargs)


def main():
    for name, config in model_configs().items():
        family = load_model_family(config.model_family)
        model = family.build_model(config)
        batch, observation, action = synthetic_batch(config, model)
        metrics = family.expert_update(model, batch)
        if not all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()):
            raise RuntimeError(f"{name} produced a non-finite update.")
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError(f"{name} produced non-finite parameters.")
        family.load_checkpoint(model, family.checkpoint(model), training=False)

        model.eval()
        decision = policy_action(config, model, observation, action)
        batch_size = next(iter(observation.values())).shape[0]
        if decision.shape != (batch_size, action.shape[-1]) or not torch.isfinite(decision).all():
            raise RuntimeError(f"{name} produced an invalid policy action.")
        device = next(model.parameters()).device
        observed, predicted = latent_rollout(
            model,
            {key: value.to(device) for key, value in observation.items()},
            action[:, :3].to(device),
            context_length=3,
        )
        if observed.shape[:2] != (batch_size, 4) or predicted.shape[:2] != (batch_size, 1):
            raise RuntimeError(f"{name} returned invalid rollout shapes {observed.shape}, {predicted.shape}.")
        print(f"{name}: passed")
        del model
        torch.cuda.empty_cache()
    print("All model smoke tests passed.")


if __name__ == "__main__":
    main()
