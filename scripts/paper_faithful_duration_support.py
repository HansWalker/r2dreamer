"""Task-independent data and diagnostics for native TS/LeWM learning curves."""

import math
import pickle
import time

import mujoco
import numpy as np
import torch

import tools
from envs.dmc import make_env
from models.shared.physical_state import STATE_KEY, readout_mode
from scripts.diagnose_planner_oracle import action_candidates, simulator_branch
from scripts.paper_faithful_followup_eval import preserve_training_state
from scripts.paper_faithful_support import BranchReplay, _digest, score_branches
from scripts.smoke_tiny_planners import native_control
from scripts.train_paper_faithful_check import synchronize


def snapshot_env(env):
    """Include integration history, episode geometry, task RNG and wrapper clocks."""
    physics = env._env.physics
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(physics.model.ptr, spec))
    mujoco.mj_getState(physics.model.ptr, physics.data.ptr, state, spec)
    return {
        "integration": state, "task": np.frombuffer(pickle.dumps(env._env.task.__dict__), dtype=np.uint8).copy(),
        "geometry": {name: getattr(physics.model, name).copy() for name in
                     ("geom_pos", "geom_size", "body_pos", "site_pos", "site_size")},
        "step_count": env._env._step_count, "reset_next_step": env._env._reset_next_step,
        "episode_step": env._episode_step, "goal_image": env._goal_image.copy(),
    }


def restore_env(env, snapshot, *, reset=True):
    if reset:
        env.reset()
    physics = env._env.physics
    for name, value in snapshot["geometry"].items():
        getattr(physics.model, name)[:] = value
    env._env.task.__dict__.clear()
    env._env.task.__dict__.update(pickle.loads(snapshot["task"].tobytes()))
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(physics.model.ptr, physics.data.ptr, snapshot["integration"], spec)
    physics.forward()
    # Forward computes derived fields but can change integration history.
    mujoco.mj_setState(physics.model.ptr, physics.data.ptr, snapshot["integration"], spec)
    env._env._step_count = snapshot["step_count"]
    env._env._reset_next_step = snapshot["reset_next_step"]
    env._episode_step = snapshot["episode_step"]
    env._goal_image = snapshot["goal_image"].copy()
    env._goal_images = None


def simulate_many(env, plans):
    """Reuse one isolated renderer; restore full integration state between plans."""
    snapshot = snapshot_env(env)
    results = []
    with simulator_branch(env) as branch:
        for actions in plans:
            restore_env(branch, snapshot, reset=False)
            images, states, rewards, success = [], [], [], []
            for action in actions:
                obs, reward, done, _ = branch.step(action)
                if done:
                    raise ValueError("Diagnostic plan crossed an episode boundary")
                images.append(obs["image"].copy())
                states.append(obs[STATE_KEY].copy())
                rewards.append(float(reward))
                success.append(float(branch._env.task.get_reward(branch._env.physics)) >= 1 - 1e-6)
            results.append({"image": torch.from_numpy(np.stack(images)), "states": torch.from_numpy(np.stack(states)),
                            "rewards": torch.tensor(rewards), "success": torch.tensor(success)})
    return results


def simulate(env, actions):
    return simulate_many(env, [actions])[0]


def collect_bank(config, *, counts, candidates, horizon, seed, progress=None):
    """Split reset seeds before collecting branches; no success-based filtering."""
    splits, offset = {}, 0
    for split, count in counts.items():
        cases = []
        for index in range(count):
            case_seed = int(seed) + offset
            offset += 1
            env = make_env(config.env, case_seed, include_physical_state=True)
            try:
                obs = env.reset()
                # Shared, prespecified roll-in rule for every task. Zero steps is
                # represented too; normal task resets are not artificially balanced.
                rollin = (0, 10, 30, 60)[index % 4]
                rng = np.random.default_rng(case_seed + 700_001)
                for step in range(rollin):
                    if step % 5 == 0:
                        action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
                    obs, _, done, _ = env.step(action)
                    if done:
                        raise ValueError("Roll-in crossed episode boundary")
                frames, labels, past, past_reward = [obs["image"].copy()], [obs[STATE_KEY].copy()], [], []
                for _ in range(int(config.jepa_model.history_size) - 1):
                    action = np.zeros(env.action_space.shape, np.float32)
                    obs, reward, done, _ = env.step(action)
                    if done:
                        raise ValueError("Observation prefix crossed episode boundary")
                    frames.append(obs["image"].copy())
                    labels.append(obs[STATE_KEY].copy())
                    past.append(action)
                    past_reward.append(float(reward))
                actions = action_candidates(candidates, math.ceil(horizon / 5),
                                            env.action_space.shape[0], case_seed + 900_001)
                actions = np.repeat(actions, 5, axis=1)[:, :horizon]
                futures = simulate_many(env, actions)
                cases.append({
                    "id": f"{config.scenario.name}/{split}/{case_seed}", "seed": case_seed,
                    "split": split, "cohort": f"reset_plus_{rollin}", "rollin": rollin,
                    "prefix": torch.from_numpy(np.stack(frames)),
                    "prefix_state": torch.from_numpy(np.stack(labels)),
                    "past_action": torch.from_numpy(np.stack(past)),
                    "past_reward": torch.tensor(past_reward), "action": torch.from_numpy(actions),
                    "goal_image": torch.from_numpy(obs["goal_image"].copy()),
                    "snapshot": snapshot_env(env),
                    **{key: torch.stack([future[key] for future in futures]) for key in futures[0]},
                })
            finally:
                env.close()
            if progress:
                progress(offset, sum(counts.values()))
        splits[split] = cases
    metadata = {"task": str(config.scenario.name), "counts": counts, "seed": seed,
                "horizon": horizon, "candidates": candidates,
                "split_unit": "reset_seed_before_rollin_and_branching",
                "native_inputs": ["image", "action"], "labels": "detached readout and diagnostics only",
                "geometry": "Each snapshot preserves its own episode target; no target is selected from evaluation outcomes",
                "split_hashes": {key: _digest(value) for key, value in splits.items()}}
    return {"splits": splits, "metadata": metadata, "sha256": _digest(metadata)}


def validate_bank(bank):
    metadata = bank["metadata"]
    if bank["sha256"] != _digest(metadata):
        raise ValueError("Bank metadata hash mismatch")
    seeds = set()
    for split, cases in bank["splits"].items():
        if _digest(cases) != metadata["split_hashes"][split]:
            raise ValueError("Bank contents hash mismatch")
        for case in cases:
            if case["split"] != split or case["seed"] in seeds:
                raise ValueError("Bank split leakage")
            seeds.add(case["seed"])


class TaskBranchReplay(BranchReplay):
    """Physical labels are already encoded by the task's production environment."""

    def sample_training_batch(self):
        obs, action = self.sample()
        cases = {case["id"]: case for case in self.cases}
        labels = []
        for entry in self.last_plan:
            case, branch, start = cases[entry["anchor"]], entry["branch"], entry["start"]
            labels.append(torch.cat((case["prefix_state"], case["states"][branch]))[
                start:start + self.sequence_length])
        obs[STATE_KEY] = torch.stack(labels).float()
        return obs, action


@torch.no_grad()
def selected_plans(model, history, past, goals, executed):
    """Read the actual solver result, including CEM's executed elite mean."""
    latent = model.encode({"image": history})
    goal = model.encode({"image": goals[:, None]})[:, 0]
    if str(model.planner.type) == "cem":
        plans = model._cem_mean.detach().clone()
    else:
        candidates = model._gradient_actions
        costs = model._goal_cost(latent, past, candidates, goal)
        plans = candidates[torch.arange(len(history), device=model.device), costs.argmin(1)].detach().clone()
    torch.testing.assert_close(plans[:, 0], executed, rtol=0, atol=0)
    return plans


@torch.no_grad()
def diagnose_plan(model, env, history, past, goal, plan, seed):
    """Simulator futures only evaluate plans AFTER the native planner has acted."""
    horizon = len(plan)
    alternatives = action_candidates(6, horizon, model.action_dim, seed)
    actions = torch.cat((plan.cpu()[None], torch.from_numpy(alternatives)))
    futures = simulate_many(env, actions.numpy())
    latent = model.encode({"image": history[None].to(model.device)})
    encoded_goal = model.encode({"image": goal[None, None].to(model.device)})[:, 0]
    images = torch.stack([future["image"] for future in futures])
    flat = images.flatten(0, 1)
    actual = torch.cat([model.encode({"image": chunk[:, None].to(model.device)})[:, 0]
                        for chunk in flat.split(32)]).reshape(len(actions), horizon, *latent.shape[2:])
    predicted = model.rollout(latent, past[None].to(model.device), actions[None].to(model.device))
    predicted_cost = model.planning_cost(predicted, encoded_goal, history=latent)[0]
    actual_cost = model.planning_cost(actual[None], encoded_goal, history=latent)[0]
    error = (predicted[0] - actual).square().flatten(2).mean(2)
    result = {"actions": actions, "predicted_cost": predicted_cost.cpu(), "actual_cost": actual_cost.cpu(),
              "prediction_mse_by_step": error.cpu(), "prefix": history.cpu(), "past_action": past.cpu(),
              "goal_image": goal.cpu(), "snapshot": snapshot_env(env),
              **{key: torch.stack([future[key] for future in futures]) for key in futures[0]},
              "selected_index": 0, "alternative_scope": "zero, opposing axes, random; finite diagnostic bank"}
    if not torch.isfinite(predicted_cost).all() or not torch.isfinite(actual_cost).all():
        raise ValueError("Non-finite plan diagnostic")
    return result


@tools.preserve_rng_state
def policy_trial(config, model, cases, steps, seed, *, diagnostics=True, zero=False):
    """Matched stored starts; bounded plan probes at entry and first loss of success."""
    envs, traces, probes = [], [], []
    setup_start = time.monotonic()
    acting_seconds = diagnostic_seconds = 0.
    try:
        for case in cases:
            env = make_env(config.env, case["seed"], include_physical_state=True)
            envs.append(env)
            restore_env(env, case["snapshot"])
            np.testing.assert_array_equal(env.render(), case["prefix"][-1].numpy())
        history = torch.stack([case["prefix"] for case in cases]).to(model.device)
        past = torch.stack([case["past_action"] for case in cases]).to(model.device)
        goals = torch.stack([case["goal_image"] for case in cases]).to(model.device)
        model._cem_mean = model._gradient_actions = None
        returns, success = np.zeros(len(cases)), []
        streak = np.zeros(len(cases), dtype=int)
        probed_failure = np.zeros(len(cases), dtype=bool)
        setup_seconds = time.monotonic() - setup_start
        with readout_mode(model):
            for step in range(steps):
                torch.manual_seed(seed + step)
                synchronize(model.device)
                tick = time.monotonic()
                action = (torch.zeros(len(cases), model.action_dim, device=model.device) if zero else
                          native_control(model, lambda: model.act(
                              {"image": history, "goal_image": goals}, past, deterministic=True,
                              first=torch.full((len(cases),), step == 0, device=model.device, dtype=torch.bool))))
                plans = selected_plans(model, history, past, goals, action) if diagnostics and not zero else None
                # Inspect the state BEFORE a first action loses a >=5-step success
                # streak. Branching before stepping keeps that exact state available.
                if plans is not None:
                    synchronize(model.device)
                    acting_seconds += time.monotonic() - tick
                    dt = time.monotonic()
                    for i, env in enumerate(envs):
                        reason = "initial" if step == 0 else None
                        if not probed_failure[i] and step > 0:
                            if streak[i] >= 5:
                                with simulator_branch(env) as branch:
                                    branch.step(action[i].cpu().numpy())
                                    if float(branch._env.task.get_reward(branch._env.physics)) < 1 - 1e-6:
                                        reason = "first_loss_after_five_successes"
                            if reason is None and step == steps - 1:
                                reason = "final_state_fallback"
                        if reason:
                            probe = diagnose_plan(model, env, history[i], past[i], goals[i], plans[i], seed + step + i)
                            probe.update(case_id=cases[i]["id"], step=step, reason=reason)
                            probes.append(probe)
                            if step > 0:
                                probed_failure[i] = True
                    synchronize(model.device)
                    diagnostic_seconds += time.monotonic() - dt
                    tick = time.monotonic()
                images, rewards, good = [], [], []
                for i, env in enumerate(envs):
                    obs, reward, done, _ = env.step(action[i].cpu().numpy())
                    if done:
                        raise ValueError("Policy trial crossed an episode boundary")
                    images.append(obs["image"])
                    rewards.append(float(reward))
                    good.append(float(env._env.task.get_reward(env._env.physics)) >= 1 - 1e-6)
                streak = np.where(good, streak + 1, 0)
                history = torch.cat((history[:, 1:], torch.from_numpy(np.stack(images)).to(model.device)[:, None]), 1)
                past = torch.cat((past[:, 1:], action[:, None]), 1)
                returns += rewards
                success.append(good)
                traces.append({"step": step + 1, "actions": action.cpu().tolist(), "rewards": rewards, "success": good})
                synchronize(model.device)
                acting_seconds += time.monotonic() - tick
        tail = max(1, math.ceil(.2 * steps))
        occupancy = np.asarray(success[-tail:]).mean(0)
        return {"case_ids": [case["id"] for case in cases], "steps": steps, "returns": returns.tolist(),
                "return_mean": float(returns.mean()), "maximum_return": steps * int(config.env.action_repeat),
                "tail_steps": tail, "maintenance_occupancy": occupancy.tolist(),
                "maintenance_rate": float((occupancy >= .9).mean()), "traces": traces, "probes": probes,
                "setup_seconds": setup_seconds, "acting_seconds": acting_seconds,
                "diagnostic_seconds": diagnostic_seconds,
                "scope": "Stored reset/roll-in starts, no success filtering; not the 50-episode benchmark"}
    finally:
        for env in envs:
            env.close()


def evaluate(config, model, cases, *, steps, policy_cases, seed, horizons, initial=False):
    with preserve_training_state(model):
        tick = time.monotonic()
        branches = score_branches(model, cases, horizons=horizons)
        synchronize(model.device)
        forecast_seconds = time.monotonic() - tick
        policy = None if initial else policy_trial(config, model, cases[:policy_cases], steps, seed)
    return {"branches": branches, "policy": policy, "forecast_seconds": forecast_seconds,
            "training_state_preserved": True}
