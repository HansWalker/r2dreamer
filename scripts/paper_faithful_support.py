"""Shared intervention data and native-only measurements for JEPA diagnostics.

These are experimental data/evaluation tools, not changes to either native loss.
Simulator states are excluded from sample(); sample_training_batch exposes them
only for the production model's detached auxiliary readout. A source means an
independent anchor, not one of its branches.
"""

import copy
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from envs.dmc import make_env
from models.shared.latent_goal import latent_goal_cost
from models.shared.physical_state import readout_mode
from scripts.diagnose_goal_objective import cart_state, feedback_gain, set_cart_state
from scripts.diagnose_planner_oracle import rank_correlation, selection, simulator_branch


PROFILES = {
    "balanced": ("balanced", [0., 0., 0., 0.]),
    "boundary": ("boundary", [.18, .08, 0., .15]),
    "recoverable": ("recoverable", [.02, .12, 0., .2]),
}


def make_split_manifest(train_anchors=24, validation_anchors=12, test_anchors=12, seed=17_000_000):
    """Assign unique simulator seeds before any sibling branches are generated."""
    counts = (int(train_anchors), int(validation_anchors), int(test_anchors))
    if any(value < 1 for value in counts) or int(seed) < 0:
        raise ValueError("All three anchor splits must be positive and the seed nonnegative.")
    result, offset = {}, 0
    profiles = tuple(PROFILES)
    for split, count in zip(("train", "validation", "test"), counts, strict=True):
        rows = []
        for index in range(count):
            current_seed = int(seed) + offset
            profile = profiles[index % len(profiles)]
            cohort, nominal = PROFILES[profile]
            rows.append({"id": f"{current_seed}/{profile}", "seed": current_seed,
                         "split": split, "profile": profile, "cohort": cohort,
                         "nominal_state": list(nominal)})
            offset += 1
        result[split] = rows
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    if set(manifest) != {"train", "validation", "test"} or not all(manifest.values()):
        raise ValueError("Require nonempty train, validation, and test anchor splits.")
    ids, seeds = set(), set()
    for split, cases in manifest.items():
        for case in cases:
            if case["split"] != split or case["id"] in ids or int(case["seed"]) in seeds:
                raise ValueError("Anchor IDs and simulator seeds must be disjoint across and within splits.")
            ids.add(case["id"])
            seeds.add(int(case["seed"]))


def _digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            visit({"dtype": str(array.dtype), "shape": list(array.shape)})
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            visit(torch.as_tensor(item))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b"\n")

    visit(value)
    return digest.hexdigest()


def intervention_candidates(count, horizon, seed):
    """Indices 0/1/2 are zero/-1/+1; last two are simulator feedback plans."""
    if int(count) < 6 or int(horizon) < 1:
        raise ValueError("Use at least six candidates and a positive horizon.")
    generator = np.random.default_rng(int(seed))
    actions = np.zeros((int(count), int(horizon), 1), np.float32)
    actions[1], actions[2] = -1, 1
    labels = ["zero", "negative", "positive"]
    for index in range(3, count - 2):
        blocks = generator.uniform(-1, 1, ((horizon + 4) // 5, 1)).astype(np.float32)
        actions[index] = np.repeat(blocks, 5, axis=0)[:horizon]
        labels.append(f"random_blocks_{index}")
    return actions, labels + ["simulator_feedback_half", "simulator_feedback"]


def collect_anchor(env, spec, *, candidates=16, horizon=16, history_size=3, gain=None):
    """Collect all successor frames from copied simulators; parent stays at anchor."""
    if history_size < 1 or history_size - 1 + horizon >= env._max_steps:
        raise ValueError("The history and branch must fit inside one simulator episode.")
    reset = env.reset()
    if "goal_image" not in reset:
        raise ValueError("Native goal diagnostics require the rendered goal observation.")
    gain = feedback_gain(env) if gain is None else gain
    generator = np.random.default_rng(int(spec["seed"]))
    state = np.asarray(spec["nominal_state"], dtype=np.float64)
    state = state + generator.uniform(-1, 1, 4) * [.005, .002, .005, .01]
    if int(spec["seed"]) % 2:
        state *= -1
    set_cart_state(env, state)
    prefix, prefix_states, past, past_rewards = [env.render().copy()], [cart_state(env)], [], []
    for _ in range(history_size - 1):
        action = np.zeros(1, np.float32)
        observation, reward, done, _ = env.step(action)
        if done:
            raise ValueError("Observed prefix crossed an episode boundary.")
        prefix.append(observation["image"].copy())
        prefix_states.append(cart_state(env))
        past.append(action)
        past_rewards.append(float(reward))
    anchor_state = cart_state(env)
    integration_state = env._env.physics.get_state().copy()
    episode_step = env._episode_step
    actions, labels = intervention_candidates(candidates, horizon, spec["seed"])
    images, states, rewards, successes = [], [], [], []
    for index, sequence in enumerate(actions):
        frames, physical, reward_trace, success_trace = [], [], [], []
        with simulator_branch(env) as branch:
            for step in range(horizon):
                if index >= candidates - 2:
                    scale = .5 if index == candidates - 2 else 1.
                    sequence[step, 0] = np.clip(-scale * (gain @ cart_state(branch)), -1, 1)
                observation, reward, done, _ = branch.step(sequence[step])
                if done:
                    raise ValueError("Candidate crossed an episode boundary.")
                frames.append(observation["image"].copy())
                physical.append(cart_state(branch))
                reward_trace.append(float(reward))
                success_trace.append(bool(branch._env.task.get_reward(branch._env.physics) >= 1 - 1e-6))
        images.append(np.stack(frames))
        states.append(np.stack(physical))
        rewards.append(reward_trace)
        successes.append(success_trace)
    np.testing.assert_array_equal(integration_state, env._env.physics.get_state())
    if episode_step != env._episode_step:
        raise RuntimeError("Branch collection changed the parent episode counter.")
    case = {**copy.deepcopy(spec), "initial_state": state.tolist(), "anchor_state": anchor_state.tolist(),
            "prefix": torch.from_numpy(np.stack(prefix)),
            "prefix_state": torch.from_numpy(np.stack(prefix_states)),
            "past_action": torch.from_numpy(np.asarray(past, np.float32).reshape(history_size - 1, 1)),
            "past_reward": torch.tensor(past_rewards, dtype=torch.float32),
            "action": torch.from_numpy(actions), "candidate_labels": labels,
            "image": torch.from_numpy(np.stack(images)), "states": torch.from_numpy(np.stack(states)),
            "rewards": torch.tensor(rewards, dtype=torch.float32), "successes": torch.tensor(successes),
            "goal_image": torch.from_numpy(reset["goal_image"].copy()), "feedback_gain": gain.tolist(),
            "physical_state_order": ["cart_position", "pole_angle", "cart_velocity", "pole_velocity"]}
    if "goal_images" in reset:
        case["goal_images"] = torch.from_numpy(reset["goal_images"].copy())
    case["sha256"] = _digest(case)
    return case


@tools.preserve_rng_state
def collect_branch_bank(config, manifest, *, candidates=16, horizon=16, progress=None):
    """Return a torch.save-compatible bank; no model or optimizer is accessed."""
    validate_manifest(manifest)
    if str(config.env.task) != "dmc_cartpole_balance_sparse":
        raise ValueError("This intervention bank supports Cartpole balance sparse only.")
    splits, gain = {}, None
    complete, total = 0, sum(map(len, manifest.values()))
    for split, specs in manifest.items():
        splits[split] = []
        for spec in specs:
            env = make_env(config.env, int(spec["seed"]), include_physical_state=False)
            try:
                case = collect_anchor(env, spec, candidates=candidates, horizon=horizon,
                                      history_size=int(config.jepa_model.history_size), gain=gain)
                gain = np.asarray(case["feedback_gain"])
                splits[split].append(case)
            finally:
                env.close()
            complete += 1
            if progress is not None:
                progress(complete, total)
    metadata = {"version": 1, "split_unit": "simulator_seed_and_anchor_before_branching",
                "manifest": copy.deepcopy(manifest), "candidates": candidates, "horizon": horizon,
                "history_size": int(config.jepa_model.history_size),
                "candidate_indices": {"zero": 0, "negative": 1, "positive": 2},
                "native_inputs": ["image", "action"],
                "physical_labels_role": "evaluation_and_detached_auxiliary_readout_only",
                "environment": OmegaConf.to_container(config.env, resolve=True),
                "cohort_note": "Recoverable is a prescribed near-failure profile, not a guarantee for every sampled anchor.",
                "split_hashes": {split: _digest(cases) for split, cases in splits.items()},
                "training_branch_transitions": len(splits["train"]) * candidates * horizon}
    return {"splits": splits, "metadata": metadata, "sha256": _digest(metadata)}


class BranchReplay:
    """Native-only, anchor-grouped sampler over the bank's TRAIN split."""

    def __init__(self, bank, batch_size=336, sequence_length=4, episodes_per_batch=16, seed=0):
        self.cases = bank["splits"]["train"] if isinstance(bank, dict) else list(bank)
        self.batch_size, self.sequence_length = int(batch_size), int(sequence_length)
        self.episodes_per_batch = int(episodes_per_batch)
        if (self.episodes_per_batch < 1 or self.batch_size < 1 or self.sequence_length < 2
                or self.batch_size % self.episodes_per_batch or len(self.cases) < self.episodes_per_batch):
            raise ValueError("Batch must divide across enough distinct positive training-anchor sources.")
        if any(case.get("split") != "train" for case in self.cases):
            raise ValueError("Only TRAIN anchors may enter native replay.")
        for field in ("id", "seed"):
            if len({case[field] for case in self.cases}) != len(self.cases):
                raise ValueError("Sibling branches cannot count as independent anchor sources.")
        if any(len(case["prefix"]) + case["image"].shape[1] < self.sequence_length for case in self.cases):
            raise ValueError("Every training anchor must contain a complete sequence.")
        self.last_plan = []
        self.reset(seed)

    def reset(self, seed=0):
        self.generator = torch.Generator().manual_seed(int(seed))
        self.last_plan = []

    reseed = reset

    def state_dict(self):
        return {"generator_state": self.generator.get_state().clone(),
                "anchor_ids": [case["id"] for case in self.cases]}

    def load_state_dict(self, state):
        if state["anchor_ids"] != [case["id"] for case in self.cases]:
            raise ValueError("Saved sampler state belongs to a different anchor pool.")
        self.generator.set_state(state["generator_state"].cpu())

    def sample(self):
        selected = torch.randperm(len(self.cases), generator=self.generator)[:self.episodes_per_batch].tolist()
        rows, actions, plan = [], [], []
        for index in selected:
            case = self.cases[index]
            length = len(case["prefix"]) + case["image"].shape[1]
            for _ in range(self.batch_size // self.episodes_per_batch):
                branch = int(torch.randint(len(case["action"]), (), generator=self.generator))
                start = int(torch.randint(length - self.sequence_length + 1, (), generator=self.generator))
                image = torch.cat((case["prefix"], case["image"][branch]))
                action = torch.cat((case["past_action"], case["action"][branch]))
                rows.append(image[start:start + self.sequence_length])
                actions.append(action[start:start + self.sequence_length - 1])
                plan.append({"anchor": case["id"], "branch": branch, "start": start})
        self.last_plan = plan
        return {"image": torch.stack(rows)}, torch.stack(actions).float()

    sample_episode_batch = sample

    def sample_training_batch(self):
        """Production update tuple; physical labels supervise only its detached head."""
        obs, action = self.sample()
        cases = {case["id"]: case for case in self.cases}
        labels, rewards = [], []
        for entry in self.last_plan:
            case, branch, start = cases[entry["anchor"]], entry["branch"], entry["start"]
            state = torch.cat((case["prefix_state"], case["states"][branch]))[start:start + self.sequence_length].float()
            labels.append(torch.stack((state[:, 0], state[:, 1].cos(), state[:, 1].sin(),
                                       state[:, 2], state[:, 3]), dim=-1))
            reward = torch.cat((case["past_reward"], case["rewards"][branch]))
            rewards.append(reward[start:start + self.sequence_length - 1, None])
        obs["physical_state"] = torch.stack(labels)
        reward = torch.stack(rewards).float()
        return obs, action, reward, torch.zeros_like(reward)


def _encode(model, images, batch_size):
    return torch.cat([model.encode({"image": chunk[:, None].to(model.device)})[:, 0]
                      for chunk in images.split(int(batch_size))])


def _ratio(numerator, denominator):
    return float(numerator / denominator) if float(denominator) > 1e-12 else None


@torch.no_grad()
@tools.preserve_rng_state
def score_branches(model, cases, horizons=(1, 5), encode_batch_size=32):
    """Score the configured native objective and action-conditioned recursive futures.

    All MSEs are endpoint errors at the stated horizon. Shuffling permutes entire
    candidate plans, keeping observed history fixed. No readout fitting/decoding.
    """
    horizons = tuple(sorted(set(map(int, horizons))))
    if not cases or not horizons or min(horizons) < 1 or encode_batch_size < 1:
        raise ValueError("Require cases, positive horizons, and a positive encoding batch size.")
    mode = str(model.planner.get("objective", "last"))
    rows, moments = [], {}
    with readout_mode(model):
        for case in cases:
            if len(case["prefix"]) != model.history_size or max(horizons) > case["action"].shape[1]:
                raise ValueError("Case history/horizon is incompatible with this model.")
            horizon = max(horizons)
            history = _encode(model, case["prefix"], encode_batch_size)[None]
            images = case["image"][:, :horizon]
            actual = _encode(model, images.flatten(0, 1), encode_batch_size).reshape(
                images.shape[0], horizon, *history.shape[2:])
            goal_images = case.get("goal_images", case["goal_image"][None])
            goal = _encode(model, goal_images, encode_batch_size)[None]
            action = case["action"][:, :horizon].to(model.device)[None]
            past = case["past_action"].to(model.device)[None]
            predicted = model.rollout(history, past, action)[0]
            shuffled = model.rollout(history, past, action.roll(1, dims=1))[0]
            zero = model.rollout(history, past, torch.zeros_like(action))[0]
            if not all(torch.isfinite(value).all() for value in (history, actual, goal, predicted, shuffled, zero)):
                raise ValueError("Non-finite native branch predictions.")
            record = {"id": case["id"], "seed": case["seed"], "cohort": case["cohort"], "metrics": {}}
            for h in horizons:
                target = actual[:, h - 1]
                coordinate_samples = target.flatten(1).double().cpu()
                moments[case["id"], str(h)] = (coordinate_samples.sum(0),
                                              coordinate_samples.square().sum(0),
                                              len(coordinate_samples))
                mse = lambda value: float((value.float() - target.float()).square().mean())
                persistence = mse(history[0, -1].expand_as(target))
                errors = {"matched_mse": mse(predicted[:, h - 1]),
                          "shuffled_mse": mse(shuffled[:, h - 1]), "zero_mse": mse(zero[:, h - 1]),
                          "persistence_mse": persistence}
                costs = []
                for future in (predicted, actual):
                    costs.append(latent_goal_cost(future[None, :, :h], goal, reduction=model.goal_reduction,
                        mode=mode, history=history, tail_steps=int(model.planner.get("tail_steps", 3)))[0].cpu().numpy())
                rewards = case["rewards"][:, :h].sum(1).cpu().numpy()
                pred_choice, actual_choice = selection(costs[0], rewards), selection(costs[1], rewards)
                response = (predicted[:, h - 1] - zero[:, h - 1]).flatten(1)
                zero_indices = torch.nonzero((action[0, :, :h] == 0).flatten(1).all(1)).flatten()
                true_response = ((target - target[zero_indices[0]]).flatten(1)
                                 if len(zero_indices) else None)
                denominator = true_response.square().mean().sqrt() if true_response is not None else None
                record["metrics"][str(h)] = {**errors,
                    "matched_over_persistence": _ratio(errors["matched_mse"], persistence),
                    "target_latent_rms_std": float(target.flatten(1).std(0, unbiased=False).square().mean().sqrt()),
                    "actual_zero_action_reference_available": true_response is not None,
                    "true_action_response_rms": float(denominator) if denominator is not None else None,
                    "predicted_action_response_rms": float(response.square().mean().sqrt()),
                    "action_response_ratio": (_ratio(response.square().mean().sqrt(), denominator)
                                              if denominator is not None else None),
                    "action_response_mse": (float((response - true_response).square().mean())
                                            if true_response is not None else None),
                    "predicted_selected_return": pred_choice["return_mean"],
                    "actual_selected_return": actual_choice["return_mean"],
                    "uniform_return": float(rewards.mean()), "best_return": float(rewards.max()),
                    "predicted_regret": pred_choice["regret"], "actual_regret": actual_choice["regret"],
                    "predicted_vs_actual_cost_rank": rank_correlation(costs[0], costs[1]),
                    "actual_cost_vs_return_rank": rank_correlation(-costs[1], rewards),
                    "predicted_cost_vs_return_rank": rank_correlation(-costs[0], rewards),
                    "reward_informative": bool(np.ptp(rewards) > 1e-6),
                    "predicted_cost": costs[0].tolist(), "actual_cost": costs[1].tolist(),
                    "returns": rewards.tolist(), "predicted_selection": pred_choice, "actual_selection": actual_choice}
            rows.append(record)
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["cohort"]].append(row)
    aggregate, informative = {}, {}
    for cohort, items in groups.items():
        aggregate[cohort], informative[cohort] = {}, {}
        for h in map(str, horizons):
            for destination, subset in ((aggregate, items), (informative, [row for row in items
                                              if row["metrics"][h]["reward_informative"]])):
                values = [row["metrics"][h] for row in subset]
                summary = {"anchors": len(values), "reward_informative_anchors": sum(v["reward_informative"] for v in values),
                           "valid_counts": {}}
                for key, example in items[0]["metrics"][h].items():
                    finite = [v[key] for v in values if v[key] is not None and isinstance(v[key], (int, float))
                              and not isinstance(v[key], bool)]
                    if finite or example is None or (isinstance(example, (int, float)) and not isinstance(example, bool)):
                        summary[key] = float(np.mean(finite)) if finite else None
                        summary["valid_counts"][key] = len(finite)
                summary["within_anchor_latent_rms_std"] = summary["target_latent_rms_std"]
                if subset:
                    statistics = [moments[row["id"], h] for row in subset]
                    total, squared, count = (sum(parts) for parts in zip(*statistics, strict=True))
                    summary["target_latent_rms_std"] = float((squared / count - (total / count).square()).clamp_min(0).mean().sqrt())
                destination[cohort][h] = summary
    result = {"objective": mode, "goal_reduction": model.goal_reduction, "horizons": list(horizons),
              "error_definition": "endpoint latent MSE; fixed actual image targets; shuffled whole candidate plans",
              "spread_definition": "per-case std spans candidates; aggregate target std spans all anchors and candidates in that cohort",
              "action_response_definition": "Actual response requires an observed zero-action candidate at this horizon; otherwise true response, ratio, and response MSE are unavailable",
              "informative_definition": "Reward-informative means at least two candidate cumulative returns differ by more than 1e-6 at this horizon",
              "physical_head_used": False, "cases": rows, "aggregate": aggregate,
              "aggregate_informative": informative}
    json.dumps(result, allow_nan=False)
    return result
