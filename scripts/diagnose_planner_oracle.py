"""Separate latent goal costs from forecast errors using identical simulator branches.

No fitting, policy optimization, dataset scan, or checkpoint writes. Legacy weights
require an explicit --latent-goals override; this does not reproduce their old policy.
"""

import argparse
import copy
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import tools
from envs.dmc import goal_relation, make_env
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_planning_models import FAMILIES, SCENARIOS
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import (
    TRAINING_RECIPE_VERSION, checkpoint_compatibility, implementation_sha256,
    upgrade_readout_config, validate_checkpoint,
)


@contextmanager
def simulator_branch(env):
    """Copy integration state, task RNG, episode counters, and per-episode geometry."""
    branch = copy.copy(env)
    branch._env = copy.copy(env._env)
    branch._env._physics = env._env.physics.copy(share_model=False)
    branch._env._physics.legacy_step = env._env.physics.legacy_step
    branch._env._task = copy.deepcopy(env._env.task)
    branch._goal_renderer = None  # A branch must never close its parent's renderer.
    try:
        yield branch
    finally:
        branch.close()


def action_candidates(count, horizon, dimensions, seed):
    generator = np.random.default_rng(seed)
    actions = generator.uniform(-1, 1, (count, horizon, dimensions)).astype(np.float32)
    actions[0] = 0
    # Include opposing constant controls as well as temporally independent actions.
    for axis in range(dimensions):
        for sign in (-1, 1):
            index = 1 + 2 * axis + (sign == 1)
            if index < count:
                actions[index] = 0
                actions[index, :, axis] = sign
    return actions


def simulate_candidates(env, actions):
    frames, rewards, relations = [], [], []
    for sequence in actions:
        images, returns, physical = [], [], []
        with simulator_branch(env) as branch:
            for action in sequence:
                obs, reward, done, _ = branch.step(action)
                images.append(obs["image"])
                returns.append(float(reward))
                physical.append(goal_relation(branch._env.physics, branch._domain, branch._task))
                if done:
                    raise ValueError("A candidate crossed an episode boundary; shorten roll-in/horizon.")
        frames.append(np.stack(images))
        rewards.append(returns)
        relations.append(physical)
    return {"image": torch.from_numpy(np.stack(frames)),
            "reward": torch.tensor(rewards), "relation": torch.tensor(np.asarray(relations))}


def environment_signature(config):
    return OmegaConf.to_container(OmegaConf.create({
        "task": config.env.task, "action_repeat": config.env.action_repeat,
        "size": config.env.size, "time_limit": config.env.time_limit,
        "goal": config.env.goal, "history": config.jepa_model.history_size,
        "horizon": config.jepa_model.planner.horizon,
    }), resolve=True)


def collect_cases(config, args):
    history, horizon = int(config.jepa_model.history_size), int(config.jepa_model.planner.horizon)
    if max(args.rollin_steps) + history - 1 + horizon >= config.env.time_limit // config.env.action_repeat:
        raise ValueError("The requested prefix/candidates would reach the episode time limit.")
    cases = []
    progress = Progress("Simulator branches", len(args.sim_seeds) * len(args.rollin_steps) * 2)
    for seed in args.sim_seeds:
        for mode in ("zero", "random"):
            env = make_env(config.env, seed, include_physical_state=False)
            try:
                obs = env.reset()
                images, past = [obs["image"]], []
                generator = np.random.default_rng(seed)
                for step in range(max(args.rollin_steps) + history):
                    if step >= history - 1 and step - history + 1 in args.rollin_steps:
                        actions = action_candidates(args.candidates, horizon, env.action_space.shape[0], seed + step)
                        outcome = simulate_candidates(env, actions)
                        cases.append({
                            "id": f"{seed}/{mode}/{step}", "seed": seed, "rollin_policy": mode,
                            "anchor_agent_step": step,
                            "anchor_simulator_state": env._env.physics.get_state().tolist(),
                            "anchor_reward": float(env._env.task.get_reward(env._env.physics)),
                            "goal_image": torch.from_numpy(obs["goal_image"].copy()),
                            "prefix": torch.from_numpy(np.stack(images[-history:])),
                            "past_action": torch.from_numpy(np.asarray(past[-(history - 1):], dtype=np.float32)),
                            "action": torch.from_numpy(actions), **outcome,
                        })
                        progress.update(len(cases), force=len(cases) == progress.total)
                    if step < max(args.rollin_steps) + history - 1:
                        action = (np.zeros(env.action_space.shape, dtype=np.float32) if mode == "zero" else
                                  generator.uniform(-1, 1, env.action_space.shape).astype(np.float32))
                        obs, _, done, _ = env.step(action)
                        if done:
                            raise ValueError("Roll-in crossed an episode boundary.")
                        images.append(obs["image"])
                        past.append(action)
            finally:
                env.close()
    return cases


def rank_correlation(left, right):
    """Spearman correlation with average ranks for ties; constants are uninformative."""
    def ranks(values):
        values = np.asarray(values, dtype=np.float64)
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]
    if any(np.allclose(v, v[0], rtol=1e-6, atol=1e-9) for v in (left, right)):
        return None
    a, b = ranks(left), ranks(right)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def selection(cost, returns):
    cost, returns = np.asarray(cost), np.asarray(returns)
    chosen = np.flatnonzero(np.isclose(cost, cost.min(), rtol=1e-6, atol=1e-9))
    actual = returns[chosen]
    return {"tied_candidates": chosen.tolist(), "return_mean": float(actual.mean()),
            "return_min": float(actual.min()), "return_max": float(actual.max()),
            "regret": float(returns.max() - actual.mean())}


def ranking_summary(predicted_cost, oracle_cost, returns):
    return {
        "forecast_vs_oracle_cost_rank": rank_correlation(predicted_cost, oracle_cost),
        "oracle_cost_vs_return_rank": rank_correlation(-np.asarray(oracle_cost), returns),
        "forecast_cost_vs_return_rank": rank_correlation(-np.asarray(predicted_cost), returns),
        "return_spread": float(np.ptp(returns)), "return_informative": bool(np.ptp(returns) > 1e-6),
        "candidate_best_return": float(np.max(returns)),
        "learned_selection": selection(predicted_cost, returns),
        "oracle_latent_selection": selection(oracle_cost, returns),
    }


@torch.no_grad()
def encode_images(model, images, chunk_size):
    result = torch.cat([model.encode({"image": chunk[:, None].to(model.device)})[:, 0]
                        for chunk in images.split(chunk_size)])
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite encoded simulator images.")
    return result


@torch.no_grad()
def score_case(model, case, chunk_size):
    actions = case["action"].to(model.device)
    count, horizon = actions.shape[:2]
    with readout_mode(model):
        prefix = model.encode({"image": case["prefix"][None].to(model.device)})
        past = case["past_action"][None].to(model.device)
        goal = encode_images(model, case["goal_image"][None], chunk_size)[0]
        actual = encode_images(model, case["image"].flatten(0, 1), chunk_size).reshape(
            count, horizon, *prefix.shape[2:])
        predicted = torch.cat([model.rollout(prefix, past, part[None])[0] for part in actions.split(chunk_size)])
        # An independent teacher-forced path detects action/history offsets in the optimized rollout.
        observed_history = torch.cat((prefix.expand(count, *prefix.shape[1:]), actual), dim=1)
        all_actions = torch.cat((past.expand(count, *past.shape[1:]), actions), dim=1)
        teacher = []
        for step in range(horizon):
            teacher.append(torch.cat([
                model.predict(observed_history[i:i + chunk_size, step:step + model.history_size],
                              all_actions[i:i + chunk_size, step:step + model.history_size])[:, -1]
                for i in range(0, count, chunk_size)
            ]))
        teacher = torch.stack(teacher, dim=1)
        if not torch.isfinite(predicted).all() or not torch.isfinite(teacher).all():
            raise ValueError("Non-finite native forecasts.")
        first_error = (teacher[:, 0] - predicted[:, 0]).abs().max().item()
        torch.testing.assert_close(teacher[:, 0], predicted[:, 0], rtol=2e-3, atol=2e-3)
        reduce = torch.sum if model.goal_reduction == "sum" else torch.mean
        predicted_cost = reduce((predicted[:, -1] - goal).square().flatten(1), dim=1).cpu().tolist()
        oracle_cost = reduce((actual[:, -1] - goal).square().flatten(1), dim=1).cpu().tolist()
        errors = {name: (value - actual).flatten(2).square().mean((0, 2)).sqrt().cpu().tolist()
                  for name, value in (("open_loop", predicted), ("teacher_forced", teacher),
                                      ("persistence", prefix[:, -1:]))}
        spread = {name: value[:, -1].flatten(1).std(0, unbiased=False).square().mean().sqrt().item()
                  for name, value in (("predicted", predicted), ("actual", actual))}
    returns = case["reward"].sum(1).tolist()
    physical_cost = (case["relation"][:, -1] / model.goal_tolerance.cpu()).square().sum(-1).tolist()
    return {"id": case["id"], "rollin_policy": case["rollin_policy"],
            "anchor_reward": case["anchor_reward"], "horizon": horizon,
            "first_step_max_error": first_error, "latent_rmse_by_horizon": errors,
            "terminal_candidate_latent_rms_std": spread,
            "predicted_cost": predicted_cost, "oracle_latent_cost": oracle_cost,
            "returns": returns, "rewards": case["reward"].tolist(),
            "physical_goal_relations": case["relation"].tolist(),
            "terminal_physical_goal_cost": physical_cost,
            "oracle_vs_physical_goal_rank": rank_correlation(oracle_cost, physical_cost),
            **ranking_summary(predicted_cost, oracle_cost, returns)}


@contextmanager
def preserved_native_state(model):
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, mode in modes:
            module.training = mode


@tools.preserve_rng_state
def native_probe(model, cases, batch_size):
    """Diagnostic real clips, not a production update or an upstream architecture parity claim."""
    result = {}
    for source in ("zero", "random"):
        selected = [case for case in cases if case["rollin_policy"] == source]
        observations, actions = [], []
        # Round-robin anchors before candidates; no single anchor fills the entire batch.
        for index in range(selected[0]["action"].shape[0]):
            for case in selected:
                observations.append(torch.cat((case["prefix"], case["image"][index, :1])))
                actions.append(torch.cat((case["past_action"], case["action"][index, :1])))
        obs = {"image": torch.stack(observations[:batch_size]).to(model.device)}
        action = torch.stack(actions[:batch_size]).to(model.device)
        with preserved_native_state(model):
            model.eval()
            with torch.no_grad():
                evaluation = model.encode(obs)
                eval_prediction = model.predict(evaluation[:, :-1], action)
            # Enable only batch statistics, including neither module nor functional dropout.
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.train()
            latent = model.encode(obs)
            loss, metrics = model.representation_loss(obs, latent, action)
            parameters = [(name, value) for name, value in model.named_parameters()
                          if value.requires_grad and not name.startswith(("state_head.", "decoder."))]
            gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
            if not torch.isfinite(loss) or any(g is not None and not torch.isfinite(g).all() for g in gradients):
                raise ValueError(f"Non-finite native loss/gradients on {source} clips.")
            norms = {}
            for (name, _), gradient in zip(parameters, gradients, strict=True):
                if gradient is not None:
                    key = name.split(".")[0]
                    norms[key] = norms.get(key, 0.) + gradient.detach().double().square().sum().item()
            result[source] = {
                "clips": len(action), "distinct_anchors": min(len(selected), len(action)),
                "loss": float(loss.detach()), "components": {k: float(v.detach()) for k, v in metrics.items()},
                "gradient_l2": {k: v**.5 for k, v in norms.items()},
                "train_eval_feature_rmse": (latent.detach() - evaluation).square().mean().sqrt().item(),
                "train_prediction_rmse": float(metrics["prediction_loss"].detach().sqrt()),
                "eval_prediction_rmse": (eval_prediction - evaluation[:, 1:]).square().mean().sqrt().item(),
                "batchnorm_layers": sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),
            }
    return result


def load_case(path, scenario, family, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["training_config"])
    config.device = args.device
    if (str(config.scenario.name), str(config.model_family), int(config.seed)) != (scenario, family, args.seed):
        raise ValueError("Checkpoint identity does not match the requested run.")
    validate_checkpoint(checkpoint, config, training=False)
    if checkpoint["compatibility"]["training_sha256"] != checkpoint_compatibility(config)["training_sha256"]:
        raise ValueError("Checkpoint training metadata is inconsistent.")
    legacy = config.jepa_model.goal.get("source") != "physical_render_v1"
    if legacy:
        if not args.latent_goals:
            raise ValueError("Legacy controller: pass --latent-goals to explicitly audit current latent costs on old weights.")
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
            current = compose(config_name=f"{family}_dmc_vision", overrides=[f"scenario={scenario}"])
        config.jepa_model.goal = OmegaConf.to_container(current.jepa_model.goal, resolve=True)
        config.env.goal = OmegaConf.to_container(config.jepa_model.goal, resolve=True)
    upgrade_readout_config(config)
    tools.configure_randomness(config.seed, bool(config.deterministic_run))
    module = load_model_family(family)
    model = module.build_model(config)
    module.load_checkpoint(model, checkpoint, training=False)
    return config, model, {"legacy_goal_override": legacy,
                           "checkpoint_recipe": checkpoint["compatibility"].get("recipe_version"),
                           "native_probe_recipe": TRAINING_RECIPE_VERSION,
                           "goal": OmegaConf.to_container(config.jepa_model.goal, resolve=True)}


def write_report(output, results, data, args):
    report = {"version": 1, "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "checkpoint_writes": False, "optimizer_updates": 0, "head_used": False,
              "policy_optimization": False, "data": data, "results": results}
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Planner oracle | same simulator branches/candidates | no fitting or checkpoint writes",
             "Run | Informative return cases | Rank forecast/oracle, oracle/return, oracle/physical | Regret learned/oracle | Time"]
    def average(values):
        usable = [v for v in values if v is not None]
        return f"{sum(usable) / len(usable):.3g}" if usable else "n/a"
    for result in results:
        name = f"{result['scenario']}/{result['model']}/{result['checkpoint']}"
        if result["status"] == "FAIL":
            lines.append(f"FAIL | {name} | {result['error']}")
            continue
        cases = result["cases"]
        lines.append(f"COMPLETE | {name} | {sum(c['return_informative'] for c in cases)}/{len(cases)} | "
                     f"{average([c['forecast_vs_oracle_cost_rank'] for c in cases])}/"
                     f"{average([c['oracle_cost_vs_return_rank'] for c in cases])}/"
                     f"{average([c['oracle_vs_physical_goal_rank'] for c in cases])} | "
                     f"{average([c['learned_selection']['regret'] for c in cases])}/"
                     f"{average([c['oracle_latent_selection']['regret'] for c in cases])} | "
                     f"{duration(result['elapsed_seconds'])}")
        for source, probe in result["native_probe"].items():
            lines.append(f"  {source} | BN train/eval feature gap={probe['train_eval_feature_rmse']:.3g} | "
                         f"prediction RMSE train/eval={probe['train_prediction_rmse']:.3g}/"
                         f"{probe['eval_prediction_rmse']:.3g}")
    lines.extend(("Rank is Spearman (higher is better); constant scores return n/a, not a pass.",
                  "Physical cost is terminal squared goal relation/tolerance, not reward; useful when sparse returns tie.",
                  "Regret = best candidate's cumulative reward minus selected reward; ties use mean return.",
                  "These are short diagnostic branches, not optimized policies or whole-episode evaluations.",
                  "Native probe: correlated simulator clips, dropout disabled, no optimizer steps; BN buffers restored.",
                  "COMPLETE means execution only. Raw actions, costs, errors, gradients, and cohort details are in report.json."))
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/dmc_vision_10k"))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["cartpole_balance_sparse"])
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--checkpoints", nargs="+", choices=("pretrained.pt", "final.pt", "best.pt"),
                        default=["pretrained.pt", "final.pt"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-seeds", nargs="+", type=int, default=[730001, 730002])
    parser.add_argument("--rollin-steps", nargs="+", type=int, default=[0, 32])
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-goals", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("planner_oracle_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.candidates < 3 or args.batch_size < 2 or min(args.rollin_steps) < 0:
        parser.error("Need at least 3 candidates, batch-size >=2, and nonnegative roll-in steps.")
    for name in ("scenarios", "models", "checkpoints", "sim_seeds", "rollin_steps"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"Duplicate {name} would repeat unchanged work.")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    results, data = [], {}
    print("Planner oracle | simulator collected once/scenario | zero native/head updates", flush=True)
    for scenario in args.scenarios:
        cases, signature = None, None
        for family in args.models:
            for filename in args.checkpoints:
                result = {"scenario": scenario, "model": family, "checkpoint": filename, "status": "FAIL"}
                started = time.monotonic()
                try:
                    path = args.run_root / scenario / family / "default" / f"seed_{args.seed}" / filename
                    config, model, metadata = load_case(path, scenario, family, args)
                    if cases is None:
                        signature = environment_signature(config)
                        cases = collect_cases(config, args)
                        data[scenario] = {"environment": signature, "cases": [
                            {**{key: case[key] for key in ("id", "seed", "anchor_agent_step", "anchor_simulator_state")},
                             "action": case["action"].tolist(), "past_action": case["past_action"].tolist(),
                             "sha256": tensor_digest({k: v for k, v in case.items() if isinstance(v, torch.Tensor)})}
                            for case in cases]}
                    elif environment_signature(config) != signature:
                        raise ValueError("Models/checkpoints differ in environment, goal, history, or horizon; use separate runs.")
                    before = tensor_digest(model.state_dict())
                    scored = [score_case(model, case, args.batch_size) for case in cases]
                    probe = native_probe(model, cases, args.batch_size)
                    json.dumps({"cases": scored, "native_probe": probe}, allow_nan=False)
                    if tensor_digest(model.state_dict()) != before:
                        raise RuntimeError("Diagnostic mutated native weights or normalization buffers.")
                    result.update(status="COMPLETE", cases=scored, native_probe=probe, model_unchanged=True,
                                  model_sha256=before, **metadata)
                    del model
                except Exception as error:  # Preserve previous checkpoints' results on a worker failure.
                    result["error"] = f"{type(error).__name__}: {error}"
                    (args.output / f"{scenario}_{family}_{filename}.log").write_text(traceback.format_exc(), encoding="utf-8")
                result["elapsed_seconds"] = time.monotonic() - started
                results.append(result)
                write_report(args.output, results, data, args)
                print(f"{result['status']} | {scenario}/{family}/{filename} | {duration(result['elapsed_seconds'])}", flush=True)
    print("\n".join(write_report(args.output, results, data, args)), flush=True)
    print(f"Reports | {args.output.resolve()}", flush=True)
    return int(any(result["status"] == "FAIL" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
