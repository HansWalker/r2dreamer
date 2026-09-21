"""Staged optimizer and offline trajectory-goal checks. Never launches online training."""

import argparse
import copy
import gc
import hashlib
import json
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import PROFILES, cart_state, observe_prefix, set_cart_state
from scripts.diagnose_planner_oracle import rank_correlation, simulator_branch
from scripts.diagnose_planning_horizons import evaluate_horizon, synchronize
from scripts.planner_recipe_support import (
    BlockReplay, NormalizedActionEncoder, action_statistics, block_actions, heldout_pairs,
    optimizer_arm, pose_distance, reference_configs, resize_images,
)
from scripts.smoke_tiny_planners import FAMILIES, native_control
from scripts.train_planner_check import build_config, new_model
from scripts.train_rollout_check import pretrain
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256


def optimizer_cases(config, seed):
    env = make_env(config.env, seed)
    cases = []
    try:
        for name, (cohort, state) in PROFILES.items():
            observation = env.reset()
            set_cart_state(env, state)
            prefix = observe_prefix(env, 3)
            cases.append({"id": name, "cohort": cohort, "seed": seed, "initial_state": state,
                          "anchor_state": cart_state(env), "goal_image": torch.from_numpy(np.ascontiguousarray(observation["goal_image"])),
                          "anchor_success": float(env._env.task.get_reward(env._env.physics)) >= 1 - 1e-6,
                          **prefix})
    finally:
        env.close()
    return cases


def optimizer_check(args, output, result, persist):
    config = build_config("temporal_straightening", args)
    config.jepa_model.planner.objective = "last"  # Keep the historical optimizer experiment paired.
    # Use the production planner budget, not the old eight-iteration tiny-test budget.
    config.jepa_model.planner.samples = 16
    config.jepa_model.planner.iterations = 32
    result["config"] = OmegaConf.to_container(config, resolve=True)
    with load_model_family(config.model_family).build_replay(config) as dataset:
        mean, std, count = action_statistics(dataset)
        result["action_statistics"] = {"mean": mean.tolist(), "std": std.tolist(), "training_transitions": count}
        result["dataset_identity"] = dataset_identity(dataset.metadata)
        settings = copy.copy(args)
        settings.expert_updates = args.optimizer_updates
        model = pretrain(config, dataset, settings, output, result)
    cases = optimizer_cases(config, args.seed + 12_000_000)
    frozen = tensor_digest(model.state_dict())
    result["arms"] = []
    for name in ("current", "tanh", "direct"):
        folder = output / name
        folder.mkdir()
        arm = {"name": name, "status": "RUNNING"}
        result["arms"].append(arm)
        try:
            with optimizer_arm(model, name, mean, std):
                arm["settings"] = {
                    "iterations": int(model.planner.iterations), "restarts": int(model.planner.samples),
                    "lr": .1 if name == "direct" else 1.,
                    "initialization": "random" if name == "current" else "zero_physical_action",
                    "noise": float(model.planner.action_noise),
                    "parameterization": "normalized_direct_projected" if name == "direct" else "tanh",
                }
                arm["policy"] = evaluate_horizon(config, model, cases, 5, args, folder)
            actions = np.asarray([json.loads(line)["actions"] for line in
                                  (folder / "policy_metrics.jsonl").read_text().splitlines()])
            arm["action_saturation"] = float((np.abs(actions) > .95).mean())
            if tensor_digest(model.state_dict()) != frozen:
                raise RuntimeError("Optimizer comparison changed frozen weights/buffers.")
            arm["status"] = "COMPLETE"
        except Exception as error:
            arm.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (folder / "error.log").write_text(traceback.format_exc())
        persist()
    result["status"] = "FAIL" if any(a["status"] == "FAIL" for a in result["arms"]) else "COMPLETE"


def reset_pair(env, case, stride):
    env.reset()
    set_cart_state(env, case["initial_state"])
    images = [env.render()]
    for action in case["prefix_actions"]:
        observation, _, done, _ = env.step(action)
        if done:
            raise ValueError("Reference prefix reached a terminal state.")
        images.append(observation["image"])
    np.testing.assert_allclose(cart_state(env), case["anchor_state"], rtol=0, atol=1e-6)
    return torch.from_numpy(np.stack(images[::stride]))


@tools.preserve_rng_state
def reference_policy(config, model, cases, args, output, name, mean, std):
    envs = []
    before = tensor_digest(model.state_dict())
    cache = model._cem_mean, model._gradient_actions
    model._cem_mean = model._gradient_actions = None
    started = time.monotonic()
    ts = config.model_family == "temporal_straightening"
    arm = optimizer_arm(model, "direct", mean, std) if ts and name == "planner" else nullcontext()
    stride = args.stride
    image_size = int(config.model_io.observations.image[0])
    distances, actions_log, returns, policy_times = [], [], np.zeros(len(cases)), []
    try:
        prefixes = []
        for case in cases:
            env = make_env(config.env, args.seed + 13_000_000)
            envs.append(env)
            prefixes.append(reset_pair(env, case, stride))
        history = resize_images(torch.stack(prefixes).to(model.device), image_size)
        goals = resize_images(torch.stack([c["goal_image"] for c in cases]).to(model.device), image_size)
        past = block_actions(torch.from_numpy(np.stack([c["prefix_actions"] for c in cases])).to(model.device), stride)
        expert = torch.from_numpy(np.stack([c["expert_actions"] for c in cases])).to(model.device)
        with arm, (output / f"{name}_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            progress = Progress(f"{config.model_family} reference {name}", args.reference_blocks)
            for step in range(args.reference_blocks):
                torch.manual_seed(args.seed + 14_000_000 + step)
                first = torch.full((len(cases),), step == 0, device=model.device, dtype=torch.bool)
                synchronize(model)
                tick = time.monotonic()
                if name == "planner":
                    with torch.no_grad():
                        action = native_control(model, lambda: model.act(
                            {"image": history, "goal_image": goals}, past, deterministic=True, first=first))
                elif name == "expert" and (step + 1) * stride <= expert.shape[1]:
                    action = expert[:, step * stride:(step + 1) * stride].flatten(1)
                else:
                    action = torch.zeros(len(cases), model.action_dim, device=model.device)
                if not torch.isfinite(action).all() or (action.abs() > 1 + 1e-6).any():
                    raise ValueError("Reference controller returned non-finite/out-of-bounds actions.")
                synchronize(model)
                policy_times.append(time.monotonic() - tick)
                controls = action.detach().cpu().numpy().reshape(len(cases), stride, -1)
                images, state, distance = [], [], []
                for index, (env, sequence) in enumerate(zip(envs, controls, strict=True)):
                    for control in sequence:
                        observation, reward, done, _ = env.step(control)
                        returns[index] += reward
                        if done:
                            raise ValueError("Reference policy exceeded episode length.")
                    images.append(observation["image"])
                    state.append(cart_state(env).tolist())
                    distance.append(float(pose_distance(state[-1], cases[index]["goal_state"], args.goal_tolerance)))
                distances.append(distance)
                actions_log.append(controls)
                history = torch.cat((history[:, 1:], resize_images(torch.from_numpy(np.stack(images)).to(model.device), image_size)[:, None]), 1)
                past = torch.cat((past[:, 1:], action.detach()[:, None]), 1)
                log.write(json.dumps({"block": step + 1, "actions": controls.tolist(), "states": state,
                                      "pose_distance": distance, "returns": returns.tolist(),
                                      "policy_seconds": policy_times[-1]}, allow_nan=False) + "\n")
                progress.update(step + 1, f"goal={(np.min(distances, axis=0) <= 1).mean():.0%}",
                                force=step + 1 == args.reference_blocks)
        if tensor_digest(model.state_dict()) != before:
            raise RuntimeError("Reference control changed frozen weights/buffers.")
        distance = np.asarray(distances)
        return {"success_rate": float((distance.min(0) <= 1).mean()),
                "final_success_rate": float((distance[-1] <= 1).mean()),
                "minimum_pose_distance_mean": float(distance.min(0).mean()),
                "final_pose_distance_mean": float(distance[-1].mean()),
                "success_by_case": (distance.min(0) <= 1).tolist(),
                "return_mean": float(returns.mean()), "actions_saturated": float((np.abs(actions_log) > .95).mean()),
                "policy_seconds_per_block": float(np.mean(policy_times)), "seconds": time.monotonic() - started,
                "initial_state_sha256": before, "final_state_sha256": tensor_digest(model.state_dict())}
    finally:
        model._cem_mean, model._gradient_actions = cache
        for env in envs:
            env.close()


@tools.preserve_rng_state
@torch.no_grad()
def reference_rankings(config, model, cases, args):
    """Known expert path is a diagnostic candidate only, never supplied to the planner."""
    rng = np.random.default_rng(args.seed + 15_000_000)
    rows = []
    image_size = int(config.model_io.observations.image[0])
    env = make_env(config.env, args.seed + 13_000_000)
    try:
        with readout_mode(model):
            for case in cases:
                prefix = reset_pair(env, case, args.stride)
                path = case["expert_actions"]
                candidates = np.concatenate((path[None], np.zeros_like(path)[None], -path[None],
                                             rng.uniform(-1, 1, (9, *path.shape)).astype(np.float32)))
                actual, distance = [], []
                for sequence in candidates:
                    with simulator_branch(env) as branch:
                        for action in sequence:
                            observation, _, done, _ = branch.step(action)
                            if done:
                                raise ValueError("Reference branch crossed episode boundary.")
                        actual.append(observation["image"])
                        distance.append(float(pose_distance(cart_state(branch), case["goal_state"], args.goal_tolerance)))
                latent = model.encode({"image": resize_images(prefix[None].to(model.device), image_size)})
                past = block_actions(torch.from_numpy(case["prefix_actions"])[None].to(model.device), args.stride)
                actions = block_actions(torch.from_numpy(candidates).to(model.device), args.stride)
                predicted = native_control(model, lambda: model.rollout(latent, past, actions[None]))[0, :, -1]
                true = model.encode({"image": resize_images(torch.from_numpy(np.stack(actual)).to(model.device), image_size)})
                goal = model.encode({"image": resize_images(case["goal_image"][None].to(model.device), image_size)})[0]
                reduce = torch.sum if model.goal_reduction == "sum" else torch.mean
                predicted_cost = reduce((predicted - goal).square().flatten(1), -1).cpu().numpy()
                actual_cost = reduce((true - goal).square().flatten(1), -1).cpu().numpy()
                if not np.isfinite(predicted_cost).all() or not np.isfinite(actual_cost).all():
                    raise ValueError("Reference ranking produced non-finite latent costs.")
                def selected(cost):
                    # Do not let candidate ordering make ties favor the expert.
                    mask = np.isclose(cost, cost.min(), rtol=1e-6, atol=1e-8)
                    return float(np.asarray(distance)[mask].mean())
                rows.append({"id": case["id"], "predicted_cost": predicted_cost.tolist(),
                             "actual_cost": actual_cost.tolist(), "physical_pose_distance": distance,
                             "forecast_vs_actual_rank": rank_correlation(predicted_cost, actual_cost),
                             "actual_vs_pose_rank": rank_correlation(actual_cost, distance),
                             "selected_pose_distance": {"predicted": selected(predicted_cost), "actual": selected(actual_cost),
                                                        "uniform": float(np.mean(distance)), "expert": distance[0]},
                             "forecast_mse": float((predicted - true).square().mean()),
                             "persistence_mse": float((latent[0, -1] - true).square().mean())})
    finally:
        env.close()
    return rows


def reference_gate(policies, count, minimum):
    if count < minimum or policies["expert"]["success_rate"] < 1:
        return "UNVALIDATED"
    model, zero = policies["planner"], policies["zero"]
    if (model["success_rate"] >= .5 and model["success_rate"] >= zero["success_rate"] + .25
            and model["minimum_pose_distance_mean"] < zero["minimum_pose_distance_mean"]):
        return "OFFLINE_CONTROL_OBSERVED"
    return "NO_CONTROL_EVIDENCE"


def reference_check(name, args, output, result, persist):
    raw_config, config = reference_configs(build_config(name, args), args.stride)
    result["config"] = OmegaConf.to_container(config, resolve=True)
    result["raw_data_config"] = OmegaConf.to_container(raw_config, resolve=True)
    with load_model_family(name).build_replay(raw_config) as dataset:
        mean, std, count = action_statistics(dataset)
        mean, std = np.tile(mean, args.stride), np.tile(std, args.stride)
        result["dataset_identity"] = dataset_identity(dataset.metadata)
        result["action_statistics"] = {"mean": mean.tolist(), "std": std.tolist(), "training_transitions": count}
        env = make_env(config.env, args.seed + 13_000_000)
        try:
            cases, attempted = heldout_pairs(dataset, env, stride=args.stride, horizon=5,
                                            count=args.pairs, seed=args.seed + 16_000_000, tolerance=args.goal_tolerance)
        finally:
            env.close()
        result["pair_restorations_attempted"] = attempted
        result["cases"] = [{key: value for key, value in case.items() if key not in
                            {"prefix", "prefix_actions", "expert_actions", "goal_image"}} for case in cases]
        if len(cases) < args.minimum_pairs:
            result.update(status="UNVALIDATED", reason="Insufficient nontrivial, reproducible heldout goal pairs; no training launched.")
            return
        adapter = BlockReplay(dataset, args.stride, int(config.model_io.observations.image[0]), torch.device(args.device))
        model = new_model(config, adapter)
        model.action_encoder = NormalizedActionEncoder(model.action_encoder, mean, std, model.device)
        settings = copy.copy(args)
        settings.expert_updates = args.reference_updates
        pretrain(config, adapter, settings, output, result, model=model)
    result["policies"] = {}
    for policy in ("expert", "zero", "planner"):
        result["policies"][policy] = reference_policy(config, model, cases, args, output, policy, mean, std)
        persist()
    before = tensor_digest(model.state_dict())
    result["rankings"] = reference_rankings(config, model, cases, args)
    if tensor_digest(model.state_dict()) != before:
        raise RuntimeError("Reference ranking changed frozen weights/buffers.")
    result["status"] = reference_gate(result["policies"], len(cases), args.minimum_pairs)


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    lines = ["Planner recipe check | offline only | no checkpoints | production unchanged"]
    for run in report["runs"]:
        if run["stage"] == "optimizer":
            lines.append("TS optimizer | return | tail success | saturated actions | seconds/call")
            for arm in run.get("arms", []):
                if arm["status"] == "COMPLETE":
                    p = arm["policy"]
                    warm = p["timing"]["policy_warm_seconds"]
                    warm_text = "n/a" if warm is None else f"{warm:.3f}"
                    lines.append(f"{arm['name']} | {p['return_mean']:.1f}/{p['maximum_return']} | {p['sustained_rate']:.0%} | "
                                 f"{arm['action_saturation']:.0%} | {warm_text}")
                else:
                    lines.append(f"{arm['name']} | {arm['status']} | {arm.get('error', '')}")
        else:
            lines.append(f"Reference {run['model']} | {run['status']} | heldout pairs={len(run.get('cases', []))}")
            for name, p in run.get("policies", {}).items():
                lines.append(f"  {name} | reached={p['success_rate']:.0%} | final={p['final_success_rate']:.0%} | "
                             f"pose min/final={p['minimum_pose_distance_mean']:.3g}/{p['final_pose_distance_mean']:.3g} | "
                             f"planner={p['policy_seconds_per_block']:.3f}s/block")
        if "error" in run or "reason" in run:
            lines.append(run.get("error", run.get("reason")))
    lines += [
        "Current TS: 16 restarts, 32 steps, random init/noise. Tanh/direct: one restart, 100 steps, zero init/no noise.",
        "Direct optimizes train-standardized coordinates, initialized at zero physical action; projection to DMC bounds is an adaptation.",
        "All arms retain the SAME terminal latent cost; TS upstream intermediate-state MPC cost is NOT introduced.",
        "Reference: train-only normalized action blocks; disjoint trajectory goals; 224px upsampling of 64px images, NOT new visual detail.",
        "Reference is a larger local vision-only adaptation, NOT an exact upstream reproduction or a parameter-matched comparison.",
        "LeWM retains local bounded physical-coordinate CEM proposals; model action inputs are train-standardized.",
        f"{report['settings']['stride']} stored actions/block means {2 * report['settings']['stride']} raw DMC steps (repeat=2). Pose success ignores velocity, NOT sustained success.",
        "Expert replay verifies reachability; zero actions reveal trivial goals. Neither supplies privileged actions/state to the planner.",
        "Control gate: >= minimum pairs, expert 100%, planner >=50% and >=25pp over zero, lower mean closest pose distance.",
        "OFFLINE_CONTROL_OBSERVED is a small diagnostic only. No automatic online run, production change, or full retraining.",
        "Detailed settings, frozen-weight hashes, per-case costs and timing are in report.json; updates/controls in JSONL.",
    ]
    if "seconds" in report:
        lines.append(f"Time | {duration(report['seconds'])}")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "optimizer", "reference"), default="all")
    parser.add_argument("--models", choices=FAMILIES, nargs="+", default=list(FAMILIES), help="Reference-stage models; optimizer stage is TS only.")
    parser.add_argument("--optimizer-updates", type=int, default=1000)
    parser.add_argument("--reference-updates", type=int, default=3000)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--reference-blocks", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--minimum-pairs", type=int, default=8)
    parser.add_argument("--goal-tolerance", nargs=2, type=float, default=[.01, .01], metavar=("METERS", "RADIANS"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("planner_recipe_check_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if min(args.optimizer_updates, args.reference_updates, args.policy_steps, args.stride, args.reference_blocks,
           args.pairs, args.minimum_pairs) < 1 or args.seed < 0:
        parser.error("Use positive budgets and nonnegative seeds.")
    if args.minimum_pairs > args.pairs or args.reference_blocks < 5 or (2 + args.reference_blocks) * args.stride >= 500 or args.policy_steps >= 498:
        parser.error("Invalid pair counts or episode length; reference requires at least five blocks.")
    if any(not np.isfinite(t) or t <= 0 for t in args.goal_tolerance) or len(args.models) != len(set(args.models)):
        parser.error("Use finite positive tolerances and unique models.")
    args.scenario = "cartpole_balance_sparse"
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA diagnostics require BF16 support.")
        torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    files = [Path(__file__), Path(__file__).with_name("planner_recipe_support.py")]
    report = {"experiment": "planner_recipe_check", "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "online_updates": 0, "checkpoint_writes": False, "production_settings_changed": False, "runs": []}
    jobs = []
    if args.stage in {"all", "optimizer"}:
        jobs.append(("optimizer", "temporal_straightening"))
    if args.stage in {"all", "reference"}:
        jobs += [("reference", name) for name in args.models]
    print(f"Planner recipe | stages={len(jobs)} | optimizer_fit={args.optimizer_updates} | reference_fit={args.reference_updates} | offline only", flush=True)
    for stage, name in jobs:
        result = {"stage": stage, "model": name, "status": "RUNNING"}
        report["runs"].append(result)
        output = args.output / stage / name
        output.mkdir(parents=True)
        tick = time.monotonic()
        try:
            if stage == "optimizer":
                optimizer_check(args, output, result, lambda: write_report(args.output, report))
            else:
                reference_check(name, args, output, result, lambda: write_report(args.output, report))
        except Exception as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc())
        result["seconds"] = time.monotonic() - tick
        print(f"{result['status']} | {stage}/{name} | {duration(result['seconds'])}", flush=True)
        write_report(args.output, report)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(r["status"] == "FAIL" for r in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
