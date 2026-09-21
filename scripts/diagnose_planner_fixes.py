"""One small offline fit per family; paired controller changes and native fitting controls."""

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity
from envs.dmc import make_env
from models.shared.latent_goal import latent_goal_cost
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import case_metadata, collect_objective_cases
from scripts.diagnose_planner_oracle import encode_images, ranking_summary
from scripts.diagnose_planning_horizons import evaluate_horizon
from scripts.predictor_fit_control import fit_control
from scripts.train_planner_check import build_config, new_model
from scripts.train_rollout_check import pretrain
from scripts.upstream_ts_probe import COMMIT, parity, sources, training_parity
from training import load_model_family
from training.progress import duration
from training.protocol import implementation_sha256


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    lines = ["Planner corrections | one small offline fit/model | no online training or checkpoint writes",
             "Model/arm | Return | Tail success | Candidate return: learned / true-latent / uniform | Status"]
    for run in report["runs"]:
        for arm in run.get("arms", []):
            rows = arm["rankings"]
            selected = [np.mean([r[k]["return_mean"] for r in rows]) for k in ("learned_selection", "oracle_latent_selection")]
            uniform = np.mean([np.mean(r["returns"]) for r in rows])
            policy = arm["policy"]
            lines.append(f"{run['model']}/{arm['name']} | {policy['return_mean']:.2f}/{policy['maximum_return']} | "
                         f"{policy['sustained_rate']:.0%} | {selected[0]:.3g}/{selected[1]:.3g}/{uniform:.3g} | COMPLETE")
        fit = run.get("fit_control", {})
        if "after" in fit:
            lines.append(f"{run['model']} fitting | weights_changed={fit['weights_changed']}")
            for split in ("train", "validation"):
                def value(stage, key):
                    return np.mean([r[key] for r in fit[stage][split]])
                lines.append(f"{run['model']} fitting/{split} | TF h1 {value('before', 'teacher_h1'):.3g}->{value('after', 'teacher_h1'):.3g} | "
                             f"recursive h{fit['before'][split][0]['horizon']} {value('before', 'recursive_final'):.3g}->{value('after', 'recursive_final'):.3g}")
        lines.append(f"{run['model']} | {run['status']}" + (f" | {run['error']}" if 'error' in run else ""))
    lines += ["Goal sets and tail costs are explicit DMC adaptations; TS native uses upstream MPC weighting, LeWM native stays terminal-only.",
              "Same offline weights/candidates/initial states per arm. Simulator futures and rewards are diagnostic targets, never planner inputs.",
              "Fitting control changes disposable predictor copies only; episode/seed splits are disjoint. No physical-head planning.",
              "COMPLETE means execution, not repair. Short intervention rollouts are not full task-success evaluations.",
              "Defaults only test cartpole. Goal-set rendering for other tasks is covered by simulator unit tests."]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def cache_predictions(model, cases):
    rows = []
    with readout_mode(model):
        for case in cases:
            history = model.encode({"image": case["prefix"][None].to(model.device)})
            past = case["past_action"][None].to(model.device)
            actions = torch.as_tensor(case["action"], device=model.device)
            prediction = torch.cat([model.rollout(history, past, c[None])[0] for c in actions.split(8)])
            actual = encode_images(model, case["image"].flatten(0, 1), 8).reshape_as(prediction)
            rows.append((history, prediction[None], actual[None]))
    return rows


def fit_cases(cases):
    return [{"id": c["id"], "episode": c["seed"], "start": 0, "prefix": c["prefix"], "past": c["past_action"],
             "actions": torch.as_tensor(c["action"]), "images": c["image"], "goal_image": c["goal_image"],
             "pose_distance": np.max(np.abs(c["states"][..., :2]) / c["tolerance"], axis=-1).tolist()}
            for c in cases]


def run_model(name, args, output, result, cases, source, persist):
    config = build_config(name, args)
    config.jepa_model.planner.horizon = args.horizon
    result["config"] = OmegaConf.to_container(config, resolve=True)
    with load_model_family(name).build_replay(config) as dataset:
        result["dataset_identity"] = dataset_identity(dataset.metadata)
        model = new_model(config, dataset)
        if name == "temporal_straightening":
            sampler = dataset.state_dict()
            try:
                batch = dataset.sample_episode_batch()
            finally:
                dataset.load_state_dict(sampler)
            with readout_mode(model):
                latent = model.encode({"image": batch[0]["image"][:2].to(model.device)})
                actions = batch[1][:2].to(model.device)
                result["parity"] = parity(model, source, latent[:1, :-1], actions[:1, :-1], actions[:1, -1:, None].expand(-1, 1, args.horizon, -1))
                result["training_parity"] = training_parity(model, source, latent, actions)
            if any(result[k]["status"] != "PASS" for k in ("parity", "training_parity")):
                raise ValueError("Upstream parity mismatch; stopped before offline training.")
        model = pretrain(config, dataset, args, output, result, model=model)
    frozen = tensor_digest(model.state_dict())
    cached = cache_predictions(model, cases)
    result["arms"] = []
    native = "ts_mpc" if name == "temporal_straightening" else "last"
    arms = [("legacy_terminal", "last", False)]
    if native != "last":
        arms.append(("native_mpc", native, False))
    arms += [("native_goal_set", native, True), ("stable_goal_set", "tail", True)]
    for label, objective, goal_set in arms:
        arm_config = copy.deepcopy(config)
        arm_config.jepa_model.planner.objective = objective
        if goal_set:
            arm_config.jepa_model.goal.alternatives = arm_config.scenario.goal_alternatives
        # build_config resolves interpolations: update env goal explicitly for this arm.
        arm_config.env.goal = copy.deepcopy(arm_config.jepa_model.goal)
        model.planner = arm_config.jepa_model.planner
        env = make_env(arm_config.env, args.seed + 12_000_000)
        try:
            observation = env.reset()
            goal_images = observation.get("goal_images", observation["goal_image"][None])
        finally:
            env.close()
        with readout_mode(model):
            goals = encode_images(model, torch.from_numpy(goal_images), 8)[None]
        rows = []
        for case, (history, predicted, actual) in zip(cases, cached, strict=True):
            costs = [latent_goal_cost(path, goals, reduction=model.goal_reduction, mode=objective,
                                      history=history, tail_steps=3)[0].cpu().tolist() for path in (predicted, actual)]
            returns = case["rewards"].sum(1).tolist()
            rows.append({"id": case["id"], "predicted_cost": costs[0], "actual_cost": costs[1], "returns": returns,
                         **ranking_summary(*costs, returns)})
        folder = output / label
        folder.mkdir()
        policy = evaluate_horizon(arm_config, model, cases, args.horizon, args, folder)
        result["arms"].append({"name": label, "objective": objective, "goal_count": goal_images.shape[0],
                               "rankings": rows, "policy": policy})
        persist()
    result["fit_control"] = fit_control(model, fit_cases(cases), 64, args.fit_updates)
    if tensor_digest(model.state_dict()) != frozen:
        raise RuntimeError("Controller/fitting checks changed the original model.")
    result["status"] = "COMPLETE"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("temporal_straightening", "leworldmodel"),
                        default=["temporal_straightening", "leworldmodel"])
    parser.add_argument("--expert-updates", type=int, default=1000)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upstream-cache", type=Path, default=Path("local/upstream_ts") / COMMIT)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("planner_fixes_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if (min(args.expert_updates, args.fit_updates, args.policy_steps) < 1 or args.horizon < 3
            or max(args.horizon, args.policy_steps) >= 498 or args.seed < 0 or len(set(args.models)) != len(args.models)):
        parser.error("Use positive budgets, horizon >= 3, horizon/policy-steps < 498, and unique models.")
    args.scenario = "cartpole_balance_sparse"
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA test requires BF16 support.")
    source = sources(args.upstream_cache) if "temporal_straightening" in args.models else None
    args.output.mkdir(parents=True, exist_ok=False)
    config = build_config(args.models[0], args)
    cases = collect_objective_cases(config, SimpleNamespace(
        sim_seeds=[args.seed + 12_000_000, args.seed + 12_000_001],
        horizons=list(range(1, args.horizon + 1)), candidates=16), history_size=3)
    files = [Path(__file__).with_name(name) for name in (
        "diagnose_planner_fixes.py", "predictor_fit_control.py", "upstream_ts_probe.py",
        "action_conditioning_support.py", "diagnose_goal_objective.py", "diagnose_planning_horizons.py",
        "train_planner_check.py", "train_rollout_check.py")]
    report = {"implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cases": [case_metadata(c) for c in cases],
              "checkpoint_reads": False, "checkpoint_writes": False, "online_updates": 0, "runs": []}
    for name in args.models:
        result = {"model": name, "status": "RUNNING"}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        started = time.monotonic()
        try:
            run_model(name, args, output, result, cases, source, lambda: write_report(args.output, report))
        except Exception as error:
            import traceback
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc())
        result["seconds"] = time.monotonic() - started
        write_report(args.output, report)
        print(f"{result['status']} | {name} | {duration(result['seconds'])}", flush=True)
    print((args.output / "summary.txt").read_text(), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(r["status"] != "COMPLETE" for r in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
