"""One native offline learning curve per small planner, then a production-schedule online prefix."""

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from envs import close_envs, make_envs
from models.shared.latent_goal import latent_goal_cost
from models.shared.physical_state import format_physical_rmse, readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import case_metadata, collect_objective_cases
from scripts.diagnose_planner_oracle import encode_images, ranking_summary
from scripts.diagnose_planning_horizons import evaluate_horizon
from scripts.smoke_online_checkpoints import online_prefix
from scripts.smoke_tiny_planners import FAMILIES
from scripts.train_planner_check import build_config, checked_update, new_model
from training import load_model_family
from training.evaluation import StateDataset, latent_rollout
from training.progress import Progress, duration
from training.protocol import implementation_sha256
from training.readout import online_readout
from training.trainer import online_update_target


def learning_rates(model):
    return {name: [group["lr"] for group in opt.param_groups] for name, opt in model.optimizers.items()}


def latent_errors(prediction, actual, anchor):
    return {
        "mse": (prediction - actual).square().flatten(2).mean((0, 2)).cpu().tolist(),
        "hold_mse": (anchor - actual).square().flatten(2).mean((0, 2)).cpu().tolist(),
        "target_rms_std": actual.flatten(2).std((0, 1), unbiased=False).square().mean().sqrt().item(),
    }


@torch.no_grad()
def score_candidates(model, cases):
    rows = []
    with readout_mode(model):
        for case in cases:
            history = model.encode({"image": case["prefix"][None].to(model.device)})
            past = case["past_action"][None].to(model.device)
            actions = torch.as_tensor(case["action"], device=model.device)
            predicted = torch.cat([model.rollout(history, past, part[None])[0] for part in actions.split(8)])
            actual = encode_images(model, case["image"].flatten(0, 1), 8).reshape_as(predicted)
            goal = encode_images(model, case["goal_image"][None], 8)
            costs = [latent_goal_cost(
                path[None], goal, reduction=model.goal_reduction,
                mode=model.planner.objective, history=history,
            )[0].cpu().tolist() for path in (predicted, actual)]
            returns = case["rewards"].sum(1).tolist()
            rows.append({"id": case["id"], "predicted_cost": costs[0], "actual_cost": costs[1],
                         "returns": returns, "latent": latent_errors(predicted, actual, history[:, -1:]),
                         **ranking_summary(*costs, returns)})
    informative = [row for row in rows if row["return_informative"]]
    ranks = [row["forecast_vs_oracle_cost_rank"] for row in informative
             if row["forecast_vs_oracle_cost_rank"] is not None]
    selected = {
        name: float(np.mean([row[key]["return_mean"] for row in informative])) if informative else None
        for name, key in (("predicted", "learned_selection"), ("true_latent", "oracle_latent_selection"))
    }
    selected.update(uniform=float(np.mean([np.mean(row["returns"]) for row in informative])) if informative else None,
                    best=float(np.mean([row["candidate_best_return"] for row in informative])) if informative else None)
    return {"cases": rows, "informative_ids": [row["id"] for row in informative],
            "cost_rank": float(np.mean(ranks)) if ranks else None, "rank_count": len(ranks),
            "selected_return": selected,
            "mse": np.mean([row["latent"]["mse"] for row in rows], axis=0).tolist()}


@torch.no_grad()
def score_expert(model, batch):
    observation, action, truth = batch
    observation = {k: v.to(model.device) for k, v in observation.items()}
    action, truth = action.to(model.device), truth.to(model.device)
    context, head = model.history_size, model.state_head
    with readout_mode(model):
        features, prediction = latent_rollout(model, observation, action, context)
        prefix = features[:, context - head.history + 1:context]
        observed = head(features[:, context - head.history + 1:])
        forecast = head(torch.cat((prefix, prediction), dim=1))
        target = truth[:, context:]
        scores = {}
        for source, value in (("observed", observed), ("forecast", forecast),
                              ("true_hold", truth[:, context - 1:context])):
            error = head.targets.metric_error(value, target).square()
            scores[source] = {str(h): head.targets.metric_summary(error[:, h - 1])
                              for h in sorted({1, prediction.shape[1]})}
        return {"latent": latent_errors(prediction, features[:, context:], features[:, context - 1:context]),
                "physical": scores}


@tools.preserve_rng_state
def measure(config, model, cases, expert_batch, args, output):
    started = time.monotonic()
    before = tensor_digest(model.state_dict())
    result = {"expert": score_expert(model, expert_batch), "candidates": score_candidates(model, cases)}
    result["policy"] = evaluate_horizon(config, model, cases, args.horizon, args, output)
    if tensor_digest(model.state_dict()) != before:
        raise RuntimeError("Learning-curve evaluation changed model weights or buffers.")
    result.update(model_sha256=before, learning_rates=learning_rates(model),
                  readout_updates=int(model.state_head.updates), seconds=time.monotonic() - started)
    json.dumps(result, allow_nan=False)
    return result


def validate_budget(config, args):
    quantum = int(config.env.env_num) * int(config.env.action_repeat)
    if args.online_steps and (args.online_steps % quantum or args.online_steps > int(config.training.online.steps)
                              or online_update_target(config, args.online_steps) < 1):
        raise ValueError(f"Online steps must be a multiple of {quantum}, exceed warmup, and fit the production schedule.")
    if any(step % quantum or online_update_target(config, step) < 1 for step in args.online_eval_steps):
        raise ValueError(f"Online evaluation steps must be multiples of {quantum} and exceed warmup.")
    limit = int(config.env.time_limit) // int(config.env.action_repeat)
    if int(config.jepa_model.history_size) - 1 + max(args.horizon, args.policy_steps) >= limit:
        raise ValueError("Policy/forecast horizon must stay within one simulator episode.")


def run_model(config, args, cases, output, result, persist, *, measurement=measure):
    family = load_model_family(config.model_family)
    total = max(args.eval_updates)
    config.training.expert.updates = total
    result.update(config=OmegaConf.to_container(config, resolve=True), snapshots=[],
                  offline_updates=0, schedule_updates=total)
    with family.build_replay(config) as dataset:
        model = new_model(config, dataset)
        result.update(parameters=sum(p.numel() for p in model.parameters()),
                      dataset_identity=dataset_identity(dataset.metadata))
        heldout = StateDataset(dataset.h5, dataset.metadata, config.model_io,
                               config.state_head.fields, model.state_head.targets)
        length = model.history_size + args.horizon
        windows = heldout.sample_windows(16, length, args.seed + 2_000_000, model.history_size, .5, 8)
        expert_batch = heldout.read_batch(windows, length)
        result["expert_windows"] = [asdict(window) for window in windows]

        def snapshot(phase, updates, env_steps=0):
            suffix = f"_env_{env_steps}" if phase == "online" else ""
            folder = output / f"{phase}_{updates}{suffix}"
            folder.mkdir()
            scores = measurement(config, model, cases, expert_batch, args, folder)
            item = {"phase": phase, "updates": updates, "env_steps": env_steps, **scores}
            result["snapshots"].append(item)
            persist()
            print(snapshot_line(str(config.model_family), item), flush=True)

        online_points = iter(args.online_eval_steps)
        next_online_point = next(online_points, None)

        def online_snapshot(steps, updates):
            nonlocal next_online_point
            if next_online_point is None or steps < next_online_point:
                return
            snapshot("online", updates, steps)
            # Measure after a completed update burst, recording the actual step count.
            while next_online_point is not None and next_online_point <= steps:
                next_online_point = next(online_points, None)

        # One schedule for the whole run, never restarted at a measurement boundary.
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(total)
        progress = Progress(f"{config.model_family} offline", total)
        started = time.monotonic()
        with (output / "offline_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for step in range(1, total + 1):
                rates = learning_rates(model)
                metrics = checked_update(model, lambda: model.update(dataset.sample_episode_batch()))
                log.write(json.dumps({"update": step, "learning_rates_before": rates, **metrics}, allow_nan=False) + "\n")
                result["offline_updates"] = step
                progress.update(step, f"prediction={metrics['prediction_loss']:.4g} state={metrics['state/loss']:.4g}",
                                force=step in args.eval_updates)
                if step in args.eval_updates:
                    snapshot("offline", step)
        result["offline_seconds_including_evaluation"] = time.monotonic() - started

        if args.online_steps:
            envs = None
            try:
                envs = make_envs(config.env)
                # Carry weights and optimizer moments into fresh on-policy replay.
                # The online scheduler keeps its FULL production budget, not this prefix's length.
                with online_readout(config, family, model, expected_dataset=result["dataset_identity"]):
                    result["online"] = online_prefix(
                        config, family, model, envs, args.online_steps, output / "online_metrics.jsonl",
                        snapshot=online_snapshot if args.online_eval_steps else None, diagnostic_every=1)
            finally:
                close_envs(envs)
            snapshot("online", result["online"]["updates"], result["online"]["env_steps"])
    result["status"] = "COMPLETE"


def number(value):
    return "n/a" if value is None else f"{value:.3g}"


def snapshot_line(name, snapshot):
    expert, candidate, policy = snapshot["expert"], snapshot["candidates"], snapshot["policy"]
    def pair(values):
        return f"{values[0]:.3g}/{values[-1]:.3g}"
    selected = "/".join(number(candidate["selected_return"][k]) for k in ("predicted", "true_latent", "uniform", "best"))
    stage = f"{snapshot['phase']}_{snapshot['updates']}"
    if snapshot["phase"] == "online":
        stage += f"[env={snapshot['env_steps']}]"
    return (f"{name}/{stage} | {pair(expert['latent']['mse'])} | "
            f"{pair(candidate['mse'])} | {number(candidate['cost_rank'])}({candidate['rank_count']}) | {selected} | "
            f"{policy['return_mean']:.2f}/{policy['maximum_return']} | {policy['sustained_rate']:.0%}")


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    horizon = report["settings"]["horizon"]
    lines = ["Planner learning curve | one fresh native fit/model, then own-policy online prefix | no checkpoints",
             f"Model/stage | Expert latent MSE h1/h{horizon} | Simulator latent MSE h1/h{horizon} | Cost rank(n) | "
             "Candidate return predicted/true/uniform/best | Policy return/max | Tail success"]
    for run in report["runs"]:
        lines.extend(snapshot_line(run["model"], item) for item in run.get("snapshots", []))
        for item in run.get("snapshots", []):
            physical = item["expert"]["physical"]
            lines.append(f"  {run['model']}/{item['phase']}_{item['updates']} expert observed h1 RMSE | "
                         + format_physical_rmse(physical["observed"]["1"]))
        online = run.get("online", {})
        lines.append(f"{run['status']} | {run['model']} | offline={run.get('offline_updates', 0)} | "
                     f"online={online.get('updates', 0)} | env_steps={online.get('env_steps', 0)} | "
                     f"completed_episodes={online.get('completed_episodes', 0)} | time={duration(run.get('seconds'))}"
                     + (f" | {run['error']}" if "error" in run else ""))
    lines += [
        "Native objectives only: LeWM terminal latent distance, TS upstream MPC weighting; one physical-rendered goal.",
        "Offline schedule spans the FINAL requested update; snapshots do not restart it. Online is an early production-schedule prefix.",
        "Online labels show optimizer updates and cumulative environment steps (summed across environments, including action repeats).",
        "Same held-out expert windows, simulator starts, candidate actions and policy RNG at every measurement; none are fitted.",
        "Latent MSE changes with encoder scale; hold errors and target spread are in JSON. It is not comparable across models.",
        "Cost rank is predicted vs actual-future latent cost on reward-informative anchors; positive is better, n excludes undefined ranks.",
        "Candidate returns use those same informative anchors. True/best use simulator futures for scoring only, never for planning.",
        "Physical RMSE uses original units and wrapped angles; observed/forecast/true-hold per-coordinate errors are in JSON.",
        "Policy returns are short controlled-start trials, not full benchmark episodes. COMPLETE means execution, not convergence or repair.",
        "No goal-objective sweep, predictor-only fitting, parity rerun, calibration, checkpoint I/O, or dataset audit.",
    ]
    if "seconds" in report:
        lines.append(f"Total | {duration(report['seconds'])}")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--eval-updates", nargs="+", type=int, default=[1000, 3000, 5000],
                        help="Measurement points; the largest is the total offline training/scheduler budget.")
    parser.add_argument("--online-steps", type=int, default=4096,
                        help="Environment steps in the ORIGINAL online schedule; 0 skips the continuation.")
    parser.add_argument("--online-eval-steps", nargs="+", type=int, default=[],
                        help="Intermediate online environment-step points, measured after an update burst. The final point is always measured.")
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("planner_learning_curve_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if (min(args.eval_updates) < 1 or args.policy_steps < 1 or args.horizon < 1
            or args.online_steps < 0 or args.seed < 0
            or any(len(v) != len(set(v)) for v in (args.models, args.eval_updates, args.online_eval_steps))):
        parser.error("Use positive unique update points/horizons, positive policy steps, nonnegative online steps/seed, and unique models.")
    if any(step <= 0 or step >= args.online_steps for step in args.online_eval_steps):
        parser.error("Online evaluation points must be positive and strictly before --online-steps; the final evaluation is automatic.")
    args.eval_updates.sort()
    args.online_eval_steps.sort()
    args.scenario = "cartpole_balance_sparse"
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA diagnostic requires BF16 support.")
        torch.cuda.set_device(device)
    configs = {name: build_config(name, args) for name in args.models}
    for name, config in configs.items():
        config.jepa_model.planner.horizon = args.horizon
        config.jepa_model.planner.objective = "ts_mpc" if name == "temporal_straightening" else "last"
        config.jepa_model.goal.alternatives = []
        config.env.goal = config.jepa_model.goal
        validate_budget(config, args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(f"Learning curve | models={len(args.models)} | offline={max(args.eval_updates)} once/model | "
          f"measure={args.eval_updates} | online_env_steps={args.online_steps} | "
          f"online_measure={args.online_eval_steps + ([args.online_steps] if args.online_steps else [])} | checkpoints=disabled", flush=True)
    first = configs[args.models[0]]
    cases = collect_objective_cases(first, SimpleNamespace(
        sim_seeds=[args.seed + 12_000_000, args.seed + 12_000_001],
        horizons=list(range(1, args.horizon + 1)), candidates=16), history_size=int(first.jepa_model.history_size))
    files = [Path(__file__).with_name(name) for name in (
        "train_planner_learning_curve.py", "train_planner_check.py", "smoke_online_checkpoints.py",
        "diagnose_goal_objective.py", "diagnose_planner_oracle.py", "diagnose_planning_horizons.py")]
    report = {"implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cases": [case_metadata(case) for case in cases], "checkpoint_reads": False,
              "checkpoint_writes": False, "fresh_initialization": True, "runs": []}
    for name, config in configs.items():
        result = {"model": name, "status": "RUNNING"}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        tick = time.monotonic()
        try:
            run_model(config, args, cases, output, result, lambda: write_report(args.output, report))
        except Exception as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        result["seconds"] = time.monotonic() - tick
        write_report(args.output, report)
        print(f"{result['status']} | {name} | {duration(result['seconds'])}", flush=True)
    report["seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(run["status"] != "COMPLETE" for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
