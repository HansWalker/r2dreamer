"""Short online-only LeWorldModel/TS checks from expert checkpoints; never save weights."""

import argparse
import copy
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity, validate_dataset
from envs import close_envs, make_envs
from models.shared.physical_state import readout_mode
from scripts.diagnose_planning_models import FAMILIES, SCENARIOS, analyze_checkpoint
from scripts.online_validation import (
    TrajectoryDataset,
    calibrate_readout,
    collect_episode,
    episode_metadata,
    validation_metadata,
)
from training import load_model_family
from training.evaluation import StateDataset
from training.progress import Progress, console, duration
from training.protocol import (
    checkpoint_compatibility,
    implementation_sha256,
    upgrade_readout_config,
    validate_checkpoint,
)
from training.readout import native_mixture, online_readout
from training.trainer import online_update_target, progress_metrics


@tools.preserve_rng_state
def diagnose(model, dataset, windows, args):
    with readout_mode(model):
        result = analyze_checkpoint(model, dataset, windows, args)
        # Reject non-finite diagnostics before adding them to the saved report.
        json.dumps(result, allow_nan=False)
        return result


@tools.preserve_rng_state
@torch.no_grad()
def migrate(model, family, checkpoint, dataset, windows, context):
    """Check the output affine on identical features before allowing any optimization."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        observation, _, _ = dataset.read_batch(windows[:1], context)
        features = model.encode({key: value.to(model.device) for key, value in observation.items()})
        before = model.state_head(features)
        mean, std = model.state_head.mean.clone(), model.state_head.std.clone()
        old_scale = model.state_head.output_scale.clone()
        legacy = model.state_head._legacy_optimizer
        family.load_checkpoint(model, checkpoint, training=True)
        after = model.state_head(features)
        torch.testing.assert_close(after, before, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(model.state_head.mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(model.state_head.std, std, rtol=0, atol=0)
        return {
            "max_prediction_difference": (after - before).abs().max().item(),
            "coordinates": model.state_head.coordinates,
            "expert_mean": mean.cpu().tolist(), "evaluation_std": std.cpu().tolist(),
            "previous_output_scale": old_scale.cpu().tolist(),
            "output_scale": model.state_head.output_scale.cpu().tolist(),
            "loss_scale": model.state_head.loss_scale.cpu().tolist(),
            "head_optimizer_reset": legacy,
        }
    finally:
        for module, mode in modes:
            module.training = mode


def online_prefix(config, family, model, envs, env_steps, metrics_path, snapshot=None, diagnostic_every=64):
    """Use fresh on-policy replay and the full run's schedule, stopping early only."""
    session = family.OnlineSession(config, model, envs)
    if session.replay.count():
        raise RuntimeError("Smoke online replay must start empty.")
    if hasattr(model, "configure_online"):
        model.configure_online(int(config.training.online.updates), resumed=False)
    model.train()
    session.start()
    initial_updates = model.state_head.updates.item()
    progress = Progress("Online check", env_steps)
    steps, updates, episodes = 0, 0, []
    metrics = {}
    next_diagnostic = diagnostic_every
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8", buffering=1) as log:
        while steps < env_steps:
            delta, finished = session.collect()
            if delta <= 0:
                raise RuntimeError("Collection did not advance the environment.")
            steps += delta
            episodes.extend(finished)
            target = online_update_target(config, steps)
            if target > updates and session.replay.ready():
                metrics = session.update(target - updates)
                metrics = {key: float(torch.as_tensor(value).detach()) for key, value in metrics.items()}
                if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("Non-finite or missing online training metrics.")
                updates = target
                if model.state_head.updates.item() != initial_updates + updates:
                    raise RuntimeError("Online head update counter does not match the scheduled updates.")
                log.write(json.dumps({"env_steps": steps, "updates": updates, "metrics": metrics}, allow_nan=False) + "\n")
                if snapshot is not None and updates >= next_diagnostic and steps < env_steps:
                    snapshot(steps, updates)
                    next_diagnostic = (updates // diagnostic_every + 1) * diagnostic_every
            progress.update(steps, f"updates={updates} replay={session.replay.count()} "
                            + progress_metrics(metrics, family.ONLINE_METRICS), force=steps >= env_steps)
    if not updates or updates != online_update_target(config, steps):
        raise RuntimeError("The online prefix did not complete its scheduled updates; replay may not be ready.")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Non-finite parameters after online training.")
    return {
        "env_steps": steps, "agent_transitions": steps // int(config.env.action_repeat),
        "updates": updates, "head_updates_before": initial_updates,
        "head_updates_after": model.state_head.updates.item(), "last_metrics": metrics,
        "replay_source": "fresh trajectories from the checkpoint's own updated policy; no expert seeding",
        "readout_expert_fraction": model.state_head.expert_fraction,
        "native_expert_fraction": float(config.training.online.get("expert_fraction", 0.0)),
        "replay_rows": session.replay.count(), "completed_episodes": len(episodes),
        "completed_episode_returns": [score for score, _ in episodes],
        "elapsed_seconds": time.monotonic() - started,
        "schedule_env_steps": int(config.training.online.steps),
        "schedule_updates": int(config.training.online.updates),
    }


def accuracy_regression(before, after, *, max_ratio=3.0, rmse_floor=0.01, rmse_floors=None,
                        source="observed", horizon="1", cohort="all"):
    """An explicit smoke alarm, not a statistical test or a tuned model-selection score."""
    baseline = before[cohort]["physical"][source][str(horizon)]
    updated = after[cohort]["physical"][source][str(horizon)]
    previous, current = baseline["rmse"], updated["rmse"]
    coordinates = {
        key: {"before": value, "after": current[key],
              "limit": max(value, (rmse_floors or {}).get(key, rmse_floor)) * max_ratio}
        for key, value in previous.items()
    }
    failed = [key for key, value in coordinates.items() if value["after"] > value["limit"]]
    normalized_limit = max(baseline["mean_normalized_mse"], 1.0) * max_ratio**2
    if updated["mean_normalized_mse"] > normalized_limit:
        failed.append("mean_normalized_mse")
    return {"passed": not failed, "failed_coordinates": failed, "coordinates": coordinates,
            "max_rmse_ratio": max_ratio, "rmse_floor_original_units": rmse_floor,
            "normalized_mse_limit": normalized_limit}


def diagnostic_guards(before, after, args):
    checks = {}
    for cohort in before:
        for source in ("observed", "forecast"):
            for horizon in args.horizons:
                checks[f"{cohort}/{source}/h{horizon}"] = accuracy_regression(
                    before, after, max_ratio=getattr(args, "max_rmse_ratio", 3.0),
                    rmse_floor=getattr(args, "rmse_floor", .01), rmse_floors=getattr(args, "rmse_floors", {}),
                    source=source, horizon=horizon, cohort=cohort,
                )
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def run_trials(config, family, model, checkpoint, dataset, windows, args, result):
    """Share fixed diagnostic data across fresh checkpoint branches, never across training replays."""
    fractions = list(dict.fromkeys(getattr(args, "native_expert_fractions", [0.0])))
    candidates = [(f"native_{fraction:g}", fraction, 0) for fraction in fractions]
    calibration_updates = getattr(args, "calibration_updates", 0)
    if calibration_updates:
        candidates.append((f"native_{fractions[-1]:g}_calibrated", fractions[-1], calibration_updates))
    for fraction in fractions:
        candidate = copy.deepcopy(config)
        candidate.training.online.expert_fraction = fraction
        native_mixture(candidate)
    unknown = set(getattr(args, "rmse_floors", {})) - set(model.state_head.coordinates)
    if unknown:
        raise ValueError(f"Unknown physical coordinates in --rmse-floors: {sorted(unknown)}")
    result["before"] = diagnose(model, dataset, windows, args)
    validation, validation_windows, calibration_episodes = None, None, []
    policy_episodes = getattr(args, "policy_episodes", 0)
    validation_seed = int(args.window_seed) + 4_000_000
    policy_seeds = list(range(validation_seed, validation_seed + policy_episodes))
    failure_seed = validation_seed + policy_episodes
    if policy_episodes:
        print("Evaluation | collecting fixed policy/failure validation episodes (once)", flush=True)
        episodes = [collect_episode(config, model, seed, "policy") for seed in policy_seeds]
        result["policy_before"] = episode_metadata(episodes)
        episodes.extend(collect_episode(config, model, failure_seed + i, mode)
                        for i, mode in enumerate(("zero", "random")))
        forbidden = range(int(config.env.seed), int(config.env.seed) + int(config.env.env_num))
        validation = TrajectoryDataset(episodes, forbidden_seeds=forbidden)
        validation_windows = validation.sample_windows(args.windows, args.context_length + max(args.horizons), args.window_seed)
        result["validation_data"] = validation_metadata(validation, validation_windows)
        result["validation_before"] = diagnose(model, validation, validation_windows, args)
        if calibration_updates:
            calibration_episodes = [collect_episode(config, model, failure_seed + 2 + i, mode)
                                    for i, mode in enumerate(("zero", "random"))]
            TrajectoryDataset(calibration_episodes, forbidden_seeds=[*forbidden, *(e["seed"] for e in episodes)])
    elif calibration_updates:
        raise ValueError("Calibration requires disjoint simulator validation; enable --policy-episodes.")

    result["trials"] = []
    for name, fraction, calibration in candidates:
        started = time.monotonic()
        print(f"Trial | {name} | native_expert={fraction:g} | head_calibration_updates={calibration}", flush=True)
        config.training.online.expert_fraction = fraction
        family.load_checkpoint(model, copy.deepcopy(checkpoint), training=True)
        tools.set_rng_state(checkpoint.get("rng_state"))
        trial = {"name": name, "before": result["before"], "snapshots": [],
                 "compatibility": checkpoint_compatibility(config), "native_expert_fraction": fraction}
        result["trials"].append(trial)

        def measure(steps, updates):
            expert = diagnose(model, dataset, windows, args)
            item = {"env_steps": steps, "updates": updates, "diagnostic": expert,
                    "accuracy": accuracy_regression(result["before"], expert,
                        max_ratio=getattr(args, "max_rmse_ratio", 3.0), rmse_floor=getattr(args, "rmse_floor", .01))}
            item["expert_guard"] = diagnostic_guards(result["before"], expert, args)
            if validation is not None:
                item["validation"] = diagnose(model, validation, validation_windows, args)
                item["validation_guard"] = diagnostic_guards(result["validation_before"], item["validation"], args)
            return item

        def snapshot(steps, updates):
            print(f"Evaluation | {name} | online_updates={updates}", flush=True)
            trial["snapshots"].append(measure(steps, updates))

        envs = None
        try:
            with online_readout(config, family, model, expected_dataset=result["dataset_identity"]) as readout_replay:
                if calibration:
                    sampler_state = copy.deepcopy(readout_replay.state_dict()) if readout_replay is not None else None
                    trial["calibration"] = calibrate_readout(model, calibration_episodes, calibration, failure_seed + 4)
                    if sampler_state is not None:
                        readout_replay.load_state_dict(sampler_state)
                    trial["snapshots"].append(measure(0, 0))
                envs = make_envs(config.env, seed=int(config.env.seed))
                metrics_path = Path(args.result_path).with_name(f"{name}_metrics.jsonl")
                trial["online"] = online_prefix(config, family, model, envs, args.env_steps, metrics_path, snapshot,
                                                getattr(args, "diagnostic_every_updates", 64))
        finally:
            close_envs(envs)
        final = measure(args.env_steps, trial["online"]["updates"])
        trial.update(after=final["diagnostic"], accuracy=final["accuracy"], final_checks=final)
        checks = [*trial["snapshots"], final]
        stable = all(check["expert_guard"]["passed"] and check.get("validation_guard", {"passed": True})["passed"]
                     for check in checks)
        trial["status"] = "PASS" if stable else "REGRESSION"
        if validation is not None:
            policy = TrajectoryDataset([collect_episode(config, model, seed, "policy") for seed in policy_seeds])
            policy_windows = policy.sample_windows(args.windows, args.context_length + max(args.horizons), args.window_seed)
            trial["policy_after"] = episode_metadata(policy.episodes)
            trial["policy_after_diagnostic"] = diagnose(model, policy, policy_windows, args)
            trial["policy_after_windows"] = [asdict(window) for window in policy_windows]
            goals = [cohort["goals"]["observed"]["1"] for cohort in final["validation"].values()]
            goals.append(trial["policy_after_diagnostic"]["all"]["goals"]["observed"]["1"])
            failures = [goal for goal in goals if goal["failure_states"]]
            trial["failure_state_quality_passed"] = bool(failures) and all(
                goal["false_success_rate"] <= args.max_false_success_rate for goal in failures
            )
            if stable and not trial["failure_state_quality_passed"]:
                trial["status"] = "UNVALIDATED"
        else:
            trial["failure_state_quality_passed"] = None
        trial["elapsed_seconds"] = time.monotonic() - started
    # Retain the single-case fields for existing report consumers.
    result.update({key: result["trials"][-1][key] for key in ("online", "after", "accuracy", "snapshots")})
    result["status"] = ("REGRESSION" if any(trial["status"] == "REGRESSION" for trial in result["trials"])
                        else "UNVALIDATED" if any(trial["status"] == "UNVALIDATED" for trial in result["trials"]) else "PASS")


def run_case(job):
    args = SimpleNamespace(**job)
    path = Path(args.checkpoint)
    result = {"scenario": args.scenario, "model": args.model, "checkpoint": str(path), "status": "FAIL"}
    started = time.monotonic()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = OmegaConf.create(checkpoint["training_config"])
        config.device = args.device
        if (str(config.scenario.name), str(config.model_family), int(config.seed)) != (args.scenario, args.model, args.seed):
            raise ValueError("Checkpoint does not belong to the requested scenario/model/seed.")
        if checkpoint.get("phase") != "expert" or int(checkpoint.get("expert_updates", -1)) != int(config.training.expert.updates):
            raise ValueError("Use a completed expert pretrained.pt checkpoint, not an online or partial checkpoint.")
        # This read-only diagnostic deliberately retains the old head's affine map;
        # it is not permission to resume production training under the new recipe.
        validate_checkpoint(checkpoint, config, training=False)
        if checkpoint.get("compatibility", {}).get("training_sha256") != checkpoint_compatibility(config)["training_sha256"]:
            raise ValueError("Checkpoint has inconsistent training recipe metadata.")
        legacy_controller = config.jepa_model.goal.get("source") != "physical_render_v1"
        if legacy_controller:
            if not getattr(args, "latent_goals", False):
                raise ValueError(
                    "Checkpoint uses physical-head planning. Pass --latent-goals to explicitly test the new "
                    "physical-render/latent-distance controller with these read-only pretrained weights."
                )
            from hydra import compose, initialize_config_dir

            with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
                current = compose(config_name=f"{args.model}_dmc_vision", overrides=[f"scenario={args.scenario}"])
                config.jepa_model.goal = OmegaConf.to_container(current.jepa_model.goal, resolve=True)
            config.env.goal = OmegaConf.to_container(config.jepa_model.goal, resolve=True)
        result["controller"] = {
            "source": config.jepa_model.goal.source,
            "explicit_legacy_override": legacy_controller,
            "physical_head_used_for_planning": False,
            "goal": OmegaConf.to_container(config.jepa_model.goal, resolve=True),
        }
        upgrade_readout_config(config)
        quantum = int(config.env.env_num) * int(config.env.action_repeat)
        if args.env_steps % quantum or not 0 < args.env_steps <= int(config.training.online.steps):
            raise ValueError(f"--env-steps must be a multiple of {quantum}, within the saved online budget.")
        if online_update_target(config, args.env_steps) < 1:
            raise ValueError("--env-steps must extend past the saved replay warmup and reach at least one update.")
        dataset_path = Path(args.dataset_root) / str(config.scenario.dataset)
        config.training.expert.data_path = str(dataset_path)
        metadata = validate_dataset(dataset_path, config, splits=("heldout",))
        if checkpoint.get("dataset_identity") != dataset_identity(metadata):
            raise ValueError("Checkpoint and held-out dataset identities differ.")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        tools.configure_randomness(config.seed, bool(config.deterministic_run))
        family = load_model_family(args.model)
        model = family.build_model(config)
        family.load_checkpoint(model, checkpoint, training=False)
        if not model.state_head.updates.item():
            raise ValueError("Checkpoint has no trained physical-state head.")
        if args.context_length < model.history_size:
            raise ValueError("Diagnostic context is shorter than the model's native history.")
        result.update(
            checkpoint_id=checkpoint["checkpoint_id"], run_identity=checkpoint["run_identity"],
            source_compatibility=checkpoint["compatibility"], compatibility=checkpoint_compatibility(config),
            production_resume=False, head_initialization="saved checkpoint affine, not a newly fitted head",
            config_yaml=OmegaConf.to_yaml(config, resolve=True), dataset_identity=dataset_identity(metadata),
        )
        with h5py.File(dataset_path / "data.hdf5", "r") as h5:
            dataset = StateDataset(h5, metadata, config.model_io, config.state_head.fields, model.state_head.targets)
            windows = dataset.sample_windows(args.windows, args.context_length + max(args.horizons),
                                             args.window_seed, args.context_length, .5, 8)
            result["windows"] = [asdict(window) for window in windows]
            result["migration"] = migrate(model, family, checkpoint, dataset, windows, args.context_length)
            tools.set_rng_state(checkpoint.get("rng_state"))
            run_trials(config, family, model, checkpoint, dataset, windows, args, result)
        if device.type == "cuda":
            result["gpu_reserved_peak_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
        result["execution_passed"] = True
    except Exception as error:  # noqa: BLE001 - Preserve each worker's failure alongside successful reports.
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        Path(args.result_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def write_summary(output, results, args):
    lines = ["Checkpoint online check | no pretraining or checkpoint writes",
             "Run / trial | Updates | Expert obs h1 / forecast h100 nMSE before->after | Simulator obs h1 nMSE | False goal | Policy return | Time"]
    for result in results:
        name = f"{result['scenario']}/{result['model']}"
        if result["status"] == "FAIL":
            lines.append(f"FAIL | {name} | {result.get('error', 'worker failed')} | {result['log']}")
            continue
        for trial in result.get("trials", [result]):
            observed = [trial[stage]["all"]["physical"]["observed"]["1"]["mean_normalized_mse"] for stage in ("before", "after")]
            forecast = [trial[stage]["all"]["physical"]["forecast"]["100"]["mean_normalized_mse"] for stage in ("before", "after")]
            validation = trial.get("final_checks", {}).get("validation")
            failure, false_goal, policy = "-", "-", "-"
            if validation is not None:
                first, last = result["validation_before"]["all"], validation["all"]
                failure = f"{first['physical']['observed']['1']['mean_normalized_mse']:.3g}->{last['physical']['observed']['1']['mean_normalized_mse']:.3g}"
                rates = [value["goals"]["observed"]["1"]["false_success_rate"] for value in (first, last)]
                false_goal = "->".join("no failures" if rate is None else f"{rate:.0%}" for rate in rates)
                returns = [sum(episode["return"] for episode in episodes) / len(episodes)
                           for episodes in (result["policy_before"], trial["policy_after"])]
                policy = f"{returns[0]:.1f}->{returns[1]:.1f}"
            lines.append(f"{trial['status']} | {name}/{trial.get('name', 'online')} | {trial['online']['updates']} | "
                         f"{observed[0]:.3g}->{observed[1]:.3g} / {forecast[0]:.3g}->{forecast[1]:.3g} | "
                         f"{failure} | {false_goal} | {policy} | {duration(trial['elapsed_seconds'])}")
            failures = set()
            for item in [*trial["snapshots"], trial.get("final_checks", {})]:
                for domain in ("expert_guard", "validation_guard"):
                    for key, check in item.get(domain, {}).get("checks", {}).items():
                        if not check["passed"]:
                            failures.add(f"{domain}/{key}")
            if failures:
                lines.append(f"  Error guards failed: {len(failures)} cohort/source/horizon checks; coordinates in report.json.")
    lines.extend(["PASS is a short-run error guard; it does NOT establish long-run stability or policy quality.",
                  "REGRESSION: intermediate/final physical errors exceeded declared limits. UNVALIDATED: insufficient failure coverage or excessive false goals.",
                  "Expert nMSE scales are unchanged. Original-unit RMSE, cohorts, and exact windows are in report.json.",
                  "Policy returns use complete episodes, separate seeds, and no training replay; few episodes are diagnostic only.",
                  "Native budgets are unchanged. Calibration adds explicitly reported head-only updates; no validation fitting.",
                  "Planning uses rendered physical goals and native latent distance; explicit legacy overrides are in JSON.",
                  "False goals describe the evaluation readout, not the planner's internal success predictions."])
    report = {
        "diagnostic_version": 4, "implementation_sha256": implementation_sha256(),
        "dataset_role": "held_out_expert_and_disjoint_simulator", "evaluation_fitting": False, "checkpoint_writes": False,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = "\n".join(lines)
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/dmc_vision_10k"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["cartpole_balance_sparse"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-goals", action="store_true",
                        help="Explicitly replace legacy physical-head control with rendered physical goals and native latent planning.")
    parser.add_argument("--env-steps", type=int, default=4096, help="Stop early without changing the saved training schedule.")
    parser.add_argument("--windows", type=int, default=16, help="Fixed held-out windows, shared before/after training.")
    parser.add_argument("--batch-size", type=int, default=4, help="Diagnostic batch size, not training batch size.")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--window-seed", type=int, default=2000000)
    parser.add_argument("--diagnostic-every-updates", type=int, default=64)
    parser.add_argument("--native-expert-fractions", nargs="+", type=float, default=[0.0, 0.5],
                        help="Independent fresh online branches; same native batch/update budget.")
    parser.add_argument("--calibration-updates", type=int, default=256,
                        help="Additional candidate at the last native fraction; head-only, training states only. Zero disables.")
    parser.add_argument("--policy-episodes", type=int, default=1,
                        help="Complete policy episodes before/after; also enable fixed zero/random failure validation.")
    parser.add_argument("--max-false-success-rate", type=float, default=.2)
    parser.add_argument("--rmse-floors", type=json.loads, default={}, help="JSON map of coordinate-specific original-unit alarm floors.")
    parser.add_argument("--max-rmse-ratio", type=float, default=3.0, help="Observed-state alarm threshold, not a model-selection metric.")
    parser.add_argument("--rmse-floor", type=float, default=.01, help="Original-unit floor before multiplying the baseline by --max-rmse-ratio.")
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / f"online_checkpoint_smoke_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        raise SystemExit(0 if run_case(json.load(sys.stdin))["status"] == "PASS" else 1)
    if args.dataset_root is None or min(args.env_steps, args.windows, args.batch_size, args.context_length) < 1:
        parser.error("Provide --dataset-root and positive step/window/batch/context sizes.")
    if args.diagnostic_every_updates < 1 or not math.isfinite(args.max_rmse_ratio) or args.max_rmse_ratio <= 1 or not math.isfinite(args.rmse_floor) or args.rmse_floor <= 0:
        parser.error("Use positive diagnostic frequency and RMSE floor, and a finite RMSE ratio greater than one.")
    if args.calibration_updates < 0 or args.policy_episodes < 0 or (args.calibration_updates and not args.policy_episodes):
        parser.error("Calibration needs validation episodes; counts must be nonnegative.")
    if not all(math.isfinite(value) and 0 <= value < 1 for value in args.native_expert_fractions):
        parser.error("Native fractions must be finite and in [0, 1).")
    if not math.isfinite(args.max_false_success_rate) or not 0 <= args.max_false_success_rate <= 1:
        parser.error("The false-success limit must be in [0, 1].")
    if not isinstance(args.rmse_floors, dict) or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                                                  for value in args.rmse_floors.values()):
        parser.error("RMSE floors must map coordinate names to finite positive values.")
    args.horizons = [1, 100]
    args.run_root, args.dataset_root, args.output = (path.expanduser().resolve() for path in (args.run_root, args.dataset_root, args.output))
    jobs = []
    for scenario in dict.fromkeys(args.scenarios):
        for name in dict.fromkeys(args.models):
            checkpoint = args.run_root / scenario / name / "default" / f"seed_{args.seed}" / "pretrained.pt"
            if not checkpoint.is_file():
                parser.error(f"Missing expert checkpoint: {checkpoint}")
            result_path = args.output / scenario / name / "result.json"
            jobs.append({**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                         "scenario": scenario, "model": name, "checkpoint": str(checkpoint), "result_path": str(result_path)})
    args.output.mkdir(parents=True, exist_ok=False)
    from main import execute

    results = []
    with console.capture(args.output / "orchestrator.log"):
        console.message(f"Online check | runs={len(jobs)} | steps/run={args.env_steps} | fresh replay | checkpoints=read-only")
        for job in jobs:
            path = Path(job["result_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.with_name("stdout.log")
            try:
                execute(f"online-check | {job['scenario']}/{job['model']}",
                        [sys.executable, "-u", "-m", "scripts.smoke_online_checkpoints", "--worker"],
                        log, input_text=json.dumps(job))
                code = 0
            except SystemExit as error:
                code = error.code
            result = json.loads(path.read_text()) if path.is_file() else {
                "scenario": job["scenario"], "model": job["model"], "status": "FAIL",
                "error": f"Worker exited {code} without a report",
            }
            if code and result["status"] not in {"REGRESSION", "UNVALIDATED"}:
                result["status"] = "FAIL"
            result["log"] = str(log)
            results.append(result)
            summary = write_summary(args.output, results, args)
        console.message(summary)
        console.message(f"Reports | {args.output}")
    raise SystemExit(0 if all(result["status"] == "PASS" for result in results) else 1)


if __name__ == "__main__":
    main()
