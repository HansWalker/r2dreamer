"""Short online-only LeWorldModel/TS checks from expert checkpoints; never save weights."""

import argparse
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
from scripts.diagnose_planning_models import FAMILIES, SCENARIOS, analyze_checkpoint
from training import load_model_family
from training.evaluation import StateDataset
from training.progress import Progress, console, duration
from training.protocol import (
    checkpoint_compatibility,
    implementation_sha256,
    upgrade_readout_config,
    validate_checkpoint,
)
from training.readout import online_readout
from training.trainer import online_update_target, progress_metrics


@tools.preserve_rng_state
def diagnose(model, dataset, windows, args):
    was_training = model.training
    try:
        result = analyze_checkpoint(model, dataset, windows, args)
        # Reject non-finite diagnostics before adding them to the saved report.
        json.dumps(result, allow_nan=False)
        return result
    finally:
        model.train(was_training)


@tools.preserve_rng_state
@torch.no_grad()
def migrate(model, family, checkpoint, dataset, windows, context):
    """Check the output affine on identical features before allowing any optimization."""
    was_training = model.training
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
        model.train(was_training)


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
        "replay_rows": session.replay.count(), "completed_episodes": len(episodes),
        "completed_episode_returns": [score for score, _ in episodes],
        "elapsed_seconds": time.monotonic() - started,
        "schedule_env_steps": int(config.training.online.steps),
        "schedule_updates": int(config.training.online.updates),
    }


def accuracy_regression(before, after, *, max_ratio=3.0, rmse_floor=0.01):
    """An explicit smoke alarm, not a statistical test or a tuned model-selection score."""
    baseline = before["all"]["physical"]["observed"]["1"]
    updated = after["all"]["physical"]["observed"]["1"]
    previous, current = baseline["rmse"], updated["rmse"]
    coordinates = {
        key: {"before": value, "after": current[key], "limit": max(value, rmse_floor) * max_ratio}
        for key, value in previous.items()
    }
    failed = [key for key, value in coordinates.items() if value["after"] > value["limit"]]
    normalized_limit = max(baseline["mean_normalized_mse"], 1.0) * max_ratio**2
    if updated["mean_normalized_mse"] > normalized_limit:
        failed.append("mean_normalized_mse")
    return {"passed": not failed, "failed_coordinates": failed, "coordinates": coordinates,
            "max_rmse_ratio": max_ratio, "rmse_floor_original_units": rmse_floor,
            "normalized_mse_limit": normalized_limit}


def run_case(job):
    args = SimpleNamespace(**job)
    path = Path(args.checkpoint)
    result = {"scenario": args.scenario, "model": args.model, "checkpoint": str(path), "status": "FAIL"}
    started = time.monotonic()
    envs = None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = OmegaConf.create(checkpoint["training_config"])
        upgrade_readout_config(config)
        config.device = args.device
        if (str(config.scenario.name), str(config.model_family), int(config.seed)) != (args.scenario, args.model, args.seed):
            raise ValueError("Checkpoint does not belong to the requested scenario/model/seed.")
        if checkpoint.get("phase") != "expert" or int(checkpoint.get("expert_updates", -1)) != int(config.training.expert.updates):
            raise ValueError("Use a completed expert pretrained.pt checkpoint, not an online or partial checkpoint.")
        validate_checkpoint(checkpoint, config, training=True)
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
            config_yaml=OmegaConf.to_yaml(config, resolve=True), dataset_identity=dataset_identity(metadata),
        )
        with h5py.File(dataset_path / "data.hdf5", "r") as h5:
            dataset = StateDataset(h5, metadata, config.model_io, config.state_head.fields, model.state_head.targets)
            windows = dataset.sample_windows(args.windows, args.context_length + max(args.horizons),
                                             args.window_seed, args.context_length, .5, 8)
            result["windows"] = [asdict(window) for window in windows]
            result["migration"] = migrate(model, family, checkpoint, dataset, windows, args.context_length)
            tools.set_rng_state(checkpoint.get("rng_state"))
            del checkpoint
            print("Evaluation | state_prediction=running | stage=before | held-out windows", flush=True)
            result["before"] = diagnose(model, dataset, windows, args)
            result["snapshots"] = []

            def snapshot(steps, updates):
                print(f"Evaluation | state_prediction=running | online_updates={updates}", flush=True)
                diagnostic = diagnose(model, dataset, windows, args)
                result["snapshots"].append({"env_steps": steps, "updates": updates, "diagnostic": diagnostic,
                    "accuracy": accuracy_regression(result["before"], diagnostic,
                        max_ratio=getattr(args, "max_rmse_ratio", 3.0), rmse_floor=getattr(args, "rmse_floor", .01))})

            envs = make_envs(config.env, seed=int(config.env.seed))
            with online_readout(config, family, model, expected_dataset=result["dataset_identity"]):
                result["online"] = online_prefix(config, family, model, envs, args.env_steps,
                    Path(args.result_path).with_name("metrics.jsonl"), snapshot,
                    getattr(args, "diagnostic_every_updates", 64))
            close_envs(envs)
            envs = None
            print("Evaluation | state_prediction=running | stage=after | same held-out windows", flush=True)
            result["after"] = diagnose(model, dataset, windows, args)
            result["accuracy"] = accuracy_regression(result["before"], result["after"],
                max_ratio=getattr(args, "max_rmse_ratio", 3.0), rmse_floor=getattr(args, "rmse_floor", .01))
        if device.type == "cuda":
            result["gpu_reserved_peak_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
        result["execution_passed"] = True
        stable = result["accuracy"]["passed"] and all(item["accuracy"]["passed"] for item in result["snapshots"])
        result["status"] = "PASS" if stable else "REGRESSION"
    except Exception as error:  # noqa: BLE001 - Preserve each worker's failure alongside successful reports.
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        close_envs(envs)
        result["elapsed_seconds"] = time.monotonic() - started
        Path(args.result_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def write_summary(output, results, args):
    lines = ["Checkpoint online check | no pretraining or checkpoint writes",
             "Run | Updates | Migration max error | Observed nMSE h1 before->after | Forecast nMSE h100 before->after | Time"]
    for result in results:
        name = f"{result['scenario']}/{result['model']}"
        if result["status"] == "FAIL":
            lines.append(f"FAIL | {name} | {result.get('error', 'worker failed')} | {result['log']}")
            continue
        observed = [result[stage]["all"]["physical"]["observed"]["1"]["mean_normalized_mse"] for stage in ("before", "after")]
        forecast = [result[stage]["all"]["physical"]["forecast"]["100"]["mean_normalized_mse"] for stage in ("before", "after")]
        lines.append(f"{result['status']} | {name} | {result['online']['updates']} | {result['migration']['max_prediction_difference']:.3g} | "
                     f"{observed[0]:.3g}->{observed[1]:.3g} | {forecast[0]:.3g}->{forecast[1]:.3g} | {duration(result['elapsed_seconds'])}")
        if result["status"] == "REGRESSION":
            failures = set(result["accuracy"]["failed_coordinates"])
            for item in result["snapshots"]:
                failures.update(item["accuracy"]["failed_coordinates"])
            lines.append("  Observed-state error regression: " + ", ".join(sorted(failures)))
    lines.extend(["PASS includes the short-run observed-state RMSE guard; it does NOT establish long-run learning stability.",
                  "REGRESSION means execution succeeded but an intermediate/final coordinate exceeded the declared error limit.",
                  "Expert nMSE scales are unchanged. Original-unit RMSE, cohorts, and exact windows are in report.json.",
                  "This is an early prefix of the original schedule, not a compressed training run or a policy success evaluation."])
    report = {
        "diagnostic_version": 2, "implementation_sha256": implementation_sha256(),
        "dataset_role": "held_out_expert", "evaluation_fitting": False, "checkpoint_writes": False,
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
    parser.add_argument("--env-steps", type=int, default=4096, help="Stop early without changing the saved training schedule.")
    parser.add_argument("--windows", type=int, default=16, help="Fixed held-out windows, shared before/after training.")
    parser.add_argument("--batch-size", type=int, default=4, help="Diagnostic batch size, not training batch size.")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--window-seed", type=int, default=2000000)
    parser.add_argument("--diagnostic-every-updates", type=int, default=64)
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
            if code and result["status"] != "REGRESSION":
                result["status"] = "FAIL"
            result["log"] = str(log)
            results.append(result)
            summary = write_summary(args.output, results, args)
        console.message(summary)
        console.message(f"Reports | {args.output}")
    raise SystemExit(0 if all(result["status"] == "PASS" for result in results) else 1)


if __name__ == "__main__":
    main()
