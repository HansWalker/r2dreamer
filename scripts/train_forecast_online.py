"""Two-model forecast diagnosis and native online learning, roughly five A100 hours.

Continue completed TS/LeWM duration checkpoints. Fixed work counts, original
goal scores, no new training loss, no timing calibration or clock cutoff.
"""

import argparse
import json
import math
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from envs import close_envs, make_envs
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.evaluate_goal_maintenance import branch_maintenance, load_source, validate_payload
from scripts.forecast_online_support import forecast_errors, probe_cases, summarize_probes
from scripts.offline_online_replay import OfflineOnlineSession, original_replay, replay_settings
from scripts.paper_faithful_duration_support import evaluate, policy_trial
from scripts.paper_faithful_followup_eval import preserve_training_state
from scripts.paper_faithful_support import _digest, score_branches
from scripts.train_paper_faithful_duration import (
    MODELS, TASKS, atomic_save, file_hash, runtime_versions, save_evaluation, source_hashes,
)
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256
from training.readout import online_readout
from training.trainer import online_update_target


FORMAT = "forecast_online_v1"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retain-offline", action="store_true",
                        help="Start native updates with 50%% original data, linearly tapering to 0%% by this run's final update")
    parser.add_argument("--dataset-root", type=Path, help="Override the source's expert-data path; identity must still match")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=TASKS, default="cartpole_balance_sparse")
    parser.add_argument("--online-steps", type=int, default=16384, help="Raw environment steps PER MODEL, a prefix of the saved online schedule")
    parser.add_argument("--online-seed", type=int, default=72_000_000)
    parser.add_argument("--policy-seed", type=int, default=61_000_000)
    parser.add_argument("--policy-cases", type=int, default=6)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--lookahead-seconds", type=float, default=.5)
    parser.add_argument("--hold-seconds", type=float, default=.3, help="Success diagnostic window only; does NOT change the native objective")
    parser.add_argument("--save-every", type=int, default=4096, help="Raw steps between recoverable weight/optimizer snapshots; this runner has no resume mode")
    parser.add_argument("--dry-run", action="store_true", help="Validate source and print budgets; no expert-data access, model execution, collection or outputs")
    args = parser.parse_args(argv)
    if (min(args.online_steps, args.save_every, args.policy_cases, args.policy_steps) < 1
            or min(args.online_seed, args.policy_seed) < 0
            or any(not math.isfinite(v) or v <= 0 for v in (args.lookahead_seconds, args.hold_seconds))
            or args.hold_seconds > args.lookahead_seconds):
        parser.error("Use positive finite budgets/windows, hold <= lookahead and nonnegative seeds")
    args.models, args.tasks = list(MODELS), [args.task]
    if args.output is None:
        prefix = "offline_online_decay" if args.retain_offline else "forecast_online"
        args.output = Path("runs") / datetime.now(timezone.utc).strftime(prefix + "_%Y%m%d_%H%M%S")
    return args


def configure(record, bank, args):
    config = OmegaConf.create(record["row"]["config"])
    config.device = config.env.device = config.replay.device = str(args.device)
    if args.dataset_root is not None:
        config.env.dataset_root = str(args.dataset_root.absolute())
        # Source configs are resolved: changing env.dataset_root alone does not
        # update the previously interpolated replay path.
        config.training.expert.data_path = str(args.dataset_root.absolute() / str(config.scenario.dataset))
    config.env.seed = args.online_seed
    config.replay.seed = args.online_seed + 1_000_003
    # This is the native_long condition from the frozen comparison, not tail.
    condition = next((row for row in record["conditions"] if row["name"] == "native_long"), record["conditions"][0])
    config.jepa_model.planner.horizon = condition["horizon"]
    quantum = int(config.env.env_num) * int(config.env.action_repeat)
    if args.online_steps % (2 * quantum):
        raise ValueError(f"Online steps must be divisible by {2 * quantum}, so midpoint and endpoint land on vector steps")
    if args.online_steps > int(config.training.online.steps):
        raise ValueError("Online budget exceeds the saved full schedule")
    midpoint = args.online_steps // 2
    if online_update_target(config, midpoint) < 1:
        raise ValueError("Midpoint must be beyond warmup and include at least one scheduled update")
    if int(config.env.env_num) < int(config.replay.episodes_per_batch):
        raise ValueError("Require enough online environments for source-episode diversity from the first updates")
    if min(midpoint // quantum, args.online_steps // quantum) < int(config.replay.sequence_length):
        raise ValueError("Online collection is too short to make complete training windows")
    online_seeds = set(range(args.online_seed, args.online_seed + int(config.env.env_num)))
    bank_seeds = {case["seed"] for cases in bank["splits"].values() for case in cases}
    if online_seeds & bank_seeds:
        raise ValueError("Online reset seeds overlap stored diagnostic/training bank seeds")
    replay = {"mode": "online_only" if not config.training.online.get("expert_fraction", 0.) else "expert_mixture"}
    if args.retain_offline:
        # Growing new-data pool for the whole experiment; original HDF5/branches
        # remain separate, read-only sources and are never evicted.
        config.replay.max_size = max(int(config.replay.max_size), args.online_steps // int(config.env.action_repeat))
        replay = replay_settings(config, bank, record["row"].get("coverage_fraction"),
                                 online_update_target(config, args.online_steps))
    return config, condition, {
        "raw_steps": args.online_steps, "agent_transitions": args.online_steps // int(config.env.action_repeat),
        "vector_collections": args.online_steps // quantum, "environments": int(config.env.env_num),
        "midpoint_raw_steps": midpoint, "midpoint_updates": online_update_target(config, midpoint),
        "updates": online_update_target(config, args.online_steps),
        "schedule_steps": int(config.training.online.steps), "schedule_updates": int(config.training.online.updates),
        "warmup_transitions": int(config.training.online.warmup_transitions),
        "native_expert_fraction": None if args.retain_offline else float(config.training.online.get("expert_fraction", 0.)),
        "native_branch_fraction": None if args.retain_offline else 0., "native_replay": replay,
        "readout_expert_fraction": float(config.state_head.online.expert_fraction),
        "planner": OmegaConf.to_container(config.jepa_model.planner, resolve=True),
    }


def load_training(record, config):
    if file_hash(record["path"]) != record["file_sha256"]:
        raise ValueError("Source checkpoint changed after validation")
    payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    validate_payload(payload, record["row"], record["dataset_identity"], record["bank_sha256"])
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    family = load_model_family(config.model_family)
    model = family.build_model(config)
    family.load_checkpoint(model, payload, training=True)
    for name, value in payload["counters"].items():
        setattr(model, name, value)
    if tensor_digest(model.state_dict()) != record["model_state_sha256"]:
        raise ValueError("Training load changed source model tensors")
    return family, model


def persist(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def summary(report):
    lines = ["Forecast diagnosis + native online learning | two models | fixed work budget",
             f"Run: {report['run_name']} | Status: {report['status']}"]
    for row in report["runs"]:
        lines.append(f"{row['model']} | {row['status']} | raw steps={row.get('raw_steps', 0)} | online updates={row.get('updates', 0)}")
        replay = row.get("budget", {}).get("native_replay", {})
        if replay.get("mode") == "offline_plus_online":
            lines.append(f"  Replay: original share {replay['offline_start_fraction']:.0%} -> 0% linearly "
                         f"over {replay['decay_updates']} native updates; uniform windows within each pool.")
            if "replay_sampling" in row.get("online", {}):
                lines.append(f"  Total native samples: {row['online']['replay_sampling']['total_samples']}")
        for snapshot in row["snapshots"]:
            p = snapshot["evaluation"]["summary"]["policy"]
            lines.append(f"  {snapshot['name']} @{snapshot['updates']}: return {p['return_mean']:.2f}/{p['maximum_return']} | "
                         f"maintained {round(p['maintenance_rate'] * p['cases'])}/{p['cases']}")
    lines += ["Same validation starts, goal scores, horizon and search settings before/midpoint/after.",
              "Fixed initial plans never enter training. Real-history predictions are diagnostics only.",
              ("Native batches hand over from 50/50 original/online to all online; the diagnostic readout mixture is unchanged."
               if report.get("settings", {}).get("retain_offline") else "Online replay starts empty."),
              "The saved full learning-rate/update schedule is not compressed.",
              "Raw latent MSE changes can reflect encoder changes: inspect normalized errors, feature spread and real control too.",
              "COMPLETE means execution, not a repair. One seed per model; this is a prefix, not the full online protocol.",
              "Checkpoints save weights and optimizers for inspection; exact simulator/replay resume is not supported."]
    if "seconds" in report:
        lines.append(f"Elapsed: {duration(report['seconds'])}")
    if "error" in report:
        lines.append(report["error"])
    return "\n".join(lines) + "\n"


def save_weights(folder, model, config, record, row, *, milestone=False):
    payload = {**load_model_family(model.model_family).checkpoint(model), "format": FORMAT,
               "resume_supported": False, "phase": "online", "training_config": OmegaConf.to_container(config, resolve=True),
               "source_checkpoint_sha256": record["file_sha256"], "dataset_identity": record["dataset_identity"],
               "bank_sha256": record["bank_sha256"], "env_steps": row["raw_steps"], "updates": row["updates"],
               "native_replay": row.get("budget", {}).get("native_replay", {}),
               "rng_state": tools.get_rng_state(),
               "counters": {name: getattr(model, name) for name in ("_gradient_updates", "_clipped_updates")}}
    path = folder / "latest.pt"
    atomic_save(payload, path)
    row["latest_checkpoint"] = {"file": path.name, "raw_steps": row["raw_steps"], "updates": row["updates"], "sha256": file_hash(path)}
    if milestone:
        path = folder / f"step_{row['raw_steps']}.pt"
        atomic_save(payload, path)
        row.setdefault("checkpoints", []).append({"file": path.name, "raw_steps": row["raw_steps"],
                                                  "updates": row["updates"], "sha256": file_hash(path)})


def save_online_data(folder, session, record, row):
    """Persist the newly added data separately from immutable original sources."""
    path = folder / "online_data.pt"
    windows = session.original.inventory(session.replay)
    atomic_save({"format": "offline_online_data_v1", "task": record["task"], "model": record["model"],
                 "dataset_identity": record["dataset_identity"], "bank_sha256": record["bank_sha256"],
                 "raw_steps": row["raw_steps"], "updates": row["updates"], "replay": session.replay.state_dict(),
                 "sampling_generator_state": session.original.generator.get_state(),
                 "sampling_updates": session.updates, "native_replay": row["budget"]["native_replay"],
                 "total_samples": dict(session.original.total_samples), "windows": windows}, path)
    row["online_data"] = {"file": path.name, "raw_steps": row["raw_steps"], "rows": session.replay.count(),
                          "online_windows": windows["online"], "sha256": file_hash(path)}


def snapshot(config, model, bank, condition, args, folder, row, name, fixed):
    print(f"Evaluate | {row['model']} | {name} | online updates={row['updates']}", flush=True)
    started = time.monotonic()
    horizon, hold = condition["branch_horizon"], condition["branch_hold_steps"]
    horizons = sorted({1, horizon, *(h for h in (5, 15) if h <= horizon)})
    cases = bank["splits"]["validation"]
    # Preserve model/optimizer/RNG/cache/gradient state, including the training
    # controller's warm starts. Evaluation uses separate simulator instances.
    with preserve_training_state(model):
        result = evaluate(config, model, cases, steps=args.policy_steps, policy_cases=args.policy_cases,
                          seed=args.policy_seed, horizons=horizons)
        probes = result["policy"]["probes"]
        if fixed is None:
            fixed = probe_cases(probes)
        result["branch_maintenance"] = branch_maintenance(result["branches"], cases, horizon, hold)
        # The source bank can be longer than the chosen diagnostic horizon.
        diagnostic_cases = [{**case, "action": case["action"][:, :horizon], "image": case["image"][:, :horizon]}
                            for case in cases]
        result["forecast_errors"] = forecast_errors(model, diagnostic_cases)
        branches = score_branches(model, fixed, horizons=horizons)
        result["fixed_initial_plans"] = {
            "branches": branches, "forecast_errors": forecast_errors(model, fixed),
            "maintenance": branch_maintenance(branches, fixed, horizon, hold),
            "data_sha256": _digest(fixed),
            "source": "before_plans.pt; identical images/actions at every milestone; never used for updates"}
        result["current_plan_diagnostics"] = summarize_probes(probes, hold)
        result["current_plan_diagnostics"]["forecast_errors"] = (
            result["fixed_initial_plans"]["forecast_errors"] if name == "before"
            else forecast_errors(model, probe_cases(probes)))
        digest = tensor_digest(model.state_dict())
    item = {"name": name, "raw_steps": row["raw_steps"], "updates": row["updates"],
            "model_state_sha256": digest, "evaluation": save_evaluation(folder, name, result),
            "seconds": time.monotonic() - started}
    row["snapshots"].append(item)
    return fixed


def train_one(config, model, family, record, bank, condition, args, output, report, row):
    folder = output / record["task"] / record["model"]
    folder.mkdir(parents=True)
    fixed = snapshot(config, model, bank, condition, args, folder, row, "before", None)
    persist(output, report)
    tools.configure_randomness(args.online_seed, bool(config.deterministic_run))
    envs, session = None, None
    try:
        retained = (original_replay(config, family, bank, row["budget"]["native_replay"], record["dataset_identity"])
                    if args.retain_offline else nullcontext(None))
        with online_readout(config, family, model, expected_dataset=record["dataset_identity"]), retained as original:
            if hasattr(model, "configure_online"):
                model.configure_online(int(config.training.online.updates), resumed=False)
            model.train()
            envs = make_envs(config.env, seed=args.online_seed)
            session = (OfflineOnlineSession(config, model, envs, original)
                       if args.retain_offline else family.OnlineSession(config, model, envs))
            session.start()
            if session.replay.count():
                raise RuntimeError("Online replay must start empty")
            if original is not None:
                row["budget"]["native_replay"]["initial_windows"] = original.inventory(session.replay)
            initial_head_updates = int(model.state_head.updates)
            initial_native_updates = model._gradient_updates
            save_weights(folder, model, config, record, row, milestone=True)
            progress = Progress(f"Online {record['model']}", args.online_steps)
            next_save, episodes, update_seconds, collection_seconds = args.save_every, [], 0., 0.
            with (folder / "online_metrics.jsonl").open("w", buffering=1) as log:
                while row["raw_steps"] < args.online_steps:
                    tick = time.monotonic()
                    delta, finished = session.collect()
                    if delta != int(config.env.env_num) * int(config.env.action_repeat):
                        raise RuntimeError("Unexpected online collection quantum")
                    collection_seconds += time.monotonic() - tick
                    row["raw_steps"] += delta
                    episodes.extend(finished)
                    target = online_update_target(config, row["raw_steps"])
                    metrics = {}
                    if target > row["updates"] and session.replay.ready():
                        tick = time.monotonic()
                        metrics = session.update(target - row["updates"])
                        metrics = {key: float(torch.as_tensor(value).detach()) for key, value in metrics.items()}
                        if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                            raise ValueError("Non-finite or missing online metrics")
                        row["updates"] = target
                        update_seconds += time.monotonic() - tick
                        if (int(model.state_head.updates) != initial_head_updates + target
                                or model._gradient_updates != initial_native_updates + target):
                            raise RuntimeError("Native/readout updates differ from the scheduled count")
                    entry = {"raw_steps": row["raw_steps"], "updates": row["updates"], "replay_rows": session.replay.count(),
                             "metrics": metrics, "completed_episodes": finished}
                    log.write(json.dumps(entry, allow_nan=False) + "\n")
                    progress.update(row["raw_steps"], f"updates={row['updates']} replay={session.replay.count()}",
                                    force=row["raw_steps"] == args.online_steps)
                    milestone = row["raw_steps"] in (args.online_steps // 2, args.online_steps)
                    if milestone or row["raw_steps"] >= next_save:
                        save_weights(folder, model, config, record, row, milestone=milestone)
                        if args.retain_offline:
                            save_online_data(folder, session, record, row)
                        next_save = (row["raw_steps"] // args.save_every + 1) * args.save_every
                        persist(output, report)
                    if milestone:
                        if row["updates"] != target:
                            raise RuntimeError("Online replay did not supply the scheduled milestone updates")
                        fixed = snapshot(config, model, bank, condition, args, folder, row,
                                         "after" if row["raw_steps"] == args.online_steps else "midpoint", fixed)
                        persist(output, report)
            row["online"] = {"replay_rows": session.replay.count(), "completed_episodes": episodes,
                             "collection_seconds": collection_seconds, "update_seconds": update_seconds,
                             "partial_episode_returns": session.returns.cpu().tolist(),
                             "partial_episode_lengths": session.lengths.cpu().tolist(),
                             "head_updates_before": initial_head_updates, "head_updates_after": int(model.state_head.updates)}
            if original is not None:
                row["online"]["replay_sampling"] = {"total_samples": dict(original.total_samples),
                                                      "final_windows": original.inventory(session.replay)}
            if row["updates"] != online_update_target(config, args.online_steps):
                raise RuntimeError("Did not complete the scheduled online prefix")
            if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("Non-finite model tensors after online training")
    finally:
        try:
            if (args.retain_offline and session is not None and session.replay.count()
                    and row.get("online_data", {}).get("raw_steps") != row["raw_steps"]):
                save_online_data(folder, session, record, row)
        finally:
            close_envs(envs)


def main(argv=None):
    args = arguments(argv)
    output = args.output.absolute()
    report = {"format": FORMAT, "run_name": output.name, "status": "PREPARING", "runs": [],
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "split": "validation", "offline_updates": 0, "online_schedule_changed": False,
              "runtime_target_hours": 5, "runtime_enforced": False}
    created, started = False, time.monotonic()
    try:
        torch.set_num_threads(1)
        source, banks, records = load_source(args)
        prepared = [(record, *configure(record, banks[record["task"]], args)) for record in records]
        report["budgets"] = [{"model": record["model"], **budget} for record, _, _, budget in prepared]
        if args.dry_run:
            print(json.dumps({"budgets": report["budgets"], "expert_dataset_checked": False}, indent=2), flush=True)
            report["status"] = "DRY_RUN"
            return 0
        versions = runtime_versions(args.device)
        for name in ("dm-control", "mujoco", "numpy"):
            if versions[name] != source["versions"][name]:
                raise ValueError(f"Source simulator/runtime version differs: {name}")
        if torch.device(args.device).type == "cuda":
            torch.cuda.set_device(args.device)
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Source full models require BF16-capable CUDA")
        # Fail for absent/wrong expert data before any expensive evaluation. The
        # default native learner uses none; its detached readout retains 50%.
        for record, config, _, _ in prepared:
            if args.retain_offline or config.state_head.online.expert_fraction or config.training.online.get("expert_fraction", 0.):
                with load_model_family(config.model_family).build_replay(config) as replay:
                    if dataset_identity(replay.metadata) != record["dataset_identity"]:
                        raise ValueError("Expert dataset identity differs from the source checkpoint")
        if output.resolve().is_relative_to(args.source_run.resolve()):
            raise ValueError("Output must be outside the source run")
        output.mkdir(parents=True, exist_ok=False)
        created = True
        report.update(status="RUNNING", versions=versions, implementation_sha256=implementation_sha256(),
                      source_report_sha256=file_hash(args.source_run / "report.json"), source_helpers=source_hashes(),
                      runner_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                                     ("train_forecast_online.py", "forecast_online_support.py", "offline_online_replay.py", "evaluate_goal_maintenance.py")},
                      source_banks=source["banks"],
                      checkpoints=[{key: str(record[key]) if isinstance(record[key], Path) else record[key]
                                    for key in ("task", "model", "path", "file_sha256", "updates", "model_state_sha256")}
                                   for record in records])
        persist(output, report)
        for record, config, condition, budget in prepared:
            row = {"task": record["task"], "model": record["model"], "status": "RUNNING", "raw_steps": 0, "updates": 0,
                   "config": OmegaConf.to_container(config, resolve=True), "budget": budget, "snapshots": []}
            report["runs"].append(row)
            tick = time.monotonic()
            try:
                family, model = load_training(record, config)
                bank = banks[record["task"]]
                if "control" not in report:
                    with preserve_training_state(model):
                        control = policy_trial(config, model, bank["splits"]["validation"][:args.policy_cases],
                                               args.policy_steps, args.policy_seed, diagnostics=False, zero=True)
                    (output / "zero_control.json").write_text(json.dumps(control, allow_nan=False) + "\n")
                    report["control"] = {key: value for key, value in control.items() if key not in ("traces", "probes")}
                train_one(config, model, family, record, bank, condition, args, output, report, row)
                row["status"] = "COMPLETE"
                del model
                if torch.device(args.device).type == "cuda":
                    torch.cuda.empty_cache()
            except BaseException as error:
                row["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL"
                raise
            finally:
                row["seconds"] = time.monotonic() - tick
                persist(output, report)
        report["status"] = "COMPLETE"
        return 0
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        return 130
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        return 1
    finally:
        report["seconds"] = time.monotonic() - started
        if created:
            persist(output, report)
        print(summary(report), end="", flush=True)
        print(f"Run | {output.name} | status={report['status']}", flush=True)
        print(f"Reports | {output}" if created else "Reports | none written", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
