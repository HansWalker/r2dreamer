"""Four-hour, six-fit native TS/LeWM learning curves across all three DMC tasks.

Offline only. Each fit gets the same predeclared update count and its own complete
optimizer/sampler/RNG checkpoint. No controller modules or training losses are added.
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from scripts.paper_faithful_duration_support import (
    TaskBranchReplay, collect_bank, evaluate, policy_trial, validate_bank,
)
from scripts.paper_faithful_followup_eval import preserve_training_state, summarize_evaluation
from scripts.paper_faithful_reference import run_reference_checks
from scripts.train_paper_faithful_check import build_config as base_config, new_model, synchronize, update
from scripts.train_paper_faithful_followup import milestone_updates, runtime_versions
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256


TASKS = ("cartpole_balance_sparse", "reacher", "ball_in_cup")
MODELS = {"temporal_straightening": "ts_patch_01", "leworldmodel": "lewm"}
FORMAT = "paper_faithful_duration_v1"
SOURCE_FILES = ("train_paper_faithful_duration.py", "paper_faithful_duration_support.py",
                "train_paper_faithful_check.py", "train_paper_faithful_followup.py",
                "paper_faithful_followup_eval.py", "paper_faithful_support.py",
                "paper_faithful_reference.py", "diagnose_planner_oracle.py",
                "diagnose_fresh_readout.py", "smoke_tiny_planners.py")


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_config(model, task, args):
    settings = SimpleNamespace(**vars(args))
    settings.scenario, settings.horizon = task, None
    config = base_config(MODELS[model], settings)
    # Hydra's production logdir contains wall-clock interpolation. It is unused
    # here and must not make otherwise identical resume configurations differ.
    config.logdir = f"paper_faithful_duration/{task}/{model}/seed_{args.seed}"
    return config


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dmc_expert_vision"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume this runner's existing run directory and fixed budget")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=60_000_000)
    parser.add_argument("--minutes", type=float, default=240.)
    parser.add_argument("--updates", type=int, help="Explicit updates per fit, overriding the time estimate")
    parser.add_argument("--min-updates", type=int, default=8192)
    parser.add_argument("--max-updates", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--train-anchors", type=int, default=64)
    parser.add_argument("--validation-anchors", type=int, default=6)
    parser.add_argument("--test-anchors", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--forecast-horizons", nargs="+", type=int, default=[1, 5, 15, 25])
    parser.add_argument("--validation-policy-cases", type=int, default=2)
    parser.add_argument("--validation-policy-steps", type=int, default=100)
    parser.add_argument("--policy-cases", type=int, default=3)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--calibration-updates", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--profile", choices=("full", "tiny"), default="full")
    parser.add_argument("--reference-cache", type=Path, default=Path("runs/reference_cache"))
    parser.add_argument("--skip-reference", action="store_true", help="Explicit smoke-only bypass")
    parser.add_argument("--no-reference-download", action="store_true")
    args = parser.parse_args(argv)
    raw = list(sys.argv[1:] if argv is None else argv)
    if args.resume:
        allowed = {"--resume", "--dry-run"}
        if any(token.split("=")[0] not in allowed for token in raw if token.startswith("--")):
            parser.error("--resume restores saved settings; only --dry-run may accompany it")
    positive = (args.minutes, args.min_updates, args.max_updates, args.batch_size, args.sources,
                args.train_anchors, args.validation_anchors, args.test_anchors, args.candidates,
                args.validation_policy_cases, args.validation_policy_steps, args.policy_cases,
                args.policy_steps, args.calibration_updates, args.save_every, *args.forecast_horizons)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("Counts, horizons and runtime must be finite and positive")
    if (args.batch_size % args.sources or args.batch_size % 2 or args.sources % 2 or
            args.sources < 2 or args.train_anchors < args.sources // 2):
        parser.error("Use an even batch/source count, divisible batch, and enough training anchors")
    if (len(set(args.tasks)) != len(args.tasks) or len(set(args.models)) != len(args.models)
            or args.min_updates > args.max_updates or (args.updates is not None and args.updates < 1)
            or min(args.seed, args.data_seed) < 0 or args.candidates < 6):
        parser.error("Invalid/duplicate models, tasks, seeds, candidates or update budget")
    if args.validation_policy_cases > args.validation_anchors or args.policy_cases > min(args.test_anchors, args.validation_anchors):
        parser.error("Policy cases must fit inside their corresponding split")
    if max(args.policy_steps, args.validation_policy_steps) + max(25, *args.forecast_horizons) + 63 >= 500:
        parser.error("Trials and diagnostic plans must fit after the longest roll-in")
    return args


def branch_replay(bank, args):
    return TaskBranchReplay(bank, batch_size=args.batch_size // 2, sequence_length=4,
                            episodes_per_batch=args.sources // 2, seed=args.seed + 12345)


def save_checkpoint(path, model, dataset, branch, row, total_updates, identity, bank_hash):
    payload = {**load_model_family(model.model_family).checkpoint(model), "format": FORMAT,
               "resume_supported": True, "total_updates": total_updates, "row": copy.deepcopy(row),
               "training_config": row["config"], "updates": row["updates"], "phase": "expert",
               "dataset_identity": identity, "bank_hash": bank_hash,
               "dataset_sampler": dataset.state_dict(), "branch_sampler": branch.state_dict(),
               "rng_state": tools.get_rng_state(),
               "counters": {name: getattr(model, name) for name in ("_gradient_updates", "_clipped_updates")}}
    atomic_save(payload, path)


def restore_checkpoint(path, model, dataset, branch, total_updates, identity, bank_hash, config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != FORMAT or not payload.get("resume_supported") or
            payload["total_updates"] != total_updates or payload["dataset_identity"] != identity or
            payload["bank_hash"] != bank_hash or
            payload["training_config"] != OmegaConf.to_container(config, resolve=True)):
        raise ValueError("Resume checkpoint does not match this run, budget, data or configuration")
    load_model_family(model.model_family).load_checkpoint(model, payload, training=True)
    if hasattr(model, "configure_pretraining"):
        model.configure_pretraining(total_updates)
    dataset.load_state_dict(payload["dataset_sampler"])
    branch.load_state_dict(payload["branch_sampler"])
    for name, value in payload["counters"].items():
        setattr(model, name, value)
    tools.set_rng_state(payload["rng_state"])
    return payload["row"]


def save_evaluation(folder, name, result):
    policy = result.get("policy")
    if policy is not None:
        (folder / f"{name}_policy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in policy.pop("traces")))
        probes = policy.pop("probes")
        atomic_save(probes, folder / f"{name}_plans.pt")
        policy["traces_file"], policy["plans_file"] = f"{name}_policy.jsonl", f"{name}_plans.pt"
        policy["plan_diagnostics"] = [{
            "case_id": p["case_id"], "step": p["step"], "reason": p["reason"],
            "predicted_cost": p["predicted_cost"].tolist(), "actual_cost": p["actual_cost"].tolist(),
            "returns": p["rewards"].sum(1).tolist(), "selected_index": 0,
            "better_predicted_alternative": bool((p["predicted_cost"][1:] < p["predicted_cost"][0] - 1e-7).any()),
        } for p in probes]
    return {"result": result, "summary": summarize_evaluation(result)}


def summary(report):
    lines = ["Native TS/LeWM training duration | offline only | one configuration per model",
             f"Run: {report['run_name']} | Status: {report['status']}",
             "Task | model | updates | training minutes | test return | maintained cases"]
    for row in report["runs"]:
        p = row.get("test", {}).get("summary", {}).get("policy", {})
        lines.append(f"{row['task']} | {row['model']} | {row['updates']} | {row['training_seconds']/60:.1f} | "
                     f"{p.get('return_mean', 'n/a')} / {p.get('maximum_return', 'n/a')} | "
                     f"{p.get('maintenance_rate', 'n/a')} ({p.get('cases', 0)} cases)")
        for snapshot in row["validation"]:
            p = snapshot["summary"]["policy"]
            h = snapshot["summary"]["horizons"]
            ratios = ", ".join(f"H{k}={v['all'].get('matched_over_persistence')}" for k, v in h.items())
            lines.append(f"  validation @{snapshot['updates']}: return={p.get('return_mean')}; prediction/persistence {ratios}")
    if "budget" in report:
        lines.append(f"Budget per fit: {report['budget']['common_updates']} updates; target {report['settings']['minutes']} minutes total")
    lines += ["Validation curves use fixed starts/duration. Final test starts are separate.",
              "One seed per model/task; not a multi-seed confirmation or full paper reproduction.",
              "Snapshots contain full training state. Resume preserves the original budget and learning-rate schedule.",
              "No online updates; this tests offline learning duration, not online retention.",
              "Actual-future goal scores still depend on the learned encoder; they are not reward oracles.",
              "COMPLETE means execution, not successful task control or convergence."]
    if "error" in report:
        lines.append(report["error"])
    return "\n".join(lines) + "\n"


def persist(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def calibrate(config, args, bank):
    with load_model_family(config.model_family).build_replay(config) as dataset:
        tick = time.monotonic()
        model = new_model(config, dataset)
        branch = branch_replay(bank, args)
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(args.max_updates)
        setup = time.monotonic() - tick
        rates = []
        for step in range(args.calibration_updates + 2):
            synchronize(model.device)
            tick = time.monotonic()
            update(model, dataset, branch, .5, step, args.seed)
            synchronize(model.device)
            if step >= 2:
                rates.append(time.monotonic() - tick)
        validation = evaluate(config, model, bank["splits"]["validation"], steps=3,
                              policy_cases=args.validation_policy_cases, seed=args.data_seed + 500_000,
                              horizons=args.forecast_horizons)
        # Runtime profiling uses validation states for BOTH batch sizes; no test outcomes.
        profile_cases = bank["splits"]["validation"][:args.policy_cases]
        if len(profile_cases) != args.policy_cases:
            raise ValueError("Runtime profiling needs at least policy-cases validation anchors")
        with preserve_training_state(model):
            test_shape = policy_trial(config, model, profile_cases, 3, args.data_seed + 500_000)
        def policy_seconds(result, steps):
            return result["setup_seconds"] + result["acting_seconds"] / result["steps"] * steps + result["diagnostic_seconds"]
        # Initialization forecasts, midpoint/final validation, then final held-out test.
        forecast = validation["forecast_seconds"]
        overhead = (setup + forecast * (3 + args.test_anchors / args.validation_anchors) +
                    2 * policy_seconds(validation["policy"], args.validation_policy_steps) +
                    policy_seconds(test_shape, args.policy_steps) + 60.)
        result = {"update_seconds": statistics.median(rates), "overhead_seconds": overhead,
                  "setup_seconds": setup, "forecast_seconds": forecast,
                  "validation_acting_seconds_per_decision": validation["policy"]["acting_seconds"] / 3,
                  "test_acting_seconds_per_decision": test_shape["acting_seconds"] / 3,
                  "scope": "Disposable model; validation-only timing; no test scores retained"}
        del model
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def choose_budget(args, calibration, elapsed):
    overhead = sum(row["overhead_seconds"] for row in calibration.values())
    rate = sum(row["update_seconds"] for row in calibration.values())
    available = args.minutes * 60 - elapsed - 1.2 * overhead - 60
    count = args.updates if args.updates is not None else min(args.max_updates, math.floor(available / (1.15 * rate)))
    if count < (1 if args.updates is not None else args.min_updates):
        raise ValueError(f"Four-hour estimate permits {count} updates per fit, below --min-updates {args.min_updates}. "
                         "No comparison fits started. Increase --minutes or explicitly lower the minimum; models are not shrunk.")
    return {"common_updates": count, "milestones": milestone_updates(count, [.5]), "fits": len(calibration),
            "estimated_remaining_seconds": count * rate + overhead, "fixed_before_training": True,
            "hard_deadline": False, "adjacent_targets_per_fit": count * args.batch_size * 3,
            "expert_target_presentations_per_fit": count * (args.batch_size // 2) * 3,
            "intervention_target_presentations_per_fit": count * (args.batch_size // 2) * 3}


def trim_log(path, updates):
    if path.exists():
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        retained = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # A process kill may leave one incomplete final line.
            if row["update"] <= updates:
                retained.append(line)
        path.write_text("".join(line + "\n" for line in retained))


def fit(config, args, bank, report, key):
    folder = args.output / key
    folder.mkdir(parents=True, exist_ok=True)
    count = report["budget"]["common_updates"]
    config.training.expert.updates = count
    with load_model_family(config.model_family).build_replay(config) as dataset:
        identity = dataset_identity(dataset.metadata)
        if identity != report["datasets"][str(config.scenario.name)]["identity"]:
            raise ValueError("Expert dataset identity changed")
        model = new_model(config, dataset)
        branch = branch_replay(bank, args)
        row = {"key": key, "task": str(config.scenario.name), "model": str(config.model_family),
               "seed": args.seed, "status": "RUNNING", "updates": 0, "training_seconds": 0.,
               "config": OmegaConf.to_container(config, resolve=True), "validation": [], "coverage_fraction": .5}
        checkpoint = folder / "latest.pt"
        if checkpoint.exists():
            row = restore_checkpoint(checkpoint, model, dataset, branch, count, identity, bank["sha256"], config)
        elif hasattr(model, "configure_pretraining"):
            model.configure_pretraining(count)
        existing = next((i for i, old in enumerate(report["runs"]) if old["key"] == key), None)
        if existing is None:
            report["runs"].append(row)
        else:
            report["runs"][existing] = row
        if row["status"] == "COMPLETE":
            return
        log_path = folder / "metrics.jsonl"
        trim_log(log_path, row["updates"])

        def save():
            save_checkpoint(checkpoint, model, dataset, branch, row, count, identity, bank["sha256"])
            persist(args.output, report)

        def measure(step):
            print(f"Validation | {key} update {step}/{count}", flush=True)
            result = evaluate(config, model, bank["splits"]["validation"], steps=args.validation_policy_steps,
                              policy_cases=args.validation_policy_cases, seed=args.data_seed + 500_000,
                              horizons=args.forecast_horizons, initial=step == 0)
            row["validation"].append({"updates": step, **save_evaluation(folder, f"validation_{step}", result)})
            if step:
                # These complete snapshots also identify the exact weights used by plan traces.
                save_checkpoint(folder / f"update_{step}.pt", model, dataset, branch, row,
                                count, identity, bank["sha256"])
            # Commit completed-measurement progress only after its archived weights
            # exist. A kill between writes then safely repeats the measurement.
            save()

        measured = {snapshot["updates"] for snapshot in row["validation"]}
        if row["updates"] in report["budget"]["milestones"] and row["updates"] not in measured:
            measure(row["updates"])
        progress = Progress(key, count)
        with log_path.open("a", buffering=1) as log:
            for step in range(row["updates"] + 1, count + 1):
                synchronize(model.device)
                tick = time.monotonic()
                values = update(model, dataset, branch, .5, step, args.seed)
                synchronize(model.device)
                row["training_seconds"] += time.monotonic() - tick
                row["updates"] = step
                log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
                progress.update(step, f"prediction={values['prediction_loss']:.4g}", force=step == count)
                if step % args.save_every == 0 or step in report["budget"]["milestones"]:
                    save()  # Safe recovery point even if the following measurement is interrupted.
                if step in report["budget"]["milestones"]:
                    measure(step)
        print(f"Final test | {key}", flush=True)
        result = evaluate(config, model, bank["splits"]["test"], steps=args.policy_steps,
                          policy_cases=args.policy_cases, seed=args.data_seed + 600_000,
                          horizons=args.forecast_horizons)
        row["test"] = save_evaluation(folder, "test", result)
        row["status"] = "COMPLETE"
        save()
        del model
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()


def main(argv=None):
    args = arguments(argv)
    report = None
    if args.resume:
        output = args.resume.absolute()
        report = json.loads((output / "report.json").read_text())
        if report.get("format") != FORMAT or "budget" not in report:
            raise ValueError("Resume requires this runner's calibrated run; old evaluation snapshots are unsupported")
        if report["implementation_sha256"] != implementation_sha256() or report["source_hashes"] != source_hashes():
            raise ValueError("Code differs from the saved run; refusing to label a changed experiment an exact resume")
        dry_run = args.dry_run
        args = argparse.Namespace(**report["settings"])
        for key in ("dataset_root", "reference_cache"):
            setattr(args, key, Path(getattr(args, key)))
        args.output, args.resume, args.dry_run = output, output, dry_run
    if args.output is None:
        args.output = Path("runs") / datetime.now(timezone.utc).strftime("paper_faithful_duration_%Y%m%d_%H%M%S")
    configs = {f"{task}/{model}": build_config(model, task, args) for task in args.tasks for model in args.models}
    if args.dry_run:
        print(json.dumps({"target_minutes": args.minutes, "offline_only": True, "seed": args.seed,
                          "resume": bool(args.resume), "fits": {key: {
                              "horizon": int(config.jepa_model.planner.horizon), "objective": str(config.jepa_model.planner.objective),
                              "samples": int(config.jepa_model.planner.samples), "iterations": int(config.jepa_model.planner.iterations),
                              "data": str(config.training.expert.data_path)} for key, config in configs.items()}}, indent=2))
        return 0
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the four-hour command on the GPU host")
    started = time.monotonic()
    if report is None:
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"format": FORMAT, "run_name": args.output.name, "status": "RUNNING", "runs": [],
                  "settings": {k: str(v.absolute()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "implementation_sha256": implementation_sha256(), "source_hashes": source_hashes(),
                  "versions": runtime_versions(args.device), "online_updates": 0, "online_schedule_changed": False,
                  "datasets": {}, "banks": {}, "controls": {}, "calibration": {}}
    else:
        report["status"] = "RUNNING"
        report.pop("error", None)
        report.pop("traceback", None)
        if runtime_versions(args.device) != report["versions"]:
            raise ValueError("Runtime versions/device changed; exact resume is not supported across runtimes")
    persist(args.output, report)
    banks = {}
    try:
        if not args.resume:
            # Fail missing/mismatched datasets before collecting any new simulator data.
            for task in args.tasks:
                config = configs[f"{task}/{args.models[0]}"]
                with load_model_family(config.model_family).build_replay(config) as dataset:
                    report["datasets"][task] = {"identity": dataset_identity(dataset.metadata),
                                                "train_transitions": int(dataset.lengths[dataset.episodes].sum()),
                                                "train_episodes": len(dataset.episodes)}
            report["reference"] = ({"status": "NOT_RUN", "reason": "Explicit smoke-only bypass"} if args.skip_reference else
                                   run_reference_checks({name: configs[f"{args.tasks[0]}/{name}"] for name in args.models},
                                                        cache=args.reference_cache, allow_download=not args.no_reference_download))
            if not args.skip_reference and report["reference"]["status"] != "PASS":
                raise RuntimeError("Pinned upstream component parity failed")
            for task in args.tasks:
                config = configs[f"{task}/{args.models[0]}"]
                bank = collect_bank(config, counts={"train": args.train_anchors, "validation": args.validation_anchors,
                                                    "test": args.test_anchors}, candidates=args.candidates,
                                    horizon=max(*args.forecast_horizons, int(config.jepa_model.planner.horizon)),
                                    seed=args.data_seed + TASKS.index(task) * 1_000_000,
                                    progress=lambda n, total: print(f"Bank | {task} {n}/{total}", flush=True))
                path = args.output / f"{task}_bank.pt"
                atomic_save(bank, path)
                report["banks"][task] = {"file": path.name, "file_sha256": file_hash(path), "sha256": bank["sha256"],
                                         "metadata": bank["metadata"]}
                banks[task] = bank
                model = new_model(config)
                report["controls"][task] = {}
                with preserve_training_state(model):
                    for split, cases, steps in (("validation", args.validation_policy_cases, args.validation_policy_steps),
                                                ("test", args.policy_cases, args.policy_steps)):
                        control = policy_trial(config, model, bank["splits"][split][:cases], steps, args.data_seed,
                                               diagnostics=False, zero=True)
                        (args.output / f"{task}_{split}_zero.json").write_text(json.dumps(control, allow_nan=False) + "\n")
                        report["controls"][task][split] = {k: v for k, v in control.items() if k not in ("traces", "probes")}
                del model
                persist(args.output, report)
            for key, config in configs.items():
                print(f"Calibrate | {key}", flush=True)
                report["calibration"][key] = calibrate(config, args, banks[str(config.scenario.name)])
                persist(args.output, report)
            report["budget"] = choose_budget(args, report["calibration"], time.monotonic() - started)
            persist(args.output, report)
        else:
            for task, entry in report["banks"].items():
                path = args.output / entry["file"]
                if file_hash(path) != entry["file_sha256"]:
                    raise ValueError("Saved bank file changed")
                banks[task] = torch.load(path, map_location="cpu", weights_only=False)
                validate_bank(banks[task])
        print(f"Budget | {len(configs)} fits x {report['budget']['common_updates']} updates | "
              f"target {args.minutes:g} minutes total", flush=True)
        for key, config in configs.items():
            fit(config, args, banks[str(config.scenario.name)], report, key)
        report["status"] = "COMPLETE"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "Interrupted; resume restores the last atomic checkpoint without extending its schedule"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["seconds"] = report.get("seconds", 0.) + time.monotonic() - started
        persist(args.output, report)
        print(summary(report), flush=True)
        print(f"Run | {args.output.name} | status={report['status']}", flush=True)
        print(f"Reports | {args.output.absolute()}", flush=True)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
