"""Fixed-length Cartpole comparison: roughly four hours for native TS and LeWM.

Offline only. Each fit gets a predeclared update count and its own complete
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
    parser.add_argument("--reuse-banks", type=Path, help="Copy validated simulator banks into a NEW run and train from scratch")
    parser.add_argument("--estimate-from", type=Path, help="Read saved timings and print budget estimates only; no GPU, data collection or training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["cartpole_balance_sparse"])
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=60_000_000)
    parser.add_argument("--minutes", type=float, default=240., help="Rough runtime label in fixed mode; target for legacy timing modes")
    parser.add_argument("--budget-mode", choices=("fixed", "equal-time", "equal-updates"), default="fixed",
                        help="Fixed counts without calibration (default); legacy timing modes are opt-in")
    parser.add_argument("--ts-updates", type=int, default=24000, help="TS updates in fixed mode")
    parser.add_argument("--lewm-updates", type=int, default=28000, help="LeWM updates in fixed mode")
    parser.add_argument("--updates", type=int, help="Explicit common updates; selects fixed mode and skips timing calibration")
    parser.add_argument("--min-updates", type=int, default=8192)
    parser.add_argument("--max-updates", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--train-anchors", type=int, default=64)
    parser.add_argument("--validation-anchors", type=int, default=6)
    parser.add_argument("--test-anchors", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--forecast-horizons", nargs="+", type=int, default=[1, 5, 15, 25])
    parser.add_argument("--validation-fractions", nargs="*", type=float, default=[.25, .5, .75],
                        help="Intermediate update fractions; initialization and final validation are always included")
    parser.add_argument("--validation-policy-cases", type=int, default=4)
    parser.add_argument("--validation-policy-steps", type=int, default=100)
    parser.add_argument("--policy-cases", type=int, default=6)
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
    if args.estimate_from:
        allowed = {"--estimate-from", "--minutes", "--updates", "--min-updates", "--max-updates",
                   "--tasks", "--models", "--budget-mode"}
        if any(token.split("=")[0] not in allowed for token in raw if token.startswith("--")):
            parser.error("--estimate-from only accepts budget overrides and task/model subsets; saved evaluation settings apply")
        if "--budget-mode=fixed" in raw or any(raw[i:i + 2] == ["--budget-mode", "fixed"] for i in range(len(raw))):
            parser.error("--estimate-from inspects legacy timing modes; fixed counts do not need a time estimate")
    elif args.updates is not None:
        args.budget_mode = "fixed"
    positive = (args.minutes, args.min_updates, args.max_updates, args.batch_size, args.sources,
                args.ts_updates, args.lewm_updates,
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
    if (any(not math.isfinite(value) or not 0 < value < 1 for value in args.validation_fractions) or
            len(set(args.validation_fractions)) != len(args.validation_fractions)):
        parser.error("Validation fractions must be unique and strictly between zero and one")
    args.validation_fractions.sort()
    return args


def fixed_budget(args):
    """Finish the declared counts regardless of elapsed time; no timing probes."""
    counts = {"temporal_straightening": args.ts_updates, "leworldmodel": args.lewm_updates}
    fits = {}
    for task in args.tasks:
        for model in args.models:
            count = args.updates if args.updates is not None else counts[model]
            fits[f"{task}/{model}"] = {
                "updates": count, "milestones": milestone_updates(count, args.validation_fractions),
                "allocation_seconds": None, "adjacent_targets": count * args.batch_size * 3,
                "expert_target_presentations": count * (args.batch_size // 2) * 3,
                "intervention_target_presentations": count * (args.batch_size // 2) * 3,
            }
    return {"mode": "fixed", "common_updates": args.updates, "fits": len(fits), "fit_budgets": fits,
            "fixed_before_training": True, "hard_deadline": False, "rough_target_minutes": args.minutes,
            "scope": "Declared update counts; no timing calibration, runtime feasibility gate or clock cutoff"}


def branch_replay(bank, args):
    return TaskBranchReplay(bank, batch_size=args.batch_size // 2, sequence_length=4,
                            episodes_per_batch=args.sources // 2, seed=args.seed + 12345)


def bank_spec(config, args):
    task = str(config.scenario.name)
    return {"task": task, "counts": {"train": args.train_anchors, "validation": args.validation_anchors,
                                     "test": args.test_anchors},
            "candidates": args.candidates,
            "horizon": max(*args.forecast_horizons, int(config.jepa_model.planner.horizon)),
            "seed": args.data_seed + TASKS.index(task) * 1_000_000}


def reuse_bank(source, config, args, identity, versions):
    """Only data reuse, never weight/timing reuse or a changed-code exact resume."""
    previous = json.loads((source / "report.json").read_text())
    if previous.get("format") != FORMAT or previous.get("implementation_sha256") != implementation_sha256():
        raise ValueError("Bank source has a different format or model/environment implementation")
    # This runner may change its budget/reporting; all bank-producing helpers must match.
    for name, digest in source_hashes().items():
        if name != Path(__file__).name and previous.get("source_hashes", {}).get(name) != digest:
            raise ValueError(f"Bank source helper changed: {name}")
    for name in ("dm-control", "mujoco", "numpy"):
        if previous.get("versions", {}).get(name) != versions[name]:
            raise ValueError(f"Bank source simulator runtime changed: {name}")
    task = str(config.scenario.name)
    if previous.get("datasets", {}).get(task, {}).get("identity") != identity:
        raise ValueError(f"Bank source expert dataset identity changed: {task}")
    entry = previous["banks"][task]
    if any(entry["metadata"].get(key) != value for key, value in bank_spec(config, args).items()):
        raise ValueError(f"Bank source collection settings differ: {task}")
    path = source / entry["file"]
    if file_hash(path) != entry["file_sha256"]:
        raise ValueError(f"Bank source file changed: {task}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    validate_bank(bank)
    if bank["sha256"] != entry["sha256"] or bank["metadata"] != entry["metadata"]:
        raise ValueError(f"Bank source manifest mismatch: {task}")
    return bank


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
        label = "rough runtime only; no time limit" if report["budget"]["mode"] == "fixed" else "runtime target"
        lines.append(f"Budget: {report['budget']['mode']}; {report['settings']['minutes']} minutes total ({label})")
        for key, entry in report["budget"]["fit_budgets"].items():
            allocation = entry.get("allocation_seconds")
            label = f"; {allocation/60:g} minutes including preparation share" if allocation is not None else ""
            lines.append(f"  {key}: {entry['updates']} updates{label}")
    if "budget_estimate" in report:
        estimate = report["budget_estimate"]
        lines.append(f"Timing at {estimate['required_updates']} updates/fit: "
                     f"preparation {estimate['preparation_seconds']/60:.1f} min; "
                     f"training {estimate['required_training_seconds']/60:.1f} min; "
                     f"evaluation/setup {estimate['nontraining_seconds']/60:.1f} min; "
                     f"total {estimate['required_total_seconds']/3600:.2f} h "
                     f"({estimate['required_total_with_margin_seconds']/3600:.2f} h with margin)")
        if estimate["mode"] == "equal-time":
            lines.append(f"Minimum target with equal time allocations and margins: {estimate['minimum_target_seconds']/3600:.2f} h")
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
        # All declared validation milestones, then the separate final held-out test.
        forecast = validation["forecast_seconds"]
        validation_points = (len(milestone_updates(args.updates, args.validation_fractions)) if args.updates is not None
                             else len(args.validation_fractions) + 2)
        overhead = (setup + forecast * (validation_points + args.test_anchors / args.validation_anchors) +
                    (validation_points - 1) * policy_seconds(validation["policy"], args.validation_policy_steps) +
                    policy_seconds(test_shape, args.policy_steps) + 60.)
        result = {"update_seconds": statistics.median(rates), "overhead_seconds": overhead,
                  "setup_seconds": setup, "forecast_seconds": forecast,
                  "validation_acting_seconds_per_decision": validation["policy"]["acting_seconds"] / 3,
                  "test_acting_seconds_per_decision": test_shape["acting_seconds"] / 3,
                  "validation_points": validation_points,
                  "validation_policy_cases": args.validation_policy_cases, "test_policy_cases": args.policy_cases,
                  "validation_policy_seconds": policy_seconds(validation["policy"], args.validation_policy_steps),
                  "test_policy_seconds": policy_seconds(test_shape, args.policy_steps),
                  "scope": "Disposable model; validation-only timing; no test scores retained"}
        del model
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    return result


class BudgetInfeasible(ValueError):
    """An expected scheduling outcome, not a model/training exception."""


def budget_estimate(args, calibration, elapsed):
    if not calibration or any(not math.isfinite(row[key]) or row[key] <= 0
                              for row in calibration.values() for key in ("overhead_seconds", "update_seconds")):
        raise ValueError("Budget estimation requires finite positive timings for every fit")
    overhead = sum(row["overhead_seconds"] for row in calibration.values())
    rate = sum(row["update_seconds"] for row in calibration.values())
    available = args.minutes * 60 - elapsed - 1.2 * overhead - 60
    permitted = max(0, min(args.max_updates, math.floor(available / (1.15 * rate))))
    required = args.updates if args.updates is not None else args.min_updates
    fit_estimates = {}
    for key, row in calibration.items():
        allocation = args.minutes * 60 / len(calibration)
        preparation = elapsed / len(calibration)
        remaining = allocation - preparation - 1.2 * row["overhead_seconds"] - 60 / len(calibration)
        fit_permitted = (max(0, min(args.max_updates, math.floor(remaining / (1.15 * row["update_seconds"]))))
                         if args.budget_mode == "equal-time" else permitted)
        fit_estimates[key] = {"allocation_seconds": allocation if args.budget_mode == "equal-time" else None,
                              "preparation_seconds": preparation, "permitted_updates": fit_permitted,
                              "required_with_margin_seconds": preparation + 1.2 * row["overhead_seconds"] +
                              1.15 * required * row["update_seconds"] + 60 / len(calibration)}
    if args.budget_mode == "equal-time":
        permitted = min(row["permitted_updates"] for row in fit_estimates.values())
    minimum_target = (max(row["required_with_margin_seconds"] for row in fit_estimates.values()) * len(calibration)
                      if args.budget_mode == "equal-time" else elapsed + 1.2 * overhead + 1.15 * required * rate + 60)
    return {"mode": args.budget_mode, "target_seconds": args.minutes * 60, "preparation_seconds": elapsed,
            "nontraining_seconds": overhead, "seconds_per_common_update": rate,
            "permitted_updates": permitted, "required_updates": required,
            "required_training_seconds": required * rate,
            "required_total_seconds": elapsed + overhead + required * rate,
            "required_total_with_margin_seconds": elapsed + 1.2 * overhead + 1.15 * required * rate + 60,
            "minimum_target_seconds": minimum_target, "fit_estimates": fit_estimates,
            "fits_target": required <= permitted, "explicit_updates_override": args.updates is not None,
            "scope": "Timing estimate, not a convergence threshold or a hard deadline"}


def choose_budget(args, calibration, elapsed):
    estimate = budget_estimate(args, calibration, elapsed)
    count = args.updates if args.updates is not None else estimate["permitted_updates"]
    if args.updates is None and not estimate["fits_target"]:
        raise BudgetInfeasible(
            f"The {args.minutes:g}-minute target permits only {count} updates in the limiting fit "
            f"under {args.budget_mode}, below --min-updates {args.min_updates}. "
            f"At that minimum, measured speeds imply {estimate['required_total_seconds']/3600:.2f} hours total "
            f"(target at least {estimate['minimum_target_seconds']/3600:.2f} hours with timing margins and this allocation). "
            "No comparison fits started. Saved banks can be reused with --reuse-banks in a new run. "
            "Use --estimate-from to inspect saved timings without starting another experiment.")
    fits = {}
    for key, entry in estimate["fit_estimates"].items():
        updates = args.updates if args.updates is not None else entry["permitted_updates"]
        fits[key] = {"updates": updates, "milestones": milestone_updates(updates, args.validation_fractions),
                     "allocation_seconds": entry["allocation_seconds"], "preparation_seconds": entry["preparation_seconds"],
                     "estimated_training_seconds": updates * calibration[key]["update_seconds"],
                     "estimated_nontraining_seconds": calibration[key]["overhead_seconds"],
                     "adjacent_targets": updates * args.batch_size * 3,
                     "expert_target_presentations": updates * (args.batch_size // 2) * 3,
                     "intervention_target_presentations": updates * (args.batch_size // 2) * 3}
    return {"mode": args.budget_mode, "common_updates": count if args.budget_mode == "equal-updates" else None,
            "fit_budgets": fits, "fits": len(fits), "fixed_before_training": True, "hard_deadline": False,
            "estimated_remaining_seconds": sum(row["estimated_training_seconds"] + row["estimated_nontraining_seconds"]
                                               for row in fits.values())}


def print_saved_estimate(args, argv):
    """Inspect historical measurements without initializing models or a CUDA device."""
    previous = json.loads((args.estimate_from / "report.json").read_text())
    if previous.get("format") != FORMAT:
        raise ValueError("Expected a duration-run report")
    saved = previous["settings"]
    raw = sys.argv[1:] if argv is None else argv
    supplied = {token.split("=")[0] for token in raw if token.startswith("--")}
    for key in ("minutes", "updates", "min_updates", "max_updates", "tasks", "models", "budget_mode"):
        if "--" + key.replace("_", "-") not in supplied:
            setattr(args, key, saved.get(key, "equal-updates" if key == "budget_mode" else getattr(args, key)))
    expected = {f"{task}/{model}" for task in args.tasks for model in args.models}
    if not expected.issubset(previous.get("calibration", {})):
        raise ValueError("Saved calibration is incomplete for the requested task/model subset")
    calibration = {key: previous["calibration"][key] for key in sorted(expected)}
    if args.min_updates > args.max_updates:
        raise ValueError("Minimum updates exceeds maximum after applying saved settings")
    result = {"source_run": previous["run_name"], "read_only": True,
              "selected_fits": sorted(expected),
              "saved_evaluation_settings": {key: saved.get(key, [.5] if key == "validation_fractions" else None) for key in
                                            ("validation_fractions", "validation_policy_cases", "validation_policy_steps",
                                             "policy_cases", "policy_steps")},
              "future_work_excluding_preparation": budget_estimate(args, calibration, 0),
              "caveat": "Historical timings and SAVED evaluation settings only, not the expanded current defaults. A new run recalibrates its actual evaluations and repeats preparation. This does not select its update budgets."}
    preparation = previous.get("preparation_seconds")
    if preparation is None and not previous.get("runs"):
        preparation = previous.get("seconds")
    if preparation is not None:
        # The old report may not break preparation down by task; retaining ALL of
        # it is conservative when inspecting a subset, never claim it was timed separately.
        result["including_recorded_preparation"] = budget_estimate(args, calibration, preparation)
    print(json.dumps(result, indent=2))
    return 0


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
    fit_budget = report["budget"]["fit_budgets"][key]
    count, milestones = fit_budget["updates"], fit_budget["milestones"]
    config.training.expert.updates = count
    with load_model_family(config.model_family).build_replay(config) as dataset:
        identity = dataset_identity(dataset.metadata)
        if identity != report["datasets"][str(config.scenario.name)]["identity"]:
            raise ValueError("Expert dataset identity changed")
        model = new_model(config, dataset)
        branch = branch_replay(bank, args)
        row = {"key": key, "task": str(config.scenario.name), "model": str(config.model_family),
               "seed": args.seed, "status": "RUNNING", "updates": 0, "training_seconds": 0.,
               "budget": copy.deepcopy(fit_budget),
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
        if row["updates"] in milestones and row["updates"] not in measured:
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
                if step % args.save_every == 0 or step in milestones:
                    save()  # Safe recovery point even if the following measurement is interrupted.
                if step in milestones:
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
    if args.estimate_from:
        return print_saved_estimate(args, argv)
    report = None
    if args.resume:
        output = args.resume.absolute()
        report = json.loads((output / "report.json").read_text())
        if report.get("format") != FORMAT or "budget" not in report:
            raise ValueError("Resume requires this runner's prepared run with saved update counts; old evaluation snapshots are unsupported")
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
        fixed = fixed_budget(args) if args.budget_mode == "fixed" else None
        print(json.dumps({"target_minutes": args.minutes, "offline_only": True, "seed": args.seed,
                          "budget_mode": args.budget_mode,
                          "timing_calibration": fixed is None, "hard_deadline": False,
                          "fixed_updates": {key: entry["updates"] for key, entry in fixed["fit_budgets"].items()} if fixed else None,
                          "minutes_per_fit_including_preparation": args.minutes / len(configs) if args.budget_mode == "equal-time" else None,
                          "validation_fractions": args.validation_fractions,
                          "validation_policy_cases": args.validation_policy_cases, "test_policy_cases": args.policy_cases,
                          "resume": bool(args.resume), "fits": {key: {
                              "horizon": int(config.jepa_model.planner.horizon), "objective": str(config.jepa_model.planner.objective),
                              "samples": int(config.jepa_model.planner.samples), "iterations": int(config.jepa_model.planner.iterations),
                              "data": str(config.training.expert.data_path)} for key, config in configs.items()}}, indent=2))
        return 0
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run training on the GPU host (or use --estimate-from for saved timings)")
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
                if args.reuse_banks:
                    print(f"Reuse bank | {task} | {args.reuse_banks}", flush=True)
                    bank = reuse_bank(args.reuse_banks, config, args, report["datasets"][task]["identity"], report["versions"])
                    report["bank_source"] = {"directory": str(args.reuse_banks.absolute()),
                                             "report_sha256": file_hash(args.reuse_banks / "report.json"),
                                             "scope": "Simulator banks only; fresh weights, reference checks and controls"}
                else:
                    spec = bank_spec(config, args)
                    bank = collect_bank(config, **{k: v for k, v in spec.items() if k != "task"},
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
            if args.budget_mode == "fixed":
                report["preparation_seconds"] = time.monotonic() - started
                report["budget"] = fixed_budget(args)
                report["calibration_status"] = "SKIPPED_FIXED_UPDATES"
            else:
                for key, config in configs.items():
                    print(f"Calibrate | {key}", flush=True)
                    report["calibration"][key] = calibrate(config, args, banks[str(config.scenario.name)])
                    persist(args.output, report)
                report["preparation_seconds"] = time.monotonic() - started
                report["budget_estimate"] = budget_estimate(args, report["calibration"], report["preparation_seconds"])
                # Keep the breakdown even when a legacy timing target is infeasible.
                persist(args.output, report)
                report["budget"] = choose_budget(args, report["calibration"], report["preparation_seconds"])
            persist(args.output, report)
        else:
            for task, entry in report["banks"].items():
                path = args.output / entry["file"]
                if file_hash(path) != entry["file_sha256"]:
                    raise ValueError("Saved bank file changed")
                banks[task] = torch.load(path, map_location="cpu", weights_only=False)
                validate_bank(banks[task])
        label = "rough runtime; no time cutoff" if args.budget_mode == "fixed" else "runtime target"
        print(f"Budget | {len(configs)} fits | {args.budget_mode} | {args.minutes:g} minutes total ({label})", flush=True)
        for key, entry in report["budget"]["fit_budgets"].items():
            allocation = entry["allocation_seconds"]
            label = f" | {allocation/60:g} minutes including preparation share" if allocation is not None else ""
            print(f"Budget | {key} | {entry['updates']} updates{label}", flush=True)
        for key, config in configs.items():
            tick = time.monotonic()
            try:
                fit(config, args, banks[str(config.scenario.name)], report, key)
            finally:
                timings = report.setdefault("fit_seconds", {})
                timings[key] = timings.get(key, 0.) + time.monotonic() - tick
                persist(args.output, report)
        report["status"] = "COMPLETE"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "Interrupted; resume restores the last atomic checkpoint without extending its schedule"
    except BudgetInfeasible as error:
        report["status"] = "BUDGET_INFEASIBLE"
        report["error"] = str(error)
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
