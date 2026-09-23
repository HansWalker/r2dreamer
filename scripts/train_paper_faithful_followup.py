"""Matched offline coverage comparison with broader starts and learning curves.

Three native models, two training seeds, longer forecast diagnostics, and a
calibrated total runtime. Policy horizon and online training settings stay fixed.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.paper_faithful_followup_eval import evaluate_snapshot, summarize_evaluation
from scripts.paper_faithful_followup_support import (
    followup_manifest_metadata, make_followup_manifest, policy_cases, simulator_controls,
)
from scripts.paper_faithful_reference import run_reference_checks
from scripts.paper_faithful_support import collect_branch_bank
from scripts.train_paper_faithful_check import (
    build_config, choose_updates, common_initial_weights, new_model, policy_trial,
    replay_for, save_checkpoint, synchronize, update,
)
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256


ARMS = {
    "ts_patch_01_coverage": "ts_patch_01",
    "ts_agg_01_coverage": "ts_agg_01_coverage",
    "lewm_coverage": "lewm_coverage",
}
SOURCE_FILES = (
    "train_paper_faithful_followup.py", "paper_faithful_followup_eval.py",
    "paper_faithful_followup_support.py", "train_paper_faithful_check.py",
    "paper_faithful_support.py", "paper_faithful_reference.py",
    "diagnose_goal_objective.py", "diagnose_planner_oracle.py", "diagnose_fresh_readout.py",
    "smoke_tiny_planners.py", "upstream_ts_probe.py",
)


def fit_arguments(args, seed):
    result = copy.copy(args)
    result.seed, result.horizon = seed, args.planner_horizon
    return result


def build_followup_config(arm, args, seed):
    return build_config(ARMS[arm], fit_arguments(args, seed))


def milestone_updates(count, fractions):
    """The schedule is fixed before fitting, never selected by validation results."""
    return sorted({0, count, *(round(count * fraction) for fraction in fractions
                              if 0 < round(count * fraction) < count)})


def estimated_fit_seconds(setup, validation_score, validation_cases, test_cases,
                          validation_trial, test_trial, fractions, validation_steps, test_steps):
    # Setup is paid once per trial; only the measured acting loop scales with steps.
    validation_policy = (validation_trial["setup_seconds"] +
                         validation_trial["acting_seconds"] / validation_trial["steps"] * validation_steps)
    test_policy = (test_trial["setup_seconds"] +
                   test_trial["acting_seconds"] / test_trial["steps"] * test_steps)
    return (setup + validation_score * (len(fractions) + 2 + test_cases / validation_cases) +
            validation_policy * (len(fractions) + 1) + test_policy + 15.)


def runtime_versions(device):
    result = {"torch": torch.__version__}
    for name in ("dm-control", "mujoco", "numpy", "scipy", "hydra-core"):
        result[name] = importlib.metadata.version(name)
    import platform
    result["python"] = platform.python_version()
    if torch.device(device).type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def summary(report):
    def number(value):
        return "n/a" if value is None else f"{value:.4g}"

    lines = ["Offline coverage followup | native losses unchanged | online updates=0",
             f"Status: {report['status']} | Reference: {report.get('reference', {}).get('status', 'NOT_RUN')}",
             f"Forecast horizons: {report['settings']['forecast_horizons']} | policy horizon: {report['settings']['planner_horizon']}",
             "Arm | seed | updates | final test return | maintenance | status"]
    for row in report["runs"]:
        policy = row.get("test", {}).get("summary", {}).get("policy") or {}
        lines.append(f"{row['arm']} | {row['seed']} | {row['updates']} | {number(policy.get('return_mean'))} | "
                     f"{number(policy.get('maintenance_rate'))} | {row['status']}")
    lines.append("Validation curves use the same starts/duration at each trained milestone:")
    for row in report["runs"]:
        for snapshot in row.get("validation", []):
            policy = snapshot["summary"].get("policy") or {}
            branches = snapshot["result"]["branches"]
            metrics = branches["aggregate"]["all"]
            errors = ", ".join(f"H{h}={number(metrics[str(h)].get('matched_over_persistence'))}"
                               for h in branches["horizons"])
            lines.append(f"{row['arm']}/seed{row['seed']} @{snapshot['updates']}: prediction/persistence {errors}; "
                         f"policy return={number(policy.get('return_mean'))}")
    for split, controls in report.get("controls", {}).items():
        lines.append(f"{split} controls: " + "; ".join(
            f"{name} return={number(value.get('return_mean'))}, maintenance={number(value.get('maintenance_rate'))}"
            for name, value in controls.items() if isinstance(value, dict) and "return_mean" in value))
    lines += ["Initial snapshot measures forecasts only; trained validation policy curves are paired.",
              "Final test uses separate starts and a longer policy trial; do not compare its raw return to validation.",
              "Horizon 25 is a forecasting/goal-cost diagnostic; policy planning remains at the recorded horizon.",
              "All fits share one predeclared bank and update count. Test is evaluated only after fitting each model.",
              "COMPLETE means execution, not successful control, broad recovery, or paper-task reproduction."]
    if "budget" in report:
        lines.append(f"Common update budget: {report['budget']['common_updates']} | estimated target {report['settings']['minutes']} minutes")
    if "error" in report:
        lines.append(report["error"])
    if "seconds" in report:
        lines.append(f"Elapsed: {duration(report['seconds'])}")
    return "\n".join(lines) + "\n"


def persist(output, report):
    path = output / "report.json.tmp"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    path.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def save_evaluation(folder, name, result):
    policy = result.get("policy")
    if policy is not None:
        traces = policy.pop("traces")
        path = folder / f"{name}_policy.jsonl"
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in traces))
        policy["traces_file"] = path.name
    return {"result": result, "summary": summarize_evaluation(result)}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dmc_expert_vision"))
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("paper_faithful_followup_%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("full", "small", "tiny"), default="full")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--data-seed", type=int, default=30_000_000)
    parser.add_argument("--minutes", type=float, default=60.)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--min-updates", type=int, default=256)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--calibration-updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--train-anchors", type=int, default=32)
    parser.add_argument("--validation-anchors", type=int, default=9)
    parser.add_argument("--test-anchors", type=int, default=9)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--forecast-horizons", nargs="+", type=int, default=[1, 5, 15, 25])
    parser.add_argument("--planner-horizon", type=int, default=5)
    parser.add_argument("--intermediate-fractions", nargs="+", type=float, default=[.5])
    parser.add_argument("--validation-policy-cases", type=int, default=3)
    parser.add_argument("--validation-policy-steps", type=int, default=25)
    parser.add_argument("--policy-cases", type=int, default=6)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--reference-cache", type=Path, default=Path("runs/reference_cache"))
    parser.add_argument("--no-reference-download", action="store_true")
    parser.add_argument("--skip-reference", action="store_true", help="Smoke-only bypass, explicitly recorded as NOT_RUN")
    args = parser.parse_args(argv)
    positive = (args.minutes, args.min_updates, args.max_updates, args.calibration_updates,
                args.batch_size, args.sources, args.candidates, args.validation_policy_cases,
                args.policy_cases, args.policy_steps, args.validation_policy_steps, args.planner_horizon)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("Use finite positive runtime, batch, and evaluation budgets")
    if args.data_seed < 0 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use unique nonnegative training seeds and a nonnegative data seed")
    if len(set(args.arms)) != len(args.arms) or args.min_updates > args.max_updates:
        parser.error("Use unique arms and min-updates <= max-updates")
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    if args.sources % 2 or args.batch_size % args.sources:
        parser.error("Use an even source count dividing the batch size")
    if min(args.train_anchors, args.validation_anchors, args.test_anchors) < 6 or args.train_anchors < args.sources:
        parser.error("Use at least six anchors per split and enough training anchors for the source count")
    if args.validation_policy_cases > args.validation_anchors or args.policy_cases > args.test_anchors:
        parser.error("Policy case counts cannot exceed their corresponding anchor split")
    if args.policy_cases > args.validation_anchors:
        parser.error("Validation anchors must also fit the final policy batch for disposable calibration")
    if (args.candidates < 6 or min(args.forecast_horizons) < 1 or max(args.forecast_horizons) < 3 or
            max(*args.forecast_horizons, args.planner_horizon, args.policy_steps, args.validation_policy_steps) > 497):
        parser.error("Use >=6 candidates, forecast horizons including >=3, and horizons/policy lengths <=497")
    if any(not 0 < value < 1 for value in args.intermediate_fractions) or len(set(args.intermediate_fractions)) != len(args.intermediate_fractions):
        parser.error("Intermediate fractions must be unique and strictly between zero and one")
    args.forecast_horizons = sorted(set(args.forecast_horizons))
    args.intermediate_fractions.sort()
    return args


def calibrate(arm, config, args, bank):
    started = time.monotonic()
    with load_model_family(str(config.model_family)).build_replay(config) as dataset:
        model = new_model(config, dataset)
        synchronize(args.device)
        setup = time.monotonic() - started
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(args.max_updates)
        branch = replay_for(bank, fit_arguments(args, args.seeds[0]), .5)
        times = []
        for step in range(args.calibration_updates):
            synchronize(args.device)
            started = time.monotonic()
            update(model, dataset, branch, .5, step, args.seeds[0])
            synchronize(args.device)
            times.append(time.monotonic() - started)
        started = time.monotonic()
        evaluate_snapshot(config, model, bank["splits"]["validation"], horizons=args.forecast_horizons,
                          policy_cases=0, policy_steps=0)
        synchronize(args.device)
        validation_score = time.monotonic() - started
        validation_trial = policy_trial(config, model, policy_cases(bank["splits"]["validation"], args.validation_policy_cases),
                                        min(3, args.validation_policy_steps), args.data_seed + 15000)
        test_trial = policy_trial(config, model, policy_cases(bank["splits"]["validation"], args.policy_cases),
                                  min(3, args.policy_steps), args.data_seed + 15000)
        rate = statistics.mean(times[min(2, len(times) - 1):])
        evaluation = estimated_fit_seconds(setup, validation_score, args.validation_anchors, args.test_anchors,
                                           validation_trial, test_trial, args.intermediate_fractions,
                                           args.validation_policy_steps, args.policy_steps)
        result = {"update_seconds": rate, "estimated_nontraining_seconds_per_fit": evaluation,
                  "setup_seconds": setup, "full_validation_forecast_seconds": validation_score,
                  "validation_policy_decision_seconds": validation_trial["acting_seconds"] / validation_trial["steps"],
                  "test_policy_decision_seconds": test_trial["acting_seconds"] / test_trial["steps"],
                  "disposable_updates": len(times), "calibration_seed": args.seeds[0],
                  "scope": "Disposable architecture profiling; speed reused across training seeds; no test outcomes inspected"}
        del model
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    print(f"Profile | {arm}: {rate:.3f}s/update; nontraining ~{evaluation:.0f}s/seed", flush=True)
    return result


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(1)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this launcher on the GPU host with the expert dataset.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "RUNNING", "format": "paper_faithful_followup_v1", "runs": [],
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "online_updates": 0, "online_schedule_changed": False, "coverage_fraction": .5,
              "implementation_sha256": implementation_sha256(), "versions": runtime_versions(args.device),
              "experiment_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in SOURCE_FILES}}
    persist(args.output, report)
    try:
        configs = {arm: build_followup_config(arm, args, args.seeds[0]) for arm in args.arms}
        first_config = next(iter(configs.values()))
        # Validate the read-only expert source before spending time on simulator collection.
        with load_model_family(str(first_config.model_family)).build_replay(first_config) as dataset:
            report["dataset_identity"] = dataset_identity(dataset.metadata)
        if args.skip_reference:
            report["reference"] = {"status": "NOT_RUN", "reason": "Explicit smoke-only --skip-reference"}
        else:
            print("Reference | verifying pinned upstream components", flush=True)
            report["reference"] = run_reference_checks(configs, fit_arguments(args, args.seeds[0]),
                                                       cache=args.reference_cache, allow_download=not args.no_reference_download)
            if report["reference"]["status"] != "PASS":
                raise RuntimeError("Upstream component parity did not pass; inspect the reference report")
        persist(args.output, report)
        manifest = make_followup_manifest(args.train_anchors, args.validation_anchors, args.test_anchors, args.data_seed)
        bank = collect_branch_bank(first_config, manifest, candidates=args.candidates, horizon=max(args.forecast_horizons),
                                   progress=lambda done, total: print(f"Branches | {done}/{total}", flush=True))
        torch.save(bank, args.output / "branches.pt")
        report["branches"] = {"file": "branches.pt", "sha256": bank["sha256"], "metadata": bank["metadata"],
                              "followup_sampling": followup_manifest_metadata(manifest)}
        report["controls"] = {}
        for split, count, steps in (("validation", args.validation_policy_cases, args.validation_policy_steps),
                                    ("test", args.policy_cases, args.policy_steps)):
            print(f"Controls | {split}: zero action and privileged state feedback", flush=True)
            controls = simulator_controls(first_config, policy_cases(bank["splits"][split], count), steps)
            path = args.output / f"{split}_controls.json"
            path.write_text(json.dumps(controls, indent=2, allow_nan=False) + "\n")
            report["controls"][split] = {name: {key: value for key, value in result.items() if key != "traces"}
                                         for name, result in controls.items()}
        persist(args.output, report)
        report["calibration"] = {}
        for arm, config in configs.items():
            report["calibration"][arm] = calibrate(arm, config, args, bank)
            persist(args.output, report)
        rates = [report["calibration"][arm]["update_seconds"] for _ in args.seeds for arm in args.arms]
        overhead = sum(row["estimated_nontraining_seconds_per_fit"] for row in report["calibration"].values()) * len(args.seeds)
        remaining = args.minutes * 60 - (time.monotonic() - started)
        count = args.updates or choose_updates(remaining, rates, overhead, args.min_updates, args.max_updates)
        milestones = milestone_updates(count, args.intermediate_fractions)
        report["budget"] = {"common_updates": count, "fits": len(rates), "milestones": milestones,
                            "adjacent_targets_per_fit": count * args.batch_size * 3,
                            "estimated_remaining_seconds": count * sum(rates) + overhead,
                            "fixed_before_training": True, "hard_deadline": False}
        print(f"Budget | {len(rates)} fits x {count} updates; milestones {milestones}; "
              f"estimated remaining {duration(report['budget']['estimated_remaining_seconds'])}", flush=True)
        persist(args.output, report)
        common_hashes = {}
        for seed in args.seeds:
            for arm in args.arms:
                config = build_followup_config(arm, args, seed)
                config.training.expert.updates = count
                folder = args.output / f"seed_{seed}" / arm
                folder.mkdir(parents=True)
                with load_model_family(str(config.model_family)).build_replay(config) as dataset:
                    identity = dataset_identity(dataset.metadata)
                    if identity != report["dataset_identity"]:
                        raise RuntimeError("Expert dataset identity changed between calibration and fitting")
                    model = new_model(config, dataset)
                    initial = tensor_digest(common_initial_weights(model))
                    key = (seed, str(config.model_family))
                    if key in common_hashes and common_hashes[key] != initial:
                        raise RuntimeError("Paired models differ in common initial weights")
                    common_hashes[key] = initial
                    branch = replay_for(bank, fit_arguments(args, seed), .5)
                    if hasattr(model, "configure_pretraining"):
                        model.configure_pretraining(count)
                    row = {"arm": arm, "seed": seed, "family": str(config.model_family), "status": "RUNNING", "updates": 0,
                           "config": OmegaConf.to_container(config, resolve=True), "validation": [], "checkpoints": [],
                           "common_initial_sha256": initial, "coverage_fraction": .5,
                           "native_parameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("state_head."))}
                    report["runs"].append(row)

                    def measure(step):
                        print(f"Validation | {arm}/seed{seed} update {step}/{count}", flush=True)
                        result = evaluate_snapshot(config, model, bank["splits"]["validation"], horizons=args.forecast_horizons,
                                                   policy_cases=args.validation_policy_cases if step else 0,
                                                   policy_steps=args.validation_policy_steps if step else 0,
                                                   policy_seed=args.data_seed + 15000)
                        row["validation"].append({"updates": step, **save_evaluation(folder, f"validation_{step}", result)})
                        if step:
                            path = folder / ("native.pt" if step == count else f"update_{step}.pt")
                            digest = save_checkpoint(path, model, config, step, identity)
                            row["checkpoints"].append({"updates": step, "file": str(path.relative_to(args.output)), "sha256": digest})
                        persist(args.output, report)

                    measure(0)
                    progress = Progress(f"{arm}/seed{seed}", count)
                    train_seconds = 0.
                    with (folder / "metrics.jsonl").open("w", buffering=1) as log:
                        for step in range(1, count + 1):
                            synchronize(args.device)
                            tick = time.monotonic()
                            values = update(model, dataset, branch, .5, step, seed)
                            synchronize(args.device)
                            train_seconds += time.monotonic() - tick
                            row["updates"] = step
                            log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
                            progress.update(step, f"prediction={values['prediction_loss']:.4g}", force=step == count)
                            if step in milestones:
                                measure(step)
                    row["training_seconds"] = train_seconds
                    print(f"Final test | {arm}/seed{seed}", flush=True)
                    result = evaluate_snapshot(config, model, bank["splits"]["test"], horizons=args.forecast_horizons,
                                               policy_cases=args.policy_cases, policy_steps=args.policy_steps,
                                               policy_seed=args.data_seed + 16000)
                    row["test"] = save_evaluation(folder, "test", result)
                    row["status"] = "COMPLETE"
                    persist(args.output, report)
                    del model
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
        report["status"] = "COMPLETE"
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["seconds"] = time.monotonic() - started
        persist(args.output, report)
        print(summary(report), end="", flush=True)
        print(f"Run | {args.output.name} | status={report['status']}", flush=True)
        print(f"Reports | {args.output.resolve()}", flush=True)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
