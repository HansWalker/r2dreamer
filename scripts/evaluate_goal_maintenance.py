"""Frozen TS/LeWM comparison of reaching and remaining near an image goal.

Uses completed duration-run checkpoints and validation snapshots. The existing
latent tail objective is an explicit planning adaptation, not a training change.
"""

import argparse
import copy
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import tools
from envs.dmc import make_env
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.paper_faithful_duration_support import evaluate, policy_trial, validate_bank
from scripts.paper_faithful_support import selection
from scripts.train_paper_faithful_duration import (
    FORMAT as SOURCE_FORMAT, MODELS, TASKS, file_hash, runtime_versions, save_evaluation, source_hashes,
)
from training import load_model_family
from training.protocol import implementation_sha256


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("goal_maintenance_%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=["cartpole_balance_sparse"])
    parser.add_argument("--models", choices=tuple(MODELS), nargs="+", default=list(MODELS))
    parser.add_argument("--lookahead-seconds", type=float, default=.5,
                        help="Minimum lookahead; never shorten the saved native horizon")
    parser.add_argument("--hold-seconds", type=float, default=.3,
                        help="Average goal distance over at least this much of the final predicted future")
    parser.add_argument("--policy-cases", type=int, default=6)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--policy-seed", type=int, default=61_000_000)
    parser.add_argument("--score-only", action="store_true", help="Score saved branches; skip acting and zero-action trials")
    parser.add_argument("--dry-run", action="store_true", help="Validate artifacts and print resolved conditions without loading models or writing outputs")
    args = parser.parse_args(argv)
    if (any(not math.isfinite(value) or value <= 0 for value in (args.lookahead_seconds, args.hold_seconds))
            or args.hold_seconds > args.lookahead_seconds
            or min(args.policy_cases, args.policy_steps) < 1 or args.policy_seed < 0):
        parser.error("Use positive finite windows with hold <= lookahead, positive trial sizes and a nonnegative seed")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.models)) != len(args.models):
        parser.error("Tasks and models must be unique")
    return args


def conditions(planner, decision_seconds, lookahead_seconds, hold_seconds):
    """Resolve one task-independent rule using its actual simulator time step."""
    if (not all(math.isfinite(value) and value > 0 for value in
                (decision_seconds, lookahead_seconds, hold_seconds)) or hold_seconds > lookahead_seconds):
        raise ValueError("Require positive finite time windows with hold <= lookahead")
    # Tolerance avoids rounding 0.3 / 0.02 to 16 due only to floating-point error.
    steps = lambda seconds: max(1, math.ceil(seconds / decision_seconds - 1e-9))
    native = int(planner.horizon)
    horizon = max(native, steps(lookahead_seconds))
    hold = steps(hold_seconds)
    if native < 1 or hold > horizon:
        raise ValueError("Invalid native horizon or holding window")
    rows = [{"name": "native_saved", "horizon": native, "objective": str(planner.objective),
             "tail_steps": int(planner.get("tail_steps", 3))}]
    if horizon != native:
        rows.append({**rows[0], "name": "native_long", "horizon": horizon})
    rows.append({"name": "goal_maintenance", "horizon": horizon, "objective": "tail", "tail_steps": hold})
    for row in rows:
        row.update(decision_seconds=decision_seconds, lookahead_seconds=row["horizon"] * decision_seconds,
                   hold_seconds=hold * decision_seconds if row["objective"] == "tail" else None,
                   branch_horizon=horizon, branch_hold_steps=hold)
    return rows


def source_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing source file or path outside source run: {relative}")
    return path


def validate_payload(payload, row, identity, bank_hash):
    if (payload.get("format") != SOURCE_FORMAT or payload.get("resume_supported") is not True
            or payload.get("training_config") != row["config"]
            or payload.get("updates") != row["updates"]
            or payload.get("total_updates") != row["updates"]
            or payload.get("dataset_identity") != identity or payload.get("bank_hash") != bank_hash):
        raise ValueError("Final checkpoint configuration, updates, dataset or bank differs from its source report")
    saved_row = payload.get("row", {})
    if any(saved_row.get(key) != row[key] for key in ("key", "task", "model", "seed", "status", "updates")):
        raise ValueError("Checkpoint is not the completed source fit")
    weights = payload.get("model_state_dict", {})
    if not weights or any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Checkpoint weights must be nonempty and finite")


def load_source(args, *, compatible_implementations=()):
    """Read artifacts, validate provenance and resolve time windows for all tasks."""
    root = args.source_run.resolve()
    report = json.loads(source_file(root, "report.json").read_text())
    if report.get("format") != SOURCE_FORMAT or report.get("status") != "COMPLETE":
        raise ValueError("Require a completed paper_faithful_duration_v1 source run")
    if (report.get("implementation_sha256") not in (implementation_sha256(), *compatible_implementations)
            or report.get("source_hashes") != source_hashes()):
        raise ValueError("Source model/environment or duration helpers differ from the current implementation")
    banks, records = {}, []
    for task in args.tasks:
        if task not in report["banks"]:
            raise ValueError(f"Source has no bank/checkpoints for {task}; no training is performed by this evaluator")
        info = report["banks"][task]
        path = source_file(root, info["file"])
        if file_hash(path) != info["file_sha256"]:
            raise ValueError("Source bank file hash mismatch")
        bank = torch.load(path, map_location="cpu", weights_only=False)
        validate_bank(bank)
        if bank["metadata"] != info["metadata"] or bank["sha256"] != info["sha256"]:
            raise ValueError("Source bank metadata differs from report")
        if set(bank["splits"]) != {"train", "validation", "test"} or bank["metadata"]["task"] != task:
            raise ValueError("Source bank task/splits differ")
        cases = bank["splits"]["validation"]
        if not cases or args.policy_cases > len(cases):
            raise ValueError("Not enough validation starts for the requested policy cases")
        banks[task] = bank
        for family in args.models:
            matches = [row for row in report["runs"] if row["task"] == task and row["model"] == family]
            if len(matches) != 1 or matches[0]["status"] != "COMPLETE" or matches[0]["updates"] < 1:
                raise ValueError(f"Require exactly one completed fit for {task}/{family}")
            row = matches[0]
            config = OmegaConf.create(row["config"])
            if (config.model_family != family or config.scenario.name != task or config.seed != row["seed"]
                    or config.jepa_model.history_size != 3
                    or config.jepa_model.planner.objective != ("ts_mpc" if family == "temporal_straightening" else "last")
                    or config.jepa_model.planner.get("aggregate_goal_weight", 0.) != 0.
                    or config.env.goal.get("alternatives", [])):
                raise ValueError("Require the saved native single-goal TS/LeWM configuration")
            env = make_env(config.env, int(row["seed"]))
            try:
                dt = float(env._env.control_timestep()) * int(config.env.action_repeat)
            finally:
                env.close()
            matrix = conditions(config.jepa_model.planner, dt, args.lookahead_seconds, args.hold_seconds)
            horizon = max(item["horizon"] for item in matrix)
            if horizon > bank["metadata"]["horizon"]:
                raise ValueError(f"Requested H{horizon} exceeds saved H{bank['metadata']['horizon']} branches for {task}")
            action_dim = math.prod(config.model_io.action.shape)
            for case in cases:
                if (tuple(case["prefix"].shape) != (3, *config.env.size, 3)
                        or tuple(case["past_action"].shape) != (2, action_dim)
                        or case["action"].shape[-1] != action_dim):
                    raise ValueError("Source image/action shapes differ from checkpoint")
                # Failure probes simulate a complete plan from the final policy state.
                if case["snapshot"]["episode_step"] + args.policy_steps + horizon >= int(config.env.time_limit) // int(config.env.action_repeat):
                    raise ValueError("Policy trial and diagnostic horizon exceed the remaining episode")
            checkpoint = source_file(root, f"{row['key']}/latest.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identity = report["datasets"][task]["identity"]
            validate_payload(payload, row, identity, bank["sha256"])
            records.append({"task": task, "model": family, "seed": row["seed"], "updates": row["updates"],
                            "path": checkpoint, "file_sha256": file_hash(checkpoint), "row": row,
                            "dataset_identity": identity, "bank_sha256": bank["sha256"],
                            "model_state_sha256": tensor_digest(payload["model_state_dict"]), "conditions": matrix})
    return report, banks, records


def load_frozen(record, device):
    if file_hash(record["path"]) != record["file_sha256"]:
        raise ValueError("Source checkpoint changed after validation")
    payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    validate_payload(payload, record["row"], record["dataset_identity"], record["bank_sha256"])
    config = OmegaConf.create(record["row"]["config"])
    config.device = config.env.device = str(device)
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    family = load_model_family(config.model_family)
    model = family.build_model(config)
    family.load_checkpoint(model, payload, training=False)
    if tensor_digest(model.state_dict()) != record["model_state_sha256"]:
        raise ValueError("Loaded model tensors differ from checkpoint")
    model.eval().requires_grad_(False)
    return config, model


def branch_maintenance(branches, cases, horizon, hold):
    """Score selection by sustained task success separately from latent distance."""
    rows = []
    for case, measured in zip(cases, branches["cases"], strict=True):
        if case["id"] != measured["id"]:
            raise ValueError("Branch score/case ordering differs")
        occupancy = case["success"][:, horizon - hold:horizon].float().mean(1).numpy()
        maintained = (occupancy >= .9).astype(np.float32)
        metrics = measured["metrics"][str(horizon)]
        row = {"id": case["id"], "occupancy": occupancy.tolist(),
               "informative": bool(np.ptp(occupancy) > 1e-6), "uniform_occupancy": float(occupancy.mean()),
               "best_occupancy": float(occupancy.max()), "best_maintenance": float(maintained.max())}
        for label in ("predicted", "actual"):
            costs = np.asarray(metrics[f"{label}_cost"])
            row[f"{label}_selected_occupancy"] = selection(costs, occupancy)["return_mean"]
            row[f"{label}_selected_maintenance"] = selection(costs, maintained)["return_mean"]
        rows.append(row)
    keys = ("uniform_occupancy", "best_occupancy", "best_maintenance", "predicted_selected_occupancy",
            "actual_selected_occupancy", "predicted_selected_maintenance", "actual_selected_maintenance")
    return {"horizon": horizon, "hold_steps": hold, "cases": rows,
            "scope": "Common forecast/holding window across conditions; finite branch bank, not a control upper bound",
            **{name: {"cases": len(subset), **{key: float(np.mean([row[key] for row in subset])) if subset else None
                                               for key in keys}}
               for name, subset in (("all", rows), ("informative", [row for row in rows if row["informative"]]))}}


def evaluate_condition(config, model, cases, condition, args):
    original, caches = model.planner, (model._cem_mean, model._gradient_actions)
    current = copy.deepcopy(config)
    with open_dict(current.jepa_model.planner):
        for key in ("horizon", "objective", "tail_steps"):
            current.jepa_model.planner[key] = condition[key]
    model.planner = current.jepa_model.planner
    model._cem_mean = model._gradient_actions = None
    before = tensor_digest(model.state_dict())
    try:
        # All conditions score the same future length and candidate sequences.
        # This forecast diagnostic is separate from native_saved's shorter policy horizon.
        result = evaluate(current, model, cases, steps=args.policy_steps, policy_cases=args.policy_cases,
                          seed=args.policy_seed, horizons=[condition["branch_horizon"]], initial=args.score_only)
        after = tensor_digest(model.state_dict())
        if before != after:
            raise RuntimeError("Frozen comparison changed model tensors")
        result.update(planner=OmegaConf.to_container(model.planner, resolve=True),
                      initial_state_sha256=before, final_state_sha256=after,
                      branch_maintenance=branch_maintenance(result["branches"], cases,
                                                           condition["branch_horizon"], condition["branch_hold_steps"]))
        return result
    finally:
        model.planner = original
        model._cem_mean, model._gradient_actions = caches


def summary(report):
    lines = ["Goal maintenance comparison | frozen models | validation only | training/online updates=0",
             f"Run: {report['run_name']} | Status: {report['status']}",
             "Task/model | condition | H/hold | return/max | maintained cases | seconds"]
    for row in report["runs"]:
        condition = row["condition"]
        p = row.get("evaluation", {}).get("summary", {}).get("policy", {})
        rate, count = p.get("maintenance_rate"), p.get("cases", 0)
        kept = "n/a" if rate is None else f"{round(rate * count)}/{count}"
        lines.append(f"{row['task']}/{row['model']} | {condition['name']} | "
                     f"{condition['horizon']}/{condition['tail_steps'] if condition['objective'] == 'tail' else '-'} | "
                     f"{p.get('return_mean')} / {p.get('maximum_return')} | {kept} | {row.get('seconds', 0):.1f} | {row['status']}")
        diagnostic = row.get("evaluation", {}).get("result", {}).get("branch_maintenance")
        if diagnostic:
            value = diagnostic["informative"]
            lines.append(f"  H{diagnostic['horizon']} branch tail occupancy (actual/predicted/uniform/best): "
                         f"{value['actual_selected_occupancy']} / {value['predicted_selected_occupancy']} / "
                         f"{value['uniform_occupancy']} / {value['best_occupancy']} ({value['cases']} informative anchors)")
    for task, control in report["controls"].items():
        lines.append(f"{task}/zero action | {control['return_mean']} / {control['maximum_return']} | "
                     f"maintenance={control['maintenance_rate']}")
    lines += ["Tail scoring is an explicit planning adaptation; native networks and training losses are unchanged.",
              "Forecast comparisons use the same longer horizon even when native_saved acts with its shorter horizon.",
              "Same validation starts, durations and planner search settings per model; no test split evaluated.",
              "Actual-image scoring still uses the learned encoder; finite-window closeness does not guarantee stability.",
              "COMPLETE means execution, not a demonstrated repair. No online-retention claim."]
    if "error" in report:
        lines.append(report["error"])
    return "\n".join(lines) + "\n"


def persist(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def main(argv=None):
    args = arguments(argv)
    # Preserve the user-supplied writable path spelling (important on WSL mounts).
    output = args.output.absolute()
    report = {"format": "goal_maintenance_v1", "run_name": output.name, "status": "PREPARING",
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "training_updates": 0, "online_updates": 0, "online_schedule_changed": False,
              "split": "validation", "runs": [], "controls": {}}
    created, started = False, time.monotonic()
    try:
        torch.set_num_threads(1)
        source, banks, records = load_source(args)
        matrix = [{"task": row["task"], "model": row["model"], "updates": row["updates"],
                   "conditions": row["conditions"]} for row in records]
        if args.dry_run:
            print(json.dumps(matrix, indent=2), flush=True)
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
        report.update(versions=versions, source_report_sha256=file_hash(args.source_run / "report.json"),
                      implementation_sha256=implementation_sha256(), source_helpers=source_hashes(),
                      evaluator_sha256=file_hash(Path(__file__)), source_banks=source["banks"], matrix=matrix,
                      checkpoints=[{key: str(row[key]) if isinstance(row[key], Path) else row[key]
                                    for key in ("task", "model", "seed", "updates", "path", "file_sha256", "model_state_sha256")}
                                   for row in records])
        output.mkdir(parents=True, exist_ok=False)
        created, report["status"] = True, "RUNNING"
        persist(output, report)
        for record in records:
            config, model = load_frozen(record, args.device)
            cases = banks[record["task"]]["splits"]["validation"]
            if record["task"] not in report["controls"] and not args.score_only:
                print(f"Zero action | {record['task']}", flush=True)
                control = policy_trial(config, model, cases[:args.policy_cases], args.policy_steps,
                                       args.policy_seed, zero=True, diagnostics=False)
                (output / f"{record['task']}_zero.json").write_text(json.dumps(control, indent=2, allow_nan=False) + "\n")
                report["controls"][record["task"]] = {key: value for key, value in control.items() if key not in ("traces", "probes")}
            for condition in record["conditions"]:
                row = {"task": record["task"], "model": record["model"], "seed": record["seed"],
                       "condition": condition, "status": "RUNNING"}
                report["runs"].append(row)
                folder = output / record["task"] / record["model"] / condition["name"]
                folder.mkdir(parents=True)
                print(f"Evaluate | {record['task']}/{record['model']} | {condition['name']} | "
                      f"H={condition['horizon']} ({condition['lookahead_seconds']:g}s)", flush=True)
                tick = time.monotonic()
                try:
                    result = evaluate_condition(config, model, cases, condition, args)
                    row.update(evaluation=save_evaluation(folder, "validation", result), status="COMPLETE")
                except BaseException as error:
                    row["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL"
                    raise
                finally:
                    row["seconds"] = time.monotonic() - tick
                    persist(output, report)
            del model
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
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
