"""Frozen TS/LeWM planning comparison: H5/H15 and TS spatial + 0.1 aggregate cost.

Consumes a completed paper_faithful_followup run, including branches.pt and native
checkpoints. Only its validation split is evaluated. No expert dataset, training,
checkpoint writes, test-set selection, or online schedule changes are involved.
"""

import argparse
import copy
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

import tools
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.paper_faithful_followup_eval import evaluate_snapshot, select_policy_cases
from scripts.paper_faithful_followup_support import simulator_controls, validate_followup_manifest
from scripts.paper_faithful_support import _digest
from scripts.train_paper_faithful_followup import runtime_versions, save_evaluation
from training import load_model_family
from training.progress import duration
from training.protocol import implementation_sha256


SOURCE_ARMS = {"ts_agg_01_coverage": "temporal_straightening", "lewm_coverage": "leworldmodel"}
PAPER = "https://arxiv.org/html/2603.12231v3"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditions(horizons=(5, 15)):
    return [
        {"name": f"{name}_h{horizon}", "source_arm": source, "horizon": int(horizon),
         "aggregate_goal_weight": weight}
        for name, source, weight in (
            ("ts_spatial", "ts_agg_01_coverage", 0.),
            ("ts_spatial_aggregate", "ts_agg_01_coverage", .1),
            ("lewm_native", "lewm_coverage", 0.),
        )
        for horizon in horizons
    ]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Completed follow-up directory containing report.json, branches.pt and native.pt files")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("paper_faithful_planning_%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizons", type=int, nargs="+", choices=(5, 15), default=[5, 15])
    parser.add_argument("--policy-cases", type=int, default=12)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--policy-seed", type=int, default=41_015_000)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Validate source artifacts and print the matrix without simulator/model evaluation or output writes")
    args = parser.parse_args(argv)
    if min(args.policy_cases, args.policy_steps, args.encode_batch_size) < 1 or min(*args.seeds, args.policy_seed) < 0:
        parser.error("Use positive case/step/batch counts and nonnegative seeds")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.horizons)) != len(args.horizons):
        parser.error("Seeds and horizons must be unique")
    args.horizons.sort()
    return args


def source_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing source file or path outside source run: {relative}")
    return path


def environment_identity(env):
    # These fields affect execution/data paths, not the recorded simulator task.
    runtime = {"device", "dataset_root", "seed", "eval_seed", "env_num", "eval_episode_num"}
    return {key: value for key, value in env.items() if key not in runtime}


def validate_payload(payload, record):
    if payload.get("format") != "paper_faithful_offline_v1" or payload.get("resume_supported") is not False:
        raise ValueError("Expected an evaluation-only paper_faithful_offline_v1 checkpoint")
    if payload.get("training_config") != record["config"] or payload.get("updates") != record["updates"]:
        raise ValueError("Checkpoint training configuration/update count differs from source report")
    if payload.get("dataset_identity") != record["dataset_identity"]:
        raise ValueError("Checkpoint expert dataset identity differs from source report")
    weights = payload.get("model_state_dict", {})
    if not weights or any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Checkpoint weights must be nonempty and finite")
    if record["source_arm"] == "ts_agg_01_coverage":
        if not any(key.startswith("encoder.agg_mlp.") for key in weights) or not any(key.startswith("encoder.agg_post_norm.") for key in weights):
            raise ValueError("Combined TS cost requires the saved trained aggregation head")


def load_source(source_run, seeds, *, horizons=(5, 15), policy_cases=12, policy_steps=200):
    """Bind checkpoints and bank contents to a completed source report, read-only."""
    root = Path(source_run).resolve()
    report = json.loads(source_file(root, "report.json").read_text())
    if report.get("format") != "paper_faithful_followup_v1" or report.get("status") != "COMPLETE":
        raise ValueError("Source must be a completed paper_faithful_followup_v1 run")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require distinct source seeds")
    branch_info = report["branches"]
    bank = torch.load(source_file(root, branch_info["file"]), map_location="cpu", weights_only=False)
    metadata = bank["metadata"]
    if metadata != branch_info["metadata"] or _digest(metadata) != bank["sha256"] or bank["sha256"] != branch_info["sha256"]:
        raise ValueError("Branch-bank metadata hash or source report mismatch")
    validate_followup_manifest(metadata["manifest"])
    if set(bank["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Require the original disjoint train/validation/test bank")
    if metadata["history_size"] != 3 or max(horizons) > metadata["horizon"]:
        raise ValueError("Source bank needs three-frame prefixes and enough future steps")
    for split, cases in bank["splits"].items():
        if _digest(cases) != metadata["split_hashes"][split]:
            raise ValueError(f"Branch-bank {split} split hash mismatch")
        specs = metadata["manifest"][split]
        if len(cases) != len(specs):
            raise ValueError("Branch count differs from source manifest")
        for case, spec in zip(cases, specs):
            if any(case.get(key) != value for key, value in spec.items()):
                raise ValueError("Branch cases differ from the assigned source manifest")
            if _digest({k: v for k, v in case.items() if k != "sha256"}) != case["sha256"]:
                raise ValueError("Branch case hash mismatch")
            count, horizon = metadata["candidates"], metadata["horizon"]
            if (tuple(case["image"].shape) != (count, horizon, 64, 64, 3)
                    or tuple(case["prefix"].shape) != (3, 64, 64, 3)
                    or tuple(case["action"].shape) != (count, horizon, 1)
                    or tuple(case["rewards"].shape) != (count, horizon)
                    or tuple(case["past_action"].shape) != (2, 1)):
                raise ValueError("Source branch image/action/reward shapes are incompatible")
            if case["past_action"].count_nonzero() or "goal_images" in case:
                raise ValueError("This controlled-policy runner requires zero-action prefixes and a single goal")
    select_policy_cases(bank["splits"]["validation"], policy_cases)
    records = []
    for seed in seeds:
        for arm, family in SOURCE_ARMS.items():
            matches = [row for row in report["runs"] if row["seed"] == seed and row["arm"] == arm]
            if len(matches) != 1 or matches[0]["status"] != "COMPLETE" or matches[0]["updates"] < 1:
                raise ValueError(f"Require exactly one completed trained source for {arm}/seed{seed}")
            row = matches[0]
            config = row["config"]
            if config["seed"] != seed or config["model_family"] != family or row["family"] != family:
                raise ValueError("Source seed/family identity mismatch")
            settings = config["jepa_model"]
            expected_objective = "ts_mpc" if family == "temporal_straightening" else "last"
            if settings["planner"]["objective"] != expected_objective or settings["planner"].get("aggregate_goal_weight", 0.) != 0.:
                raise ValueError("Source must use the original native spatial/terminal objective")
            if arm.startswith("ts_") and settings.get("curvature_mode") != "agg":
                raise ValueError("TS source must have a trained aggregation head")
            if settings["history_size"] != 3 or config["env"]["task"] != "dmc_cartpole_balance_sparse":
                raise ValueError("Require source Cartpole three-frame models")
            if environment_identity(config["env"]) != environment_identity(metadata["environment"]):
                raise ValueError("Source model and branch simulator settings differ")
            if config["env"].get("goal", {}).get("alternatives"):
                raise ValueError("This controlled-policy runner requires a single goal")
            if policy_steps + 2 > config["env"]["time_limit"] // config["env"]["action_repeat"]:
                raise ValueError("Policy duration exceeds the source simulator episode limit")
            checkpoints = [cp for cp in row["checkpoints"] if cp["updates"] == row["updates"]]
            if len(checkpoints) != 1:
                raise ValueError("Require exactly one final checkpoint per source fit")
            checkpoint = checkpoints[0]
            path = source_file(root, checkpoint["file"])
            if file_sha256(path) != checkpoint["sha256"]:
                raise ValueError(f"Checkpoint file hash mismatch: {path}")
            record = {"source_arm": arm, "seed": seed, "path": path, "sha256": checkpoint["sha256"],
                      "config": copy.deepcopy(config), "updates": row["updates"],
                      "dataset_identity": copy.deepcopy(report["dataset_identity"])}
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_payload(payload, record)
            record["model_state_sha256"] = tensor_digest(payload["model_state_dict"])
            records.append(record)
    return report, bank, records


def load_frozen(record, device):
    """Strictly load every saved tensor; restore no optimizer or training resume."""
    if file_sha256(record["path"]) != record["sha256"]:
        raise ValueError("Checkpoint changed after source validation")
    payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    validate_payload(payload, record)
    config = OmegaConf.create(record["config"])
    config.device = config.env.device = str(device)
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    family = load_model_family(str(config.model_family))
    model = family.build_model(config)
    family.load_checkpoint(model, payload, training=False)
    if tensor_digest(model.state_dict()) != record["model_state_sha256"]:
        raise ValueError("Loaded model differs from the saved checkpoint tensors")
    model.eval().requires_grad_(False)
    return config, model


def evaluate_condition(config, model, cases, condition, args):
    """Change only planning horizon/cost, then restore caller settings and caches."""
    original = model.planner
    caches = model._cem_mean, model._gradient_actions
    evaluation_config = copy.deepcopy(config)
    planner = copy.deepcopy(original)
    with open_dict(planner):
        planner.horizon = int(condition["horizon"])
        planner.aggregate_goal_weight = float(condition["aggregate_goal_weight"])
    evaluation_config.jepa_model.planner = planner
    model.planner = evaluation_config.jepa_model.planner
    model._cem_mean = model._gradient_actions = None
    before = tensor_digest(model.state_dict())
    try:
        result = evaluate_snapshot(evaluation_config, model, cases, horizons=args.horizons,
                                   policy_cases=args.policy_cases, policy_steps=args.policy_steps,
                                   policy_seed=args.policy_seed, encode_batch_size=args.encode_batch_size)
        after = tensor_digest(model.state_dict())
        if after != before:
            raise RuntimeError("Frozen planning evaluation changed checkpoint tensors")
        result.update(planner=OmegaConf.to_container(model.planner, resolve=True),
                      initial_state_sha256=before, final_state_sha256=after,
                      scope="Frozen checkpoint; validation split only; explicit planning horizon and optional TS aggregate cost. Native training is unchanged and no updates occur.")
        return result
    finally:
        model.planner = original
        model._cem_mean, model._gradient_actions = caches


def summary(report):
    def number(value):
        return "n/a" if value is None else f"{value:.4g}"

    lines = ["Frozen planning comparison | validation only | native/online updates=0",
             f"Status: {report['status']}",
             "Condition | seed | return/max | maintenance | H(actual/predicted/uniform/best candidate return) | seconds"]
    for row in report["runs"]:
        if row["status"] != "COMPLETE":
            lines.append(f"{row['condition']['name']} | {row['seed']} | {row['status']}")
            continue
        value = row["evaluation"]["summary"]
        policy = value["policy"]
        h = str(row["condition"]["horizon"])
        score = value["horizons"][h]["informative"]
        ranks = "/".join(number(score.get(key)) for key in ("actual_selected_return", "predicted_selected_return", "uniform_return", "best_return"))
        lines.append(f"{row['condition']['name']} | {row['seed']} | {number(policy['return_mean'])}/{policy['maximum_return']} | "
                     f"{number(policy['maintenance_rate'])} | H{h}({ranks}; n={score['anchors']}) | {duration(row['seconds'])}")
    for name, control in report.get("controls", {}).items():
        lines.append(f"Control {name}: return={number(control['return_mean'])}/{control['maximum_return']}, maintenance={number(control['maintenance_rate'])}")
    lines += ["All conditions use the same validation starts, duration and per-call RNG seeds; horizon-dependent random tensor shapes differ.",
              "Forecast scoring uses both H5/H15 (or the requested subset); acting uses the condition's recorded horizon.",
              "TS combined cost uses spatial + 0.1 * aggregate distance with the existing TS temporal weighting for both terms.",
              "Actual-future native cost is not a reward oracle. Candidate best is only the best in the finite bank.",
              "COMPLETE means execution, not successful balance or original-paper task reproduction.",
              "Confirm a selected configuration on fresh held-out starts; the original test bank is not evaluated here."]
    if "error" in report:
        lines.append(report["error"])
    if "seconds" in report:
        lines.append(f"Elapsed: {duration(report['seconds'])}")
    return "\n".join(lines) + "\n"


def persist(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(1)
    source = args.source_run.resolve()
    if args.output.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the read-only source run")
    if args.output.exists() and not args.dry_run:
        raise ValueError("Output already exists; choose a new directory")
    print("Source | verifying saved bank, final checkpoints, configurations and hashes", flush=True)
    source_report, bank, records = load_source(source, args.seeds, horizons=args.horizons,
                                              policy_cases=args.policy_cases, policy_steps=args.policy_steps)
    matrix = conditions(args.horizons)
    selected = select_policy_cases(bank["splits"]["validation"], args.policy_cases)
    if args.dry_run:
        print(json.dumps({"status": "SOURCE_VALIDATED", "source": str(source), "seeds": args.seeds,
                          "conditions": matrix, "evaluations": len(matrix) * len(args.seeds),
                          "validation_forecast_cases": len(bank["splits"]["validation"]),
                          "policy_case_ids": [case["id"] for case in selected], "policy_steps": args.policy_steps,
                          "training_updates": 0, "source_checkpoints": [str(record["path"]) for record in records]}, indent=2))
        return 0
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on Lambda, or use --dry-run to validate sources without a GPU.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "RUNNING", "format": "paper_faithful_planning_v1", "runs": [],
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "training_updates": 0, "online_updates": 0, "online_schedule_changed": False,
              "implementation_sha256": implementation_sha256(), "versions": runtime_versions(args.device),
              "experiment_source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
                  "evaluate_paper_faithful_planning.py", "paper_faithful_followup_eval.py", "paper_faithful_followup_support.py",
                  "paper_faithful_support.py", "train_paper_faithful_check.py", "train_paper_faithful_followup.py",
                  "diagnose_goal_objective.py", "diagnose_planner_oracle.py", "smoke_tiny_planners.py", "diagnose_fresh_readout.py")},
              "source": {"directory": str(source), "report_sha256": file_sha256(source / "report.json"),
                         "implementation_sha256": source_report["implementation_sha256"],
                         "reference": source_report.get("reference"), "branches_sha256": bank["sha256"],
                         "validation_sha256": bank["metadata"]["split_hashes"]["validation"]},
              "protocol": {"split": "validation", "conditions": matrix, "case_ids": [case["id"] for case in selected],
                           "policy_steps": args.policy_steps, "policy_rng_seed": args.policy_seed,
                           "native_parameters_frozen": True, "solver_budgets": "unchanged from each source checkpoint",
                           "paper_variant": PAPER, "aggregate_weight": .1,
                           "aggregate_cost_definition": "Per-state learned aggregation; same mean coordinate reduction and ts_mpc history/time weighting as spatial cost; combine terms before choosing a goal.",
                           "test_evaluated": False, "fresh_confirmation_required": True}}
    persist(args.output, report)
    try:
        control_config = OmegaConf.create(records[0]["config"])
        control_config.device = control_config.env.device = args.device
        print(f"Controls | {len(selected)} validation starts x {args.policy_steps} decisions", flush=True)
        controls = simulator_controls(control_config, selected, args.policy_steps)
        (args.output / "validation_controls.json").write_text(json.dumps(controls, indent=2, allow_nan=False) + "\n")
        report["controls"] = {name: {key: value for key, value in row.items() if key != "traces"} for name, row in controls.items()}
        persist(args.output, report)
        for record in records:
            config, model = load_frozen(record, args.device)
            for condition in matrix:
                if condition["source_arm"] != record["source_arm"]:
                    continue
                folder = args.output / f"seed_{record['seed']}" / condition["name"]
                folder.mkdir(parents=True)
                row = {"status": "RUNNING", "condition": condition, "seed": record["seed"],
                       "source_checkpoint": {key: str(value) if isinstance(value, Path) else value
                                             for key, value in record.items() if key != "config"},
                       "source_planner": copy.deepcopy(record["config"]["jepa_model"]["planner"])}
                report["runs"].append(row)
                persist(args.output, report)
                print(f"Evaluate | {condition['name']}/seed{record['seed']} | {len(selected)} starts x {args.policy_steps} decisions", flush=True)
                tick = time.monotonic()
                result = evaluate_condition(config, model, bank["splits"]["validation"], condition, args)
                if result["policy"]["case_ids"] != report["protocol"]["case_ids"]:
                    raise RuntimeError("Native policy starts differ from paired controls")
                row.update(evaluation=save_evaluation(folder, "validation", result), status="COMPLETE", seconds=time.monotonic() - tick)
                persist(args.output, report)
                policy = row["evaluation"]["summary"]["policy"]
                print(f"Complete | {condition['name']}/seed{record['seed']} | return={policy['return_mean']:.2f}/{policy['maximum_return']} | maintenance={policy['maintenance_rate']:.3f} | {duration(row['seconds'])}", flush=True)
            del model
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        report["status"] = "COMPLETE"
    except Exception as error:
        if report["runs"] and report["runs"][-1]["status"] == "RUNNING":
            report["runs"][-1].update(status="FAILED", error=f"{type(error).__name__}: {error}")
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["seconds"] = time.monotonic() - started
        persist(args.output, report)
    print(summary(report), flush=True)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
