"""Locate action-conditioning failures after one offline reference-sized fit per model."""

import argparse
import gc
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity
from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.action_conditioning_support import collect_branches, probe_case
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.planner_recipe_support import BlockReplay, NormalizedActionEncoder, action_statistics, heldout_pairs, reference_configs
from scripts.train_planner_check import build_config, new_model
from scripts.train_rollout_check import pretrain
from scripts.upstream_ts_probe import COMMIT, parity, sources, training_parity
from scripts.predictor_fit_control import fit_control
from training import load_model_family
from training.progress import duration
from training.protocol import implementation_sha256

MODELS = ("temporal_straightening", "leworldmodel")


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summary(report):
    lines = ["Action conditioning | one offline fit/model | no online updates, policy optimization or checkpoints",
             "Model / precision / h | TF matched/shuffled/zero MSE | Recursive matched/shuffled/zero MSE | response ratio | goal range ratio"]
    for run in report["runs"]:
        lines.append(f"{run['model']} | {run['status']} | {duration(run.get('seconds', 0))}")
        if "error" in run:
            lines.append(run["error"])
        if "upstream" in run:
            for stage, item in run["upstream"].items():
                lines.append(f"  TS upstream {stage}: {item['status']} | " + ", ".join(
                    f"{k} max={v['max_abs_error']:.3g}" for k, v in item["checks"].items()))
        fit = run.get("fit_control")
        if fit:
            lines.append(f"  Native fitting control: {fit['status']} (copy only) | weights_changed={fit.get('weights_changed', 'n/a')}")
            for split in ("train", "validation") if "after" in fit else ():
                def average(stage, key):
                    return mean([r[key] for r in fit[stage][split]])
                lines.append(f"  {split}: TF h1 {average('before', 'teacher_h1'):.3g}->{average('after', 'teacher_h1'):.3g}; "
                             f"recursive h5 {average('before', 'recursive_final'):.3g}->{average('after', 'recursive_final'):.3g}")
        cases = run.get("cases", [])
        if not cases:
            continue
        for precision in cases[0]["precision"]:
            for h in range(5):
                rows = [c["precision"][precision]["horizons"][h] for c in cases]
                def losses(mode):
                    return "/".join(f"{mean([r[mode][arm]['mse'] for r in rows]):.3g}" for arm in ("matched", "shuffled", "zero"))
                def number(key):
                    value = mean([r["recursive"][key] for r in rows])
                    return "n/a" if value is None else f"{value:.3g}"
                lines.append(f"{run['model']} / {precision} / {h + 1} | {losses('teacher_forced')} | "
                             f"{losses('recursive')} | {number('response_ratio')} | {number('goal_cost_range_ratio')}")
        geometry = {stage: mean([c["geometry"][stage]["nonexpert_pose_rank"] for c in cases]) for stage in ("encoder", "projector")}
        lines.append("  Real-image goal/pose rank (excluding exact goal candidate): " + json.dumps(geometry))
    lines += ["TF = real preceding states at every step. Recursive = predictions fed back; no future observations.",
              "Goals are heldout trajectory endpoints. This is not a task-success or equilibrium-goal policy evaluation.",
              "Summary averages the per-pair metrics. MSE is raw latent error within each model, NOT a cross-model quality metric.",
              "Ratios/ranks have no universal pass threshold; null ratios (zero denominator) are excluded from summary means.",
              "Shuffled/zero replace only future controls, holding histories and targets fixed. Past observed actions never change.",
              "Response ratio compares predicted versus real-encoded +1/-1 action effects; goal range ratio compares candidate cost ranges.",
              "FP32/BF16 predictor probes share cached native-precision encoder features, isolating predictor numerics.",
              "TS parity includes cached-latent loss/backprop and a matched optimizer step; NOT vision encoder, dropout-mask parity or proprioception.",
              "FP32 probes disable TF32 and use math SDPA; native fitting controls use disjoint episode splits on disposable copies.",
              "COMPLETE means diagnostics ran, not that planning is fixed. Per-layer traces, gradients and exact branch data are in report.json.",
              "Reference-sized vision-only adapters and action blocks match the last diagnostic; production models/objectives are unchanged."]
    return "\n".join(lines) + "\n"


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--expert-updates", type=int, default=3000)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--parity-only", action="store_true", help="Stop after initial forward/training parity; no offline fitting.")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--minimum-pairs", type=int, default=8)
    parser.add_argument("--gradient-pairs", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--goal-tolerance", type=float, nargs=2, default=[.01, .01])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upstream-cache", type=Path, default=Path("local/upstream_ts") / COMMIT)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("action_conditioning_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if (min(args.expert_updates, args.pairs, args.minimum_pairs, args.gradient_pairs, args.stride) < 1 or args.fit_updates < 0
            or not args.gradient_pairs <= args.minimum_pairs <= args.pairs or args.seed < 0 or args.stride * 7 >= 500
            or len(set(args.models)) != len(args.models)
            or any(not np.isfinite(t) or t <= 0 for t in args.goal_tolerance)):
        parser.error("Use positive budgets/tolerances, unique models, gradient-pairs <= minimum-pairs <= pairs, and 7*stride < 500.")
    args.scenario = "cartpole_balance_sparse"
    return args


def run_model(name, args, output, result, bank, source, persist):
    raw, config = reference_configs(build_config(name, args), args.stride)
    result["config"] = OmegaConf.to_container(config, resolve=True)
    result["raw_data_config"] = OmegaConf.to_container(raw, resolve=True)
    size = int(config.model_io.observations.image[0])
    with load_model_family(name).build_replay(raw) as dataset:
        result["dataset_identity"] = dataset_identity(dataset.metadata)
        action_mean, std, count = action_statistics(dataset)
        action_mean, std = np.tile(action_mean, args.stride), np.tile(std, args.stride)
        result["action_statistics"] = {"mean": action_mean.tolist(), "std": std.tolist(), "training_transitions": count}
        if bank is None:
            env = make_env(config.env, args.seed + 13_000_000)
            try:
                cases, attempted = heldout_pairs(dataset, env, stride=args.stride, horizon=5, count=args.pairs,
                                                seed=args.seed + 16_000_000, tolerance=args.goal_tolerance)
                result["pair_restorations_attempted"] = attempted
                if len(cases) < args.minimum_pairs:
                    raise ValueError(f"Only {len(cases)} reproducible heldout pairs; need {args.minimum_pairs}. No fit launched.")
                bank = collect_branches(env, cases, args.stride, args.goal_tolerance, args.seed + 15_000_000)
            finally:
                env.close()
        adapter = BlockReplay(dataset, args.stride, size, torch.device(args.device))
        model = new_model(config, adapter)
        model.action_encoder = NormalizedActionEncoder(model.action_encoder, action_mean, std, model.device)
        initial = tensor_digest(model.state_dict())
        result["initial_probe"], encoded = probe_case(model, bank[0], size, gradients=True)
        if name == "temporal_straightening":
            result["upstream"] = {}
            with readout_mode(model):
                result["upstream"]["initial"] = parity(model, source, encoded["history"], encoded["past"], encoded["actions"][None])
                latent = torch.cat((encoded["history"], encoded["future"][:1, :1]), dim=1)
                actions = torch.cat((encoded["past"], encoded["actions"][:1, :1]), dim=1)
                result["upstream"]["training"] = training_parity(model, source, latent, actions)
            if any(v["status"] != "PASS" for v in result["upstream"].values()):
                raise ValueError("Initial upstream mismatch; stopped before spending the offline training budget.")
        if initial != tensor_digest(model.state_dict()):
            raise RuntimeError("Pre-fit probes mutated weights/buffers.")
        persist()
        if args.parity_only:
            result["status"] = "COMPLETE"
            return bank
        pretrain(config, adapter, args, output, result, model=model)
    frozen = tensor_digest(model.state_dict())
    result["cases"] = []
    for index, case in enumerate(bank):
        probe, encoded = probe_case(model, case, size, gradients=index < args.gradient_pairs)
        result["cases"].append(probe)
        if index == 0 and name == "temporal_straightening":
            with readout_mode(model):
                result["upstream"]["trained"] = parity(model, source, encoded["history"], encoded["past"], encoded["actions"][None])
        print(f"Probe | {name} | heldout pairs={index + 1}/{len(bank)}", flush=True)
        persist()
    result["frozen_state_sha256"] = tensor_digest(model.state_dict())
    if frozen != result["frozen_state_sha256"]:
        raise RuntimeError("Post-fit probes mutated weights/buffers.")
    if args.fit_updates:
        result["fit_control"] = fit_control(model, bank, size, args.fit_updates)
        if tensor_digest(model.state_dict()) != frozen:
            raise RuntimeError("Copy-only fitting control changed original model tensors.")
    result["status"] = ("MISMATCH" if any(x["status"] != "PASS" for x in result.get("upstream", {}).values()) else "COMPLETE")
    return bank


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA diagnostics require BF16 support.")
    # Fail on unavailable reference code before loading data or training either model.
    source = sources(args.upstream_cache) if "temporal_straightening" in args.models else None
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).with_name(name) for name in
             ("diagnose_action_conditioning.py", "action_conditioning_support.py", "upstream_ts_probe.py", "planner_recipe_support.py", "predictor_fit_control.py")]
    report = {"experiment": "action_conditioning", "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "online_updates": 0, "checkpoint_writes": False, "production_settings_changed": False,
              "candidate_names": ["expert", "zero", "negative_one", "positive_one", "negated_expert", "time_reversed_expert",
                                  "random_0", "random_1", "random_2", "random_3"], "runs": []}
    bank = None
    started = time.monotonic()
    print(f"Action conditioning | models={len(args.models)} | offline updates/model={args.expert_updates} | no online training", flush=True)
    for name in args.models:
        folder = args.output / name
        folder.mkdir()
        result = {"model": name, "status": "RUNNING"}
        report["runs"].append(result)
        tick = time.monotonic()
        try:
            bank = run_model(name, args, folder, result, bank, source, lambda: write_report(args.output, report))
        except Exception as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (folder / "error.log").write_text(traceback.format_exc())
        if bank is not None and "branches" not in report:
            report["branches"] = [{k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in case.items()
                                   if k not in {"prefix", "images", "goal_image"}} for case in bank]
        result["seconds"] = time.monotonic() - tick
        write_report(args.output, report)
        print(f"{result['status']} | {name} | {duration(result['seconds'])}", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["seconds"] = time.monotonic() - started
    write_report(args.output, report)
    print(summary(report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(r["status"] != "COMPLETE" for r in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
