"""Short paired TS training check: legacy flattened versus corrected patchwise curvature."""

import argparse
import copy
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity
from models.shared.physical_state import format_physical_rmse
from scripts.diagnose_fixed_replay import FixedBatches
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.online_validation import TrajectoryDataset, collect_episode, episode_metadata, validation_metadata
from scripts.train_planner_check import build_config, checked_update, diagnostic_settings, measure_errors, new_model
from training import load_model_family
from training.evaluation import StateDataset
from training.progress import Progress, duration
from training.protocol import implementation_sha256

VARIANTS = ("legacy_flattened", "fixed_patchwise")


def legacy_loss(self, obs, latent, action):
    """Diagnostic-only reproduction of the old loss; never restore it in production."""
    if self.decoder is not None:
        raise ValueError("The paired diagnostic requires the optional decoder to be disabled.")
    prediction = self.predict(latent[:, :-1], action)
    prediction_loss = F.mse_loss(prediction, latent[:, 1:].detach())
    trajectory = latent.flatten(-2)
    velocity = trajectory[:, 1:] - trajectory[:, :-1]
    previous, current = velocity[:, :-1], velocity[:, 1:]
    curvature = 1 - F.cosine_similarity(previous, current, dim=-1, eps=1e-6)
    moving = (previous.norm(dim=-1) > 1e-6) & (current.norm(dim=-1) > 1e-6)
    curvature_loss = curvature[moving].mean() if moving.any() else curvature.new_zeros(())
    return self.prediction_weight * prediction_loss + self.curvature_weight * curvature_loss, {
        "prediction_loss": prediction_loss, "visual_prediction_loss": prediction_loss,
        "curvature_loss": curvature_loss,
    }


def sampler_digest(dataset):
    state = copy.deepcopy(dataset.state_dict())
    state["episode_order"] = state["episode_order"].tolist()
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def physical_checks(before, after):
    """Screen original-unit errors; tiny expert variances must not decide this verdict."""
    checks = {}
    for cohort in before:
        old, new = before[cohort]["all"]["physical"], after[cohort]["all"]["physical"]
        for source, horizon in (("observed", "1"), ("forecast", "5"), ("forecast", "100")):
            coordinates = {}
            for key, value in old[source][horizon]["physical_rmse"].items():
                limit = 3 * max(value, .01)
                current = new[source][horizon]["physical_rmse"][key]
                coordinates[key] = {"before": value, "after": current, "limit": limit,
                                    "passed": current <= limit}
            checks[f"{cohort}/{source}/h{horizon}"] = coordinates
    return checks


def collect_shared(config, reference, seed):
    # Short fixed-behavior trajectories isolate adaptation from controller/data feedback.
    settings = copy.deepcopy(config)
    settings.env.time_limit = 200 * int(config.env.action_repeat)
    count = int(config.replay.episodes_per_batch)
    training = [collect_episode(settings, reference, 7_000_000 + seed + i,
                                "zero" if i % 2 == 0 else "random") for i in range(count)]
    validation = [collect_episode(config, reference, 8_000_000 + seed + i, mode)
                  for i, mode in enumerate(("zero", "random"))]
    data = TrajectoryDataset(validation, forbidden_seeds=[episode["seed"] for episode in training])
    return training, data


def run_variant(config, dataset, sampler_start, replay, plans, sources, settings, args, variant, output, persist):
    dataset.load_state_dict(copy.deepcopy(sampler_start))
    model = new_model(config, dataset)
    if variant == "legacy_flattened":
        model.representation_loss = MethodType(legacy_loss, model)
    elif variant != "fixed_patchwise":
        raise ValueError(f"Unknown curvature variant: {variant}")
    result = {"variant": variant, "status": "RUNNING", "snapshots": [], "phases": {},
              "initial_state_sha256": tensor_digest(model.state_dict()),
              "parameters": sum(p.numel() for p in model.parameters())}
    started = time.monotonic()
    output.mkdir()
    baseline = None
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)

    def snapshot(phase, step):
        scores = measure_errors(model, sources, settings)
        entry = {"phase": phase, "updates": step, "scores": scores}
        if baseline is not None:
            entry["guards"] = physical_checks(baseline, scores)
        result["snapshots"].append(entry)
        persist(result)

    expert = {"batch": None}
    try:
        with (output / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for phase, count in (("offline", args.expert_updates), ("adaptation", args.online_updates)):
                if phase == "adaptation":
                    model.state_head.configure_online(lambda: model.readout_features(expert["batch"]))
                phase_start = time.monotonic()
                progress = Progress(f"TS {variant} {phase}", count)
                norms, clipped = [], 0
                for step in range(1, count + 1):
                    torch.manual_seed(args.seed + (0 if phase == "offline" else 1_000_000) + step)
                    if phase == "offline":
                        batch = dataset.sample_episode_batch()
                    else:
                        batch = replay.batch(plans[step - 1])
                        expert["batch"] = dataset.sample_episode_batch()
                    values = checked_update(model, lambda: model.update(batch))
                    norms.append(values["grad_norm"])
                    clipped += int(values["grad_clipped"])
                    log.write(json.dumps({"phase": phase, "update": step, **values}, allow_nan=False) + "\n")
                    progress.update(step, f"prediction={values['prediction_loss']:.3g} "
                                    f"curvature={values['curvature_loss']:.3g} state={values['state/loss']:.3g}",
                                    force=step == count)
                    if phase == "adaptation" and step in {max(1, count // 4), count}:
                        snapshot(phase, step)
                result["phases"][phase] = {
                    "updates": count, "seconds": time.monotonic() - phase_start,
                    "sampler_end_sha256": sampler_digest(dataset), "grad_norm_mean": sum(norms) / count,
                    "grad_norm_max": max(norms), "clipped_fraction": clipped / count,
                }
                if phase == "offline":
                    snapshot(phase, count)
                    baseline = result["snapshots"][-1]["scores"]
        regression = any(not coordinate["passed"] for snap in result["snapshots"]
                         for check in snap.get("guards", {}).values() for coordinate in check.values())
        result["status"] = "REGRESSION" if regression else "NO_REGRESSION"
        if model.device.type == "cuda":
            result["gpu_reserved_peak_gib"] = torch.cuda.max_memory_reserved(model.device) / 1024**3
    except Exception as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        model.state_head._expert_source = None
    result["seconds"] = time.monotonic() - started
    persist(result)
    return result


def comparison(results):
    if len(results) != 2 or any(result["status"] in {"RUNNING", "FAIL"} for result in results):
        return "INCOMPLETE"
    old, new = results
    if old["initial_state_sha256"] != new["initial_state_sha256"] or any(
        old["phases"][phase]["sampler_end_sha256"] != new["phases"][phase]["sampler_end_sha256"]
        for phase in ("offline", "adaptation")
    ):
        return "INVALID: starting weights or expert sampling differed"
    if new["status"] == "REGRESSION":
        return "REGRESSION REMAINS: the corrected arm still exceeded a physical-error guard"
    if old["status"] == "REGRESSION":
        return "ENCOURAGING: legacy regressed and corrected did not, on this short fixed-replay test only"
    return "INCONCLUSIVE: the legacy failure was not reproduced; neither arm exceeded the error guards"


def write_report(output, report):
    report["comparison"] = comparison(report["runs"])
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["TS curvature A/B | tiny models from scratch | shared expert batches + fixed simulator replay",
             "No planner optimization, policy evaluation, calibration, or checkpoint writes."]
    for result in report["runs"]:
        lines.append(f"{result['status']} | {result['variant']} | {duration(result.get('seconds', 0))}")
        if result["status"] in {"RUNNING", "FAIL"}:
            if "error" in result:
                lines.append(f"  {result['error']}")
            continue
        before, after = [snap["scores"] for snap in (result["snapshots"][0], result["snapshots"][-1])]
        for cohort in before:
            old, new = before[cohort]["all"], after[cohort]["all"]
            lines.append(f"  {cohort} latent std: {old['representation']['rms_std']:.3g}"
                         f"->{new['representation']['rms_std']:.3g}")
            for source, horizon in (("observed", "1"), ("forecast", "5"), ("forecast", "100")):
                lines.append(f"  {cohort} {source} h{horizon} RMSE | " + format_physical_rmse(
                    new["physical"][source][horizon], before=old["physical"][source][horizon]))
    lines += [report["comparison"],
              "Guard: any coordinate RMSE > 3 * max(post-offline RMSE, 0.01 original units), at either adaptation snapshot.",
              "nMSE is secondary in JSON and does not decide the guards. Both arms use all other CURRENT fixes.",
              "Fixed zero/random replay isolates the loss change; it is NOT own-policy online training or proof of recovery."]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expert-updates", type=int, default=1000)
    parser.add_argument("--online-updates", type=int, default=512, help="Fixed-replay adaptation updates per arm.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("ts_ablation_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if min(args.expert_updates, args.online_updates) < 1 or args.seed < 0:
        parser.error("Update counts must be positive; seed must be nonnegative.")
    args.scenario = "cartpole_balance_sparse"
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.device(args.device).type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This diagnostic requires a BF16-capable CUDA GPU (or --device cpu).")
        torch.cuda.set_device(args.device)
    config = build_config("temporal_straightening", args)
    if config.jepa_model.decoder.enabled or float(config.training.online.expert_fraction) != 0:
        raise ValueError("Expected decoder-disabled TS with native online-only updates.")
    settings = diagnostic_settings(args.seed)
    settings.horizons = [1, 5, 100]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": OmegaConf.to_container(config, resolve=True),
              "budget": {"expert_updates": args.expert_updates, "adaptation_updates": args.online_updates},
              "checkpoint_writes": False, "runs": []}
    family = load_model_family("temporal_straightening")
    with family.build_replay(config) as dataset:
        sampler_start = copy.deepcopy(dataset.state_dict())
        reference = new_model(config, dataset)
        heldout = StateDataset(dataset.h5, dataset.metadata, config.model_io, config.state_head.fields, reference.state_head.targets)
        windows = heldout.sample_windows(8, settings.context_length + 100, settings.window_seed,
                                         settings.context_length, .5, 8)
        print("Data | collecting shared zero/random replay once; expert and simulator validation stay held out", flush=True)
        episodes, validation = collect_shared(config, reference, args.seed)
        failure_windows = validation.sample_windows(2, settings.context_length + 100, settings.window_seed)
        sources = {"expert": (heldout, windows), "simulator": (validation, failure_windows)}
        replay = FixedBatches(episodes, config, 9_000_000 + args.seed)
        plans = [replay.draw() for _ in range(args.online_updates)]
        (args.output / "replay_batches.json").write_text(json.dumps(plans) + "\n", encoding="utf-8")
        report.update(dataset_identity=dataset_identity(dataset.metadata),
                      training_simulator_episodes=episode_metadata(episodes),
                      expert_windows=[asdict(window) for window in windows],
                      simulator_validation=validation_metadata(validation, failure_windows),
                      initial_sampler_sha256=sampler_digest(dataset),
                      replay_batches_sha256=hashlib.sha256(json.dumps(plans).encode()).hexdigest())
        del reference
        print(f"Train | two TS arms | each: {args.expert_updates} expert + {args.online_updates} replay updates", flush=True)
        for variant in VARIANTS:
            report["runs"].append({"variant": variant, "status": "RUNNING"})

            def persist(result):
                report["runs"][-1] = result
                write_report(args.output, report)

            run_variant(config, dataset, sampler_start, replay, plans, sources, settings, args,
                        variant, args.output / variant, persist)
    report["elapsed_seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(result["status"] == "FAIL" for result in report["runs"])
               or report["comparison"].startswith("INVALID") or report["runs"][-1]["status"] == "REGRESSION")


if __name__ == "__main__":
    raise SystemExit(main())
