"""Frozen original-checkpoint history ablation; no fitting, policies or simulation.

All conditions end their observed prefix at the SAME recorded frame. Only its
left edge moves. Real future images are diagnostic targets, never forecast inputs.
"""
import argparse
import gc
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.analyze_world_model_capabilities import MODELS, records, write_csv


def tensor_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(f"{key}:{tuple(value.shape)}:{value.dtype}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def crop_history(observation, actions, targets, maximum, context):
    if not 1 <= context <= maximum or actions.shape[1] != targets.shape[1] - 1:
        raise ValueError("Invalid history or misaligned image/action lengths")
    start = maximum - context
    return ({k: v[:, start:] for k, v in observation.items()}, actions[:, start:], targets[:, start:])


@torch.no_grad()
def evaluate_batch(model, config, observation, actions, targets, *, context, horizons, samples, seed):
    from training.evaluation import latent_rollout
    from models.dreamer import Dreamer
    from models.storm import StormModel
    head = model.state_head
    stochastic = isinstance(model, (Dreamer, StormModel))
    samples = samples if stochastic else 1
    device = next(model.parameters()).device
    observation = {k: v.to(device) for k, v in observation.items()}
    actions, targets = actions.to(device), targets.to(device)
    kwargs = {"storm_context_length": int(config.storm_train.context_length)} if isinstance(model, StormModel) else {}
    sums = {}
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for _ in range(samples):
            features, predicted = latent_rollout(model, observation, actions, context, **kwargs)
            prefix = features[:, context - head.history + 1:context]
            estimates = {
                "forecast": head(torch.cat((prefix, predicted), dim=1)),
                "observed": head(features[:, context - head.history + 1:]),
                "current": head(features[:, context - head.history:context]),
            }
            for k, value in estimates.items():
                sums[k] = sums.get(k, 0) + value
    estimates = {k: v / samples for k, v in sums.items()}
    current_truth, future_truth = targets[:, context - 1:context], targets[:, context:]
    indices = torch.tensor(horizons, device=device) - 1
    result = {}
    for label, estimate, truth in [
        ("current", estimates["current"], current_truth),
        ("forecast", estimates["forecast"], future_truth),
        ("observed", estimates["observed"], future_truth),
        ("decoded_hold", estimates["current"].expand_as(future_truth), future_truth),
        ("true_hold", current_truth.expand_as(future_truth), future_truth),
    ]:
        if label != "current":
            estimate, truth = estimate[:, indices], truth[:, indices]
        error = head.targets.metric_error(estimate, truth).square()
        if not torch.isfinite(error).all():
            raise ValueError(f"Non-finite {label} error")
        result[label] = error.cpu()
    # Native-coordinate forecast values retained for a fixed-history control.
    result["forecast_values"] = estimates["forecast"][:, indices].cpu()
    result["current_values"] = estimates["current"].cpu()
    return result


def summarize(errors, windows, horizons, coordinates):
    result = {}
    for cohort in ("all", "uniform", "motion"):
        indices = [i for i, w in enumerate(windows) if cohort == "all" or w.cohort == cohort]
        if not indices:
            continue
        row = {"windows": len(indices)}
        for key in ("current", "forecast", "observed", "decoded_hold", "true_hold"):
            rmse = errors[key][indices].mean(0).sqrt()
            hs = [0] if key == "current" else horizons
            row[key] = {str(h): dict(zip(coordinates, rmse[i].tolist())) for i, h in enumerate(hs)}
        result[cohort] = row
    return result


def select_windows(archive, count):
    from training.evaluation import Window
    if count < 2 or count % 2:
        raise ValueError("Use a positive even window count for equal uniform/motion cohorts")
    selected = []
    for cohort in ("uniform", "motion"):
        candidates = [w for w in archive["windows"] if w["cohort"] == cohort]
        if len(candidates) < count // 2:
            raise ValueError("Insufficient archived windows")
        selected.extend(Window(w["episode"], w["start"], cohort, w["motion_score"]) for w in candidates[:count // 2])
    if len({(w.episode, w.start) for w in selected}) != len(selected):
        raise ValueError("Repeated evaluation window")
    return selected


def preflight(args):
    archive_root = args.archive_root or args.run_root
    archived = records(archive_root, ["cartpole_balance_sparse"])
    selected = [(t, m, r, p) for t, m, r, p in archived if m in args.models]
    missing = [str(args.run_root / t / m / "seed_0/final.pt") for t, m, _, _ in selected
               if not (args.run_root / t / m / "seed_0/final.pt").is_file()]
    if missing:
        raise FileNotFoundError("Missing original checkpoints:\n" + "\n".join(missing))
    q = selected[0][2]["physical_state_prediction"]
    if q["context_length"] != 64 or max(args.contexts) > 64 or max(args.horizons) > max(q["horizons"]):
        raise ValueError("Requested conditions exceed the archived observed prefix / future")
    windows = select_windows(q, args.windows)
    if not (args.dataset_root / "cartpole_balance_sparse/data.hdf5").is_file():
        raise FileNotFoundError("Missing original Cartpole expert dataset")
    return selected, windows


def evaluate_model(args, task, name, archived, windows, directory):
    import h5py
    import tools
    from dmc_expert.storage import dataset_identity, validate_dataset
    from training import load_model_family
    from training.evaluation import StateDataset
    from training.protocol import validate_checkpoint
    path = args.run_root / task / name / "seed_0/final.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    counters = payload.get("trainer_state", {})
    if (payload.get("checkpoint_id"), payload.get("expert_updates"), counters.get("world_model_updates"),
            counters.get("env_steps"), payload.get("phase")) != (archived["checkpoint_id"], 10000, 10000, 80000, "online"):
        raise ValueError(f"Checkpoint is not the original archived final: {name}")
    if payload["run_identity"] != archived["run_identity"] or payload["dataset_identity"] != archived["dataset_identity"]:
        raise ValueError(f"Source provenance mismatch: {name}")
    config = OmegaConf.create(payload["training_config"])
    config.device = args.device
    validate_checkpoint(payload, config, training=False)
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    family = load_model_family(config.model_family)
    model = family.build_model(config)
    family.load_checkpoint(model, payload, training=False)
    model.eval().requires_grad_(False)
    head = model.state_head
    if min(args.contexts) < max(head.history, getattr(model, "history_size", 1), getattr(model, "frame_stack", 1)):
        raise ValueError("Context shorter than native model/readout requirements")
    if not head.updates.item():
        raise ValueError("Untrained physical readout")
    before = tensor_digest(model.state_dict())
    del payload
    gc.collect()
    dataset_path = args.dataset_root / task
    metadata = validate_dataset(dataset_path, config, splits=("heldout",))
    if dataset_identity(metadata) != archived["dataset_identity"]:
        raise ValueError("Wrong expert dataset")
    coordinates = list(head.targets.metric_coordinates)
    result = {"model": name, "checkpoint_id": archived["checkpoint_id"], "source_run_identity": archived["run_identity"],
              "model_state_before": before, "physical_coordinates": coordinates,
              "physical_units": dict(head.targets.metric_units), "readout_history": head.history,
              "metric_version": "physical_rmse_wrapped_angles_v1",
              "state_samples": args.samples if name.startswith(("dreamer/", "storm/")) else 1,
              "conditions": {}, "status": "RUNNING"}
    all_errors, data_hashes = {}, []
    with h5py.File(dataset_path / "data.hdf5", "r") as h5:
        dataset = StateDataset(h5, metadata, config.model_io, config.state_head.fields, head.targets)
        # Read each maximum-prefix batch once; crop only its left edge per condition.
        for batch_number, (obs, actions, truth) in enumerate(dataset.batches(windows, 64 + max(args.horizons), args.batch_size)):
            data_hashes.append(tensor_digest({**obs, "actions": actions, "targets": truth}))
            for context in args.contexts:
                shortened = crop_history(obs, actions, truth, 64, context)
                values = evaluate_batch(model, config, *shortened, context=context, horizons=args.horizons,
                                        samples=args.samples, seed=args.seed + batch_number)
                for key, value in values.items():
                    all_errors.setdefault(context, {}).setdefault(key, []).append(value)
            print(f"{name} | windows={min((batch_number + 1) * args.batch_size, len(windows))}/{len(windows)}", flush=True)
    all_errors = {k: {label: torch.cat(v) for label, v in errors.items()} for k, errors in all_errors.items()}
    for context, errors in all_errors.items():
        result["conditions"][str(context)] = summarize(errors, windows, args.horizons, coordinates)
    result["data_sha256"] = hashlib.sha256(json.dumps(data_hashes).encode()).hexdigest()
    # These deterministic models cannot use observations older than their native context.
    if name in ("leworldmodel/default", "temporal_straightening/default", "tdmpc2/default"):
        reference = all_errors[max(args.contexts)]
        result["fixed_history_control_max_abs"] = {}
        for context, values in all_errors.items():
            delta = max(float((values[k] - reference[k]).abs().max()) for k in ("current_values", "forecast_values"))
            result["fixed_history_control_max_abs"][str(context)] = delta
            for key in ("current_values", "forecast_values"):
                torch.testing.assert_close(values[key], reference[key], atol=2e-4, rtol=2e-4)
    result["model_state_after"] = tensor_digest(model.state_dict())
    if before != result["model_state_after"]:
        raise RuntimeError("Frozen evaluation changed weights or buffers")
    saved = directory / (name.replace("/", "_") + "_errors.pt")
    torch.save({"windows": [asdict(w) for w in windows], "horizons": args.horizons,
                "coordinates": coordinates, "conditions": all_errors}, saved)
    result.update(status="COMPLETE", errors_file=saved.name)
    del model, all_errors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(report, output):
    rows = []
    lines = [f"Run: {report['run_name']} | Status: {report['status']}",
             "Frozen final checkpoints; Cartpole; no training or policy execution.",
             "Same endpoint/future/actions; vary only observed prefix length."]
    for r in report["models"]:
        for context, cohorts in r.get("conditions", {}).items():
            for cohort, scores in cohorts.items():
                for metric in ("current", "forecast", "observed", "decoded_hold", "true_hold"):
                    for horizon, coords in scores[metric].items():
                        for coordinate, rmse in coords.items():
                            rows.append(dict(model=r["model"], context=int(context), cohort=cohort, metric=metric,
                                             horizon=int(horizon), coordinate=coordinate, rmse=rmse))
            scores = cohorts["all"]["current"]["0"]
            lines.append(f"{r['model']} | history={context} | current physical RMSE={scores}")
    if rows:
        write_csv(output / "metrics.csv", rows)
    lines += [f"Elapsed seconds: {report.get('seconds', 0):.1f}", f"Run: {report['run_name']} | Status: {report['status']}"]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/dmc_vision_10k"))
    parser.add_argument("--archive-root", type=Path, help="Original evaluation.json tree; defaults to run-root")
    parser.add_argument("--dataset-root", type=Path, default=Path("/lambda/nfs/DMC/data/dmc_expert_vision"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--contexts", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 25])
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=76000000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Check archive, paths and window selection; do not execute models")
    args = parser.parse_args(argv)
    args.contexts, args.horizons = sorted(set(args.contexts)), sorted(set(args.horizons))
    if min(args.contexts + args.horizons + [args.batch_size, args.samples]) < 1 or args.seed < 0 or len(set(args.models)) != len(args.models):
        parser.error("Positive sizes, unique models and a nonnegative seed are required")
    output = args.output or Path("runs") / datetime.now(timezone.utc).strftime("history_context_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"run_name": output.name, "status": "RUNNING", "models": [],
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "training_updates": 0, "source": "original 39-run experiment final checkpoints",
              "notes": ["Same forecast endpoint and future actions for every history length.",
                        "Current RMSE is at the final observed frame; observed future RMSE sees real future images.",
                        "True-state hold is scoring only, never an inference input.",
                        "Stochastic models average physical predictions across native samples before scoring.",
                        "Same seeds give reproducible conditions, not identical random draws after unequal prefixes.",
                        "Original physical readout is frozen; this is not a retrained probe of all latent information.",
                        "One training seed and expert windows; not general on-policy robustness."]}
    try:
        selected, windows = preflight(args)
        report["windows"] = [asdict(w) for w in windows]
        report["forecast_start"] = [w.start + 64 for w in windows]
        if args.dry_run:
            report["status"] = "PREFLIGHT_ONLY"
            return
        from training.protocol import implementation_sha256, runtime_sha256
        report.update(evaluation_implementation=implementation_sha256(), evaluation_runtime=runtime_sha256(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for task, name, archive, _ in selected:
            report['active_model'] = name
            result = evaluate_model(args, task, name, archive, windows, output)
            report["models"].append(result)
            if len({r["data_sha256"] for r in report["models"]}) != 1:
                raise ValueError("Model families received different evaluation data")
            report["seconds"] = time.monotonic() - started
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            write_summary(report, output)
        report["status"] = "COMPLETE"
        report.pop('active_model', None)
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        write_summary(report, output)
        print(f"Run: {output.name} | Status: {report['status']} | Output: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
