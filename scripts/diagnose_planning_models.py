"""Compare fixed physical readouts and latent representations before/after online training.

Uses only held-out expert windows. No fitting, environment steps, or action optimization.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import torch
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity, validate_dataset
from training import load_model_family
from training.evaluation import StateDataset, latent_rollout
from training.protocol import implementation_sha256, validate_checkpoint

FAMILIES = ("leworldmodel", "temporal_straightening")
SCENARIOS = ("cartpole_balance_sparse", "reacher", "ball_in_cup")


def latent_statistics(frames):
    """Centered frame-level variance and entropy effective rank of covariance eigenvalues."""
    values = frames.flatten(1).double().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite encoded latents.")
    centered = values - values.mean(0)
    variance = centered.square().mean(0)
    # Use the smaller Gram matrix; no spatial pooling that could hide token collapse.
    gram = centered @ centered.T if len(values) <= values.shape[1] else centered.T @ centered
    eigenvalues = torch.linalg.eigvalsh(gram / len(values)).clamp_min(0)
    threshold = eigenvalues.max() * max(values.shape) * torch.finfo(torch.float64).eps
    positive = eigenvalues[eigenvalues > threshold]
    weights = positive / positive.sum() if positive.numel() else positive
    return {
        "frames": len(values), "dimensions": values.shape[1],
        "maximum_rank": min(len(values) - 1, values.shape[1]),
        "rms_std": variance.mean().sqrt().item(),
        "mean_std": variance.sqrt().mean().item(),
        "zero_variance_fraction": (variance == 0).double().mean().item(),
        "effective_rank": (-torch.sum(weights * weights.log())).exp().item() if positive.numel() else 0.0,
        "participation_ratio": weights.square().sum().reciprocal().item() if positive.numel() else 0.0,
    }


@torch.no_grad()
def analyze_batch(model, observation, actions, targets, context, horizons, random_actions):
    features, prediction = latent_rollout(model, observation, actions, context)
    head = model.state_head
    indices = torch.tensor(horizons, device=actions.device) - 1
    prefix = features[:, context - head.history + 1:context]
    forecast = head(torch.cat((prefix, prediction), dim=1))
    observed = head(features[:, context - head.history + 1:])
    anchor = head(features[:, context - head.history:context])
    truth = targets[:, context:]
    estimates = {
        "observed": observed, "forecast": forecast,
        "decoded_persistence": anchor, "true_persistence": targets[:, context - 1:context],
    }
    errors = {name: (value - truth).square()[:, indices].cpu() for name, value in estimates.items()}
    future = features[:, context:]
    latent_error = (prediction - future).flatten(2).square().mean(-1)[:, indices].cpu()
    temporal = features.diff(dim=1).flatten(2).square().mean((1, 2)).cpu()
    physical_motion = targets.diff(dim=1).square().mean(1).cpu()
    # Four fixed frames per window bound the CPU covariance calculation.
    frame_indices = sorted({context - 1, context, context + (prediction.shape[1] - 1) // 2, features.shape[1] - 1})
    frames = features[:, frame_indices].flatten(2).cpu()
    history = features[:, context - model.history_size:context]
    past = actions[:, context - model.history_size:context - 1]
    actual = actions[:, context - 1:]
    responses = {}
    for name, alternative in (("zero", torch.zeros_like(actual)), ("random", random_actions)):
        alternative_prediction = model.rollout(history, past, alternative[:, None])[:, 0]
        alternative_physical = head(torch.cat((prefix, alternative_prediction), dim=1))
        responses[name] = {
            "latent_squared_delta": (alternative_prediction - prediction).flatten(2).square().mean(-1)[:, indices].cpu(),
            "physical_squared_delta": (alternative_physical - forecast).square()[:, indices].cpu(),
            "action_squared_delta": (alternative - actual).square().mean((1, 2)).cpu(),
        }
    goals = {}
    if hasattr(model, "goal_tolerance"):
        true_relation = head.targets.goal_relation(truth)[:, indices]

        def inside(relation):
            scaled = relation / model.goal_tolerance
            return scaled.norm(dim=-1) <= 1 if model.goal_geometry == "radial" else (scaled.abs() <= 1).all(-1)

        actual_success = inside(true_relation)
        for name in ("observed", "forecast"):
            relation = head.targets.goal_relation(estimates[name])[:, indices]
            difference = relation - true_relation
            if head.targets.task == "dmc_cartpole_balance_sparse":
                difference[..., 1] = torch.atan2(difference[..., 1].sin(), difference[..., 1].cos())
            goals[name] = {
                "squared_error": difference.square().cpu(),
                "false_success": (inside(relation) & ~actual_success).cpu(),
                "failure": (~actual_success).cpu(),
                "predicted_success": inside(relation).cpu(),
            }
    return {"errors": errors, "latent_error": latent_error, "temporal": temporal,
            "physical_motion": physical_motion, "frames": frames, "responses": responses, "goals": goals}


def summarize_batches(batches, windows, head, horizons):
    coordinates = list(head.coordinates)
    std = head.std.detach().float().cpu()
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Readout scales must be finite and positive.")
    errors = {name: torch.cat([batch["errors"][name] for batch in batches]) for name in batches[0]["errors"]}
    frames = torch.cat([batch["frames"] for batch in batches])
    latent_error = torch.cat([batch["latent_error"] for batch in batches])
    temporal = torch.cat([batch["temporal"] for batch in batches])
    motion = torch.cat([batch["physical_motion"] for batch in batches])
    result = {}
    selections = {"all": list(range(len(windows)))}
    selections.update({name: [i for i, window in enumerate(windows) if window.cohort == name]
                       for name in dict.fromkeys(window.cohort for window in windows) if name != "all"})
    for cohort, indices in selections.items():
        if not indices:
            continue
        physical = {}
        for name, values in errors.items():
            mse = values[indices].mean(0)
            normalized = mse / std.square()
            physical[name] = {
                str(horizon): {
                    "rmse": dict(zip(coordinates, mse[i].sqrt().tolist(), strict=True)),
                    "normalized_mse": dict(zip(coordinates, normalized[i].tolist(), strict=True)),
                    "mean_normalized_mse": normalized[i].mean().item(),
                    "normalized_loss_fraction": dict(zip(
                        coordinates, (normalized[i] / normalized[i].sum().clamp_min(1e-30)).tolist(), strict=True,
                    )),
                } for i, horizon in enumerate(horizons)
            }
        responses = {}
        for name in batches[0]["responses"]:
            latent = torch.cat([batch["responses"][name]["latent_squared_delta"] for batch in batches])[indices]
            decoded = torch.cat([batch["responses"][name]["physical_squared_delta"] for batch in batches])[indices]
            action = torch.cat([batch["responses"][name]["action_squared_delta"] for batch in batches])[indices]
            responses[name] = {
                "action_rms_delta": action.mean().sqrt().item(),
                "latent_rms_delta": dict(zip(map(str, horizons), latent.mean(0).sqrt().tolist(), strict=True)),
                "physical_rms_delta": {str(h): dict(zip(coordinates, decoded.mean(0)[i].sqrt().tolist(), strict=True))
                                       for i, h in enumerate(horizons)},
            }
        representation = latent_statistics(frames[indices].flatten(0, 1))
        representation["temporal_rms_delta"] = temporal[indices].mean().sqrt().item()
        representation["forecast_latent_rmse"] = dict(zip(
            map(str, horizons), latent_error[indices].mean(0).sqrt().tolist(), strict=True,
        ))
        result[cohort] = {
            "windows": len(indices), "physical": physical, "representation": representation,
            "action_sensitivity": responses,
            "physical_temporal_rms_delta": dict(zip(coordinates, motion[indices].mean(0).sqrt().tolist(), strict=True)),
        }
        goals = {}
        for source in batches[0].get("goals", {}):
            values = {key: torch.cat([batch["goals"][source][key] for batch in batches])[indices]
                      for key in batches[0]["goals"][source]}
            goals[source] = {
                str(horizon): {
                    "relation_rmse": values["squared_error"][:, i].mean(0).sqrt().tolist(),
                    "failure_states": int(values["failure"][:, i].sum()),
                    "false_success_rate": (values["false_success"][:, i].sum().item()
                                           / values["failure"][:, i].sum().item())
                        if values["failure"][:, i].any() else None,
                    "predicted_success_fraction": values["predicted_success"][:, i].float().mean().item(),
                } for i, horizon in enumerate(horizons)
            }
        if goals:
            result[cohort]["goals"] = goals
    return result


def analyze_checkpoint(model, dataset, windows, args):
    model.eval()
    context, horizons = args.context_length, args.horizons
    length = context + max(horizons)
    random = torch.Generator().manual_seed(args.window_seed + 1)
    action_dim = model.action_dim
    alternatives = torch.rand(len(windows), max(horizons), action_dim, generator=random) * 2 - 1
    batches = []
    for offset in range(0, len(windows), args.batch_size):
        observation, action, target = dataset.read_batch(windows[offset:offset + args.batch_size], length)
        batches.append(analyze_batch(
            model, {key: value.to(model.device) for key, value in observation.items()},
            action.to(model.device), target.to(model.device), context, horizons,
            alternatives[offset:offset + len(action)].to(model.device),
        ))
    return summarize_batches(batches, windows, model.state_head, horizons)


def write_reports(output, results, args):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "diagnostic_version": 1, "implementation_sha256": implementation_sha256(),
        "dataset_role": "held_out_expert", "evaluation_fitting": False,
        "context_length": args.context_length, "horizons": args.horizons, "window_seed": args.window_seed,
        "notes": [
            "Observed decoding uses real images; forecasting uses only the observed prefix and remaining recorded actions.",
            "Rank uses centered, flattened frames with ordered TS patches; compare checkpoints within a model.",
            "Action sensitivity is response, not correctness: counterfactual ground truth is unavailable.",
            "Low temporal motion can be natural in successful expert data; inspect the motion cohort and physical motion.",
            "These expert windows do not establish readout accuracy on failed online-policy states.",
        ],
        "results": results,
    }
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    with (output / "physical_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "model", "checkpoint", "cohort", "horizon", "source", "coordinate",
                         "rmse_original_units", "expert_std", "normalized_mse", "normalized_loss_fraction"])
        for result in results:
            for cohort, values in result["cohorts"].items():
                for source, by_horizon in values["physical"].items():
                    for horizon, metrics in by_horizon.items():
                        for coordinate, rmse in metrics["rmse"].items():
                            writer.writerow([result["scenario"], result["model"], result["checkpoint_name"],
                                             cohort, horizon, source, coordinate, rmse,
                                             result["readout_std"][coordinate], metrics["normalized_mse"][coordinate],
                                             metrics["normalized_loss_fraction"][coordinate]])
    first, last = map(str, (min(args.horizons), max(args.horizons)))
    lines = ["Held-out expert diagnostics | no fitting, simulator, or planner optimization",
             f"Scenario/model/checkpoint | observed nMSE h{first} | forecast nMSE h{first}/h{last} | rank/max | latent std/motion | random-action latent delta h{first}/h{last}"]
    for result in results:
        values = result["cohorts"]["all"]
        physical, representation = values["physical"], values["representation"]
        response = values["action_sensitivity"]["random"]["latent_rms_delta"]
        lines.append(
            f"{result['scenario']}/{result['model']}/{result['checkpoint_name']} | "
            f"{physical['observed'][first]['mean_normalized_mse']:.3g} | "
            f"{physical['forecast'][first]['mean_normalized_mse']:.3g}/{physical['forecast'][last]['mean_normalized_mse']:.3g} | "
            f"{representation['effective_rank']:.2f}/{representation['maximum_rank']} | "
            f"{representation['rms_std']:.3g}/{representation['temporal_rms_delta']:.3g} | "
            f"{response[first]:.3g}/{response[last]:.3g}"
        )
    lines.append("nMSE uses checkpoint expert scales; original-unit errors and coordinate contributions are in physical_errors.csv.")
    lines.append("Rank/action sensitivity have no universal pass threshold. Cohort details and exact windows are in report.json.")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/dmc_vision_10k"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--window-seed", type=int, default=2000000)
    parser.add_argument("--output", type=Path, default=Path("runs/readout_diagnostics"))
    args = parser.parse_args()
    args.horizons = sorted(set(args.horizons))
    if min(args.windows, args.batch_size, args.context_length, *args.horizons) < 1:
        parser.error("Window count, batch size, context length, and horizons must be positive.")
    torch.set_num_threads(1)
    results = []
    for scenario in args.scenarios:
        metadata, windows, reference = None, None, None
        for name in args.models:
            for filename in ("pretrained.pt", "final.pt"):
                path = args.run_root / scenario / name / "default" / f"seed_{args.seed}" / filename
                print(f"START | {scenario}/{name}/{filename}", flush=True)
                started = time.monotonic()
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                config = OmegaConf.create(checkpoint["training_config"])
                config.device = args.device
                if str(config.model_family) != name or str(config.scenario.name) != scenario or int(config.seed) != args.seed:
                    raise ValueError(f"Checkpoint identity does not match its requested run: {path}")
                validate_checkpoint(checkpoint, config, training=False)
                dataset_path = args.dataset_root / str(config.scenario.dataset)
                if metadata is None:
                    metadata = validate_dataset(dataset_path, config, splits=("heldout",))
                if checkpoint.get("dataset_identity") != dataset_identity(metadata):
                    raise ValueError(f"Checkpoint and held-out dataset identities differ: {path}")
                family = load_model_family(name)
                model = family.build_model(config)
                family.load_checkpoint(model, checkpoint, training=False)
                if not model.state_head.updates.item():
                    raise ValueError(f"Checkpoint has no trained readout: {path}")
                if args.context_length < model.history_size:
                    raise ValueError("Observed prefix must cover the model's native history.")
                with h5py.File(dataset_path / "data.hdf5", "r") as h5:
                    dataset = StateDataset(h5, metadata, config.model_io, config.state_head.fields, model.state_head.targets)
                    signature = (str(dataset_path.resolve()), dataset.episodes.tolist(), model.state_head.coordinates)
                    if reference is not None and signature != reference:
                        raise ValueError("Compared models must share their dataset, held-out split, and physical targets.")
                    reference = signature
                    if windows is None:
                        windows = dataset.sample_windows(args.windows, args.context_length + max(args.horizons),
                                                         args.window_seed, args.context_length, 0.5, 8)
                    cohorts = analyze_checkpoint(model, dataset, windows, args)
                head = model.state_head
                results.append({
                    "scenario": scenario, "model": name, "checkpoint_name": filename,
                    "checkpoint": str(path.resolve()), "checkpoint_id": checkpoint["checkpoint_id"],
                    "run_identity": checkpoint["run_identity"], "checkpoint_phase": checkpoint["phase"],
                    "dataset": str(dataset_path.resolve()), "dataset_identity": dataset_identity(metadata),
                    "readout_updates": int(head.updates),
                    "readout_mean": dict(zip(head.coordinates, head.mean.cpu().tolist(), strict=True)),
                    "readout_std": dict(zip(head.coordinates, head.std.cpu().tolist(), strict=True)),
                    "windows": [asdict(window) for window in windows], "cohorts": cohorts,
                    "elapsed_seconds": time.monotonic() - started,
                })
                write_reports(args.output, results, args)
                print(f"DONE | {scenario}/{name}/{filename} | {time.monotonic() - started:.1f}s", flush=True)
                del model, checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(write_reports(args.output, results, args))
    print(f"Reports | {args.output.resolve()}")


if __name__ == "__main__":
    main()
