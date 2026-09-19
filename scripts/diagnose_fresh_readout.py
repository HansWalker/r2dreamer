"""Fit fresh physical heads on frozen planner checkpoints, without policy optimization.

Expert development episodes come from the original TRAIN split, partitioned again
for head fitting/validation. The benchmark held-out split is never read or fitted.
"""

import argparse
import csv
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import (
    dataset_identity,
    observation_indices,
    read_image_window,
    read_physical_state,
    split_episode_indices,
    validate_dataset,
)
from models.shared.physical_state import PhysicalStateHead, readout_mode
from scripts.diagnose_planning_models import FAMILIES, SCENARIOS
from scripts.online_validation import (
    collect_episode,
    episode_metadata,
    sample_trajectory_windows,
)
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import (
    checkpoint_compatibility,
    implementation_sha256,
    validate_checkpoint,
)


def tensor_digest(values):
    digest = hashlib.sha256()
    for key, value in values.items():
        digest.update(f"{key}:{tuple(value.shape)}:{value.dtype}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_partition(training, validation):
    """Reject exact duplicates as well as episode/seed overlap, not repeated individual states."""
    for key in ("id", "sha256"):
        left, right = [episode[key] for episode in training], [episode[key] for episode in validation]
        if len(set(left)) != len(left) or len(set(right)) != len(right) or set(left) & set(right):
            raise ValueError(f"Training and validation need unique, disjoint episode {key}s.")


def expert_partition(path, metadata, config, targets, args):
    eligible = split_episode_indices(metadata, "train", int(metadata["num_episodes"]))
    count = args.expert_train + args.expert_validation
    with h5py.File(path / "data.hdf5", "r") as h5:
        eligible = eligible[np.asarray(h5["lengths"])[eligible] >= args.context_length + max(args.horizons) - 1]
        if len(eligible) < count:
            raise ValueError(f"Need {count} sufficiently long expert TRAIN episodes, found {len(eligible)}.")
        selected = np.random.default_rng(args.data_seed).choice(eligible, count, replace=False)
        indices = observation_indices(metadata, config.state_head.fields)
        episodes = []
        for index in selected.tolist():
            length = int(h5["lengths"][index])
            episode = {
                "id": f"expert:{index}", "episode_index": index, "policy": "expert",
                "image": torch.from_numpy(read_image_window(h5["images"], index, 0, length + 1)),
                "state": torch.from_numpy(read_physical_state(h5, index, slice(0, length + 1), indices, targets)),
                "action": torch.from_numpy(np.asarray(h5["actions"][index, :length], dtype=np.float32)),
                "agent_steps": length, "return": float(h5["returns"][index]),
            }
            episode["sha256"] = tensor_digest({key: episode[key] for key in ("image", "state", "action")})
            episodes.append(episode)
    return episodes[:args.expert_train], episodes[args.expert_train:]


def collect_pool(config, model, metadata, path, args):
    training, validation = expert_partition(path, metadata, config, model.state_head.targets, args)
    seed = args.data_seed + 100_000 * SCENARIOS.index(str(config.scenario.name))
    count = 2 * (args.sim_train + args.sim_validation)
    forbidden = set(range(int(config.env.seed), int(config.env.seed) + int(config.env.env_num)))
    if metadata.get("episode_seed_rule") == "seed + episode_index":
        forbidden.update(range(int(metadata["seed"]), int(metadata["seed"]) + int(metadata["num_episodes"])))
    if forbidden.intersection(range(seed, seed + count)):
        raise ValueError("Supplemental simulator seeds overlap original collection or online training seeds.")
    progress = Progress("Shared simulator data", count)
    completed = 0
    for split, episodes, per_mode in (("train", training, args.sim_train),
                                      ("validation", validation, args.sim_validation)):
        for mode in ("zero", "random"):
            for _ in range(per_mode):
                episode = collect_episode(config, model, seed, mode)
                episode["id"] = f"simulator:{seed}"
                episode["sha256"] = tensor_digest({key: episode[key] for key in ("image", "state", "action")})
                episodes.append(episode)
                seed += 1
                completed += 1
                progress.update(completed, f"{split}/{mode}", force=completed == count)
    check_partition(training, validation)
    return training, validation


@torch.no_grad()
def encode_pool(model, episodes, chunk_size):
    features = []
    progress = Progress("Cache frozen features", len(episodes))
    with readout_mode(model):
        for index, episode in enumerate(episodes):
            encoded = torch.cat([model.encode({"image": chunk[None].to(model.device)})[0]
                                 for chunk in episode["image"].split(chunk_size)])
            if not torch.isfinite(encoded).all() or not torch.isfinite(episode["state"]).all():
                raise ValueError("Non-finite cached features or physical labels.")
            features.append(encoded.detach())
            progress.update(index + 1, force=index + 1 == len(episodes))
    return features


class FeatureBank:
    """Sample complete causal histories, balancing source first and episode second."""

    def __init__(self, episodes, features, history):
        self.history = history
        self.features = torch.cat(features)
        self.labels = torch.cat([episode["state"] for episode in episodes]).to(self.features.device)
        lengths = torch.tensor([len(value) for value in features])
        if (lengths < history).any():
            raise ValueError("Each fitting episode must contain a complete head history.")
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
        self.choices = lengths - history + 1
        self.groups = {source: torch.tensor([i for i, episode in enumerate(episodes) if episode["policy"] == source])
                       for source in ("expert", "zero", "random")}
        if any(not len(group) for group in self.groups.values()):
            raise ValueError("Fitting needs expert, zero-action, and random-action episodes.")

    def sample(self, count, generator):
        if count < 4 or count % 4:
            raise ValueError("Balanced head batches must be positive multiples of four.")
        selected, starts = [], []
        for source, size in (("expert", count // 2), ("zero", count // 4), ("random", count // 4)):
            group = self.groups[source]
            episode = group[torch.randint(len(group), (size,), generator=generator)]
            start = (torch.rand(size, generator=generator) * self.choices[episode]).long()
            selected.append(episode)
            starts.append(start)
        selected, starts = torch.cat(selected), torch.cat(starts)
        indices = (self.offsets[selected] + starts)[:, None] + torch.arange(self.history)
        indices = indices.to(self.features.device)
        return self.features[indices], self.labels[indices], {"episodes": selected.tolist(), "starts": starts.tolist()}


def fresh_head(original, config, labels, seed):
    """Fresh weights; optional training labels retain the older calibrated diagnostic."""
    projection = original.project[0].out_features
    tokens = original.readout[0].in_features // (original.history * projection)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = PhysicalStateHead(original.project[0].in_features, config,
                                 history=original.history, tokens=tokens).to(original.mean.device)
    scales = torch.ones_like(original.std)
    if labels is not None:
        floors = torch.full_like(original.std, .1)
        floors[original.targets.velocities] = 1
        floors[original.targets.trigonometric] = 1
        scales = torch.maximum(labels.std(0, unbiased=False), floors)
        scales[original.targets.trigonometric] = 1
    with torch.no_grad():
        head.mean.copy_(original.mean)
        head.std.copy_(original.std)
        # No affine migration: these are genuinely fresh weights in physical output units.
        head.output_scale.copy_(scales)
        head.loss_scale.copy_(scales)
    return head


def inside_goal(relation, tolerance, geometry):
    scaled = relation / tolerance
    return scaled.norm(dim=-1) <= 1 if geometry == "radial" else (scaled.abs() <= 1).all(-1)


def pool_coverage(episodes, model):
    targets = model.state_head.targets
    result = {}
    for source in ("expert", "zero", "random"):
        selected = [episode for episode in episodes if episode["policy"] == source]
        labels = torch.cat([episode["state"] for episode in selected])
        success = inside_goal(targets.goal_relation(labels), model.goal_tolerance.cpu(), model.goal_geometry)
        result[source] = {
            "episodes": len(selected), "states": len(labels), "goal_fraction": success.float().mean().item(),
            "minimum": dict(zip(targets.coordinates, labels.amin(0).tolist(), strict=True)),
            "maximum": dict(zip(targets.coordinates, labels.amax(0).tolist(), strict=True)),
            "std": dict(zip(targets.coordinates, labels.std(0, unbiased=False).tolist(), strict=True)),
        }
    return result


def physical_metrics(prediction, truth, head, tolerance, geometry):
    mse = (prediction - truth).square().mean(0)
    normalized = mse / head.std.square()
    if not torch.isfinite(normalized).all():
        raise ValueError("Non-finite physical errors.")
    relation, target = head.targets.goal_relation(prediction), head.targets.goal_relation(truth)
    difference = relation - target
    if head.targets.task == "dmc_cartpole_balance_sparse":
        difference[..., 1] = torch.atan2(difference[..., 1].sin(), difference[..., 1].cos())
    actual, predicted = inside_goal(target, tolerance, geometry), inside_goal(relation, tolerance, geometry)
    successes, failures = int(actual.sum()), int((~actual).sum())
    return {
        "samples": len(truth), "rmse": dict(zip(head.coordinates, mse.sqrt().tolist(), strict=True)),
        "normalized_mse": dict(zip(head.coordinates, normalized.tolist(), strict=True)),
        "mean_normalized_mse": normalized.mean().item(),
        "relation_rmse": difference.square().mean(0).sqrt().tolist(),
        "success_states": successes, "failure_states": failures,
        "false_success_count": int((predicted & ~actual).sum()),
        "missed_success_count": int((~predicted & actual).sum()),
        "false_success_rate": int((predicted & ~actual).sum()) / failures if failures else None,
        "missed_success_rate": int((~predicted & actual).sum()) / successes if successes else None,
    }


@torch.no_grad()
def cache_forecasts(model, episodes, features, args):
    """Compute native forecasts once; changing a detached head cannot change these tensors."""
    windows = sample_trajectory_windows(episodes, args.windows_per_episode,
                                        args.context_length + max(args.horizons), args.data_seed)
    context, history = args.context_length, model.state_head.history
    horizons = torch.tensor(args.horizons, device=model.device)
    offsets = torch.arange(1 - history, 1, device=model.device)
    cached = []
    progress = Progress("Cache fixed-action forecasts", len(windows))
    with readout_mode(model):
        for start in range(0, len(windows), args.eval_batch_size):
            batch = windows[start:start + args.eval_batch_size]
            observed = torch.stack([features[w.episode][w.start:w.start + context + max(args.horizons)] for w in batch])
            actions = torch.stack([episodes[w.episode]["action"][w.start:w.start + context + max(args.horizons) - 1]
                                   for w in batch]).to(model.device)
            truth = torch.stack([episodes[w.episode]["state"][w.start:w.start + context + max(args.horizons)]
                                 for w in batch]).to(model.device)
            prediction = model.rollout(observed[:, context - model.history_size:context],
                                       actions[:, context - model.history_size:context - 1],
                                       actions[:, None, context - 1:])[:, 0]
            prefix = observed[:, context - history + 1:context]
            forecast = torch.cat((prefix, prediction), dim=1)
            if not torch.isfinite(forecast).all():
                raise ValueError("Non-finite native forecast.")
            cached.append({
                "observed": observed[:, (context + horizons - 1)[:, None] + offsets],
                "forecast": forecast[:, (history + horizons - 2)[:, None] + offsets],
                "anchor": observed[:, context - history:context],
                "truth": truth[:, context + horizons - 1], "anchor_truth": truth[:, context - 1],
            })
            progress.update(start + len(batch), force=start + len(batch) == len(windows))
    return windows, {key: torch.cat([batch[key] for batch in cached]) for key in cached[0]}


@torch.no_grad()
def score_cache(head, cache, windows, horizons, model):
    truth = cache["truth"]
    count, horizon_count = truth.shape[:2]
    estimates = {source: torch.cat([head(batch.flatten(0, 1)).reshape(len(batch), horizon_count, -1)
                                   for batch in cache[source].split(32)]) for source in ("observed", "forecast")}
    estimates["decoded_persistence"] = head(cache["anchor"]).expand_as(truth)
    estimates["true_persistence"] = cache["anchor_truth"][:, None].expand_as(truth)
    groups = {"all": list(range(count))}
    groups.update({source: [i for i, window in enumerate(windows) if window.cohort == source]
                   for source in dict.fromkeys(window.cohort for window in windows)})
    return {source: {name: {str(h): physical_metrics(value[indices, i], truth[indices, i], head,
                                                    model.goal_tolerance, model.goal_geometry)
                            for i, h in enumerate(horizons)} for name, value in estimates.items()}
            for source, indices in groups.items()}


def fit_head(head, bank, updates, seed, log, *, fixed=None, label="Fresh head"):
    generator = torch.Generator().manual_seed(seed)
    progress = Progress(label, updates)
    for step in range(1, updates + 1):
        feature, target, _ = fixed if fixed is not None else bank.sample(head.samples_per_update, generator)
        metrics = head.fit(feature, target)
        loss = float(metrics["state/loss"])
        if not np.isfinite(loss):
            raise ValueError("Non-finite head fitting loss.")
        log.write(json.dumps({"phase": label, "update": step, "loss": loss,
                              "lr": head.optimizer.param_groups[0]["lr"], "examples": int(head.examples)}) + "\n")
        progress.update(step, f"loss={loss:.5g}", force=step == updates)


def fit_diagnostic(model, config, training, validation, args, output):
    started = time.monotonic()
    model.eval().requires_grad_(False)
    unchanged = tensor_digest(model.state_dict())
    original = model.state_head
    train_features = encode_pool(model, training, args.encode_batch_size)
    valid_features = encode_pool(model, validation, args.encode_batch_size)
    bank = FeatureBank(training, train_features, original.history)
    cached = {
        "train": cache_forecasts(model, training, train_features, args),
        "validation": cache_forecasts(model, validation, valid_features, args),
    }
    before = {split: score_cache(original, cache, windows, args.horizons, model)
              for split, (windows, cache) in cached.items()}
    small = fresh_head(original, config.state_head, bank.labels, args.fit_seed)
    small.samples_per_update = args.fit_batch_size
    fixed = bank.sample(args.fit_batch_size, torch.Generator().manual_seed(args.fit_seed + 1))
    with (output / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        fit_head(small, bank, args.fit_updates, args.fit_seed, log, fixed=fixed, label="Small-batch fit")
        with torch.no_grad():
            prediction, truth = small(fixed[0])[:, 0], fixed[1][:, -1]
            small_metrics = physical_metrics(prediction, truth, small, model.goal_tolerance, model.goal_geometry)
            scaled = ((prediction - truth).square().mean(0).sqrt() / small.loss_scale).tolist()
        small_result = {"updates": args.fit_updates, "examples": int(small.examples), "batch": fixed[2],
                        "physical": small_metrics, "scaled_rmse": dict(zip(small.coordinates, scaled, strict=True)),
                        "max_scaled_rmse_limit": args.fit_tolerance, "fitted": max(scaled) <= args.fit_tolerance}
        del small
        # The generalization experiment starts afresh, never from the memorized small batch.
        head = fresh_head(original, config.state_head, bank.labels, args.fit_seed)
        fit_head(head, bank, args.updates, args.fit_seed + 2, log)
    after = {split: score_cache(head, cache, windows, args.horizons, model)
             for split, (windows, cache) in cached.items()}
    if tensor_digest(model.state_dict()) != unchanged or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("The frozen checkpoint changed during head-only fitting.")
    torch.testing.assert_close(head.std, original.std, rtol=0, atol=0)
    return {
        "small_batch": small_result, "before": before, "after": after,
        "windows": {split: [asdict(window) for window in windows] for split, (windows, _) in cached.items()},
        "head_updates": int(head.updates), "head_examples": int(head.examples),
        "head_batch_size": head.samples_per_update, "head_lr": head.optimizer.param_groups[0]["lr"],
        "head_history": head.history, "head_parameters": sum(p.numel() for p in head.parameters()),
        "goal_geometry": model.goal_geometry, "goal_tolerance": model.goal_tolerance.tolist(),
        "conditioning": {name: dict(zip(head.coordinates, getattr(head, name).tolist(), strict=True))
                         for name in ("mean", "std", "output_scale", "loss_scale")},
        "native_unchanged": True, "native_sha256": unchanged, "native_updates": 0,
        "elapsed_seconds": time.monotonic() - started,
    }


def load_checkpoint(path, scenario, name, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["training_config"])
    config.device = args.device
    if (str(config.scenario.name), str(config.model_family), int(config.seed)) != (scenario, name, args.seed):
        raise ValueError("Checkpoint identity does not match the requested run.")
    if checkpoint.get("phase") != "expert" or checkpoint.get("expert_updates") != int(config.training.expert.updates):
        raise ValueError("Use a completed expert pretrained.pt checkpoint, not an online or partial checkpoint.")
    validate_checkpoint(checkpoint, config, training=False)
    if checkpoint.get("compatibility", {}).get("training_sha256") != checkpoint_compatibility(config)["training_sha256"]:
        raise ValueError("Checkpoint has inconsistent training recipe metadata.")
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    family = load_model_family(name)
    model = family.build_model(config)
    family.load_checkpoint(model, checkpoint, training=False)
    if not model.state_head.updates.item() or args.context_length < model.history_size:
        raise ValueError("Need a trained checkpoint head and a prefix covering native history.")
    if model.state_head.samples_per_update % 4 or model.state_head.samples_per_update < 4:
        raise ValueError("The checkpoint's head batch size must be a positive multiple of four.")
    return config, model, checkpoint


def write_reports(output, results, data, args):
    report = {"diagnostic_version": 1, "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()
                  + Path(__file__).with_name("online_validation.py").read_bytes()).hexdigest(),
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "checkpoint_writes": False, "native_updates": 0, "policy_optimization": False,
              "benchmark_heldout_used": False, "validation_fitting": False,
              "head_training_mixture": {"expert": .5, "zero": .25, "random": .25},
              "data": data, "results": results}
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Fresh readout diagnostic | frozen native models | no online updates or checkpoint writes",
             "Run | Small-batch fit | Expert validation nMSE h1 | Simulator validation nMSE h1 | False/missed goals | Time"]
    for result in results:
        name = f"{result['scenario']}/{result['model']}"
        if result["status"] == "FAIL":
            lines.append(f"FAIL | {name} | {result['error']}")
            continue
        def metric(stage, source, field="mean_normalized_mse", prediction="observed", horizon="1", result=result):
            values = result[stage]["validation"]
            if source == "expert":
                return values[source][prediction][horizon][field]
            rows = [values[group][prediction][horizon] for group in ("zero", "random")]
            return sum(row[field] * row["samples"] for row in rows) / sum(row["samples"] for row in rows)
        def rate(stage, numerator, denominator, result=result):
            rows = [result[stage]["validation"][group]["observed"]["1"] for group in ("expert", "zero", "random")]
            total = sum(row[denominator] for row in rows)
            return f"{sum(row[numerator] for row in rows) / total:.1%}" if total else "no coverage"
        expert = "->".join(f"{metric(stage, 'expert', 'mean_normalized_mse'):.3g}" for stage in ("before", "after"))
        simulator = "->".join(f"{metric(stage, 'simulator', 'mean_normalized_mse'):.3g}" for stage in ("before", "after"))
        rates = " / ".join("->".join(rate(stage, num, den) for stage in ("before", "after"))
                           for num, den in (("false_success_count", "failure_states"), ("missed_success_count", "success_states")))
        fitted = "FIT" if result["small_batch"]["fitted"] else "NOT FIT"
        lines.append(f"COMPLETE | {name} | {fitted} | {expert} | {simulator} | {rates} | {duration(result['elapsed_seconds'])}")
        forecasts = []
        for source in ("expert", "simulator"):
            pair = " -> ".join("/".join(f"{metric(stage, source, prediction='forecast', horizon=h):.3g}"
                                        for h in ("5", "100")) for stage in ("before", "after"))
            forecasts.append(f"{source}={pair}")
        lines.append("  Forecast nMSE h5/h100 | " + " | ".join(forecasts))
        relations = []
        for stage in ("before", "after"):
            rows = [result[stage]["validation"][group]["observed"]["1"] for group in ("zero", "random")]
            rmse = [(sum(row["relation_rmse"][i]**2 * row["samples"] for row in rows)
                     / sum(row["samples"] for row in rows))**.5 for i in range(len(rows[0]["relation_rmse"]))]
            relations.append("[" + ", ".join(f"{value:.4g}" for value in rmse) + "]")
        lines.append("  Simulator observed goal-relation RMSE (original units) | " + " -> ".join(relations))
    lines.extend([
        "COMPLETE means the diagnostic executed, not that the model is repaired. Small-batch FIT is only a fitting check.",
        "nMSE keeps the original expert scales. Inspect coordinate/angle errors and forecast h5 in physical_errors.csv/report.json.",
        "Expert validation episodes were seen during native pretraining, but never used to fit these fresh heads.",
        "Simulator train/validation episodes are disjoint and identical across models. Zero/random actions do not guarantee recovery coverage.",
        "This does not validate online stability or planning quality; no production settings or weights were changed.",
    ])
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    with (output / "physical_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "model", "stage", "split", "cohort", "source", "horizon", "coordinate", "rmse", "normalized_mse"])
        for result in results:
            if result["status"] == "FAIL":
                continue
            for stage in ("before", "after"):
                for split, cohorts in result[stage].items():
                    for cohort, sources in cohorts.items():
                        for source, horizons in sources.items():
                            for horizon, values in horizons.items():
                                for coordinate, rmse in values["rmse"].items():
                                    writer.writerow([result["scenario"], result["model"], stage, split, cohort, source,
                                                     horizon, coordinate, rmse, values["normalized_mse"][coordinate]])
    return summary


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/dmc_vision_10k"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["cartpole_balance_sparse"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-seed", type=int, default=8_000_000)
    parser.add_argument("--fit-seed", type=int, default=9_000_000)
    parser.add_argument("--expert-train", type=int, default=32)
    parser.add_argument("--expert-validation", type=int, default=8)
    parser.add_argument("--sim-train", type=int, default=8, help="Complete training episodes per zero/random policy.")
    parser.add_argument("--sim-validation", type=int, default=4, help="Complete validation episodes per zero/random policy.")
    parser.add_argument("--updates", type=int, default=5000, help="Fresh head updates; checkpoint head batch size and expert LR.")
    parser.add_argument("--fit-updates", type=int, default=1000, help="Separate fresh-head small-batch fitting check.")
    parser.add_argument("--fit-batch-size", type=int, default=32)
    parser.add_argument("--fit-tolerance", type=float, default=.05, help="Small-batch maximum RMSE / fixed training scale.")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--windows-per-episode", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / f"fresh_readout_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    counts = (args.expert_train, args.expert_validation, args.sim_train, args.sim_validation, args.updates,
              args.fit_updates, args.fit_batch_size, args.context_length, args.windows_per_episode,
              args.eval_batch_size, args.encode_batch_size)
    if min(counts) < 1 or args.fit_batch_size % 4 or args.fit_batch_size < 4:
        parser.error("Counts must be positive; small fitting batch size must be a multiple of four.")
    if not np.isfinite(args.fit_tolerance) or args.fit_tolerance <= 0 or min(args.seed, args.data_seed, args.fit_seed) < 0:
        parser.error("Use nonnegative seeds and a finite positive fitting tolerance.")
    args.horizons = [1, 5, 10, 100]
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    results, data = [], {}
    print("Fresh heads | native checkpoints read-only | shared simulator data | no planner or online training", flush=True)
    for scenario in dict.fromkeys(args.scenarios):
        pool, reference = None, None
        for name in dict.fromkeys(args.models):
            output = args.output / scenario / name
            output.mkdir(parents=True)
            path = args.run_root / scenario / name / "default" / f"seed_{args.seed}" / "pretrained.pt"
            result = {"scenario": scenario, "model": name, "checkpoint": str(path.resolve()), "status": "FAIL"}
            model = checkpoint = None
            print(f"START | {scenario}/{name}", flush=True)
            try:
                config, model, checkpoint = load_checkpoint(path, scenario, name, args)
                dataset_path = args.dataset_root / str(config.scenario.dataset)
                if pool is None:
                    metadata = validate_dataset(dataset_path, config, splits=("train",))
                if checkpoint.get("dataset_identity") != dataset_identity(metadata):
                    raise ValueError("Checkpoint and expert dataset identities differ.")
                signature = (str(dataset_path.resolve()), model.state_head.coordinates,
                             str(config.env.task), list(config.env.size), int(config.env.time_limit),
                             int(config.env.action_repeat), OmegaConf.to_container(config.state_head.fields),
                             model.goal_geometry, model.goal_tolerance.tolist())
                if reference is not None and signature != reference:
                    raise ValueError("Compared models must share environment, expert dataset, and physical targets.")
                if pool is None:
                    pool = collect_pool(config, model, metadata, dataset_path, args)
                    reference = signature
                    data[scenario] = {"dataset_identity": dataset_identity(metadata),
                                      "train": episode_metadata(pool[0]), "validation": episode_metadata(pool[1]),
                                      "coverage": {split: pool_coverage(episodes, model)
                                                   for split, episodes in zip(("train", "validation"), pool, strict=True)}}
                result.update(checkpoint_id=checkpoint["checkpoint_id"], source_compatibility=checkpoint["compatibility"])
                result.update(fit_diagnostic(model, config, *pool, args, output))
                result["status"] = "COMPLETE"
            except Exception as error:  # noqa: BLE001 - Retain completed diagnostics when another model fails.
                result["error"] = f"{type(error).__name__}: {error}"
                with (output / "error.log").open("w", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
                print(f"FAIL | {scenario}/{name} | {result['error']}", flush=True)
            finally:
                results.append(result)
                summary = write_reports(args.output, results, data, args)
                del model, checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"{result['status']} | {scenario}/{name}", flush=True)
    print(summary)
    print(f"Reports | {args.output.resolve()}")
    raise SystemExit(1 if any(result["status"] == "FAIL" for result in results) else 0)


if __name__ == "__main__":
    main()
