"""Compare planner adaptation on saved replay, without collection or checkpoint writes."""

import argparse
import copy
import csv
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import tools
from buffer import SequenceBuffer
from dmc_expert.storage import dataset_identity, validate_dataset
from scripts.diagnose_fresh_readout import (
    cache_forecasts,
    check_partition,
    encode_pool,
    expert_partition,
    fresh_head,
    load_checkpoint,
    score_cache,
    tensor_digest,
)
from scripts.diagnose_planning_models import FAMILIES, SCENARIOS
from scripts.online_validation import episode_metadata
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import (
    implementation_sha256,
    upgrade_readout_config,
    validate_checkpoint,
)

MODES = ("native", "frozen_bn", "frozen_encoder")


def replay_partition(payload, validation_count, context, horizon, seed):
    saved = payload["replay_state"]["replay"]
    episodes = []
    for index, rows in enumerate((*saved.get("completed", ()), *saved.get("truncated", ()))):
        if len(rows) < context + horizon:
            continue
        episode = {
            "id": f"replay:{index}", "policy": "replay", "agent_steps": len(rows) - 1,
            "image": rows["image"].cpu(), "state": rows["physical_state"].float().cpu(),
            # Replay stores (observation_t, action_t), without the final successor image.
            "action": rows["action"][:-1].float().cpu(),
        }
        episode["sha256"] = tensor_digest({key: episode[key] for key in ("image", "state", "action")})
        episodes.append(episode)
    if len(episodes) <= validation_count:
        raise ValueError("Saved replay needs separate training and validation episodes.")
    order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(seed)).tolist()
    validation = [episodes[i] for i in order[:validation_count]]
    training = [episodes[i] for i in order[validation_count:]]
    check_partition(training, validation)
    return training, validation


class FixedBatches:
    """Use the production window sampler; save indices instead of duplicating images."""

    def __init__(self, episodes, config, seed):
        settings = copy.deepcopy(config.replay)
        settings.device = settings.storage_device = "cpu"
        self.sampler = SequenceBuffer(settings)
        self.length = int(settings.sequence_length)
        self.episodes = episodes
        for episode in episodes:
            count = len(episode["image"])
            self.sampler.completed.append(TensorDict({
                "image": episode["image"], "physical_state": episode["state"],
                "action": torch.cat((episode["action"], torch.zeros_like(episode["action"][:1]))),
                "reward": torch.zeros(count, 1), "terminal": torch.zeros(count, 1),
            }, batch_size=[count]))
        self.sampler._generator.manual_seed(seed)
        self.indices = {id(episode): i for i, episode in enumerate(self.sampler.completed)}
        if not self.sampler.ready():
            raise ValueError(f"Need at least {settings.episodes_per_batch} usable training episodes per source.")

    def draw(self):
        return [[self.indices[id(episode)], list(starts)] for episode, starts in self.sampler.sample_groups(
            self.sampler.batch_size, self.length, self.sampler.episodes_per_batch)]

    def batch(self, plan):
        rows = torch.stack([self.sampler.completed[index][start:start + self.length]
                            for index, starts in plan for start in starts])
        return ({key: rows[key] for key in ("image", "physical_state")},
                rows["action"][:, :-1], rows["reward"][:, :-1], rows["terminal"][:, :-1])

    def features(self, plan, encoded):
        return (torch.stack([encoded[index][start:start + self.length] for index, starts in plan for start in starts]),
                torch.stack([self.episodes[index]["state"][start:start + self.length]
                             for index, starts in plan for start in starts]).to(encoded[0].device))


def representation_state(model):
    return {key: value for key, value in model.state_dict().items() if key.startswith(("encoder.", "projector."))}


def batchnorm_state(model):
    return {key: value for key, value in model.state_dict().items()
            if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}


@tools.preserve_rng_state
def measure(model, episodes, args):
    features = encode_pool(model, episodes, args.encode_batch_size)
    windows, cache = cache_forecasts(model, episodes, features, args)
    scores = score_cache(model.state_head, cache, windows, args.horizons, model)
    return scores, cache["anchor"].detach().cpu(), windows


def run_ablation(config, model, checkpoint, training, validation, args, output):
    """All modes share native weights/moments, a fitted fresh head, and exact batches."""
    if float(config.training.online.expert_fraction) != 0 or model.state_head.expert_fraction != .5:
        raise ValueError("This isolation test requires native online-only batches and 50% expert head retention.")
    if args.updates > int(config.training.online.updates):
        raise ValueError("Diagnostic updates must fit within the original online schedule.")
    expert, replay = training
    online = FixedBatches(replay, config, args.data_seed)
    retained = FixedBatches(expert, config, args.data_seed + 1)
    plans = {phase: [{"online": online.draw(), "expert": retained.draw()} for _ in range(count)]
             for phase, count in (("head_fit", args.head_updates), ("native_updates", args.updates))}
    encoded = json.dumps(plans, separators=(",", ":"))
    (output / "batches.json").write_text(encoded + "\n", encoding="utf-8")
    scores, _, windows = measure(model, validation, args)
    result = {
        "original_head": scores, "batch_plan_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "windows": [asdict(window) for window in windows], "trials": [],
        "native_batch_size": int(config.replay.batch_size), "sequence_length": online.length,
        "source_episodes_per_update": int(config.replay.episodes_per_batch),
        "head_updates_before_ablation": args.head_updates,
        "head_fit_lr": float(config.state_head.lr), "head_online_lr": model.state_head.online_lr,
        "head_online_warmup": model.state_head.online_warmup,
        "native_expert_fraction": 0.0, "head_expert_fraction": .5,
        "coordinates": model.state_head.coordinates, "goal_geometry": model.goal_geometry,
        "goal_tolerance": model.goal_tolerance.tolist(),
        "native_schedule_updates": int(config.training.online.updates),
        "native_optimizer_moments": "reused from expert checkpoint; online schedule restarted",
        "native_grad_clip": model.grad_clip,
    }
    native_digest = tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith("state_head.")})
    online_features = encode_pool(model, replay, args.encode_batch_size)
    expert_features = encode_pool(model, expert, args.encode_batch_size)
    model.state_head = fresh_head(model.state_head, config.state_head, None, args.fit_seed)
    progress = Progress("Fit shared fresh head", args.head_updates)
    with (output / "head_fit.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for step, plan in enumerate(plans["head_fit"], 1):
            x, y = online.features(plan["online"], online_features)
            ex, ey = retained.features(plan["expert"], expert_features)
            metrics = model.state_head.fit(torch.cat((x, ex)), torch.cat((y, ey)))
            log.write(json.dumps({"update": step, "loss": float(metrics["state/loss"])}, allow_nan=False) + "\n")
            progress.update(step, force=step == args.head_updates)
    del online_features, expert_features
    if tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith("state_head.")}) != native_digest:
        raise RuntimeError("Fresh-head fitting changed the native model.")
    head_weights = copy.deepcopy(model.state_head.state_dict())
    result["conditioning"] = {key: getattr(model.state_head, key).tolist()
                              for key in ("mean", "std", "output_scale", "loss_scale")}
    result["head_examples_per_update"] = model.state_head.samples_per_update
    baseline, reference, _ = measure(model, validation, args)
    result["fresh_head_baseline"] = baseline
    family = load_model_family(str(config.model_family))
    for mode in MODES:
        started = time.monotonic()
        model.set_adaptation_mode("native")
        family.load_checkpoint(model, copy.deepcopy(checkpoint), training=True)
        model.state_head.load_state_dict(head_weights)
        model._gradient_updates = model._clipped_updates = 0
        model.set_adaptation_mode(mode)
        model.train()
        expert_source = {"batch": None}
        model.state_head.configure_online(lambda source=expert_source: model.readout_features(source["batch"]))
        if hasattr(model, "configure_online"):
            model.configure_online(int(config.training.online.updates), resumed=False)
        bn_before = tensor_digest(batchnorm_state(model))
        encoder_before = tensor_digest(representation_state(model))
        trial = {"mode": mode, "status": "COMPLETE", "snapshots": []}
        result["trials"].append(trial)
        progress = Progress(f"Adapt {mode}", args.updates)
        norms, clipped, losses = [], 0, []
        try:
            with (output / f"{mode}.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                for step, plan in enumerate(plans["native_updates"], 1):
                    # Repeated data and stochastic seeds, regardless of evaluation cadence or trial order.
                    torch.manual_seed(args.fit_seed + step)
                    batch, expert_batch = online.batch(plan["online"]), retained.batch(plan["expert"])
                    expert_source["batch"] = expert_batch
                    metrics = model.update(batch)
                    values = {key: float(value) for key, value in metrics.items()}
                    log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
                    norms.append(values["grad_norm"])
                    clipped += int(values["grad_clipped"])
                    losses.append(values["loss"])
                    progress.update(step, f"loss={values['loss']:.4g}", force=step == args.updates)
                    if step % args.eval_every == 0 or step == args.updates:
                        scores, anchors, checked = measure(model, validation, args)
                        if checked != windows:
                            raise RuntimeError("Validation windows changed during adaptation.")
                        drift = (anchors - reference).square().mean().sqrt().item()
                        trial["snapshots"].append({"update": step, "scores": scores, "latent_drift_rms": drift})
            trial.update(updates=len(losses), loss_first=losses[0], loss_last=losses[-1],
                         preclip_grad_norm_mean=sum(norms) / len(norms), preclip_grad_norm_max=max(norms),
                         clipped_fraction=clipped / len(losses))
            bn_same = tensor_digest(batchnorm_state(model)) == bn_before
            encoder_same = tensor_digest(representation_state(model)) == encoder_before
            trial.update(batchnorm_unchanged=bn_same, encoder_and_projector_unchanged=encoder_same)
            if mode != "native" and not bn_same or mode == "frozen_encoder" and not encoder_same:
                raise RuntimeError("An ablation changed frozen parameters or running statistics.")
        except Exception as error:  # noqa: BLE001 - Preserve completed trials and try the remaining controls.
            trial.update(status="FAIL", error=f"{type(error).__name__}: {error}", updates=len(losses))
            (output / f"{mode}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            model.state_head._expert_source = None
        trial["elapsed_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def write_report(output, results, args):
    report = {
        "diagnostic_version": 1, "implementation_sha256": implementation_sha256(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_writes": False, "collection": False, "planner_optimization": False,
        "benchmark_heldout_used": False, "validation_fitting": False, "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Fixed replay | identical batches across ablations | fresh heads | checkpoints read-only",
             "Run / mode | Updates | Expert observed nMSE | Replay observed / forecast h5 nMSE | Clip % | Latent drift | Time"]
    rows = []
    for result in results:
        name = f"{result['scenario']}/{result['model']}"
        if "error" in result:
            lines.append(f"FAIL | {name} | {result['error']}")
            continue
        before = result["fresh_head_baseline"]
        for trial in result["trials"]:
            if trial["status"] == "FAIL":
                lines.append(f"FAIL | {name}/{trial['mode']} | {trial['error']}")
                continue
            last = trial["snapshots"][-1]
            def change(cohort, source, horizon, before=before, after=last["scores"]):
                return "->".join(f"{scores[cohort][source][horizon]['mean_normalized_mse']:.3g}"
                                 for scores in (before, after))
            lines.append(f"COMPLETE | {name}/{trial['mode']} | {trial['updates']} | "
                         f"{change('expert', 'observed', '1')} | {change('replay', 'observed', '1')} / "
                         f"{change('replay', 'forecast', '5')} | {100 * trial['clipped_fraction']:.1f} | "
                         f"{last['latent_drift_rms']:.3g} | {duration(trial['elapsed_seconds'])}")
            for snapshot in [{"update": 0, "scores": before}, *trial["snapshots"]]:
                for cohort, sources in snapshot["scores"].items():
                    for source, horizons in sources.items():
                        for horizon, values in horizons.items():
                            for coordinate, rmse in values["rmse"].items():
                                rows.append([name, trial["mode"], snapshot["update"], cohort, source, horizon,
                                             coordinate, rmse, values["normalized_mse"][coordinate]])
    lines.extend([
        "COMPLETE means finite execution, not repaired models. Inspect physical_errors.csv and goal/angle errors in report.json.",
        "Native budgets/losses unchanged; this is a short schedule prefix on static saved replay, not a policy evaluation.",
        "Head fitting uses only training episodes; expert validation was seen during original native pretraining.",
        "BatchNorm/encoder freezes are diagnostic only. No production freeze or multi-step loss is enabled.",
    ])
    with (output / "physical_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "mode", "update", "cohort", "source", "horizon", "coordinate", "rmse", "normalized_mse"])
        writer.writerows(rows)
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
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
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--head-updates", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--expert-train", type=int, default=32)
    parser.add_argument("--expert-validation", type=int, default=8)
    parser.add_argument("--replay-validation", type=int, default=8)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / f"fixed_replay_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    if min(args.updates, args.head_updates, args.eval_every, args.expert_train, args.expert_validation,
           args.replay_validation, args.windows_per_episode, args.encode_batch_size, args.eval_batch_size) < 1:
        parser.error("Counts must be positive.")
    if min(args.seed, args.data_seed, args.fit_seed) < 0:
        parser.error("Seeds must be nonnegative.")
    args.context_length, args.horizons = 3, [1, 5]
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for scenario in dict.fromkeys(args.scenarios):
        for name in dict.fromkeys(args.models):
            output = args.output / scenario / name
            output.mkdir(parents=True)
            root = args.run_root / scenario / name / "default" / f"seed_{args.seed}"
            result = {"scenario": scenario, "model": name, "checkpoint": str(root / "pretrained.pt"),
                      "replay_checkpoint": str(root / "latest.pt")}
            model = checkpoint = saved = training = validation = None
            print(f"START | fixed-replay {scenario}/{name}", flush=True)
            try:
                config, model, checkpoint = load_checkpoint(root / "pretrained.pt", scenario, name, args)
                saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
                replay_config = OmegaConf.create(saved["training_config"])
                validate_checkpoint(saved, replay_config, training=False)
                identity = lambda cfg: (str(cfg.scenario.name), str(cfg.model_family), int(cfg.seed))
                if identity(replay_config) != identity(config) or saved.get("dataset_identity") != checkpoint["dataset_identity"]:
                    raise ValueError("Replay checkpoint does not match the expert checkpoint's run and dataset.")
                if saved.get("phase") != "online" or "replay_state" not in saved:
                    raise ValueError("latest.pt must contain saved online replay; no collection is performed.")
                upgrade_readout_config(config)
                if name == "temporal_straightening":
                    config.jepa_model.optim.grad_clip = model.grad_clip = 1.0
                path = args.dataset_root / str(config.scenario.dataset)
                metadata = validate_dataset(path, config, splits=("train",))
                if dataset_identity(metadata) != checkpoint["dataset_identity"]:
                    raise ValueError("Expert dataset identity differs from the checkpoint.")
                expert_train, expert_valid = expert_partition(path, metadata, config, model.state_head.targets, args)
                replay_train, replay_valid = replay_partition(saved, args.replay_validation, args.context_length,
                                                              max(args.horizons), args.data_seed)
                training, validation = (expert_train, replay_train), expert_valid + replay_valid
                check_partition(expert_train + replay_train, validation)
                result.update(checkpoint_id=checkpoint["checkpoint_id"], replay_checkpoint_id=saved["checkpoint_id"],
                              source_compatibility=checkpoint["compatibility"],
                              data={"train": episode_metadata(expert_train + replay_train),
                                    "validation": episode_metadata(validation)})
                del saved
                saved = None
                result.update(run_ablation(config, model, checkpoint, training, validation, args, output))
            except Exception as error:  # noqa: BLE001 - Keep the other model's diagnostics.
                result["error"] = f"{type(error).__name__}: {error}"
                (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                results.append(result)
                summary = write_report(args.output, results, args)
                del model, checkpoint, saved, training, validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(summary)
    print(f"Reports | {args.output.resolve()}")
    raise SystemExit(int(any("error" in r or any(t["status"] == "FAIL" for t in r["trials"]) for r in results)))


if __name__ == "__main__":
    main()
