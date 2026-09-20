"""Fresh, CPU-sized latent-planner training on real DMC trajectories, without checkpoints."""

import argparse
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import tools
from envs import close_envs, make_envs
from models.shared.physical_state import format_physical_rmse
from scripts.diagnose_fixed_replay import FixedBatches, measure
from scripts.online_validation import collect_episode, episode_metadata, TrajectoryDataset
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256, validate_training_recipe

FAMILIES = ("leworldmodel", "temporal_straightening")


def tiny_config(name, args):
    # Change size and diagnostic budgets only: retain native losses, LRs and optimizer types.
    overrides = [
        f"scenario={args.scenario}", f"device={args.device}", f"seed={args.seed}",
        "deterministic_run=true", "jepa_model.use_amp=false",
        "env.env_num=2", f"env.time_limit={2 * args.episode_steps}",
        "replay.batch_size=8", "replay.sequence_length=4", "replay.episodes_per_batch=2",
        f"replay.max_size={4 * args.episode_steps}",
        "state_head.projection_dim=4", "state_head.hidden_dim=16", "state_head.samples_per_update=16",
        "jepa_model.encoder.embedding_dim=16", "jepa_model.encoder.vision.base_channels=2",
        "jepa_model.encoder.vision.layers=1", "jepa_model.encoder.vision.heads=1",
        "jepa_model.encoder.vision.mlp_dim=32", "jepa_model.projector_dim=32",
        "jepa_model.predictor.layers=1", "jepa_model.predictor.heads=1",
        "jepa_model.predictor.dim_head=16", "jepa_model.predictor.mlp_dim=32",
        "jepa_model.predictor.action_embedding_dim=4",
        "jepa_model.planner.horizon=5", "jepa_model.planner.samples=8",
        "jepa_model.planner.elites=2", "jepa_model.planner.iterations=2",
        f"training.expert.updates={args.offline_updates}",
        f"training.online.updates={args.online_updates}",
        f"training.online.steps={4 * (16 + args.online_updates)}",
        "training.online.warmup_transitions=32",
    ]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name=f"{name}_dmc_vision", overrides=overrides)
    OmegaConf.resolve(config)
    validate_training_recipe(config)
    return config


def parameter_copies(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def changed_parameters(model, before):
    changed = {name for name, value in model.named_parameters() if not torch.equal(before[name], value.detach().cpu())}
    result = {"native_changed": any(not name.startswith("state_head.") for name in changed),
              "readout_changed": any(name.startswith("state_head.") for name in changed)}
    if not all(result.values()):
        raise RuntimeError(f"An optimizer phase did not change both native and readout parameters: {result}")
    return result


def native_control(model, operation):
    """A tripwire: physical decoding is allowed for fitting/evaluation, never action selection."""
    with patch.object(model.state_head, "forward", side_effect=RuntimeError("Planner used the physical readout")):
        return operation()


@tools.preserve_rng_state
def snapshot(model, validation, args, phase, updates):
    diagnostics = SimpleNamespace(context_length=3, horizons=[1, 5, 20], windows_per_episode=4,
                                  data_seed=args.seed + 1000, encode_batch_size=16, eval_batch_size=8)
    scores, anchor, windows = measure(model, validation, diagnostics)
    return {"phase": phase, "online_updates": updates, "scores": scores,
            "latent_std": anchor.flatten(2).std(dim=(0, 1), unbiased=False).mean().item(),
            "windows": [{"episode": w.episode, "start": w.start} for w in windows]}


def train_one(config, model, training, validation, args, output, result):
    started = time.monotonic()
    offline = FixedBatches(training, config, args.seed + 10)
    retained = FixedBatches(training, config, args.seed + 20)
    labels = torch.cat([episode["state"] for episode in training])
    model.state_head.set_stats(labels.mean(0), labels.std(0, unbiased=False))
    head_stats = {key: getattr(model.state_head, key).clone() for key in ("mean", "std", "output_scale", "loss_scale")}
    result.update(config=OmegaConf.to_container(config, resolve=True),
                  parameters=sum(p.numel() for p in model.parameters()),
                  head_parameters=sum(p.numel() for p in model.state_head.parameters()),
                  evaluation_std=model.state_head.std.tolist(), snapshots=[], phases={}, policy={})
    result["snapshots"].append(snapshot(model, validation, args, "initial", 0))
    print(f"Model | {config.model_family} | parameters={result['parameters']:,} | batch=8x4 | FP32", flush=True)

    def evaluate_policy(phase):
        episodes = [native_control(model, lambda seed=args.seed + 2000 + i:
                                   collect_episode(config, model, seed, "policy")) for i in range(2)]
        result["policy"][phase] = episode_metadata(episodes)

    def update(log, phase, step, operation):
        before = int(model.state_head.updates)
        model.train()
        metrics = {key: float(value) for key, value in operation().items()}
        if not metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"{phase}/{step}: non-finite training metrics")
        if int(model.state_head.updates) != before + 1:
            raise RuntimeError(f"{phase}/{step}: physical head did not update")
        log.write(json.dumps({"phase": phase, "update": step, **metrics}, allow_nan=False) + "\n")
        return metrics

    with (output / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        before = parameter_copies(model)
        phase_start = time.monotonic()
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(args.offline_updates)
        progress = Progress(f"{config.model_family} offline", args.offline_updates)
        losses = []
        for step in range(1, args.offline_updates + 1):
            metrics = update(log, "offline", step, lambda: model.update(offline.batch(offline.draw())))
            losses.append(metrics["loss"])
            progress.update(step, f"loss={metrics['loss']:.4g} state={metrics['state/loss']:.4g}",
                            force=step == args.offline_updates)
        result["phases"]["offline"] = {"updates": len(losses), "seconds": time.monotonic() - phase_start,
                                       "loss_first": losses[0], "loss_last": losses[-1],
                                       **changed_parameters(model, before)}
        result["snapshots"].append(snapshot(model, validation, args, "offline", 0))
        evaluate_policy("offline")

        envs = None
        try:
            envs = make_envs(config.env, seed=args.seed + 3000)
            session = load_model_family(config.model_family).OnlineSession(config, model, envs)
            session.start()
            if session.replay.count() != 0:
                raise RuntimeError("Online replay must start empty")
            before = parameter_copies(model)
            if hasattr(model, "configure_online"):
                model.configure_online(args.online_updates, resumed=False)
            model.state_head.configure_online(lambda: model.readout_features(retained.batch(retained.draw())))
            progress = Progress(f"{config.model_family} online", args.online_updates)
            phase_start = time.monotonic()
            episodes, losses = [], []
            # Two live environments supply 32 fresh transitions before the first update.
            for call in range(16 + args.online_updates):
                _, completed = native_control(model, session.collect)
                episodes.extend({"return": r, "agent_steps": length} for r, length in completed)
                if call < 16:
                    continue
                if not session.replay.ready():
                    raise RuntimeError("Fresh replay is not ready after warmup")
                step = call - 15
                metrics = update(log, "online", step, lambda: session.update(1))
                losses.append(metrics["loss"])
                progress.update(step, f"loss={metrics['loss']:.4g} state={metrics['state/loss']:.4g}",
                                force=step == args.online_updates)
                if step % 64 == 0 or step == args.online_updates:
                    result["snapshots"].append(snapshot(model, validation, args, "online", step))
            result["phases"]["online"] = {"updates": len(losses), "seconds": time.monotonic() - phase_start,
                                          "agent_transitions": 2 * (16 + args.online_updates),
                                          "replay_rows": session.replay.count(), "episodes": episodes,
                                          "loss_first": losses[0], "loss_last": losses[-1],
                                          **changed_parameters(model, before)}
            if "goal_image" in session.replay._obs_keys:
                raise RuntimeError("Goal observations leaked into replay")
        finally:
            close_envs(envs)
            model.state_head._online = False
            model.state_head._expert_source = None
    evaluate_policy("online")
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite final model state: {name}")
    for key, before in head_stats.items():
        torch.testing.assert_close(getattr(model.state_head, key), before, rtol=0, atol=0)
    if int(model.state_head.updates) != args.offline_updates + args.online_updates:
        raise RuntimeError("Total readout update count differs from requested training budget")
    if int(model.state_head.online_updates) != args.online_updates:
        raise RuntimeError("Online readout update count differs from requested training budget")
    result.update(status="PASS", seconds=time.monotonic() - started,
                  checks="finite metrics/weights; both optimizers update; fresh replay; no head in control; fixed scales")


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Tiny planner training | fresh initialization | no checkpoints | PASS = execution checks only",
             "Data: real random-action simulator episodes, NOT expert data; validation episodes/seeds are disjoint.",
             "Model | Params | Offline/online updates | Time"]
    for result in report["runs"]:
        if result["status"] != "PASS":
            lines.append(f"FAIL | {result['model']} | {result.get('error', 'incomplete')}")
            continue
        initial, offline, final = result["snapshots"][0], result["snapshots"][1], result["snapshots"][-1]
        def score(snap, source, horizon):
            return snap["scores"]["all"][source][str(horizon)]
        phases = result["phases"]
        lines.append(f"PASS | {result['model']} | {result['parameters']:,} | "
                     f"{phases['offline']['updates']}/{phases['online']['updates']} | {duration(result['seconds'])}")
        lines.append("  Initial observed h1 RMSE | " + format_physical_rmse(score(initial, "observed", 1)))
        for source, horizon in (("observed", 1), ("forecast", 20)):
            lines.append(f"  {source} h{horizon} RMSE offline->online | " + format_physical_rmse(
                score(final, source, horizon), before=score(offline, source, horizon),
                baseline=score(final, "true_persistence", horizon) if source == "forecast" else None))
    lines.extend(["Angles are wrapped radians; hold=true-state persistence (scoring only). nMSE remains secondary in JSON with fixed training scales.",
                  "Images remain 64x64; native losses/LRs are unchanged. Batches, widths, depth, planner and schedules are tiny.",
                  "Short policy episodes are diagnostic, not task-success benchmarks. This does not validate full-size learning.",
                  "Per-coordinate RMSE, persistence baselines, exact windows, policy returns and resolved settings: report.json."])
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--scenario", choices=("cartpole_balance_sparse", "reacher", "ball_in_cup"),
                        default="cartpole_balance_sparse")
    parser.add_argument("--offline-updates", type=int, default=128)
    parser.add_argument("--online-updates", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=64, help="Agent steps per diagnostic episode.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("tiny_planners_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.offline_updates < 2 or args.online_updates < 2 or args.episode_steps < 24:
        parser.error("Need >=2 offline/online updates and >=24 agent steps for held-out h20 forecasts.")
    if len(set(args.models)) != len(args.models) or args.seed < 0:
        parser.error("Models must be unique and seed nonnegative.")
    return args


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {"diagnostic_version": 1, "implementation_sha256": implementation_sha256(),
              "scenario": args.scenario, "seed": args.seed, "checkpoint_writes": False, "runs": []}
    training = validation = None
    for name in args.models:
        result = {"model": name, "status": "FAIL"}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        try:
            config = tiny_config(name, args)
            tools.configure_randomness(args.seed, True)
            model = load_model_family(name).build_model(config)
            if training is None:
                print("Data | collecting 4 random-action training + 2 held-out episodes once", flush=True)
                training = [collect_episode(config, model, args.seed + 100 + i, "random") for i in range(4)]
                validation = [collect_episode(config, model, args.seed + 1000 + i, "random") for i in range(2)]
                TrajectoryDataset(validation, forbidden_seeds=[e["seed"] for e in training])
                if {e["sha256"] for e in training} & {e["sha256"] for e in validation}:
                    raise RuntimeError("Duplicate training/validation trajectory")
                report["data"] = {"training": episode_metadata(training), "validation": episode_metadata(validation)}
            train_one(config, model, training, validation, args, output, result)
        except Exception as error:  # Preserve the other model's results when a diagnostic fails.
            result["error"] = f"{type(error).__name__}: {error}"
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"FAIL | {name} | {result['error']}", flush=True)
        write_report(args.output, report)
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(run["status"] != "PASS" for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
