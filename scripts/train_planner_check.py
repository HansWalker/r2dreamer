"""Hour-scale, from-scratch expert/online training diagnostic for both latent planners."""

import argparse
import copy
import json
import math
import statistics
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from envs import close_envs, make_envs, make_eval_envs
from models.shared.physical_state import format_physical_rmse
from scripts.online_validation import TrajectoryDataset, collect_episode, validation_metadata
from scripts.smoke_online_checkpoints import diagnose, diagnostic_guards
from scripts.smoke_tiny_planners import FAMILIES, changed_parameters, native_control, parameter_copies
from scripts.smoke_training import EvaluationTimer
from training import load_model_family
from training.evaluation import StateDataset
from training.progress import Progress, duration
from training.protocol import implementation_sha256, validate_training_recipe
from training.readout import online_readout
from training.trainer import online_update_target


def build_config(name, args):
    overrides = [
        f"scenario={args.scenario}", f"device={args.device}", f"seed={args.seed}",
        f"env.dataset_root={json.dumps(str(args.dataset_root.resolve()))}",
        f"env.seed={4_000_000 + args.seed}", f"env.eval_seed={5_000_000 + args.seed}",
        "env.eval_episode_num=2", "replay.batch_size=128",
        "jepa_model.encoder.embedding_dim=64", "jepa_model.encoder.vision.base_channels=4",
        "jepa_model.encoder.vision.layers=2", "jepa_model.encoder.vision.heads=2",
        "jepa_model.encoder.vision.mlp_dim=128", "jepa_model.projector_dim=128",
        "jepa_model.predictor.layers=2", "jepa_model.predictor.heads=2",
        "jepa_model.predictor.dim_head=32", "jepa_model.predictor.mlp_dim=128",
        "jepa_model.predictor.action_embedding_dim=4",
        # Keep the scenario horizon, native objectives, optimizers and learning rates.
        "jepa_model.planner.samples=64" if name == "leworldmodel" else "jepa_model.planner.samples=4",
        "jepa_model.planner.elites=8",
        "jepa_model.planner.iterations=6" if name == "leworldmodel" else "jepa_model.planner.iterations=8",
    ]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name=f"{name}_dmc_vision", overrides=overrides)
    OmegaConf.resolve(config)
    validate_training_recipe(config)
    return config


def set_budget(config, updates):
    config.training.expert.updates = config.training.online.updates = int(updates)
    # Preserve one common data/update ratio, independent of each model's measured speed.
    transitions = int(config.training.online.warmup_transitions) + 4 * int(updates)
    envs = int(config.env.env_num)
    transitions = math.ceil(transitions / envs) * envs
    config.training.online.steps = transitions * int(config.env.action_repeat)
    validate_training_recipe(config)


def choose_budget(configs, timings, minutes, elapsed):
    fixed, per_update = 0.0, 0.0
    breakdown = {}
    for name, config in configs.items():
        rate = timings[name]
        warmup_calls = int(config.training.online.warmup_transitions) / int(config.env.env_num)
        episode_steps = int(config.env.time_limit) // int(config.env.action_repeat)
        # Initial, post-offline and four online snapshots, plus two complete policy evaluations.
        overhead = (warmup_calls * rate["collect"] + 6 * rate["diagnostics"]
                    + 2 * (rate["evaluation_reset"] + episode_steps * rate["evaluation_step"]))
        unit = rate["offline"] + rate["online"] + 4 / int(config.env.env_num) * rate["collect"]
        breakdown[name] = {"fixed_seconds": overhead, "seconds_per_shared_update": unit}
        fixed += overhead
        per_update += unit
    reserve = max(60.0, minutes * 60 * .1)
    remaining = minutes * 60 - elapsed - fixed - reserve
    if not math.isfinite(per_update) or per_update <= 0:
        raise ValueError("Calibration did not produce positive, finite training rates")
    updates = math.floor(remaining / per_update / 64) * 64
    if updates < 2048:
        raise ValueError("Time target cannot fit 2,048 updates/phase and full online episodes; increase --minutes.")
    for name, config in configs.items():
        rate = timings[name]
        calls = (int(config.training.online.warmup_transitions) + 4 * updates) / int(config.env.env_num)
        breakdown[name].update(offline_seconds=updates * rate["offline"],
                               online_seconds=updates * rate["online"] + calls * rate["collect"])
    return {"updates_per_phase_per_model": updates, "target_minutes_total": minutes,
            "setup_and_calibration_seconds": elapsed, "reserve_seconds": reserve,
            "projected_seconds_total": elapsed + fixed + updates * per_update,
            "models": breakdown,
            "note": "Fixed equal budgets chosen before training; no wall-clock stopping or padding. Runtime is approximate."}


def new_model(config, dataset):
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    model = load_model_family(config.model_family).build_model(config)
    model.state_head.set_stats(dataset.state_mean, dataset.state_std)
    if int(model.state_head.updates):
        raise RuntimeError("Diagnostic must start with fresh model/head weights")
    return model


def checked_update(model, operation):
    model.train()
    before = int(model.state_head.updates)
    metrics = {key: float(value) for key, value in operation().items()}
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite or missing training metrics")
    if int(model.state_head.updates) != before + 1:
        raise RuntimeError("Readout update counter differs from native update count")
    return metrics


def diagnostic_settings(seed):
    return SimpleNamespace(context_length=64, horizons=[1, 5, 25, 100], window_seed=2_000_000 + seed,
                           batch_size=8, max_rmse_ratio=3.0, rmse_floor=.01, rmse_floors={})


def measure_errors(model, sources, settings):
    return {name: diagnose(model, dataset, windows, settings) for name, (dataset, windows) in sources.items()}


@tools.preserve_rng_state
def evaluate_policy(config, model, *, timing_steps=None):
    settings = copy.deepcopy(config)
    if timing_steps is not None:
        settings.env.time_limit = timing_steps * int(settings.env.action_repeat)
    tools.configure_randomness(int(settings.env.eval_seed), bool(settings.deterministic_run))
    envs = None
    try:
        envs = make_eval_envs(settings.env)
        timer = EvaluationTimer(envs, model.device)
        score, length, extra = native_control(
            model, lambda: load_model_family(settings.model_family).evaluate(settings, model, timer))
        return {"return": score, "agent_steps": length,
                **{key: float(value) for key, value in extra.items()}, "timing": timer.finish()}
    finally:
        close_envs(envs)


def calibrate(config, dataset, sources, settings):
    """Disposable weights/replay; the actual run is reinitialized after budgets are fixed."""
    family = load_model_family(config.model_family)
    model = new_model(config, dataset)
    envs = None
    timings = {}

    def timed(operation):
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        start = time.monotonic()
        operation()
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        return time.monotonic() - start

    try:
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(int(config.training.expert.updates))
        offline = [timed(lambda: checked_update(model, lambda: model.update(dataset.sample_episode_batch())))
                   for _ in range(20)]
        timings["offline"] = statistics.mean(offline[4:])
        envs = make_envs(config.env)
        session = family.OnlineSession(config, model, envs)
        session.start()
        calls = math.ceil(int(config.training.online.warmup_transitions) / int(config.env.env_num))
        collect = [timed(lambda: native_control(model, session.collect)) for _ in range(calls)]
        if not session.replay.ready():
            raise RuntimeError("Calibration replay is not ready after the configured warmup")
        timings["collect"] = statistics.mean(collect[-8:])
        if hasattr(model, "configure_online"):
            model.configure_online(int(config.training.online.updates), resumed=False)
        with online_readout(config, family, model):
            online = []
            for step in range(20):
                if step % 4 == 0:
                    native_control(model, session.collect)
                online.append(timed(lambda: checked_update(model, lambda: session.update(1))))
        timings["online"] = statistics.mean(online[4:])
        close_envs(envs)
        envs = None
        timings["diagnostics"] = timed(lambda: measure_errors(model, sources, settings))
        policy = evaluate_policy(config, model, timing_steps=12)["timing"]
        timings["evaluation_step"] = policy["warm_step_seconds"]
        timings["evaluation_reset"] = policy["reset_seconds"] + policy["first_step_seconds"]
        timings["parameters"] = sum(p.numel() for p in model.parameters())
        return timings
    finally:
        close_envs(envs)


def run_training(config, dataset, sources, settings, output, result, persist):
    family = load_model_family(config.model_family)
    model = new_model(config, dataset)
    count = int(config.training.expert.updates)
    started = time.monotonic()
    result.update(status="RUNNING", config=OmegaConf.to_container(config, resolve=True), snapshots=[], policy={},
                  parameters=sum(p.numel() for p in model.parameters()),
                  evaluation_std=model.state_head.std.tolist(), phases={})
    scales = {key: getattr(model.state_head, key).clone() for key in ("mean", "std", "loss_scale", "output_scale")}
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)

    def snapshot(phase, updates, env_steps=0):
        scores = measure_errors(model, sources, settings)
        item = {"phase": phase, "updates": updates, "env_steps": env_steps, "scores": scores}
        if phase == "online":
            before = result["snapshots"][1]["scores"]
            item["guards"] = {name: diagnostic_guards(before[name], score, settings) for name, score in scores.items()}
        result["snapshots"].append(item)
        persist()
        expert = scores["heldout_expert"]["all"]["physical"]
        print(f"Check | {config.model_family} | {phase}={updates} | "
              f"observed h1 RMSE: {format_physical_rmse(expert['observed']['1'])}", flush=True)

    snapshot("initial", 0)
    with (output / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        before = parameter_copies(model)
        phase_start = time.monotonic()
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(count)
        progress = Progress(f"{config.model_family} offline", count)
        for step in range(1, count + 1):
            metrics = checked_update(model, lambda: model.update(dataset.sample_episode_batch()))
            log.write(json.dumps({"phase": "offline", "update": step, **metrics}, allow_nan=False) + "\n")
            progress.update(step, f"loss={metrics['loss']:.4g} state={metrics['state/loss']:.4g}", force=step == count)
        result["phases"]["offline"] = {"updates": count, "seconds": time.monotonic() - phase_start,
                                       **changed_parameters(model, before)}
        snapshot("offline", count)
        result["policy"]["offline"] = evaluate_policy(config, model)
        persist()
        envs = None
        try:
            envs = make_envs(config.env)
            session = family.OnlineSession(config, model, envs)
            session.start()
            if session.replay.count():
                raise RuntimeError("Online replay must start empty")
            before = parameter_copies(model)
            phase_start = time.monotonic()
            if hasattr(model, "configure_online"):
                model.configure_online(count, resumed=False)
            steps = updates = 0
            metrics = {}
            episodes = []
            progress = Progress(f"{config.model_family} online", int(config.training.online.steps))
            next_snapshot = max(1, count // 4)
            with online_readout(config, family, model):
                while steps < int(config.training.online.steps):
                    delta, completed = native_control(model, session.collect)
                    steps += delta
                    episodes.extend(completed)
                    target = online_update_target(config, steps)
                    if session.replay.ready():
                        while updates < target:
                            metrics = checked_update(model, lambda: session.update(1))
                            updates += 1
                            log.write(json.dumps({"phase": "online", "update": updates, "env_steps": steps,
                                                  **metrics}, allow_nan=False) + "\n")
                    if updates >= next_snapshot:
                        snapshot("online", updates, steps)
                        next_snapshot = (updates // max(1, count // 4) + 1) * max(1, count // 4)
                    detail = (f" loss={metrics['loss']:.4g} state={metrics['state/loss']:.4g}" if metrics else "")
                    progress.update(steps, f"updates={updates}/{count} replay={session.replay.count()}{detail}",
                                    force=steps == int(config.training.online.steps))
            if updates != count or int(model.state_head.updates) != 2 * count:
                raise RuntimeError("Native/readout updates did not meet the fixed budget")
            if result["snapshots"][-1]["updates"] != updates or result["snapshots"][-1]["phase"] != "online":
                snapshot("online", updates, steps)
            result["phases"]["online"] = {"updates": updates, "env_steps": steps,
                                          "agent_transitions": steps // int(config.env.action_repeat),
                                          "seconds": time.monotonic() - phase_start,
                                          "replay_rows": session.replay.count(), "episode_returns": episodes,
                                          **changed_parameters(model, before)}
            if "goal_image" in session.replay._obs_keys:
                raise RuntimeError("Goal images leaked into replay")
        finally:
            close_envs(envs)
    result["policy"]["online"] = evaluate_policy(config, model)
    for key, value in scales.items():
        torch.testing.assert_close(getattr(model.state_head, key), value, rtol=0, atol=0)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("Non-finite model state after training")
    stable = all(guard["passed"] for snap in result["snapshots"] for guard in snap.get("guards", {}).values())
    result.update(status="PASS" if stable else "REGRESSION", seconds=time.monotonic() - started)
    if model.device.type == "cuda":
        result["gpu_peak_gib"] = torch.cuda.max_memory_reserved(model.device) / 1024**3


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Small planner training | from scratch | expert offline + own-policy online | no checkpoints",
             "Model | Parameters | Offline/online updates | Policy return offline->online | Time"]
    for result in report["runs"]:
        if result["status"] not in {"PASS", "REGRESSION"}:
            lines.append(f"{result['status']} | {result['model']} | {result.get('error', 'in progress')}")
            continue
        before, after = [s["scores"]["heldout_expert"]["all"]["physical"]
                         for s in (result["snapshots"][1], result["snapshots"][-1])]
        phases, policy = result["phases"], result["policy"]
        lines.append(f"{result['status']} | {result['model']} | {result['parameters']:,} | "
                     f"{phases['offline']['updates']}/{phases['online']['updates']} | "
                     f"{policy['offline']['return']:.1f}->{policy['online']['return']:.1f} | {duration(result['seconds'])}")
        for source, horizon in (("observed", "1"), ("forecast", "5"), ("forecast", "100")):
            if horizon not in after[source]:
                continue
            lines.append(f"  Expert {source} h{horizon} RMSE offline->online | " + format_physical_rmse(
                after[source][horizon], before=before[source][horizon],
                baseline=after["true_persistence"][horizon] if source == "forecast" else None))
    lines += ["PASS checks execution and declared short-run error limits, not task mastery or full-size stability.",
              "REGRESSION means an intermediate/final physical-error guard failed; inspect per-coordinate RMSE in report.json.",
              "Angles are wrapped radians; hold=true-state persistence (scoring only). nMSE remains secondary in JSON; guards unchanged.",
              "Expert HELDOUT and zero/random failure trajectories are never fitted; training losses are unchanged.",
              "Policy returns use two complete, fixed-seed episodes per stage. The time target covers both models, sequentially."]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scenario", choices=("cartpole_balance_sparse", "reacher", "ball_in_cup"), default="cartpole_balance_sparse")
    parser.add_argument("--minutes", type=float, default=60, help="Approximate TOTAL runtime, including both models.")
    parser.add_argument("--updates", type=int, help="Skip calibration; use this fixed number of updates per phase per model.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("planner_training_check_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or args.minutes <= 0 or args.seed < 0:
        parser.error("Minutes must be positive and finite; seed must be nonnegative.")
    if args.updates is not None and args.updates < 2048:
        parser.error("Use at least 2,048 updates per phase so online collection includes complete episodes.")
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU diagnostic requires CUDA with BF16 support")
        torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    configs = {name: build_config(name, args) for name in FAMILIES}
    first = configs[FAMILIES[0]]
    family = load_model_family(FAMILIES[0])
    settings = diagnostic_settings(args.seed)
    report = {"diagnostic_version": 1, "implementation_sha256": implementation_sha256(),
              "scenario": args.scenario, "seed": args.seed, "checkpoint_writes": False,
              "fresh_initialization": True, "calibration": {}, "runs": []}
    persist = lambda: write_report(args.output, report)
    with family.build_replay(first) as dataset:
        # Open and normalize on TRAIN once. The same sampler state is restored for each actual model.
        state = copy.deepcopy(dataset.state_dict())
        mean, std = dataset.state_stats
        report["data"] = {"path": str(dataset.path), "identity": dataset_identity(dataset.metadata),
                          "training_episodes": dataset.num_episodes, "state_mean": mean.tolist(), "state_std": std.tolist()}
        reference = new_model(first, dataset)
        heldout = StateDataset(dataset.h5, dataset.metadata, first.model_io, first.state_head.fields, reference.state_head.targets)
        windows = heldout.sample_windows(16, settings.context_length + max(settings.horizons),
                                         settings.window_seed, settings.context_length, .5, 8)
        print("Data | expert training split | 16 fixed held-out windows | collecting two failure-validation episodes once", flush=True)
        failures = [collect_episode(first, reference, 6_000_000 + args.seed + i, mode)
                    for i, mode in enumerate(("zero", "random"))]
        failure_data = TrajectoryDataset(failures)
        failure_windows = failure_data.sample_windows(4, settings.context_length + max(settings.horizons), settings.window_seed)
        sources = {"heldout_expert": (heldout, windows), "simulator_failure": (failure_data, failure_windows)}
        report["validation"] = {"expert_windows": [asdict(w) for w in windows],
                                "simulator": validation_metadata(failure_data, failure_windows)}
        del reference
        if args.updates is None:
            for name, config in configs.items():
                print(f"Timing | {name} | disposable weights; no trained weights carried into the run", flush=True)
                dataset.load_state_dict(copy.deepcopy(state))
                report["calibration"][name] = calibrate(config, dataset, sources, settings)
                persist()
            report["budget"] = choose_budget(configs, report["calibration"], args.minutes, time.monotonic() - started)
            updates = report["budget"]["updates_per_phase_per_model"]
            print(f"Estimate | total={duration(report['budget']['projected_seconds_total'])} "
                  f"plus {duration(report['budget']['reserve_seconds'])} timing margin", flush=True)
        else:
            updates = args.updates
            report["budget"] = {"updates_per_phase_per_model": updates, "source": "explicit --updates; runtime not targeted"}
        print(f"Budget | each model: offline={updates:,} + online={updates:,} updates | models=2 | sequential", flush=True)
        for name, config in configs.items():
            set_budget(config, updates)
            dataset.load_state_dict(copy.deepcopy(state))
            result = {"model": name, "status": "RUNNING"}
            report["runs"].append(result)
            output = args.output / name
            output.mkdir()
            try:
                run_training(config, dataset, sources, settings, output, result, persist)
            except Exception as error:
                result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"FAIL | {name} | {result['error']}", flush=True)
            persist()
    report["elapsed_seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(r["status"] != "PASS" for r in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
