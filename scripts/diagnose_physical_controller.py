"""Paired image/physical controllers on identical frozen tiny cartpole world models.

Native offline/online learning is unchanged. An independent controller readout
is fitted on separate training clips; the benchmark evaluation head is untouched.
"""

import argparse
import hashlib
import json
import math
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import tools
from models.shared.physical_state import format_physical_rmse, readout_mode
from scripts.diagnose_fresh_readout import (
    FeatureBank, collect_pool, encode_pool, fresh_head, inside_goal, tensor_digest,
)
from scripts.diagnose_goal_objective import case_metadata, collect_objective_cases
from scripts.diagnose_planner_oracle import encode_images, rank_correlation, selection
from scripts.diagnose_planning_horizons import evaluate_horizon
from scripts.online_validation import episode_metadata
from scripts.smoke_tiny_planners import FAMILIES
from scripts.train_planner_check import build_config
from scripts.train_planner_learning_curve import measure, run_model, validate_budget
from training.progress import Progress, duration
from training.protocol import implementation_sha256


@dataclass(frozen=True)
class CartpoleCost:
    cart_limit: float = .25
    angle_scale: float = .1000417136
    cart_velocity_scale: float = .5
    pole_velocity_scale: float = 1.
    arrival_velocity_weight: float = .1
    tail_steps: int = 3

    def __call__(self, state):
        """[... , H, (x, cos(theta), sin(theta), vx, omega)] -> [...]."""
        x, cosine, sine, vx, omega = state.float().unbind(-1)
        boundary = ((x.abs() - self.cart_limit).relu() / self.cart_limit).square()
        # Chord distance avoids atan2's undefined gradient at a zero orientation.
        # A zero/invalid orientation is costly, not mistaken for an upright pole.
        tilt = ((cosine - 1).square() + sine.square()) / (2 * (1 - math.cos(self.angle_scale)))
        velocity = (vx / self.cart_velocity_scale).square() + (omega / self.pole_velocity_scale).square()
        return (boundary + tilt).mean(-1) + self.arrival_velocity_weight * velocity[..., -self.tail_steps:].mean(-1)


def physical_labels(state):
    """Convert simulator qpos/qvel to the existing head's cartpole label layout."""
    x, angle, vx, omega = state.unbind(-1)
    return torch.stack((x, angle.cos(), angle.sin(), vx, omega), dim=-1)


def decode_futures(head, history, future):
    batch, samples, horizon = future.shape[:3]
    prefix = history[:, history.shape[1] - head.history + 1:]
    prefix = prefix[:, None].expand(-1, samples, -1, *history.shape[2:])
    features = torch.cat((prefix, future), dim=2).flatten(0, 1)
    return head(features).reshape(batch, samples, horizon, -1)


@contextmanager
def physical_controller(model, head, cost):
    """Diagnostic-only scoring override; preserve native solvers and action gradients."""
    previous = model.__dict__.get("_goal_cost")
    modes = [(module, module.training) for module in head.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in head.parameters()]

    def score(history, past_action, candidates, goal):
        future = model.rollout(history, past_action, candidates)
        return cost(decode_futures(head, history, future))

    try:
        head.eval()
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        model._goal_cost = score
        yield
    finally:
        if previous is None:
            del model._goal_cost
        else:
            model._goal_cost = previous
        for module, mode in modes:
            module.training = mode
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


@torch.no_grad()
def cache_windows(model, episodes, count, horizon, seed):
    """Fixed causal train/validation clips; every forecast starts from real history only."""
    context = model.history_size
    features = encode_pool(model, episodes, 64)
    sampler = FeatureBank(episodes, features, context + horizon)
    observed, truth, sampled = sampler.sample(count, torch.Generator().manual_seed(seed))
    windows = [{"episode": episodes[index]["id"], "start": start, "source": episodes[index]["policy"]}
               for index, start in zip(sampled["episodes"], sampled["starts"], strict=True)]
    actions = torch.stack([episodes[index]["action"][start:start + context + horizon - 1]
                           for index, start in zip(sampled["episodes"], sampled["starts"], strict=True)]).to(model.device)
    predicted = []
    with readout_mode(model):
        for start in range(0, count, 32):
            part, action = observed[start:start + 32], actions[start:start + 32]
            future = model.rollout(part[:, :context], action[:, :context - 1], action[:, None, context - 1:])[:, 0]
            predicted.append(torch.cat((part[:, :context], future), dim=1))
    return {"observed": observed.detach(), "forecast": torch.cat(predicted).detach(),
            "truth": truth.detach(), "windows": windows}


def fit_controller(model, config, bank, args, output):
    head = fresh_head(model.state_head, config.state_head, None, args.seed + 21_000_000)
    head.samples_per_update = 2 * args.head_batch * (args.horizon + 1)
    generator = torch.Generator().manual_seed(args.seed + 22_000_000)
    progress = Progress("Controller head only", args.head_updates)
    started = time.monotonic()
    initial = tensor_digest(head.state_dict())
    with (output / "controller_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for step in range(1, args.head_updates + 1):
            indices = torch.randint(len(bank["truth"]), (args.head_batch,), generator=generator).to(model.device)
            features = torch.cat([bank[key][indices] for key in ("observed", "forecast")])
            labels = bank["truth"][indices].repeat(2, 1, 1)
            values = {key: float(value) for key, value in head.fit(features, labels).items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError("Non-finite controller-head training metrics.")
            log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
            progress.update(step, f"loss={values['state/loss']:.4g}", force=step == args.head_updates)
    return head, {"updates": int(head.updates), "examples": int(head.examples),
                  "parameters": sum(p.numel() for p in head.parameters()), "seconds": time.monotonic() - started,
                  "initial_sha256": initial, "final_sha256": tensor_digest(head.state_dict()),
                  "loss": "smooth_l1, unit physical scales, equal observed/forecast clips",
                  "loss_scale": head.loss_scale.tolist(), "windows": bank["windows"]}


@torch.no_grad()
def validate_head(head, bank, tolerance):
    truth = bank["truth"][:, head.history:]
    result = {}
    sources = np.asarray([window["source"] for window in bank["windows"]])
    for source in sorted(set(sources)):
        mask = torch.as_tensor(sources == source, device=truth.device)
        target = truth[mask]
        result[source] = {}
        for kind in ("observed", "forecast"):
            prediction = head(bank[kind][mask])[:, 1:]
            result[source][kind] = physical_errors(head, prediction, target, tolerance)
    return {"scores": result, "windows": bank["windows"]}


def physical_errors(head, prediction, truth, tolerance):
    error = head.targets.metric_error(prediction, truth).square()
    actual = inside_goal(head.targets.goal_relation(truth), tolerance, "box")
    predicted = inside_goal(head.targets.goal_relation(prediction), tolerance, "box")
    failures, successes = int((~actual).sum()), int(actual.sum())
    return {"rmse": {str(h): head.targets.metric_summary(error[:, h - 1])
                     for h in sorted({1, truth.shape[1]})},
            "false_goal_fraction": float((predicted & ~actual).sum() / failures) if failures else None,
            "missed_goal_fraction": float((~predicted & actual).sum() / successes) if successes else None,
            "failure_states": failures, "success_states": successes}


@torch.no_grad()
def score_physical_candidates(model, head, cases, cost):
    rows = []
    with readout_mode(model):
        for case in cases:
            history = model.encode({"image": case["prefix"][None].to(model.device)})
            past = case["past_action"][None].to(model.device)
            actions = torch.as_tensor(case["action"], device=model.device)
            future = torch.cat([model.rollout(history, past, part[None])[0] for part in actions.split(8)])[None]
            observed = encode_images(model, case["image"].flatten(0, 1), 8).reshape_as(future)
            truth = physical_labels(torch.as_tensor(case["states"], device=model.device)).float()
            states = {"forecast": decode_futures(head, history, future)[0],
                      "observed": decode_futures(head, history, observed)[0], "true": truth}
            costs = {key: cost(value).cpu().tolist() for key, value in states.items()}
            returns = case["rewards"].sum(1).tolist()
            rows.append({"id": case["id"], "costs": costs, "returns": returns,
                         "informative": float(np.ptp(returns)) > 1e-6,
                         "selection": {key: selection(value, returns) for key, value in costs.items()},
                         "cost_rank": {key: rank_correlation(value, costs["true"])
                                       for key, value in costs.items() if key != "true"},
                         "physical_errors": {key: physical_errors(head, value, truth, model.goal_tolerance)
                                             for key, value in states.items() if key != "true"}})
    informative = [row for row in rows if row["informative"]]
    selected = {key: float(np.mean([row["selection"][key]["return_mean"] for row in informative])) if informative else None
                for key in ("forecast", "observed", "true")}
    selected.update(uniform=float(np.mean([np.mean(r["returns"]) for r in informative])) if informative else None,
                    best=float(np.mean([max(r["returns"]) for r in informative])) if informative else None)
    return {"cases": rows, "informative_ids": [r["id"] for r in informative], "selected_return": selected}


@tools.preserve_rng_state
def compare(config, model, cases, expert_batch, args, output, pools):
    started = time.monotonic()
    if model.state_head.coordinates != ["position[0]", "position[1]", "position[2]", "velocity[0]", "velocity[1]"]:
        raise ValueError("Physical controller requires the cartpole cosine/sine label layout.")
    before = tensor_digest(model.state_dict())
    native = measure(config, model, cases, expert_batch, args, output)
    train, validation = pools
    bank = cache_windows(model, train, args.head_windows, args.horizon, args.seed + 23_000_000)
    head, fitting = fit_controller(model, config, bank, args, output)
    del bank
    validation_bank = cache_windows(model, validation, args.validation_windows, args.horizon, args.seed + 24_000_000)
    validation_scores = validate_head(head, validation_bank, model.goal_tolerance)
    del validation_bank
    # The benchmark held-out expert windows are a further evaluation-only check.
    obs, action, truth = expert_batch
    with readout_mode(model):
        features = model.encode({"image": obs["image"].to(model.device)})
        action = action.to(model.device)
        context = model.history_size
        future = model.rollout(features[:, :context], action[:, :context - 1], action[:, None, context - 1:])
        heldout = {key: physical_errors(head, decode_futures(head, features[:, :context], path)[:, 0],
                                       truth[:, context:].to(model.device), model.goal_tolerance)
                   for key, path in (("forecast", future), ("observed", features[:, None, context:]))}
    ranking = score_physical_candidates(model, head, cases, CartpoleCost())
    folder = output / "physical_policy"
    folder.mkdir()
    head_before = tensor_digest(head.state_dict())
    with physical_controller(model, head, CartpoleCost()):
        policy = evaluate_horizon(config, model, cases, args.horizon, args, folder)
    if tensor_digest(model.state_dict()) != before or tensor_digest(head.state_dict()) != head_before:
        raise RuntimeError("Controller comparison changed protected model/head weights or buffers.")
    native["physical_controller"] = {"fitting": fitting, "validation": validation_scores, "heldout_expert": heldout,
                                     "candidates": ranking, "policy": policy, "protected_sha256": before}
    native["seconds"] = time.monotonic() - started
    print(f"Controllers | {config.model_family} | image={native['policy']['return_mean']:.2f} "
          f"physical={policy['return_mean']:.2f}/{policy['maximum_return']} | "
          f"tail={native['policy']['sustained_rate']:.0%}/{policy['sustained_rate']:.0%}", flush=True)
    return native


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Physical controller check | frozen paired controllers | separate readout | no checkpoints",
             "Model/stage | Policy image/physical (max) | Tail image/physical | Candidate physical forecast/observed/true/uniform/best"]
    for run in report["runs"]:
        for snapshot in run.get("snapshots", []):
            physical = snapshot["physical_controller"]
            old, new = snapshot["policy"], physical["policy"]
            values = physical["candidates"]["selected_return"]
            selected = "/".join("n/a" if values[k] is None else f"{values[k]:.2f}"
                                for k in ("forecast", "observed", "true", "uniform", "best"))
            lines.append(f"{run['model']}/{snapshot['phase']}_{snapshot['updates']} | "
                         f"{old['return_mean']:.2f}/{new['return_mean']:.2f} ({old['maximum_return']}) | "
                         f"{old['sustained_rate']:.0%}/{new['sustained_rate']:.0%} | {selected}")
            scores = physical["heldout_expert"]["forecast"]["rmse"]
            lines.append("  Controller held-out expert forecast h1 RMSE | " + format_physical_rmse(scores["1"]))
        lines.append(f"{run['status']} | {run['model']} | time={duration(run.get('seconds'))}"
                     + (f" | {run['error']}" if "error" in run else ""))
    lines += ["Physical cost: mean cart-boundary + pole chord error, plus last-three-step velocity cost; fixed scales/weights in JSON.",
              "Controller fit: separate fresh head on TRAIN expert/zero/random clips, 50% observed and 50% frozen forecasts; no native/evaluation-head updates.",
              "Native online data uses the original image controller. This isolates scoring, not physical-controller online learning or long-run stability.",
              "Candidate scores use identical informative anchors; true states/feedback candidates are diagnostic only, never fed to either deployed controller.",
              "Policy starts, budgets, solvers and per-call RNG are paired. Returns are short controlled-start tests, not full benchmark scores.",
              "COMPLETE means execution, not improvement. Source-specific RMSE, false goals, exact clips, timings and policy traces are in JSON/logs."]
    if "seconds" in report:
        lines.append(f"Total | {duration(report['seconds'])}")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--expert-updates", type=int, default=5000)
    parser.add_argument("--online-steps", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--head-updates", type=int, default=2000)
    parser.add_argument("--head-windows", type=int, default=2048)
    parser.add_argument("--validation-windows", type=int, default=512)
    parser.add_argument("--head-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("physical_controller_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if (min(args.expert_updates, args.horizon, args.policy_steps, args.head_updates, args.head_windows,
            args.validation_windows, args.head_batch) < 1 or min(args.seed, args.online_steps) < 0
            or args.head_windows % 4 or args.validation_windows % 4 or len(args.models) != len(set(args.models))):
        parser.error("Use positive budgets, window counts divisible by four, nonnegative seed/online steps, and unique models.")
    args.scenario = "cartpole_balance_sparse"
    args.eval_updates, args.online_eval_steps = [args.expert_updates], []
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA diagnostic requires BF16 support.")
        torch.cuda.set_device(device)
    configs = {name: build_config(name, args) for name in args.models}
    for config in configs.values():
        config.jepa_model.planner.horizon = args.horizon
        validate_budget(config, args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(f"Physical controller | models={len(configs)} | offline={args.expert_updates} | "
          f"online_env_steps={args.online_steps} | head_updates/snapshot={args.head_updates} | no checkpoints", flush=True)
    cases = collect_objective_cases(configs[args.models[0]], SimpleNamespace(
        sim_seeds=[args.seed + 12_000_000, args.seed + 12_000_001],
        horizons=list(range(1, args.horizon + 1)), candidates=16), history_size=3)
    files = [Path(__file__).with_name(name) for name in (
        "diagnose_physical_controller.py", "train_planner_learning_curve.py", "train_planner_check.py",
        "diagnose_fresh_readout.py", "online_validation.py", "diagnose_planning_horizons.py",
        "diagnose_goal_objective.py", "diagnose_planner_oracle.py", "smoke_online_checkpoints.py")]
    report = {"implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest(),
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "physical_cost": asdict(CartpoleCost()), "cases": [case_metadata(case) for case in cases],
              "checkpoint_reads": False, "checkpoint_writes": False, "runs": []}
    pools = None

    @tools.preserve_rng_state
    def measurement(config, model, cases, expert_batch, args, output):
        nonlocal pools
        if pools is None:
            path = args.dataset_root / config.scenario.dataset
            pool_args = SimpleNamespace(expert_train=12, expert_validation=4, sim_train=4, sim_validation=2,
                                        context_length=model.history_size, horizons=[args.horizon], data_seed=args.seed + 20_000_000)
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            pools = collect_pool(config, model, metadata, path, pool_args)
            report["controller_data"] = {"train": episode_metadata(pools[0]), "validation": episode_metadata(pools[1]),
                                         "settings": vars(pool_args)}
        return compare(config, model, cases, expert_batch, args, output, pools)

    for name, config in configs.items():
        result = {"model": name, "status": "RUNNING"}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        tick = time.monotonic()
        try:
            run_model(config, args, cases, output, result, lambda: write_report(args.output, report), measurement=measurement)
        except Exception as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        result["seconds"] = time.monotonic() - tick
        write_report(args.output, report)
        print(f"{result['status']} | {name} | {duration(result['seconds'])}", flush=True)
    report["seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(run["status"] != "COMPLETE" for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
