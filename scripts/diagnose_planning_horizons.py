"""Paired native-planner horizon diagnostic: train once, then freeze all model weights."""

import argparse
import copy
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import (
    PROFILES,
    cart_state,
    case_metadata,
    collect_objective_cases,
    observe_prefix,
    score_horizon,
    score_objective,
    set_cart_state,
)
from scripts.diagnose_planner_oracle import encode_images, rank_correlation
from scripts.smoke_tiny_planners import FAMILIES, native_control
from scripts.train_planner_check import build_config
from scripts.train_rollout_check import pretrain
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256


@torch.no_grad()
def score_rankings(model, cases, horizons, batch_size, action_repeat):
    """One recursive forecast per candidate; reuse its endpoints for every horizon."""
    oracle = {(row["id"], row["horizon"]): row for row in
              score_objective(model, cases, horizons, batch_size, action_repeat)}
    rows = []
    with readout_mode(model):
        for case in cases:
            prefix = model.encode({"image": case["prefix"][None].to(model.device)})
            past = case["past_action"][None].to(model.device)
            goal = encode_images(model, case["goal_image"][None], batch_size)[0]
            actions = torch.as_tensor(case["action"], device=model.device)
            prediction = native_control(model, lambda prefix=prefix, past=past, actions=actions: torch.cat([
                model.rollout(prefix, past, chunk[None])[0] for chunk in actions.split(batch_size)
            ]))
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite open-loop forecasts.")
            actual = encode_images(model, case["image"].flatten(0, 1), batch_size)
            actual = actual.reshape(len(actions), len(horizons), *actual.shape[1:])
            for index, horizon in enumerate(horizons):
                terminal = prediction[:, horizon - 1]
                error = (terminal - goal).square().flatten(1)
                cost = error.sum(-1) if model.goal_reduction == "sum" else error.mean(-1)
                real = oracle[case["id"], horizon]
                rows.append({"id": case["id"], "horizon": horizon, "oracle": real,
                             "forecast": score_horizon(case, cost.cpu().numpy(), horizon, index, action_repeat),
                             "forecast_vs_oracle_rank": rank_correlation(cost.cpu().numpy(), real["latent_cost"]),
                             "latent_rmse": (terminal - actual[:, index]).square().mean().sqrt().item()})
    return rows


def summarize_rankings(rows, horizons):
    # Compare the SAME informative anchors across horizons, not a changing cohort.
    common = set.intersection(*[
        {row["id"] for row in rows if row["horizon"] == h and row["oracle"]["reward_informative"]}
        for h in horizons
    ])

    def returns(group):
        methods = {"forecast": ("forecast", "latent"), "oracle": ("oracle", "latent"),
                   "uniform": ("oracle", "uniform"), "best": ("oracle", "reward_oracle")}
        return {name: float(np.mean([r[source]["selection"][method]["normalized_return"] for r in group]))
                if group else None for name, (source, method) in methods.items()}

    result = {}
    for h in horizons:
        group = [row for row in rows if row["horizon"] == h]
        matched = [row for row in group if row["id"] in common]
        ranks = [r["forecast_vs_oracle_rank"] for r in matched if r["forecast_vs_oracle_rank"] is not None]
        result[str(h)] = {"common_ids": sorted(common), "common_count": len(common),
                          "informative_count": sum(r["oracle"]["reward_informative"] for r in group),
                          "anchor_count": len(group), "common_returns": returns(matched),
                          "all_anchor_returns": returns(group),
                          "forecast_vs_oracle_rank": float(np.mean(ranks)) if ranks else None,
                          "rank_count": len(ranks),
                          "coverage": "CONTRAST" if len(common) >= 4 else "LOW_CONTRAST"}
    return result


def synchronize(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


@tools.preserve_rng_state
def evaluate_horizon(config, model, cases, horizon, args, output):
    """Batched native action selection; no simulator state/reward enters the model."""
    envs = []
    original_planner = model.planner
    caches = model._cem_mean, model._gradient_actions
    model.planner = copy.deepcopy(original_planner)
    model.planner.horizon = horizon
    model._cem_mean = model._gradient_actions = None
    started = time.monotonic()
    before = tensor_digest(model.state_dict())
    try:
        for case in cases:
            env = make_env(config.env, case["seed"], include_physical_state=False)
            envs.append(env)
            observation = env.reset()
            set_cart_state(env, case["initial_state"])
            prefix = observe_prefix(env, model.history_size)
            # Check once per arm: environment state and input history must be paired.
            np.testing.assert_array_equal(cart_state(env), case["anchor_state"])
            np.testing.assert_array_equal(observation["goal_image"], case["goal_image"].numpy())
            for key in ("prefix", "past_action"):
                torch.testing.assert_close(prefix[key], case[key], rtol=0, atol=0)
        history = torch.stack([c["prefix"] for c in cases]).to(model.device)
        past = torch.stack([c["past_action"] for c in cases]).to(model.device)
        goals = torch.stack([c["goal_image"] for c in cases]).to(model.device)
        reset_seconds = time.monotonic() - started
        returns = np.zeros(len(cases))
        successes = []
        policy_times, env_times = [], []
        progress = Progress(f"{config.model_family} horizon {horizon}", args.policy_steps)
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        with (output / "policy_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for step in range(args.policy_steps):
                # Same per-call seed across arms; horizon-dependent tensor shapes still differ.
                torch.manual_seed(args.seed + 14_000_000 + step)
                first = torch.full((len(cases),), step == 0, device=model.device, dtype=torch.bool)
                synchronize(model)
                tick = time.monotonic()
                with torch.no_grad():
                    action = native_control(model, lambda history=history, past=past, first=first: model.act(
                        {"image": history, "goal_image": goals}, past, deterministic=True, first=first))
                actions = action.detach().cpu().numpy()
                synchronize(model)
                policy_times.append(time.monotonic() - tick)
                tick = time.monotonic()
                images, rewards, successful, states = [], [], [], []
                for env, control in zip(envs, actions, strict=True):
                    obs, reward, done, _ = env.step(control)
                    if done:
                        raise ValueError("Closed-loop trial reached an unexpected episode boundary.")
                    images.append(obs["image"])
                    rewards.append(float(reward))
                    successful.append(float(env._env.task.get_reward(env._env.physics)) >= 1 - 1e-6)
                    states.append(cart_state(env).tolist())
                history = torch.cat((history[:, 1:], torch.from_numpy(np.stack(images)).to(model.device)[:, None]), dim=1)
                past = torch.cat((past, action.detach()[:, None]), dim=1)[:, -(model.history_size - 1):]
                synchronize(model)
                env_times.append(time.monotonic() - tick)
                returns += rewards
                successes.append(successful)
                log.write(json.dumps({"agent_step": step + 1, "case_ids": [c["id"] for c in cases],
                                      "actions": actions.tolist(), "rewards": rewards, "success": successful,
                                      "states": states, "policy_seconds": policy_times[-1],
                                      "environment_seconds": env_times[-1]}, allow_nan=False) + "\n")
                progress.update(step + 1, f"return={returns.mean():.1f}", force=step + 1 == args.policy_steps)
        after = tensor_digest(model.state_dict())
        if before != after:
            raise RuntimeError("Horizon evaluation changed model weights or buffers.")
        tail = min(10, args.policy_steps)
        sustained = np.asarray(successes[-tail:]).all(0)
        maximum = args.policy_steps * int(config.env.action_repeat)
        per_case = [{"id": c["id"], "cohort": c["cohort"], "anchor_success": c["anchor_success"],
                     "return": float(returns[i]), "normalized_return": float(returns[i] / maximum),
                     "sustained": bool(sustained[i]), "success_fraction": float(np.mean(np.asarray(successes)[:, i]))}
                    for i, c in enumerate(cases)]
        return {"horizon": horizon, "seconds_lookahead": horizon * int(config.env.action_repeat) * envs[0]._env.control_timestep(),
                "agent_steps": args.policy_steps, "maximum_return": maximum, "cases": per_case,
                "return_mean": float(returns.mean()), "normalized_return": float(returns.mean() / maximum),
                "sustained_rate": float(sustained.mean()), "sustained_tail_steps": tail,
                "cohort_return": {cohort: float(np.mean([r["return"] for r in per_case if r["cohort"] == cohort]))
                                  for cohort in sorted({c["cohort"] for c in cases})},
                "initial_state_sha256": before, "final_state_sha256": after,
                "timing": {"reset_seconds": reset_seconds, "policy_first_seconds": policy_times[0],
                           "policy_warm_seconds": float(np.mean(policy_times[1:])) if len(policy_times) > 1 else None,
                           "policy_total_seconds": sum(policy_times), "environment_total_seconds": sum(env_times),
                           "wall_seconds": time.monotonic() - started},
                "gpu_reserved_peak_gib": torch.cuda.max_memory_reserved(model.device) / 2**30 if model.device.type == "cuda" else None}
    finally:
        model.planner = original_planner
        model._cem_mean, model._gradient_actions = caches
        for env in envs:
            env.close()


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Planning horizons | one offline fit/model | frozen weights | unchanged native goal cost",
             "Model | H(seconds) | Policy return/max | Tail success | Policy s/call | Common anchors | Rank-test return predicted/true/uniform/best | Status"]
    def number(value):
        return "n/a" if value is None else f"{value:.3f}"
    for run in report["runs"]:
        for arm in run.get("arms", []):
            if arm["status"] != "COMPLETE":
                lines.append(f"{run['model']} | {arm['horizon']} | {arm['status']} | {arm.get('error', '')}")
                continue
            policy = arm["policy"]
            ranking = run["ranking_summary"][str(arm["horizon"])]
            selected = "/".join(number(ranking["common_returns"][k]) for k in ("forecast", "oracle", "uniform", "best"))
            lines.append(f"{run['model']} | {arm['horizon']}({policy['seconds_lookahead']:.2f}s) | "
                         f"{policy['return_mean']:.1f}/{policy['maximum_return']} | {policy['sustained_rate']:.0%} | "
                         f"{number(policy['timing']['policy_warm_seconds'])} | "
                         f"{ranking['common_count']}/{ranking['anchor_count']} | {selected} | COMPLETE/{ranking['coverage']}")
        if run["status"] == "FAIL":
            lines.append(f"FAIL | {run['model']} | {run.get('error', 'One or more horizon arms failed')}")
    lines += ["Policy = actual closed-loop control on identical constructed starts; short fixed-length trials, not 500-step benchmark episodes.",
              "Tail success = every endpoint of the last min(10, policy_steps) actions. Policy returns exclude the shared zero-action prefix.",
              "Ranking uses the same anchors informative at ALL tested horizons and identical candidate prefixes; returns are normalized by H * action_repeat.",
              "True = encode actual future images; predicted = recursive learned dynamics. Uniform averages candidates; best is their maximum return.",
              "Two privileged simulator-feedback candidates are ranking controls only; neither deployed planner receives them, rewards, or physical state.",
              "Same planner samples/restarts/iterations and per-call RNG seeds across horizons; tensor shapes and total compute differ.",
              "Policy s/call is warm planning time for the whole anchor batch; environment/reset time and allocator peaks are in JSON.",
              "No online updates, physical-head planning, checkpoint reads/writes, dataset audit, or production changes. COMPLETE means execution, not improvement."]
    if "seconds" in report:
        lines.append(f"Time | {duration(report['seconds'])}")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--expert-updates", type=int, default=1000)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 15, 25])
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--sim-seeds", nargs="+", type=int, default=[12_000_000, 12_000_001])
    parser.add_argument("--candidates", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=128, help="Ranking encoder/forecast chunk size, not the training batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("planning_horizon_check_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if min(args.expert_updates, args.policy_steps, args.batch_size, *args.horizons) < 1 or args.candidates < 13:
        parser.error("Use positive budgets and at least 13 candidates.")
    if args.seed < 0 or min(args.sim_seeds) < 0:
        parser.error("Seeds must be nonnegative.")
    if any(len(v) != len(set(v)) for v in (args.models, args.horizons, args.sim_seeds)):
        parser.error("Models, horizons and simulator seeds must be unique.")
    args.horizons.sort()
    args.scenario = "cartpole_balance_sparse"
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA test requires BF16 support.")
        torch.cuda.set_device(device)
    configs = {name: build_config(name, args) for name in args.models}
    histories = {int(c.jepa_model.history_size) for c in configs.values()}
    if len(histories) != 1 or min(histories) < 2:
        raise ValueError("Horizon pairing requires a shared history of at least two real frames.")
    history_size = histories.pop()
    for config in configs.values():
        if history_size - 1 + max(args.policy_steps, *args.horizons) >= int(config.env.time_limit) // int(config.env.action_repeat):
            raise ValueError("Requested trial would reach the episode time limit; shorten --policy-steps/--horizons.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"experiment": "paired_planning_horizons", "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "checkpoint_reads": False, "checkpoint_writes": False, "production_settings_changed": False,
              "online_updates": 0, "runs": []}
    print(f"Planning horizons | models={len(configs)} | offline={args.expert_updates} once/model | "
          f"horizons={args.horizons} | anchors={len(args.sim_seeds) * len(PROFILES)} | "
          f"policy_steps={args.policy_steps} | no checkpoints", flush=True)
    cases = collect_objective_cases(next(iter(configs.values())), args, history_size=history_size)
    report["cases"] = [case_metadata(case) for case in cases]
    write_report(args.output, report)
    for name, config in configs.items():
        result = {"model": name, "status": "RUNNING", "config": OmegaConf.to_container(config, resolve=True), "arms": []}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        model = None
        try:
            with load_model_family(name).build_replay(config) as dataset:
                result["dataset_identity"] = dataset_identity(dataset.metadata)
                model = pretrain(config, dataset, args, output, result)
            before = tensor_digest(model.state_dict())
            result["rankings"] = score_rankings(model, cases, args.horizons, args.batch_size, int(config.env.action_repeat))
            if tensor_digest(model.state_dict()) != before:
                raise RuntimeError("Ranking changed model weights or buffers.")
            result["ranking_summary"] = summarize_rankings(result["rankings"], args.horizons)
            write_report(args.output, report)
            for horizon in args.horizons:
                arm = {"horizon": horizon, "status": "RUNNING"}
                result["arms"].append(arm)
                folder = output / f"horizon_{horizon}"
                folder.mkdir()
                try:
                    if tensor_digest(model.state_dict()) != before:
                        raise RuntimeError("Horizon arms did not start from identical weights.")
                    arm["policy"] = evaluate_horizon(config, model, cases, horizon, args, folder)
                    arm["status"] = "COMPLETE"
                except Exception as error:  # noqa: BLE001 - persist errors without hiding successful arms.
                    arm.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                    (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                write_report(args.output, report)
            result["status"] = "FAIL" if any(a["status"] == "FAIL" for a in result["arms"]) else "COMPLETE"
        except Exception as error:  # noqa: BLE001 - still test the other model.
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            del model
        write_report(args.output, report)
    report["seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(run["status"] == "FAIL" for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
