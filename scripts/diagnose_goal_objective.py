"""Audit native latent goal selection with true simulator futures, not learned dynamics."""

import argparse
import hashlib
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.linalg import solve_discrete_are

from dmc_expert.storage import dataset_identity
from envs.dmc import goal_relation, goal_relation_spec, make_env
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_planner_oracle import (
    encode_images,
    rank_correlation,
    selection,
    simulator_branch,
)
from scripts.smoke_tiny_planners import FAMILIES
from scripts.train_planner_check import build_config
from scripts.train_rollout_check import pretrain
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256

# x, angle, x velocity, angular velocity. These are controlled interventions,
# not a sample of the task's reset distribution or of an agent's replay.
PROFILES = {
    "balanced": ("balanced", [0., 0., 0., 0.]),
    "boundary": ("boundary", [.18, .08, 0., .15]),
    "moving_out": ("boundary", [0., .06, .25, .6]),
    "pole_outside": ("failure", [.02, .12, 0., .2]),
    "cart_outside": ("failure", [.28, .03, .05, 0.]),
    "far_failure": ("failure", [0., .6, 0., 0.]),
}


def cart_state(env):
    physics = env._env.physics
    return np.concatenate((physics.data.qpos, physics.data.qvel)).copy()


def set_cart_state(env, state):
    if (env._domain, env._task) != ("cartpole", "balance_sparse") or len(state) != 4:
        raise ValueError("This diagnostic only supports single-pole balance_sparse.")
    with env._env.physics.reset_context():
        env._env.physics.data.qpos[:] = state[:2]
        env._env.physics.data.qvel[:] = state[2:]


def feedback_gain(env):
    """A local simulator LQR supplies candidates only; neither model uses this policy."""
    def successor(state, action):
        with simulator_branch(env) as branch:
            branch._observation = lambda time_step: {}
            set_cart_state(branch, state)
            branch.step(np.asarray([action], dtype=np.float32))
            return cart_state(branch)

    epsilon = 1e-4
    basis = np.eye(4) * epsilon
    a = np.stack([(successor(v, 0) - successor(-v, 0)) / (2 * epsilon) for v in basis], axis=1)
    b = ((successor(np.zeros(4), epsilon) - successor(np.zeros(4), -epsilon)) / (2 * epsilon))[:, None]
    q, r = np.diag([4., 20., 1., 1.]), np.eye(1)
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    if not np.isfinite(gain).all() or max(abs(np.linalg.eigvals(a - b @ gain))) >= 1:
        raise ValueError("Could not construct a stable local feedback candidate.")
    return gain[0]


def candidate_bank(count, horizon, seed):
    rng = np.random.default_rng(seed)
    values, labels = [], []
    for value in (0., -1., 1., -.5, .5):
        values.append(np.full((horizon, 1), value, dtype=np.float32))
        labels.append(f"constant_{value:g}")
    for switch in (.25, .5, .75):
        for sign in (-1., 1.):
            sequence = np.full((horizon, 1), sign, dtype=np.float32)
            sequence[max(1, int(horizon * switch)):] *= -1
            values.append(sequence)
            labels.append(f"switch_{switch:g}_{sign:g}")
    while len(values) < count - 2:
        blocks = rng.uniform(-1, 1, (math.ceil(horizon / 5), 1)).astype(np.float32)
        values.append(np.repeat(blocks, 5, axis=0)[:horizon])
        labels.append(f"random_blocks_{len(values)}")
    # Last two rows are filled by simulator feedback during collection, then
    # treated as ordinary fixed open-loop sequences by every scoring method.
    values.extend([np.zeros((horizon, 1), dtype=np.float32) for _ in range(2)])
    labels.extend(["simulator_feedback_half", "simulator_feedback"])
    return np.stack(values), labels


def simulate_case(env, actions, horizons, gain):
    images, rewards, successes, states = [], [], [], []
    for index, sequence in enumerate(actions):
        frames, reward_trace, success_trace, state_trace = [], [], [], []
        with simulator_branch(env) as branch:
            # Keep native stepping/rewards, but render only the scored endpoints.
            branch._observation = lambda time_step: {}
            for step in range(len(sequence)):
                if index >= len(actions) - 2:
                    scale = .5 if index == len(actions) - 2 else 1.
                    sequence[step, 0] = np.clip(-scale * (gain @ cart_state(branch)), -1., 1.)
                _, reward, done, _ = branch.step(sequence[step])
                if done:
                    raise ValueError("Simulator branch reached an episode boundary.")
                reward_trace.append(float(reward))
                success_trace.append(float(branch._env.task.get_reward(branch._env.physics)) >= 1 - 1e-6)
                if step + 1 in horizons:
                    frames.append(np.asarray(branch.render(), dtype=np.uint8))
                    state_trace.append(cart_state(branch))
        images.append(np.stack(frames))
        rewards.append(reward_trace)
        successes.append(success_trace)
        states.append(state_trace)
    return {"image": torch.from_numpy(np.stack(images)), "rewards": np.asarray(rewards),
            "successes": np.asarray(successes), "states": np.asarray(states)}


def collect_objective_cases(config, args):
    if args.horizons[-1] >= int(config.env.time_limit) // int(config.env.action_repeat):
        raise ValueError("Diagnostic horizon would reach the episode time limit.")
    cases, gain = [], None
    progress = Progress("True simulator futures", len(args.sim_seeds) * len(PROFILES))
    for seed in args.sim_seeds:
        env = make_env(config.env, seed, include_physical_state=False)
        try:
            env.reset()
            if gain is None:
                gain = feedback_gain(env)
            for index, (name, (cohort, nominal)) in enumerate(PROFILES.items()):
                observation = env.reset()
                rng = np.random.default_rng(seed + index)
                state = np.asarray(nominal) + rng.uniform(-1, 1, 4) * [.005, .002, .005, .01]
                if seed % 2:
                    state *= -1
                set_cart_state(env, state)
                before = env._env.physics.get_state().copy()
                actions, labels = candidate_bank(args.candidates, args.horizons[-1], seed + index)
                result = simulate_case(env, actions, args.horizons, gain)
                np.testing.assert_array_equal(before, env._env.physics.get_state())
                cases.append({"id": f"{seed}/{name}", "seed": seed, "profile": name, "cohort": cohort,
                              "anchor_state": state.tolist(),
                              "anchor_relation": goal_relation(env._env.physics, env._domain, env._task).tolist(),
                              "anchor_success": float(env._env.task.get_reward(env._env.physics)) >= 1 - 1e-6,
                              "goal_image": torch.from_numpy(observation["goal_image"].copy()),
                              "tolerance": goal_relation_spec(env._env.physics, env._domain, env._task)["tolerance"],
                              "feedback_gain": gain.tolist(), "action": actions, "candidate_labels": labels, **result})
                progress.update(len(cases), force=len(cases) == progress.total)
        finally:
            env.close()
    return cases


def selection_metrics(cost, returns, sustained, terminal_success, maximum):
    result = selection(cost, returns)
    chosen = result["tied_candidates"]
    spread = float(np.ptp(returns))
    result.update(normalized_return=result["return_mean"] / maximum,
                  regret_fraction=result["regret"] / spread if spread > 1e-6 else None,
                  sustained_rate=float(np.mean(sustained[chosen])),
                  terminal_success_rate=float(np.mean(terminal_success[chosen])))
    return result


def score_horizon(case, cost, horizon, index, action_repeat):
    returns = case["rewards"][:, :horizon].sum(1)
    tail = min(10, horizon)
    sustained = case["successes"][:, horizon - tail:horizon].all(1)
    terminal = case["successes"][:, horizon - 1]
    relation = case["states"][:, index, :2].copy()
    relation[:, 1] = np.arctan2(np.sin(relation[:, 1]), np.cos(relation[:, 1]))
    physical = np.square(relation / np.asarray(case["tolerance"])).sum(1)
    maximum = float(horizon * action_repeat)
    spread = float(np.ptp(returns))
    methods = {"latent": cost, "uniform": np.zeros_like(cost), "physical_terminal": physical,
               "reward_oracle": -returns}
    return {"id": case["id"], "profile": case["profile"], "cohort": case["cohort"], "horizon": horizon,
            "anchor_success": case["anchor_success"], "latent_cost": np.asarray(cost).tolist(),
            "physical_cost": physical.tolist(), "returns": returns.tolist(), "maximum_return": maximum,
            "sustained": sustained.tolist(), "tail_agent_steps": tail, "return_spread": spread,
            "reward_informative": spread >= max(float(action_repeat), .1 * maximum) - 1e-6,
            "recovery_available": bool(not case["anchor_success"] and sustained.any()),
            "sustain_choice_informative": bool(sustained.any() and not sustained.all()),
            "cost_return_rank": rank_correlation(-np.asarray(cost), returns),
            "cost_physical_rank": rank_correlation(cost, physical),
            "selection": {name: selection_metrics(values, returns, sustained, terminal, maximum)
                          for name, values in methods.items()}}


@torch.no_grad()
def score_objective(model, cases, horizons, chunk_size, action_repeat):
    rows = []
    with readout_mode(model):
        for case in cases:
            goal = encode_images(model, case["goal_image"][None], chunk_size)[0]
            encoded = encode_images(model, case["image"].flatten(0, 1), chunk_size)
            encoded = encoded.reshape(len(case["action"]), len(horizons), *encoded.shape[1:])
            error = (encoded - goal).float().square().flatten(2)
            cost = error.sum(-1) if model.goal_reduction == "sum" else error.mean(-1)
            if not torch.isfinite(cost).all():
                raise ValueError("Non-finite latent goal costs.")
            for index, horizon in enumerate(horizons):
                rows.append(score_horizon(case, cost[:, index].cpu().numpy(), horizon, index, action_repeat))
    return rows


def aggregate(rows):
    informative = [r for r in rows if r["reward_informative"]]
    recovery = [r for r in rows if r["recovery_available"]]
    methods = {}
    def mean(group, name, key):
        return float(np.mean([r["selection"][name][key] for r in group])) if group else None

    for name in ("latent", "uniform", "physical_terminal", "reward_oracle"):
        methods[name] = {"normalized_return": mean(informative, name, "normalized_return"),
                         "regret_fraction": mean(informative, name, "regret_fraction"),
                         "recovery_rate": mean(recovery, name, "sustained_rate")}
    return {"cases": len(rows), "reward_informative": len(informative),
            "failure_cases": sum(not r["anchor_success"] for r in rows),
            "recoverable_failures": len(recovery), "selection": methods,
            "coverage": "CONTRAST" if len(informative) >= 4 else "LOW_CONTRAST"}


def case_metadata(case):
    result = {key: value for key, value in case.items() if key not in ("image", "goal_image")}
    result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    result["images_sha256"] = tensor_digest({"future": case["image"], "goal": case["goal_image"]})
    return result


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Goal objective check | true simulator futures | frozen encoders | no learned dynamics or physical-head scoring",
             "Model | Horizon | Reward cases | Return latent/random/best | Regret fraction | Recoverable failures | Recovery latent/random | Coverage"]
    def number(value):
        return "n/a" if value is None else f"{value:.3f}"
    for run in report["runs"]:
        for horizon, group in run.get("summary", {}).items():
            selected = group["selection"]
            returns = "/".join(number(selected[key]["normalized_return"]) for key in ("latent", "uniform", "reward_oracle"))
            recovery = "/".join(number(selected[key]["recovery_rate"]) for key in ("latent", "uniform"))
            lines.append(f"{run['model']} | {horizon} | {group['reward_informative']}/{group['cases']} | {returns} | "
                         f"{number(selected['latent']['regret_fraction'])} | "
                         f"{group['recoverable_failures']}/{group['failure_cases']} | {recovery} | {group['coverage']}")
        lines.append(f"{run['status']} | {run['model']}" + (f" | {run['error']}" if "error" in run else ""))
    lines += ["Returns are normalized by horizon * action_repeat and averaged only over reward-informative anchors.",
              "Informative = reward spread >= max(action_repeat, 10% of maximum return). LOW_CONTRAST = fewer than four anchors.",
              "Regret fraction = (candidate-best return - selected return) / candidate return spread; lower is better.",
              "Recovery = success at every endpoint of the last min(10, horizon) agent steps, starting outside the success set.",
              "Recovery rates include only anchors with at least one successful candidate; unavailable recovery is not an objective failure.",
              "Random averages all candidates; cost ties are averaged, not broken using reward. Best is only a finite-candidate upper bound.",
              "Candidates include two privileged simulator-feedback sequences, shared across models; this is NOT a deployable controller.",
              "Horizon 5 is the current recipe; other horizons are diagnostic lookahead, not production changes. No action optimizer or online training.",
              "Constructed mirrored/jittered boundary and failure states are not the natural reset/replay distribution.",
              "Physical-distance baseline, velocities, actions, per-anchor scores and cohort summaries are in report.json.",
              "COMPLETE means execution, not policy success. Tiny expert-trained encoders and short branches do not validate full-run stability."]
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
    parser.add_argument("--sim-seeds", nargs="+", type=int, default=[12_000_000, 12_000_001])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 25, 100])
    parser.add_argument("--candidates", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("goal_objective_check_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if min(args.expert_updates, args.batch_size, *args.horizons) < 1 or args.candidates < 13:
        parser.error("Use positive updates, batch size and horizons, and at least 13 candidates.")
    if args.seed < 0 or min(args.sim_seeds) < 0:
        parser.error("Seeds must be nonnegative.")
    if any(len(values) != len(set(values)) for values in (args.models, args.sim_seeds, args.horizons)):
        parser.error("Models, simulator seeds and horizons must be unique.")
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
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"experiment": "true_future_goal_objective", "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "checkpoint_reads": False, "checkpoint_writes": False, "production_settings_changed": False, "runs": []}
    config = build_config(args.models[0], args)
    print(f"Goal objective | models={len(args.models)} | offline={args.expert_updates} once/model | "
          f"anchors={len(args.sim_seeds) * len(PROFILES)} | candidates={args.candidates} | "
          f"horizons={args.horizons} | no checkpoints", flush=True)
    cases = collect_objective_cases(config, args)
    report["cases"] = [case_metadata(case) for case in cases]
    write_report(args.output, report)
    for name in args.models:
        result = {"model": name, "status": "RUNNING"}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        model = None
        try:
            config = build_config(name, args)
            result["config"] = OmegaConf.to_container(config, resolve=True)
            family = load_model_family(name)
            with family.build_replay(config) as dataset:
                result["dataset_identity"] = dataset_identity(dataset.metadata)
                model = pretrain(config, dataset, args, output, result)
            before = tensor_digest(model.state_dict())
            rows = score_objective(model, cases, args.horizons, args.batch_size, int(config.env.action_repeat))
            if before != tensor_digest(model.state_dict()):
                raise RuntimeError("Goal scoring changed model weights or buffers.")
            result.update(status="COMPLETE", model_unchanged_during_scoring=True, cases=rows,
                          summary={str(h): aggregate([r for r in rows if r["horizon"] == h]) for h in args.horizons},
                          cohorts={str(h): {cohort: aggregate([r for r in rows if r["horizon"] == h and r["cohort"] == cohort])
                                             for cohort in ("balanced", "boundary", "failure")} for h in args.horizons})
        except Exception as error:  # noqa: BLE001 - persist failures and still test the other model.
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
