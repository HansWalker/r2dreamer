"""True-state feedback positive control and initial-anchor cost/search comparison.

No model fitting, image rendering, or checkpoint access. The reference is a
privileged simulator feedback policy, not a replacement benchmark controller.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.diagnose_goal_objective import cart_state, feedback_gain
from scripts.diagnose_physical_controller import CartpoleCost, physical_labels
from scripts.diagnose_physical_oracle import OraclePlanner, SimulatorFutures, make_cases
from scripts.diagnose_planner_oracle import simulator_branch


FAMILIES = ("leworldmodel", "temporal_straightening")


def step_policy(env, action):
    _, reward, done, _ = env.step(action)
    if done:
        raise ValueError("Controller comparison crossed an episode boundary.")
    return float(reward), bool(env._env.task.get_reward(env._env.physics) >= 1 - 1e-6)


def feedback_action(env, gain):
    return np.asarray([np.clip(-gain @ cart_state(env), -1, 1)], dtype=np.float32)


def feedback_policy(config, cases, steps, output):
    with ExitStack() as stack:
        envs = make_cases(config, cases, int(config.jepa_model.history_size), stack)
        gain = feedback_gain(envs[0])
        rewards, successes = [], []
        with (output / "feedback_metrics.jsonl").open("w", encoding="utf-8") as log:
            for step in range(steps):
                actions = [feedback_action(env, gain) for env in envs]
                values = [step_policy(env, action) for env, action in zip(envs, actions, strict=True)]
                rewards.append([value[0] for value in values])
                successes.append([value[1] for value in values])
                log.write(json.dumps({"agent_step": step + 1, "case_ids": [c["id"] for c in cases],
                                      "actions": np.asarray(actions).tolist(), "rewards": rewards[-1],
                                      "success": successes[-1], "states": [cart_state(env).tolist() for env in envs]}) + "\n")
        returns = np.asarray(rewards).sum(0)
        tail = min(10, steps)
        sustained = np.asarray(successes[-tail:]).all(0)
        per_case = [{"id": case["id"], "cohort": case["cohort"], "return": float(returns[i]),
                     "sustained": bool(sustained[i]), "success_fraction": float(np.asarray(successes)[:, i].mean())}
                    for i, case in enumerate(cases)]
        return gain, {"return_mean": float(returns.mean()), "maximum_return": steps * int(config.env.action_repeat),
                      "sustained_rate": float(sustained.mean()), "sustained_tail_steps": tail, "cases": per_case,
                      "cohort_return": {group: float(np.mean([c["return"] for c in per_case if c["cohort"] == group]))
                                        for group in sorted({c["cohort"] for c in cases})},
                      "trace_file": "feedback_metrics.jsonl"}


def simulate_sequence(env, horizon, cost, *, gain=None, actions=None):
    """Record a feedback-generated sequence or replay a solver's fixed sequence."""
    if (gain is None) == (actions is None):
        raise ValueError("Supply exactly one of feedback gain or fixed actions.")
    selected, states, rewards, successes = [], [], [], []
    with simulator_branch(env) as branch:
        for step in range(horizon):
            action = feedback_action(branch, gain) if gain is not None else np.asarray(actions[step], dtype=np.float32)
            reward, success = step_policy(branch, action)
            selected.append(action.tolist())
            states.append(cart_state(branch).tolist())
            rewards.append(reward)
            successes.append(success)
    value = cost(physical_labels(torch.tensor(states, dtype=torch.float64))).item()
    return {"cost": value, "return": sum(rewards), "first_action": selected[0], "actions": selected,
            "states": states, "rewards": rewards, "success": successes}


@torch.no_grad()
def selected_sequence(planner, action):
    """Extract the plan whose first action the production solver actually executes."""
    if planner.planner.type == "cem":
        sequence = planner._cem_mean
    else:
        candidates = planner._gradient_actions
        indices = torch.arange(candidates.shape[0], device=planner.device)[:, None]
        unused = torch.zeros(candidates.shape[0], 1, device=planner.device)
        costs = planner._goal_cost(indices, unused, candidates, unused)
        best = costs.argmin(dim=1)
        sequence = candidates[torch.arange(candidates.shape[0], device=planner.device), best]
    torch.testing.assert_close(sequence[:, 0], action, rtol=0, atol=0)
    if not torch.isfinite(sequence).all() or (sequence.abs() > 1).any():
        raise ValueError("Invalid solver-selected action sequence.")
    return sequence.detach().cpu().numpy()


def compare_initial(config, cases, gain, cost, args, seed):
    started = time.monotonic()
    with ExitStack() as stack:
        envs = make_cases(config, cases, int(config.jepa_model.history_size), stack)
        oracle = SimulatorFutures(envs, stack, args.fd_epsilon)
        planner = OraclePlanner(config.jepa_model.planner, oracle, cost, args.device)
        torch.manual_seed(seed + 14_000_000)
        sequence = selected_sequence(planner, planner.action(0))
        horizon = int(config.jepa_model.planner.horizon)
        records = []
        for case, env, actions in zip(cases, envs, sequence, strict=True):
            feedback = simulate_sequence(env, horizon, cost, gain=gain)
            selected = simulate_sequence(env, horizon, cost, actions=actions)
            records.append({"id": case["id"], "cohort": case["cohort"], "feedback": feedback,
                            "solver": selected, "solver_minus_feedback_cost": selected["cost"] - feedback["cost"]})
        return {"cases": records, "seconds": time.monotonic() - started,
                "counterfactual_sequences": oracle.sequences, "counterfactual_agent_steps": oracle.agent_steps,
                "seconds_lookahead": horizon * int(config.env.action_repeat) * envs[0]._env.control_timestep()}


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    feedback = report["feedback"]
    lines = ["Simulator feedback positive control; no neural model or training",
             f"Feedback | return={feedback['return_mean']:.4f}/{feedback['maximum_return']} | sustained={feedback['sustained_rate']:.1%}",
             "Initial anchors only | family | horizon | case | feedback/solver cost | feedback/solver return"]
    for run in report["runs"]:
        for case in run["comparison"]["cases"]:
            feedback, solver = case["feedback"], case["solver"]
            lines.append(f"{run['model']} | {run['horizon']} | {case['id']} | "
                         f"{feedback['cost']:.6g}/{solver['cost']:.6g} | {feedback['return']:g}/{solver['return']:g}")
    lines.extend(report["interpretation_limits"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5])
    parser.add_argument("--policy-steps", type=int)
    parser.add_argument("--fd-epsilon", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if min(args.horizons) < 1 or len(set(args.horizons)) != len(args.horizons):
        parser.error("Horizons must be unique positive integers.")
    if args.policy_steps is not None and args.policy_steps < 1:
        parser.error("Policy steps must be positive.")
    if not math.isfinite(args.fd_epsilon) or not 0 < args.fd_epsilon < .1:
        parser.error("Finite-difference epsilon must be finite and between zero and .1.")
    return args


def main(argv=None):
    args = arguments(argv)
    contents = args.source_report.read_bytes()
    source = json.loads(contents)
    args.policy_steps = args.policy_steps or int(source["settings"]["policy_steps"])
    cost = CartpoleCost(**source["physical_cost"])
    runs = {run["model"]: run for run in source["runs"]}
    cases = [{key: case[key] for key in ("id", "seed", "cohort", "initial_state", "anchor_state", "anchor_success")}
             for case in source["cases"]]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Expected nonempty uniquely identified source cases.")
    configs = {name: OmegaConf.create(runs[name]["config"]) for name in FAMILIES}
    reference = configs[FAMILIES[0]]
    for name, config in configs.items():
        if (runs[name]["status"] != "COMPLETE" or config.env.task != "dmc_cartpole_balance_sparse"
                or config.jepa_model.planner.type != ("cem" if name == "leworldmodel" else "gradient")):
            raise ValueError("Expected completed Cartpole source runs with CEM and gradient solvers.")
        if config.env != reference.env or config.jepa_model.history_size != reference.jepa_model.history_size:
            raise ValueError("Both solvers must use identical environments and history lengths.")
        if max(args.policy_steps, max(args.horizons)) + int(config.jepa_model.history_size) - 1 >= config.env.time_limit // config.env.action_repeat:
            raise ValueError("Requested policy or plan would cross an episode boundary.")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    gain, feedback = feedback_policy(reference, cases, args.policy_steps, args.output)
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "torch", "scipy", "mujoco", "dm-control")}
    versions["python"] = platform.python_version()
    dependencies = [Path(__file__), Path(__file__).with_name("diagnose_goal_objective.py"),
                    Path(__file__).with_name("diagnose_physical_controller.py"),
                    Path(__file__).with_name("diagnose_physical_oracle.py"),
                    Path(__file__).with_name("diagnose_planner_oracle.py"), Path("models/planning.py"), Path("envs/dmc.py")]
    report = {"source_report": str(args.source_report), "source_sha256": hashlib.sha256(contents).hexdigest(),
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "versions": versions, "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
              "cases": cases, "physical_cost": source["physical_cost"], "feedback_gain": gain.tolist(),
              "feedback": feedback, "runs": [], "training_updates": 0, "checkpoint_reads": False, "checkpoint_writes": False,
              "interpretation_limits": [
                  "Feedback uses privileged true state and a simulator-derived LQR gain; it is a diagnostic positive control.",
                  "Closed-loop success is checked for feedback; solver comparisons use fresh initial anchors only, without warm starts.",
                  "The comparison scores feedback-generated and solver-selected sequences under the identical source physical cost.",
                  "A lower feedback cost exposes a missed feasible sequence at that anchor; the reverse is not proof of global solver optimality.",
                  "A successful feedback policy can have higher short-horizon cost than a failing policy: short-term cost need not favor long-term recovery.",
                  "CPU and CUDA generate different candidate draws even with identical seeds; these runs need not reproduce saved CUDA solver actions.",
                  "CEM extraction uses its final elite mean; gradient extraction uses the minimum-cost final restart, matching the executed first action."]}
    write_report(args.output, report)
    print(f"Feedback | return={feedback['return_mean']:.4f}/{feedback['maximum_return']} | tail={feedback['sustained_rate']:.1%}", flush=True)
    for name, source_config in configs.items():
        for horizon in args.horizons:
            config = copy.deepcopy(source_config)
            config.jepa_model.planner.horizon = horizon
            print(f"Initial-anchor comparison | {name} | horizon={horizon}", flush=True)
            result = compare_initial(config, cases, gain, cost, args, int(source["settings"]["seed"]))
            report["runs"].append({"model": name, "horizon": horizon, "status": "COMPLETE",
                                   "planner": OmegaConf.to_container(config.jepa_model.planner, resolve=True),
                                   "source_planner_device": str(config.device),
                                   "matched_rng_backend": torch.device(args.device).type == torch.device(config.device).type,
                                   "comparison": result})
            write_report(args.output, report)
    report["seconds"] = time.monotonic() - started
    write_report(args.output, report)
    print(f"Report | {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
