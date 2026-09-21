"""Closed-loop physical planning with true MuJoCo states and futures; no fitting."""

import argparse
import copy
import hashlib
import json
import math
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from envs.dmc import make_env
from models.planning import LatentPlanner
from scripts.diagnose_goal_objective import cart_state, set_cart_state
from scripts.diagnose_physical_controller import CartpoleCost, physical_labels
from scripts.diagnose_planner_oracle import simulator_branch
from training.progress import Progress, duration


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def integration_state(physics):
    state = np.empty(mujoco.mj_stateSize(physics.model.ptr, STATE_SPEC))
    mujoco.mj_getState(physics.model.ptr, physics.data.ptr, state, STATE_SPEC)
    return state


class SimulatorFutures:
    """Reuse private physics copies, restoring full integration state per candidate."""

    def __init__(self, envs, stack, epsilon):
        if any((env._domain, env._task) != ("cartpole", "balance_sparse") for env in envs):
            raise ValueError("The physics-only rollout is specific to cartpole/balance_sparse.")
        self.envs, self.epsilon = envs, epsilon
        self.branches = [stack.enter_context(simulator_branch(env)) for env in envs]
        self.sequences = self.agent_steps = 0
        self.refresh()

    def refresh(self):
        self.anchors = tuple(integration_state(env._env.physics) for env in self.envs)

    def rollout(self, indices, actions, anchors=None):
        anchors = self.anchors if anchors is None else anchors
        batch, samples, horizon, _ = actions.shape
        result = np.empty((batch, samples, horizon, 4), dtype=np.float64)
        for row, index in enumerate(indices.reshape(-1)):
            branch = self.branches[int(index)]
            physics = branch._env.physics
            for sample, sequence in enumerate(actions[row]):
                mujoco.mj_setState(physics.model.ptr, physics.data.ptr, anchors[index], STATE_SPEC)
                # DM Control's legacy Euler step expects position/velocity fields
                # to be current. mj_step1 preserves the restored warmstart inputs.
                mujoco.mj_step1(physics.model.ptr, physics.data.ptr)
                for step, action in enumerate(sequence):
                    control = (np.clip(action, -1, 1) + 1) / 2 * (branch._action_high - branch._action_low) + branch._action_low
                    # Cartpole before_step only sets control; after_step only
                    # changes visualization. No rewards, observations or resets
                    # are needed for counterfactual physics trajectories.
                    physics.set_control(control)
                    for _ in range(branch._action_repeat):
                        physics.step(branch._env._n_sub_steps)
                    result[row, sample, step] = cart_state(branch)
        self.sequences += batch * samples
        self.agent_steps += batch * samples * horizon
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite simulator future.")
        return result


class SimulatorRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, actions, indices, oracle):
        ctx.actions = actions.detach().cpu().double().numpy().copy()
        ctx.indices = indices.detach().cpu().numpy().astype(np.int64)
        ctx.oracle, ctx.anchors = oracle, oracle.anchors
        ctx.device, ctx.dtype = actions.device, actions.dtype
        value = oracle.rollout(ctx.indices, ctx.actions, ctx.anchors)
        return torch.as_tensor(value, device=actions.device)

    @staticmethod
    def backward(ctx, output_gradient):
        gradient = np.empty_like(ctx.actions)
        upstream = output_gradient.detach().cpu().double().numpy()
        # Candidates are independent: perturb one time coordinate across every
        # candidate at once, then contract the state Jacobian with the cost VJP.
        for step in range(ctx.actions.shape[2]):
            plus, minus = ctx.actions.copy(), ctx.actions.copy()
            plus[:, :, step, 0] = np.minimum(1, plus[:, :, step, 0] + ctx.oracle.epsilon)
            minus[:, :, step, 0] = np.maximum(-1, minus[:, :, step, 0] - ctx.oracle.epsilon)
            denominator = plus[:, :, step, 0] - minus[:, :, step, 0]
            difference = (ctx.oracle.rollout(ctx.indices, plus, ctx.anchors)
                          - ctx.oracle.rollout(ctx.indices, minus, ctx.anchors))
            gradient[:, :, step, 0] = (difference * upstream).sum(axis=(2, 3)) / denominator
        return torch.as_tensor(gradient, device=ctx.device, dtype=ctx.dtype), None, None


class OraclePlanner(LatentPlanner):
    """Use the existing CEM/Adam planning loops without constructing a neural model."""

    def __init__(self, settings, oracle, cost, device):
        torch.nn.Module.__init__(self)
        self.planner, self.oracle, self.cost = copy.deepcopy(settings), oracle, cost
        self.action_dim = 1
        self.register_buffer("_device_anchor", torch.empty(0, device=device))
        self._cem_mean = self._gradient_actions = None

    @property
    def device(self):
        return self._device_anchor.device

    def encode(self, history):
        return history["case_index"]

    def _goal_cost(self, history, past_action, candidates, goal):
        future = SimulatorRollout.apply(candidates, history, self.oracle)
        return self.cost(physical_labels(future))

    def action(self, step):
        batch = len(self.oracle.envs)
        indices = torch.arange(batch, device=self.device)[:, None]
        history = {"case_index": indices}
        unused = torch.zeros(batch, 1, device=self.device)
        first = torch.full((batch,), step == 0, dtype=torch.bool, device=self.device)
        method = self._cem if self.planner.type == "cem" else self._gradient_plan
        with torch.no_grad():
            return method(history, unused, True, first, unused)


def make_cases(config, cases, context, stack):
    settings = copy.deepcopy(config.env)
    settings.goal = None
    envs = []
    for case in cases:
        env = make_env(settings, case["seed"], include_physical_state=False)
        stack.callback(env.close)
        env._observation = lambda time_step: {}
        env.reset()
        set_cart_state(env, case["initial_state"])
        for _ in range(context - 1):
            env.step(np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(cart_state(env), case["anchor_state"], rtol=0, atol=1e-10)
        envs.append(env)
    return envs


def evaluate(config, cases, source_settings, cost, args, output):
    started = time.monotonic()
    with ExitStack() as stack:
        envs = make_cases(config, cases, int(config.jepa_model.history_size), stack)
        oracle = SimulatorFutures(envs, stack, args.fd_epsilon)
        planner = OraclePlanner(config.jepa_model.planner, oracle, cost, args.device)
        returns = np.zeros(len(cases))
        success_trace, action_trace, planning_times = [], [], []
        progress = Progress(f"True-state {config.model_family}", args.policy_steps)
        with (output / "policy_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for step in range(args.policy_steps):
                torch.manual_seed(int(source_settings["seed"]) + 14_000_000 + step)
                oracle.refresh()
                tick = time.monotonic()
                actions = planner.action(step).detach().cpu().numpy()
                planning_times.append(time.monotonic() - tick)
                if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                    raise RuntimeError("Planner produced invalid actions.")
                rewards, successes, states = [], [], []
                for env, action in zip(envs, actions, strict=True):
                    _, reward, done, _ = env.step(action)
                    if done:
                        raise RuntimeError("Unexpected episode boundary in closed-loop test.")
                    rewards.append(float(reward))
                    successes.append(float(env._env.task.get_reward(env._env.physics)) >= 1 - 1e-6)
                    states.append(cart_state(env).tolist())
                returns += rewards
                action_trace.append(actions)
                success_trace.append(successes)
                log.write(json.dumps({"agent_step": step + 1, "case_ids": [c["id"] for c in cases],
                                      "actions": actions.tolist(), "rewards": rewards, "success": successes,
                                      "states": states, "policy_seconds": planning_times[-1]}, allow_nan=False) + "\n")
                progress.update(step + 1, f"return={returns.mean():.2f}", force=step + 1 == args.policy_steps)
        tail = min(10, args.policy_steps)
        sustained = np.asarray(success_trace[-tail:]).all(0)
        maximum = args.policy_steps * int(config.env.action_repeat)
        per_case = [{"id": c["id"], "cohort": c["cohort"], "anchor_success": c["anchor_success"],
                     "return": float(returns[i]), "sustained": bool(sustained[i]),
                     "success_fraction": float(np.mean(np.asarray(success_trace)[:, i]))}
                    for i, c in enumerate(cases)]
        return {"return_mean": float(returns.mean()), "maximum_return": maximum, "cases": per_case,
                "sustained_rate": float(sustained.mean()), "sustained_tail_steps": tail,
                "cohort_return": {cohort: float(np.mean([c["return"] for c in per_case if c["cohort"] == cohort]))
                                  for cohort in sorted({c["cohort"] for c in cases})},
                "action_saturation_095": float((np.abs(action_trace) >= .95).mean()),
                "planning_seconds": sum(planning_times), "seconds": time.monotonic() - started,
                "counterfactual_sequences": oracle.sequences, "counterfactual_agent_steps": oracle.agent_steps,
                "seconds_lookahead": int(planner.planner.horizon) * int(config.env.action_repeat) * envs[0]._env.control_timestep()}


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["True-state physical controller | true MuJoCo futures | no fitting or checkpoint access",
             "Model/solver | Return/max | Balanced/boundary/failure | Sustained tail | Time"]
    for run in report["runs"]:
        if run["status"] != "COMPLETE":
            lines.append(f"{run['status']} | {run['model']} | {run.get('error', '')}")
            continue
        policy = run["policy"]
        cohorts = "/".join(f"{policy['cohort_return'].get(key, float('nan')):.2f}" for key in ("balanced", "boundary", "failure"))
        lines.append(f"{run['model']}/{run['planner']['type']} | {policy['return_mean']:.2f}/{policy['maximum_return']} | "
                     f"{cohorts} | {policy['sustained_rate']:.0%} | {duration(policy['seconds'])}")
        if run["matched_policy_length"]:
            for phase, baseline in run["learned_physical_baselines"].items():
                lines.append(f"  Saved {phase} physical controller | return={baseline['return_mean']:.2f} | "
                             f"sustained={baseline['sustained_rate']:.0%}")
        else:
            lines.append("  Different policy length: saved returns are not directly comparable.")
        if not run["matched_rng_backend"]:
            lines.append("  Different CPU/CUDA RNG backend: seeds match, but planner random draws do not.")
    lines += ["COMPLETE means execution, not successful balancing. Inspect per-case traces, not only mean return.",
              "Same source starts, action repeats, physical cost, planner budgets, warm starts and per-call seeds.",
              "TS uses finite-difference simulator derivatives with the existing Adam/tanh planner; not exact autograd.",
              "MuJoCo rollouts are CPU work even with CUDA planner tensors; timing is not a neural-planner speed comparison.",
              "No learned encoder, state head, dataset, native latent objective, training update or checkpoint is used.",
              "Success would isolate errors in the learned route; failure would leave cost/horizon/solver limitations unresolved."]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True, help="Report from the physical learning-curve test.")
    parser.add_argument("--models", nargs="+", choices=("leworldmodel", "temporal_straightening"),
                        default=["leworldmodel", "temporal_straightening"])
    parser.add_argument("--device", default="cuda:0", help="Planner tensor/RNG device; MuJoCo remains CPU-based.")
    parser.add_argument("--fd-epsilon", type=float, default=1e-3)
    parser.add_argument("--policy-steps", type=int, help="Defaults to the source test length; overrides break length matching.")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("physical_oracle_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        parser.error("Models must be unique.")
    if not math.isfinite(args.fd_epsilon) or not 0 < args.fd_epsilon < .1:
        parser.error("Finite-difference epsilon must be finite and between 0 and 0.1.")
    if args.policy_steps is not None and args.policy_steps < 1:
        parser.error("Policy steps must be positive.")
    return args


def main(argv=None):
    args = arguments(argv)
    data = args.source_report.read_bytes()
    source = json.loads(data)
    args.policy_steps = args.policy_steps or int(source["settings"]["policy_steps"])
    cost = CartpoleCost(**source["physical_cost"])
    runs = {run["model"]: run for run in source["runs"]}
    cases = [{key: case[key] for key in ("id", "seed", "cohort", "initial_state", "anchor_state", "anchor_success")}
             for case in source["cases"]]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Source must contain nonempty, uniquely identified cases.")
    configs = {}
    for name in args.models:
        run = runs[name]
        config = OmegaConf.create(run["config"])
        horizon = int(config.jepa_model.planner.horizon)
        if (run["status"] != "COMPLETE" or config.env.task != "dmc_cartpole_balance_sparse"
                or horizon != source["settings"]["horizon"] or config.jepa_model.planner.type not in ("cem", "gradient")):
            raise ValueError("Expected a completed cartpole physical-controller run with matched planner settings.")
        if args.policy_steps + horizon + int(config.jepa_model.history_size) - 1 >= config.env.time_limit // config.env.action_repeat:
            raise ValueError("Requested policy and forecast horizon could reach the episode boundary.")
        configs[name] = config
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu (different planner random draws).")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"source_report": str(args.source_report), "source_sha256": hashlib.sha256(data).hexdigest(),
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "physical_cost": source["physical_cost"], "cases": cases, "runs": [],
              "checkpoint_reads": False, "checkpoint_writes": False, "training_updates": 0}
    print(f"Physical oracle | models={len(configs)} | cases={len(cases)} | policy_steps={args.policy_steps} | no training", flush=True)
    started = time.monotonic()
    for name, config in configs.items():
        output = args.output / name
        output.mkdir()
        endpoints = {s["phase"]: s for s in runs[name]["snapshots"]}
        result = {"model": name, "status": "RUNNING", "planner": OmegaConf.to_container(config.jepa_model.planner),
                  "matched_policy_length": args.policy_steps == source["settings"]["policy_steps"],
                  "matched_rng_backend": torch.device(args.device).type == torch.device(config.device).type,
                  "source_planner_device": config.device,
                  "learned_physical_baselines": {phase: s["physical_controller"]["policy"] for phase, s in endpoints.items()}}
        report["runs"].append(result)
        write_report(args.output, report)
        try:
            result["policy"] = evaluate(config, cases, source["settings"], cost, args, output)
            result["status"] = "COMPLETE"
        except Exception as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_report(args.output, report)
        print(f"{result['status']} | {name}" + (f" | {result['error']}" if "error" in result else ""), flush=True)
    report["seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(run["status"] != "COMPLETE" for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
