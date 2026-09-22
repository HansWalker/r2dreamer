"""Runtime-calibrated offline TS/LeWM recipe and action-coverage comparison.

No online collection, online optimizer phase, or online retention/schedule changes.
The default profile retains production model widths and native planner costs.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

import tools
from dmc_expert.storage import dataset_identity
from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import observe_prefix, set_cart_state
from scripts.diagnose_planner_oracle import simulator_branch
from scripts.paper_faithful_support import (
    BranchReplay, collect_branch_bank, make_split_manifest, score_branches,
)
from scripts.smoke_tiny_planners import native_control
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256


ARMS = {
    "ts_patch_01": ("temporal_straightening", "patch", .1, 0.),
    "ts_patch_001": ("temporal_straightening", "patch", .01, 0.),
    "ts_agg_01": ("temporal_straightening", "agg", .1, 0.),
    "ts_agg_01_coverage": ("temporal_straightening", "agg", .1, .5),
    "lewm": ("leworldmodel", "patch", .1, 0.),
    "lewm_coverage": ("leworldmodel", "patch", .1, .5),
}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def build_config(arm, args):
    family, mode, weight, _ = ARMS[arm]
    overrides = [f"device={args.device}", f"seed={args.seed}", "scenario=cartpole_balance_sparse",
                 f"env.dataset_root={json.dumps(str(args.dataset_root.resolve()))}",
                 f"replay.batch_size={args.batch_size}", f"replay.episodes_per_batch={args.sources}"]
    if args.profile == "small":
        overrides += ["jepa_model.encoder.embedding_dim=64", "jepa_model.encoder.vision.base_channels=4",
                      "jepa_model.encoder.vision.layers=2", "jepa_model.encoder.vision.heads=2",
                      "jepa_model.encoder.vision.mlp_dim=128", "jepa_model.projector_dim=128",
                      "jepa_model.predictor.layers=2", "jepa_model.predictor.heads=2",
                      "jepa_model.predictor.dim_head=32", "jepa_model.predictor.mlp_dim=128",
                      "jepa_model.predictor.action_embedding_dim=4"]
    elif args.profile == "tiny":
        overrides += ["jepa_model.encoder.embedding_dim=16", "jepa_model.encoder.vision.base_channels=2",
                      "jepa_model.encoder.vision.layers=1", "jepa_model.encoder.vision.heads=1",
                      "jepa_model.encoder.vision.mlp_dim=32", "jepa_model.projector_dim=32",
                      "jepa_model.predictor.layers=1", "jepa_model.predictor.heads=1",
                      "jepa_model.predictor.dim_head=16", "jepa_model.predictor.mlp_dim=32",
                      "jepa_model.predictor.action_embedding_dim=4"]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name=f"{family}_dmc_vision", overrides=overrides)
    with open_dict(config):
        config.jepa_model.curvature_mode = mode
        config.jepa_model.curvature_weight = weight
        config.jepa_model.aggregation = {"hidden_dim": 512, "output_dim": 128}
    # Head size is upstream even in small-model diagnostics; label model widths separately.
    config.state_head.samples_per_update = min(int(config.state_head.samples_per_update), 2 * args.batch_size)
    config.jepa_model.planner.horizon = args.horizon
    config.training.expert.batch_size = args.batch_size
    OmegaConf.resolve(config)
    return config


def new_model(config, dataset=None):
    tools.configure_randomness(int(config.seed), bool(config.deterministic_run))
    model = load_model_family(str(config.model_family)).build_model(config)
    if dataset is not None:
        model.state_head.set_stats(dataset.state_mean, dataset.state_std)
    return model


def native_weights(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("state_head.")}


def common_initial_weights(model):
    return {k: v for k, v in native_weights(model).items()
            if not k.startswith(("encoder.agg_mlp.", "encoder.agg_post_norm."))}


def mixed_batch(expert, branches, fraction):
    """Use the production native+detached-readout update without label leakage."""
    if not fraction:
        return expert
    obs, actions = expert[:2]
    count = round(actions.shape[0] * fraction)
    branch_obs, branch_actions = branches[:2]
    if count != branch_actions.shape[0] or set(obs) != set(branch_obs):
        raise ValueError("Coverage batch must preserve the configured image/label layout and row count.")
    keep = actions.shape[0] - count
    return ({key: torch.cat((obs[key][:keep], branch_obs[key]), 0) for key in obs},
            torch.cat((actions[:keep], branch_actions), 0))


def replay_for(bank, args, fraction):
    return BranchReplay(bank, batch_size=max(1, round(args.batch_size * fraction)), sequence_length=4,
                        episodes_per_batch=max(1, round(args.sources * fraction)), seed=args.seed + 12345)


def update(model, expert, branch, fraction, step, seed):
    torch.manual_seed(seed + step)
    model.train()
    batch = mixed_batch(expert.sample_episode_batch(),
                        branch.sample_training_batch() if fraction else None, fraction)
    values = model.update(batch)
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Non-finite training metric")
    return {key: float(value) for key, value in values.items()}


def choose_updates(seconds_remaining, rates, evaluation_seconds, minimum, maximum):
    """One fixed count for ALL arms; runtime must never select the better-trained arm."""
    per_round = sum(rates)
    available = seconds_remaining - 1.25 * evaluation_seconds - 60
    count = min(maximum, math.floor(max(0., available) / (1.2 * per_round))) if per_round > 0 else 0
    if count < minimum:
        raise ValueError(f"Runtime budget permits only {count} common updates (< {minimum}); "
                         "increase --minutes or explicitly choose --profile small. No arms were trained.")
    return count


def planner_settings(model, budget):
    settings = copy.deepcopy(model.planner)
    if budget == "source":
        settings.samples = 64 if settings.type == "cem" else 4
        settings.iterations = 6 if settings.type == "cem" else 8
        if settings.type == "cem":
            settings.elites = 8
    return settings


def solver_case(cases):
    # Prespecified cohort, selected without inspecting any model predictions.
    return next((case for case in cases if case["profile"] == "boundary"), cases[0])


@tools.preserve_rng_state
def solver_probe(model, case, budget="production"):
    """Score the executed CEM mean / selected GD restart with the real native cost."""
    original, caches = model.planner, (model._cem_mean, model._gradient_actions)
    before = tensor_digest(model.state_dict())
    model.planner = planner_settings(model, budget)
    model._cem_mean = model._gradient_actions = None
    try:
        frames = case["prefix"][None].to(model.device)
        past = case["past_action"][None].to(model.device)
        goal_image = case["goal_image"][None].to(model.device)
        with readout_mode(model):
            latent = model.encode({"image": frames})
            goal = model.encode({"image": goal_image[:, None]})[:, 0]
            synchronize(model.device)
            start = time.monotonic()
            action = native_control(model, lambda: model.act(
                {"image": frames, "goal_image": goal_image}, past,
                deterministic=True, first=torch.ones(1, dtype=torch.bool, device=model.device)))
            synchronize(model.device)
            elapsed = time.monotonic() - start
            if model.planner.type == "cem":
                sequence = model._cem_mean.clone()
            else:
                candidates = model._gradient_actions
                costs = model._goal_cost(latent, past, candidates, goal)
                sequence = candidates[torch.arange(1, device=model.device), costs.argmin(1)]
            torch.testing.assert_close(action, sequence[:, 0], rtol=0, atol=0)
            bank_actions = case["action"][:, :int(model.planner.horizon)].to(model.device)
            predicted_costs = model._goal_cost(latent, past, bank_actions[None], goal)[0]
            selected_cost = float(model._goal_cost(latent, past, sequence[:, None], goal)[0, 0])
        if tensor_digest(model.state_dict()) != before:
            raise RuntimeError("Solver probe mutated native state or readout.")
        return {"budget": budget, "case_id": case["id"], "profile": case["profile"],
                "settings": OmegaConf.to_container(model.planner, resolve=True),
                "seconds": elapsed, "actions": sequence[0].cpu().tolist(),
                "predicted_cost": selected_cost, "bank_best_predicted_cost": float(predicted_costs.min()),
                "search_gap_to_bank": selected_cost - float(predicted_costs.min())}
    finally:
        model.planner = original
        model._cem_mean, model._gradient_actions = caches


@tools.preserve_rng_state
def replay_solver(config, case, probe):
    """Render every actual future of the selected sequence; simulator state is diagnostic-only."""
    env = make_env(config.env, int(case["seed"]), include_physical_state=False)
    try:
        env.reset()
        set_cart_state(env, case["initial_state"])
        prefix = observe_prefix(env, 3)
        torch.testing.assert_close(prefix["prefix"], case["prefix"], rtol=0, atol=0)
        images, rewards = [], []
        with simulator_branch(env) as branch:
            for action in probe["actions"]:
                obs, reward, done, _ = branch.step(np.asarray(action, dtype=np.float32))
                if done:
                    raise ValueError("Solver branch crossed an episode boundary")
                images.append(torch.from_numpy(obs["image"].copy()))
                rewards.append(float(reward))
        return {**case, "action": torch.tensor(probe["actions"], dtype=torch.float32)[None],
                "image": torch.stack(images)[None], "rewards": torch.tensor(rewards)[None]}
    finally:
        env.close()


@tools.preserve_rng_state
def policy_trial(config, model, cases, steps, seed):
    """Batched controlled-start trial, with production scoring/search and endpoint maintenance."""
    envs, caches = [], (model._cem_mean, model._gradient_actions)
    before = tensor_digest(model.state_dict())
    started = time.monotonic()
    try:
        for case in cases:
            env = make_env(config.env, int(case["seed"]), include_physical_state=False)
            envs.append(env)
            env.reset()
            set_cart_state(env, case["initial_state"])
            prefix = observe_prefix(env, 3)
            torch.testing.assert_close(prefix["prefix"], case["prefix"], rtol=0, atol=0)
        history = torch.stack([c["prefix"] for c in cases]).to(model.device)
        past = torch.stack([c["past_action"] for c in cases]).to(model.device)
        goals = torch.stack([c["goal_image"] for c in cases]).to(model.device)
        returns = np.zeros(len(cases))
        success, traces = [], []
        model._cem_mean = model._gradient_actions = None
        with readout_mode(model):
            synchronize(model.device)
            acting_started = time.monotonic()
            setup_seconds = acting_started - started
            for step in range(steps):
                torch.manual_seed(seed + step)
                action = native_control(model, lambda: model.act(
                    {"image": history, "goal_image": goals}, past, deterministic=True,
                    first=torch.full((len(cases),), step == 0, dtype=torch.bool, device=model.device)))
                images, rewards, good = [], [], []
                for env, control in zip(envs, action.cpu().numpy(), strict=True):
                    obs, reward, done, _ = env.step(control)
                    if done and step != steps - 1:
                        raise ValueError("Policy evaluation crossed the environment time limit")
                    images.append(obs["image"])
                    rewards.append(float(reward))
                    good.append(float(env._env.task.get_reward(env._env.physics)) >= 1 - 1e-6)
                history = torch.cat((history[:, 1:], torch.from_numpy(np.stack(images)).to(model.device)[:, None]), 1)
                past = torch.cat((past[:, 1:], action[:, None]), 1)
                returns += rewards
                success.append(good)
                traces.append({"step": step + 1, "actions": action.cpu().tolist(), "rewards": rewards, "success": good})
            synchronize(model.device)
            acting_seconds = time.monotonic() - acting_started
        if tensor_digest(model.state_dict()) != before:
            raise RuntimeError("Policy trial changed model weights or buffers")
        tail = max(1, math.ceil(.2 * steps))
        occupancy = np.asarray(success[-tail:]).mean(0)
        return {"steps": steps, "case_ids": [c["id"] for c in cases], "returns": returns.tolist(),
                "return_mean": float(returns.mean()), "maximum_return": steps * int(config.env.action_repeat),
                "tail_steps": tail, "maintenance_occupancy": occupancy.tolist(),
                "maintenance_rate": float((occupancy >= .9).mean()), "seconds": time.monotonic() - started,
                "setup_seconds": setup_seconds, "acting_seconds": acting_seconds,
                "scope": "Controlled starts; not ordinary-reset benchmark episodes", "traces": traces}
    finally:
        model._cem_mean, model._gradient_actions = caches
        for env in envs:
            env.close()


def save_checkpoint(path, model, config, updates, identity):
    payload = {**load_model_family(str(config.model_family)).checkpoint(model),
               "format": "paper_faithful_offline_v1", "training_config": OmegaConf.to_container(config, resolve=True),
               "updates": updates, "phase": "expert", "dataset_identity": identity, "resume_supported": False}
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(path, device):
    """Evaluation only: old archives and diagnostic snapshots retain their own architecture."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"training_config", "model_state_dict"} <= payload.keys():
        raise ValueError("Expected a latent-planner checkpoint with training_config and model_state_dict")
    config = OmegaConf.create(payload["training_config"])
    if config.model_family not in {"leworldmodel", "temporal_straightening"}:
        raise ValueError("Only LeWM/TS native snapshots are supported")
    if config.env.task != "dmc_cartpole_balance_sparse":
        raise ValueError("This branch evaluator requires a Cartpole balance sparse checkpoint")
    config.device = str(device)
    saved_objective = str(config.jepa_model.planner.objective)
    config.jepa_model.planner.objective = "ts_mpc" if config.model_family == "temporal_straightening" else "last"
    model = load_model_family(str(config.model_family)).build_model(config)
    load_model_family(str(config.model_family)).load_checkpoint(model, payload, training=False)
    return config, model, {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "format": payload.get("format", "archive"), "saved_objective": saved_objective,
                           "evaluated_objective": str(model.planner.objective), "training_resume": False}


def summary(report):
    def number(value):
        return "n/a" if value is None else f"{value:.4g}"

    lines = ["Paper-faithful offline check | NO ONLINE TRAINING OR SCHEDULE CHANGES",
             f"Status: {report['status']} | profile={report['settings']['profile']}",
             f"Reference: {report.get('reference', {}).get('status', 'NOT_RUN')}",
             "Arm | updates | native parameters | status | full-budget policy return"]
    for row in report.get("runs", []):
        policy = row.get("policy", {})
        lines.append(f"{row['arm']} | {row.get('updates', 0)} | {row.get('native_parameters', '?')} | "
                     f"{row.get('status', 'PENDING')} | {policy.get('return_mean', 'n/a')}")
    lines.append("Validation: H | informative anchors | actual/predicted/uniform selected return (informative) | matched/persistence MSE (all anchors)")
    for row in report.get("runs", []):
        validation = row.get("validation", {})
        for horizon, values in validation.get("aggregate_informative", {}).get("all", {}).items():
            all_values = validation["aggregate"]["all"][horizon]
            selected = "/".join(number(values.get(key)) for key in
                                ("actual_selected_return", "predicted_selected_return", "uniform_return"))
            lines.append(f"{row['arm']}: {horizon} | {values['anchors']}/{all_values['anchors']} | "
                         f"{selected} | {number(all_values.get('matched_over_persistence'))}")
    if report.get("scope"):
        lines.append(report["scope"])
    lines += ["COMPLETE means execution, not repaired control or full paper reproduction.",
              "Matched offline update counts; coverage changes dataset composition, never the native loss.",
              "The original-task upstream reference is separate; component parity is not task reproduction.",
              "Full-size defaults; small/tiny profiles are explicit size adaptations."]
    if "error" in report:
        lines.append(report["error"])
    if "seconds" in report:
        lines.append(f"Elapsed: {duration(report['seconds'])}")
    return "\n".join(lines) + "\n"


def persist(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")
    (output / "summary.txt").write_text(summary(report))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dmc_expert_vision"))
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("paper_faithful_%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=60.)
    parser.add_argument("--profile", choices=("full", "small", "tiny"), default="full")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--updates", type=int, help="Fixed common offline updates instead of runtime calibration")
    parser.add_argument("--min-updates", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--calibration-updates", type=int, default=8)
    parser.add_argument("--train-anchors", type=int, default=32)
    parser.add_argument("--validation-anchors", type=int, default=8)
    parser.add_argument("--test-anchors", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--policy-cases", type=int, default=3)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--reference-cache", type=Path, default=Path("runs/reference_cache"))
    parser.add_argument("--no-reference-download", action="store_true")
    parser.add_argument("--skip-reference", action="store_true", help="Explicit smoke-only bypass; reported as NOT_RUN")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="Evaluate saved native snapshot(s); no fitting")
    args = parser.parse_args(argv)
    positive = [args.minutes, args.batch_size, args.sources, args.min_updates, args.max_updates,
                args.calibration_updates, args.train_anchors, args.validation_anchors, args.test_anchors,
                args.candidates, args.horizon, args.policy_cases, args.policy_steps]
    if min(positive) <= 0 or args.seed < 0 or len(set(args.arms)) != len(args.arms):
        parser.error("Use positive budgets, a nonnegative seed, and unique arms")
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    if args.batch_size % args.sources or args.sources % 2 or args.batch_size % 2:
        parser.error("Use an even source count and a batch size divisible by it")
    if args.train_anchors < args.sources or args.policy_cases > min(args.validation_anchors, args.test_anchors):
        parser.error("Need enough training anchors for distinct sources and evaluation anchors for policy cases")
    if args.candidates < 6 or args.horizon < 3 or args.horizon > 497 or args.policy_steps > 497:
        parser.error("Use >=6 candidates, horizon 3..497, and policy-steps <=497 (the observed prefix consumes two steps)")
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(1)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the GPU training host, or explicit --device cpu for small smoke checks.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "RUNNING", "settings": {k: str(v) if isinstance(v, Path) else
               [str(x) for x in v] if k == "checkpoint" else v for k, v in vars(args).items()},
              "online_updates": 0, "online_schedule_changed": False, "runs": [],
              "implementation_sha256": implementation_sha256(),
              "experiment_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ("train_paper_faithful_check.py", "paper_faithful_support.py", "paper_faithful_reference.py",
                               "diagnose_fresh_readout.py", "diagnose_goal_objective.py", "diagnose_planner_oracle.py",
                               "smoke_tiny_planners.py", "upstream_ts_probe.py")},
              "versions": {"python": platform.python_version(), "torch": torch.__version__}}
    persist(args.output, report)
    try:
        configs = {arm: build_config(arm, args) for arm in args.arms}
        if args.skip_reference:
            report["reference"] = {"status": "NOT_RUN", "reason": "Explicit --skip-reference; not a parity pass"}
        else:
            from scripts.paper_faithful_reference import run_reference_checks
            report["reference"] = run_reference_checks(configs, args, cache=args.reference_cache,
                                                       allow_download=not args.no_reference_download)
            persist(args.output, report)
            if report["reference"]["status"] != "PASS":
                raise RuntimeError("Upstream component checks did not pass; inspect reference report before fitting")
        manifest = make_split_manifest(train_anchors=args.train_anchors, validation_anchors=args.validation_anchors,
                                       test_anchors=args.test_anchors, seed=args.seed + 20_000_000)
        bank = collect_branch_bank(next(iter(configs.values())), manifest, candidates=args.candidates,
                                   horizon=args.horizon, progress=lambda done, total: print(f"Branches | {done}/{total}", flush=True))
        torch.save(bank, args.output / "branches.pt")
        report["branches"] = {"metadata": bank["metadata"], "sha256": bank["sha256"], "file": "branches.pt"}
        persist(args.output, report)
        if args.checkpoint:
            report["scope"] = "Saved-checkpoint branch scoring only; no fitting, policy trial, or solver probe."
            for index, path in enumerate(args.checkpoint):
                config, model, identity = load_snapshot(path, args.device)
                row = {"arm": f"checkpoint_{index}", "status": "RUNNING", "checkpoint": identity,
                       "family": str(config.model_family), "config": OmegaConf.to_container(config, resolve=True)}
                report["runs"].append(row)
                row["validation"] = score_branches(model, bank["splits"]["validation"], horizons=(1, args.horizon))
                row["test"] = score_branches(model, bank["splits"]["test"], horizons=(1, args.horizon))
                row["status"] = "COMPLETE"
                persist(args.output, report)
                del model
        else:
            rates, eval_times = [], []
            report["calibration"] = {}
            for arm, config in configs.items():
                fraction = ARMS[arm][3]
                setup_started = time.monotonic()
                with load_model_family(str(config.model_family)).build_replay(config) as dataset:
                    state = copy.deepcopy(dataset.state_dict())
                    model = new_model(config, dataset)
                    synchronize(args.device)
                    setup_seconds = time.monotonic() - setup_started
                    if hasattr(model, "configure_pretraining"):
                        model.configure_pretraining(args.max_updates)
                    branch = replay_for(bank, args, fraction) if fraction else None
                    times = []
                    for step in range(args.calibration_updates):
                        synchronize(args.device)
                        tick = time.monotonic()
                        update(model, dataset, branch, fraction, step, args.seed)
                        synchronize(args.device)
                        times.append(time.monotonic() - tick)
                    rate = statistics.mean(times[min(2, len(times) - 1):])
                    tick = time.monotonic()
                    score_branches(model, bank["splits"]["validation"][:1], horizons=(1, args.horizon))
                    scoring = time.monotonic() - tick
                    probe = solver_probe(model, solver_case(bank["splits"]["validation"]))
                    # Calibrate real batched native acting; do not infer GPU latency from update speed.
                    trial = policy_trial(config, model, bank["splits"]["validation"][:args.policy_cases],
                                         min(3, args.policy_steps), args.seed + 15000)
                    evaluation = (setup_seconds + scoring * (args.validation_anchors + args.test_anchors + 4) +
                                  trial["setup_seconds"] + trial["acting_seconds"] / trial["steps"] * args.policy_steps +
                                  3 * probe["seconds"])
                    rates.append(rate)
                    eval_times.append(evaluation)
                    report["calibration"][arm] = {"update_seconds": rate, "estimated_evaluation_seconds": evaluation,
                                                  "planner_call_seconds": probe["seconds"], "model_dataset_setup_seconds": setup_seconds,
                                                  "policy_setup_seconds": trial["setup_seconds"],
                                                  "policy_decision_seconds": trial["acting_seconds"] / trial["steps"],
                                                  "disposable_updates": len(times)}
                    dataset.load_state_dict(state)
                    del model
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"Profile | {arm}: {rate:.3f}s/update; evaluation ~{evaluation:.0f}s", flush=True)
                    persist(args.output, report)
            remaining = 60 * args.minutes - (time.monotonic() - started)
            count = args.updates or choose_updates(remaining, rates, sum(eval_times), args.min_updates, args.max_updates)
            report["budget"] = {"common_updates": count, "adjacent_targets_per_arm": count * args.batch_size * 3,
                                "estimated_remaining_seconds": count * sum(rates) + sum(eval_times),
                                "runtime_target_seconds": 60 * args.minutes,
                                "fixed_before_training": True, "hard_deadline": False}
            print(f"Budget | {len(configs)} arms x {count} updates; estimated remaining "
                  f"{duration(report['budget']['estimated_remaining_seconds'])}", flush=True)
            common_hashes = {}
            for arm, config in configs.items():
                config.training.expert.updates = count
                fraction = ARMS[arm][3]
                folder = args.output / arm
                folder.mkdir()
                with load_model_family(str(config.model_family)).build_replay(config) as dataset:
                    model = new_model(config, dataset)
                    initial = tensor_digest(common_initial_weights(model))
                    family = str(config.model_family)
                    if family in common_hashes and common_hashes[family] != initial:
                        raise RuntimeError("Paired arms differ in their common initial parameters/buffers")
                    common_hashes[family] = initial
                    row = {"arm": arm, "family": family, "status": "RUNNING", "updates": 0,
                           "coverage_fraction": fraction, "config": OmegaConf.to_container(config, resolve=True),
                           "common_initial_sha256": initial,
                           "native_parameters": sum(p.numel() for name, p in model.named_parameters()
                                                    if not name.startswith("state_head.")),
                           "dataset_identity": dataset_identity(dataset.metadata)}
                    report["runs"].append(row)
                    branch = replay_for(bank, args, fraction) if fraction else None
                    if hasattr(model, "configure_pretraining"):
                        model.configure_pretraining(count)
                    progress = Progress(arm, count)
                    train_start = time.monotonic()
                    with (folder / "metrics.jsonl").open("w", buffering=1) as log:
                        for step in range(1, count + 1):
                            values = update(model, dataset, branch, fraction, step, args.seed)
                            log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
                            progress.update(step, f"prediction={values['prediction_loss']:.4g}", force=step == count)
                            row["updates"] = step
                    row["training_seconds"] = time.monotonic() - train_start
                    row["checkpoint_sha256"] = save_checkpoint(folder / "native.pt", model, config, count, row["dataset_identity"])
                    row["validation"] = score_branches(model, bank["splits"]["validation"], horizons=(1, args.horizon))
                    row["solver"] = []
                    for budget in ("source", "production"):
                        case = solver_case(bank["splits"]["validation"])
                        probe = solver_probe(model, case, budget)
                        actual = replay_solver(config, case, probe)
                        probe["actual_sequence"] = score_branches(model, [actual], horizons=(args.horizon,))
                        row["solver"].append(probe)
                    row["policy"] = policy_trial(config, model, bank["splits"]["test"][:args.policy_cases],
                                                  args.policy_steps, args.seed + 16000)
                    (folder / "policy_metrics.jsonl").write_text("".join(json.dumps(t) + "\n" for t in row["policy"].pop("traces")))
                    row["test"] = score_branches(model, bank["splits"]["test"], horizons=(1, args.horizon))
                    row["status"] = "COMPLETE"
                    persist(args.output, report)
                    del model
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
            report["interpretation"] = "Screening only. Test metrics must not be reused to tune this run's hyperparameters."
        report["status"] = "COMPLETE"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["seconds"] = time.monotonic() - started
        persist(args.output, report)
        print(summary(report), end="", flush=True)
        print(f"Reports | {args.output.resolve()}", flush=True)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
