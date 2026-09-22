"""Native snapshot evaluation that leaves the next training update unchanged."""

import copy
import json
from contextlib import contextmanager

import torch

import tools
from scripts.paper_faithful_support import score_branches
from scripts.paper_faithful_followup_support import policy_cases as _policy_cases
from scripts.train_paper_faithful_check import policy_trial


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def preserve_training_state(model):
    """Restore transient state on every exit; reject unexpected training mutations.

    Copies include optimizer moments and existing gradients. Normally no weights or
    optimizer state need restoring; the snapshots also protect a failed evaluation.
    Parameter and gradient objects, mixed module modes and cache objects are retained.
    """
    rng = tools.get_rng_state()
    weights = copy.deepcopy(model.state_dict())
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    optimizer_objects = list(getattr(model, "optimizers", {}).values())
    head_optimizer = getattr(getattr(model, "state_head", None), "optimizer", None)
    if head_optimizer is not None:
        optimizer_objects.append(head_optimizer)
    optimizer_objects = list({id(optimizer): optimizer for optimizer in optimizer_objects}.values())
    optimizers = [(optimizer, copy.deepcopy(optimizer.state_dict())) for optimizer in optimizer_objects]
    scheduler = getattr(model, "scheduler", None)
    scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    counters = {name: copy.deepcopy(getattr(model, name)) for name in ("_gradient_updates", "_clipped_updates") if hasattr(model, name)}
    caches = {name: (getattr(model, name), copy.deepcopy(getattr(model, name)))
              for name in ("_cem_mean", "_gradient_actions") if hasattr(model, name)}
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        changed = []
        try:
            if not _equal(weights, model.state_dict()):
                changed.append("weights or buffers")
                model.load_state_dict(weights, strict=True)
            for optimizer, state in optimizers:
                if not _equal(state, optimizer.state_dict()):
                    changed.append("optimizer state")
                    optimizer.load_state_dict(state)
            if scheduler is not None and not _equal(scheduler_state, scheduler.state_dict()):
                changed.append("scheduler state")
                scheduler.load_state_dict(scheduler_state)
            for parameter, requires_grad, original, value in gradients:
                if not _equal(value, parameter.grad):
                    changed.append("parameter gradients")
                if original is not None and not _equal(original, value):
                    with torch.no_grad():
                        original.copy_(value)
                parameter.grad = original
                parameter.requires_grad_(requires_grad)
            for module, mode in modes:
                module.training = mode
            for name, value in counters.items():
                if getattr(model, name) != value:
                    changed.append("training counters")
                setattr(model, name, value)
            for name, (original, value) in caches.items():
                if isinstance(original, torch.Tensor) and not _equal(original, value):
                    with torch.no_grad():
                        original.copy_(value)
                setattr(model, name, original)
        finally:
            tools.set_rng_state(rng)
        if changed and not failed:
            raise RuntimeError("Snapshot evaluation changed training state (restored): " + ", ".join(sorted(set(changed))))


def select_policy_cases(cases, count):
    """Use exactly the shared simulator-control subset, independent of model seed."""
    cases = list(cases)
    if int(count) != count or count < 0 or count > len(cases):
        raise ValueError("Policy case count must be an integer between zero and the available anchor count.")
    ids = [str(case["id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Policy anchors must have unique IDs.")
    return _policy_cases(cases, int(count)) if count else []


def evaluate_snapshot(config, model, cases, *, horizons=(1, 5, 15, 25), policy_cases=3,
                      policy_steps=50, policy_seed=1701, encode_batch_size=32):
    """Evaluate recursive forecasts at several horizons and the unchanged native policy.

    Every forecast uses the same held-out branch bank. Forecast horizons do not
    change the configured control horizon or solver budget. `policy_cases=0` or
    `policy_steps=0` explicitly omits the policy and returns policy=None.
    """
    cases = list(cases)
    horizons = tuple(sorted(set(int(horizon) for horizon in horizons)))
    if not cases or not horizons or min(horizons) < 1 or policy_steps < 0 or encode_batch_size < 1:
        raise ValueError("Require nonempty cases, positive forecast horizons, steps and encoding batch size.")
    if any(max(horizons) > case["action"].shape[1] for case in cases):
        raise ValueError("Forecast horizon exceeds the shared branch bank.")
    selected = select_policy_cases(cases, policy_cases if policy_steps else 0)
    planning_horizon = int(model.planner.horizon)
    with preserve_training_state(model):
        branches = score_branches(model, cases, horizons=horizons, encode_batch_size=encode_batch_size)
        policy = policy_trial(config, model, selected, int(policy_steps), int(policy_seed)) if selected else None
    result = {"branches": branches, "policy": policy,
              "policy_selection": {"method": "shared_controls_cohort_round_robin_with_opposite_signs", "model_seed_independent": True,
                                   "policy_rng_seed": int(policy_seed),
                                   "case_ids": [case["id"] for case in selected],
                                   "cohorts": [case["cohort"] for case in selected],
                                   "available_cohorts": sorted({str(case["cohort"]) for case in cases})},
              "policy_planning_horizon": planning_horizon,
              "training_state_preserved": True,
              "scope": "Fixed held-out simulator anchors; native image/action forecasts and native policy with traces. No training, loss, horizon or solver changes."}
    json.dumps(result, allow_nan=False)
    return result


def summarize_evaluation(evaluation):
    """Compact numeric schema for learning curves; missing values remain None.

    Counts and metric-specific valid_counts are retained so empty informative
    subsets and undefined ratios/correlations cannot be mistaken for zero errors.
    """
    branches = evaluation["branches"]
    result = {"horizons": {}}
    for horizon in branches["horizons"]:
        key = str(horizon)
        result["horizons"][key] = {
            "all": copy.deepcopy(branches["aggregate"]["all"][key]),
            "informative": copy.deepcopy(branches["aggregate_informative"]["all"][key]),
        }
    policy = evaluation["policy"] or {}
    result["policy"] = {key: copy.deepcopy(policy.get(key)) for key in (
        "steps", "returns", "return_mean", "maximum_return", "tail_steps", "maintenance_occupancy", "maintenance_rate")}
    result["policy"]["cases"] = len(policy.get("case_ids", []))
    json.dumps(result, allow_nan=False)
    return result
