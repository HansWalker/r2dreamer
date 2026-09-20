"""Frozen action probes: identical histories, counterfactual futures, no planner or readout."""

import math

import numpy as np
import torch

from models.shared.physical_state import readout_mode
from scripts.diagnose_goal_objective import cart_state
from scripts.diagnose_planner_oracle import rank_correlation, simulator_branch
from scripts.diagnose_planner_recipe import reset_pair
from scripts.planner_recipe_support import block_actions, pose_distance, resize_images
from scripts.upstream_ts_probe import frozen_precision


def collect_branches(env, cases, stride, tolerance, seed):
    rng = np.random.default_rng(seed)
    bank = []
    for case in cases:
        prefix = reset_pair(env, case, stride)
        expert = case["expert_actions"]
        candidates = np.stack((expert, np.zeros_like(expert), -np.ones_like(expert), np.ones_like(expert),
                               -expert, expert[::-1], *rng.uniform(-1, 1, (4, *expert.shape)).astype(np.float32)))
        images, states, distances = [], [], []
        for sequence in candidates:
            frames, physical, distance = [], [], []
            with simulator_branch(env) as branch:
                for step, action in enumerate(sequence, 1):
                    obs, _, done, _ = branch.step(action)
                    if done:
                        raise ValueError("Counterfactual branch crossed an episode boundary.")
                    if step % stride == 0:
                        frames.append(obs["image"])
                        physical.append(cart_state(branch))
                        distance.append(float(pose_distance(physical[-1], case["goal_state"], tolerance)))
            images.append(np.stack(frames))
            states.append(np.stack(physical))
            distances.append(distance)
        np.testing.assert_allclose(cart_state(env), case["anchor_state"], atol=1e-6, rtol=0)
        bank.append({"id": case["id"], "episode": case["episode"], "start": case["start"],
                     "prefix": prefix, "past": block_actions(torch.from_numpy(case["prefix_actions"]), stride),
                     "actions": block_actions(torch.from_numpy(candidates), stride),
                     "images": torch.from_numpy(np.stack(images)), "goal_image": case["goal_image"],
                     "states": np.stack(states).tolist(), "pose_distance": distances,
                     "initial_state": case["initial_state"], "goal_state": case["goal_state"],
                     "restoration_max_error": case["restoration_max_error"]})
    return bank


def ratio(numerator, denominator):
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else None


def spread(value):
    value = value.float()
    return float((value - value.mean(0, keepdim=True)).square().mean().sqrt())


@torch.no_grad()
def encode_case(model, case, image_size):
    counts = [len(case["prefix"]), case["images"].shape[0] * case["images"].shape[1], 1]
    images = torch.cat((case["prefix"], case["images"].flatten(0, 1), case["goal_image"][None]))
    encoder, projected = [], []
    for chunk in images.split(8):
        with model.amp_context():
            raw = model.encoder({"image": resize_images(chunk.to(model.device), image_size)})
            latent = model.projector(raw)
        encoder.append(raw.float())
        projected.append(latent.float())
    result = {"past": case["past"][None].to(model.device), "actions": case["actions"].to(model.device)}
    geometry = {}
    for name, tensors in (("encoder", encoder), ("projector", projected)):
        prefix, future, goal = torch.cat(tensors).split(counts)
        future = future.reshape(*case["images"].shape[:2], *future.shape[1:])
        costs = (future[:, -1] - goal[0]).square().flatten(1).mean(-1).cpu().numpy()
        distance = np.asarray(case["pose_distance"])[:, -1]
        geometry[name] = {"goal_cost": costs.tolist(), "actual_pose_rank": rank_correlation(costs, distance),
                          "nonexpert_pose_rank": rank_correlation(costs[1:], distance[1:]),
                          "branch_spread": [spread(future[:, h]) for h in range(future.shape[1])]}
        if name == "projector":
            result.update(history=prefix[None], future=future, goal=goal[0])
    return result, geometry


def action_variants(actions):
    # A fixed derangement across candidates, never across unrelated initial states.
    return {"matched": actions, "shuffled": actions.roll(1, 0), "zero": torch.zeros_like(actions)}


def teacher_forced(model, history, past, actions, future):
    count, horizon = actions.shape[:2]
    observed = torch.cat((history.expand(count, *history.shape[1:]), future), 1)
    controls = torch.cat((past.expand(count, -1, -1), actions), 1)
    size = model.history_size
    return torch.stack([model.predict(observed[:, h:h + size], controls[:, h:h + size])[:, -1]
                        for h in range(horizon)], 1)


def horizon_scores(predictions, future, history, goal, reduction):
    result = []
    for h in range(future.shape[1]):
        truth = future[:, h]
        actual_cost = (truth - goal).square().flatten(1)
        actual_cost = actual_cost.sum(-1) if reduction == "sum" else actual_cost.mean(-1)
        row = {"horizon": h + 1, "actual_branch_spread": spread(truth),
               "persistence_mse": float((history[0, -1] - truth).square().mean()),
               "last_observed_mse": float(((history[0, -1] if h == 0 else future[:, h - 1]) - truth).square().mean())}
        for mode, arms in predictions.items():
            scores = {}
            for name, prediction in arms.items():
                errors = (prediction[:, h] - truth).square().flatten(1).mean(-1)
                scores[name] = {"mse": float(errors.mean()), "expert_mse": float(errors[0]),
                                "per_candidate_mse": errors.tolist()}
            predicted = arms["matched"][:, h]
            costs = (predicted - goal).square().flatten(1)
            costs = costs.sum(-1) if reduction == "sum" else costs.mean(-1)
            delta, true_delta = predicted[3] - predicted[2], truth[3] - truth[2]
            scores.update(branch_spread=spread(predicted), response_ratio=ratio(delta.norm(), true_delta.norm()),
                          response_cosine=ratio((delta * true_delta).sum(), delta.norm() * true_delta.norm()),
                          goal_cost_range_ratio=ratio(costs.max() - costs.min(), actual_cost.max() - actual_cost.min()),
                          goal_cost_rank=rank_correlation(costs.cpu().numpy(), actual_cost.cpu().numpy()),
                          goal_cost=costs.tolist(), actual_goal_cost=actual_cost.tolist())
            row[mode] = scores
        result.append(row)
    return result


@torch.no_grad()
def forward_scores(model, encoded):
    history, past, actions, future, goal = [encoded[k] for k in ("history", "past", "actions", "future", "goal")]
    predictions = {"teacher_forced": {}, "recursive": {}}
    for name, controls in action_variants(actions).items():
        predictions["teacher_forced"][name] = teacher_forced(model, history, past, controls, future)
        predictions["recursive"][name] = model.rollout(history, past, controls[None])[0]
    result = horizon_scores(predictions, future, history, goal, model.goal_reduction)
    if not all(torch.isfinite(p).all() for arms in predictions.values() for p in arms.values()):
        raise ValueError("Non-finite branch predictions.")
    return result


@torch.no_grad()
def layer_trace(model, encoded):
    """Opposing current actions, identical observations and identical previous actions."""
    ts = hasattr(model.predictor, "state_dim")
    patches = model.predictor._mask_patches if ts else 1
    dim = model.predictor.state_dim if ts else encoded["history"].shape[-1]
    rows, handles = [], []

    def record(name, stage):
        def hook(_module, _inputs, output):
            if stage == "action":
                visual, total = output[:, -1], output[:, -1]
            elif stage == "projector":
                visual = total = output[:, -1]
            else:
                total = output[:, -patches:]
                visual = total[..., :dim]
            rows.append({"stage": name, "total_delta_rms": float((total[1].float() - total[0].float()).square().mean().sqrt()),
                         "visual_delta_rms": float((visual[1].float() - visual[0].float()).square().mean().sqrt())})
        return hook

    modules = [("action_encoder", model.action_encoder, "action"),
               ("position_dropout", model.predictor.dropout, "tokens")]
    modules += [(f"block_{i + 1}", block, "tokens") for i, block in enumerate(model.predictor.blocks)]
    modules += [("final_norm", model.predictor.norm, "tokens"), ("pred_projector", model.pred_projector, "projector")]
    try:
        for name, module, stage in modules:
            handles.append(module.register_forward_hook(record(name, stage)))
        controls = torch.cat((encoded["past"].expand(2, -1, -1), encoded["actions"][[2, 3], :1]), 1)
        model.predict(encoded["history"].expand(2, *encoded["history"].shape[1:]), controls)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def action_gradients(model, encoded):
    with torch.enable_grad():
        controls = encoded["actions"][:1].detach()[None].requires_grad_()
        predicted = model.rollout(encoded["history"], encoded["past"], controls)[0, 0]
        rows = []
        generator = torch.Generator().manual_seed(273)
        for h, value in enumerate(predicted):
            direction = torch.randn(value.shape, generator=generator).to(value.device) / math.sqrt(value.numel())
            projection = (value * direction).sum()
            error = (value - encoded["goal"]).square()
            goal = error.sum() if model.goal_reduction == "sum" else error.mean()
            def gradient(scalar):
                grad = torch.autograd.grad(scalar, controls, retain_graph=True, allow_unused=True)[0] if scalar.requires_grad else None
                return torch.zeros_like(controls) if grad is None else grad
            jac, cost = gradient(projection), gradient(goal)
            if not torch.isfinite(jac).all() or not torch.isfinite(cost).all():
                raise ValueError("Non-finite action gradient.")
            rows.append({"horizon": h + 1,
                         "latent_projection_gradient_by_block": jac[0, 0].norm(dim=-1).tolist(),
                         "goal_gradient_by_block": cost[0, 0].norm(dim=-1).tolist()})
            if h == 0 and not model.use_amp:
                # Independent central difference along the first physical action block.
                perturb = torch.zeros_like(controls)
                perturb[..., 0, :] = 1 / math.sqrt(controls.shape[-1])
                with torch.no_grad():
                    values = [model.rollout(encoded["history"], encoded["past"], controls.detach() + sign * 1e-3 * perturb)[0, 0, 0]
                              for sign in (-1, 1)]
                finite_difference = float(((values[1] - values[0]) * direction).sum() / .002)
                rows[-1]["directional_derivative"] = {"autograd": float((jac * perturb).sum()),
                                                       "finite_difference": finite_difference, "epsilon": .001}
    return rows


def probe_case(model, case, size, *, gradients):
    with readout_mode(model):
        encoded, geometry = encode_case(model, case, size)
        result = {"id": case["id"], "episode": case["episode"], "start": case["start"], "geometry": geometry,
                  "encoder_precision": "bf16_mixed" if model.use_amp and model.device.type == "cuda" else "fp32",
                  "precision": {}}
        precisions = [("fp32", False)]
        if model.device.type == "cuda":
            precisions.append(("bf16_mixed", True))
        for name, amp in precisions:
            with frozen_precision(model, amp):
                row = {"horizons": forward_scores(model, encoded), "layers": layer_trace(model, encoded)}
                if gradients:
                    row["gradients"] = action_gradients(model, encoded)
                result["precision"][name] = row
    return result, encoded
