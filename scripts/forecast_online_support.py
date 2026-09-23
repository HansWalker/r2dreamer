"""Evaluation-only separation of recursive prediction and goal-score errors."""

import json

import numpy as np
import torch

from models.shared.physical_state import readout_mode
from scripts.paper_faithful_support import _encode, selection


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


@torch.no_grad()
def forecast_errors(model, cases, *, encode_batch_size=32):
    """Compare recursive forecasts with one-step forecasts given real histories.

    Actions are identical in both paths. Future images are supplied only to this
    diagnostic, never to control or training. Targets are re-encoded at each
    checkpoint, so raw latent errors must be read with persistence and spread.
    """
    rows = []
    with readout_mode(model):
        for case in cases:
            actions = case["action"].to(model.device)
            count, horizon, _ = actions.shape
            history = _encode(model, case["prefix"], encode_batch_size)
            targets = _encode(model, case["image"].flatten(0, 1), encode_batch_size).reshape(
                count, horizon, *history.shape[1:])
            past = case["past_action"].to(model.device)
            recursive = model.rollout(history[None], past[None], actions[None])[0]
            real_history = torch.cat((history[None].expand(count, *history.shape), targets), 1)
            all_actions = torch.cat((past[None].expand(count, *past.shape), actions), 1)
            # Each row predicts the next observation from K real observations and
            # K aligned actions, including the action leading to that target.
            states = torch.stack([real_history[:, step:step + model.history_size]
                                  for step in range(horizon)], 1).flatten(0, 1)
            aligned = torch.stack([all_actions[:, step:step + model.history_size]
                                   for step in range(horizon)], 1).flatten(0, 1)
            observed = torch.cat([model.predict(state, action)[:, -1]
                                  for state, action in zip(states.split(encode_batch_size),
                                                           aligned.split(encode_batch_size), strict=True)])
            observed = observed.reshape_as(targets)
            mse = lambda prediction: (prediction - targets).square().flatten(2).mean(2)
            recursive_mse, observed_mse = mse(recursive), mse(observed)
            persistence = mse(history[-1])
            observed_persistence = mse(real_history[:, model.history_size - 1:model.history_size - 1 + horizon])
            if not all(torch.isfinite(value).all() for value in (recursive_mse, observed_mse, persistence, observed_persistence)):
                raise ValueError("Non-finite forecast diagnostic")
            mean = lambda value: value.mean(0).cpu().tolist()
            r, o, p = mean(recursive_mse), mean(observed_mse), mean(persistence)
            observed_p = mean(observed_persistence)
            rows.append({"id": case["id"], "horizon": horizon,
                         "recursive_mse_by_step": r, "observed_history_mse_by_step": o,
                         "persistence_mse_by_step": p,
                         "observed_history_persistence_mse_by_step": observed_p,
                         "recursive_over_persistence_by_step": [ratio(a, b) for a, b in zip(r, p)],
                         "observed_over_persistence_by_step": [ratio(a, b) for a, b in zip(o, observed_p)],
                         "target_latent_rms_std": float(targets.flatten(0, 1).flatten(1).std(0, unbiased=False).square().mean().sqrt()),
                         "candidate_recursive_mse_by_step": recursive_mse.cpu().tolist(),
                         "candidate_observed_history_mse_by_step": observed_mse.cpu().tolist(),
                         "candidate_persistence_mse_by_step": persistence.cpu().tolist(),
                         "candidate_observed_history_persistence_mse_by_step": observed_persistence.cpu().tolist()})
    return {"cases": rows, "scope": "Real-history inputs are diagnostic only; no future observations enter the controller",
            "persistence_definition": "Recursive: copy initial last observation. Real-history: copy most recent real observation at each step",
            "comparison": "Same stored images/actions across checkpoints, encoded with each checkpoint's own encoder"}


def probe_cases(probes, split="validation"):
    """Adapt saved simulator replays, preserving candidate zero as the old plan."""
    cases = []
    for index, probe in enumerate(probes):
        if f"/{split}/" not in probe["case_id"]:
            raise ValueError("Fixed probes must come from the declared held-out split")
        cases.append({"id": f"{probe['case_id']}/probe_{index}_step_{probe['step']}",
                      "seed": index, "split": split, "cohort": probe["reason"],
                      "prefix": probe["prefix"], "past_action": probe["past_action"],
                      "action": probe["actions"], "goal_image": probe["goal_image"],
                      "image": probe["image"], "rewards": probe["rewards"], "success": probe["success"]})
    return cases


def summarize_probes(probes, hold_steps):
    """Retain each new selected plan's actual outcome, ordering and error curve."""
    rows = []
    for probe in probes:
        predicted, actual = (probe[key].cpu().numpy() for key in ("predicted_cost", "actual_cost"))
        rewards = probe["rewards"].sum(1).cpu().numpy()
        occupancy = probe["success"][:, -hold_steps:].float().mean(1).numpy()
        rows.append({"case_id": probe["case_id"], "step": probe["step"], "reason": probe["reason"],
                     "selected_return": float(rewards[0]), "best_available_return": float(rewards.max()),
                     "selected_occupancy": float(occupancy[0]),
                     "predicted_selected_occupancy": selection(predicted, occupancy)["return_mean"],
                     "actual_selected_occupancy": selection(actual, occupancy)["return_mean"],
                     "uniform_occupancy": float(np.mean(occupancy)), "best_occupancy": float(occupancy.max()),
                     "selected_prediction_mse_by_step": probe["prediction_mse_by_step"][0].tolist(),
                     "selected_cost_underestimate": float(actual[0] - predicted[0])})
    result = {"cases": rows, "scope": "Complete plan replays; the controller executes only the first action and replans"}
    json.dumps(result, allow_nan=False)
    return result
