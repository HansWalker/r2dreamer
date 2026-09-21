"""Latent-only controller objectives, shared with simulator-oracle diagnostics."""

import torch


def latent_goal_cost(prediction, goal, *, reduction, mode="last", history=None, tail_steps=3):
    """Score [B, candidates, future, ...] against [B, ...] or [B, goals, ...]."""
    if mode not in {"last", "ts_mpc", "tail"}:
        raise ValueError(f"Unknown latent goal objective: {mode}")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unknown latent coordinate reduction: {reduction}")
    if goal.ndim == prediction.ndim - 2:
        goal = goal[:, None]
    if (goal.ndim != prediction.ndim - 1 or goal.shape[0] != prediction.shape[0]
            or goal.shape[2:] != prediction.shape[3:] or goal.shape[1] == 0):
        raise ValueError("Goal embeddings must match the batch and latent shape of predictions.")
    if mode == "last":
        path = prediction[:, :, -1:]
    elif mode == "tail":
        if not 1 <= tail_steps <= prediction.shape[2]:
            raise ValueError("Stable-arrival tail_steps must be between one and the planning horizon.")
        path = prediction[:, :, -tail_steps:]
    else:
        if history is None:
            raise ValueError("TS MPC scoring requires the observed prefix, as in upstream rollout.")
        prefix = history.detach()[:, None].expand(-1, prediction.shape[1], -1, *history.shape[2:])
        path = torch.cat((prefix, prediction), dim=2)
    weights = None
    if mode == "ts_mpc":
        # Upstream normalizes base**i, then MEANS the weighted errors over time.
        # Shift exponents to avoid overflow without changing those coefficients.
        weights = torch.exp2(torch.arange(1 - path.shape[2], 1, device=path.device, dtype=torch.float32))
        weights = weights / weights.sum()
    costs = []
    for target in goal.detach().unbind(1):
        error = (path.float() - target.float()[:, None, None]).square().flatten(3)
        error = error.sum(-1) if reduction == "sum" else error.mean(-1)
        costs.append((error if weights is None else error * weights).mean(-1))
    # Min AFTER temporal aggregation: a candidate must aim for one coherent goal.
    return torch.stack(costs, dim=-1).min(-1).values
