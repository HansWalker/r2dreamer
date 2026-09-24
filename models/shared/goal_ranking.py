"""Optional simulator-supervised ordering of the existing image-goal distance.

No learned head, new model input, or planning objective. Labels select pairs
outside this module; only images enter the encoder.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from models.shared.latent_goal import latent_goal_cost


@contextmanager
def planning_representation(model):
    """Use deployment BN/dropout behavior while retaining encoder gradients."""
    modules = {module for root in (model.encoder, model.projector) for module in root.modules()}
    modes = {module: module.training for module in modules}
    try:
        model.encoder.eval()
        model.projector.eval()
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode


def goal_ranking_loss(model, images, margin):
    """Images are [pairs, (good, bad, matching goal), H, W, C].

Compare the same per-state squared feature distances used by the planner.
Divide LeWM's sum by feature count so the auxiliary weight/margin are in
per-coordinate MSE units for both models; action rankings are unchanged.
The goal is re-encoded each call and detached, as in the native cost.
"""
    if images.ndim != 5 or images.shape[1] != 3 or not len(images):
        raise ValueError("Goal ranking requires nonempty [pairs, 3, H, W, C] images")
    if model._aggregate_goal_weight():
        raise ValueError("Goal ranking currently supports the native spatial goal distance only")
    with planning_representation(model):
        latent = model.encode({"image": images.to(model.device, non_blocking=True)})
    cost = latent_goal_cost(latent[:, :2, None], latent[:, 2],
                            mode="last", reduction=model.goal_reduction)
    if model.goal_reduction == "sum":
        cost = cost / latent[0, 0].numel()
    gap = cost[:, 1] - cost[:, 0]
    loss = F.relu(float(margin) - gap).mean()
    return loss, {
        "goal_ranking/loss": loss.detach(),
        "goal_ranking/pair_accuracy": (gap > 0).float().mean(),
        "goal_ranking/margin_satisfied": (gap >= float(margin)).float().mean(),
        "goal_ranking/good_distance": cost[:, 0].detach().mean(),
        "goal_ranking/bad_distance": cost[:, 1].detach().mean(),
        "goal_ranking/pairs": cost.new_tensor(len(images)),
    }
