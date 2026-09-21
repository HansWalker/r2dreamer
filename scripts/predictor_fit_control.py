"""A bounded native-loss fitting control, on copies with disjoint episode validation."""

import copy

import torch

import tools
from scripts.action_conditioning_support import encode_case, forward_scores
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.upstream_ts_probe import full_precision


@tools.preserve_rng_state
def fit_control(model, cases, image_size, updates):
    if updates < 1:
        raise ValueError("Fitting control needs at least one update.")
    episodes = sorted({case["episode"] for case in cases})
    if len(episodes) < 2:
        return {"status": "UNVALIDATED", "reason": "Need two distinct episodes for fitting/validation."}
    training_episodes = set(episodes[:max(1, len(episodes) // 2)])
    local = copy.deepcopy(model)
    if hasattr(local, "configure_online"):
        # A completed cosine pretraining schedule ends at zero LR.
        local.configure_online(updates, resumed=False)
    local.set_adaptation_mode("frozen_encoder")
    local.eval()
    local.use_amp = False
    local.decoder = None
    parameters = [p for module in (local.predictor, local.action_encoder, local.pred_projector)
                  for p in module.parameters() if p.requires_grad]
    optimizers = [opt for name, opt in local.optimizers.items() if name not in {"encoder", "decoder"}]
    initial = tensor_digest(local.state_dict())
    protected = {k: v for k, v in local.state_dict().items()
                 if not k.startswith(("predictor.", "action_encoder.", "pred_projector."))}
    frozen = tensor_digest(protected)
    with full_precision():
        banks = [(case, encode_case(local, case, image_size)[0]) for case in cases]

        @torch.no_grad()
        def score():
            result = {"train": [], "validation": []}
            for case, bank in banks:
                row = forward_scores(local, bank)
                result["train" if case["episode"] in training_episodes else "validation"].append({
                    "id": case["id"], "episode": case["episode"],
                    "teacher_h1": row[0]["teacher_forced"]["matched"]["mse"],
                    "horizon": len(row),
                    "teacher_final": row[-1]["teacher_forced"]["matched"]["mse"],
                    "recursive_final": row[-1]["recursive"]["matched"]["mse"],
                    "shuffled_final": row[-1]["recursive"]["shuffled"]["mse"],
                    "endpoint_cost_rank": row[-1]["recursive"]["goal_cost_rank"],
                })
            return result

        before = score()
        windows = []
        for case, bank in banks:
            if case["episode"] not in training_episodes:
                continue
            count, horizon = bank["actions"].shape[:2]
            states = torch.cat((bank["history"].expand(count, *bank["history"].shape[1:]), bank["future"]), 1)
            controls = torch.cat((bank["past"].expand(count, -1, -1), bank["actions"]), 1)
            size = local.history_size
            windows.append((torch.cat([states[:, t:t + size + 1] for t in range(horizon)]),
                            torch.cat([controls[:, t:t + size] for t in range(horizon)])))
        losses, learning_rates = [], []
        for step in range(updates):
            latent, actions = windows[step % len(windows)]
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss, _ = local.representation_loss({}, latent, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, local.grad_clip, error_if_nonfinite=True)
            learning_rates.append([group["lr"] for opt in optimizers for group in opt.param_groups])
            for opt in optimizers:
                opt.step()
            if getattr(local, "scheduler", None) is not None:
                local.scheduler.step()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite native fitting-control loss.")
            losses.append(float(loss.detach()))
        after = score()
    if tensor_digest(protected) != frozen:
        raise RuntimeError("Fitting control changed frozen encoder/readout tensors.")
    return {"status": "COMPLETE", "updates": updates, "before": before, "after": after,
            "weights_changed": tensor_digest(local.state_dict()) != initial, "learning_rates": learning_rates,
            "loss_first": losses[0], "loss_last": losses[-1], "train_episodes": sorted(training_episodes),
            "validation_episodes": sorted(set(episodes) - training_episodes),
            "scope": "Copy only; native one-step loss, existing optimizer moments, fresh native adaptation LR schedule, frozen encoder, dropout/AMP/TF32 off. No rollout-loss adaptation.",
            "note": "Originally heldout episodes in train_episodes are fitting data for this diagnostic only, never validation. COMPLETE is not a learning-quality pass."}
