"""Train-only goal pairs and fixed held-out diagnostics for TS and LeWM.

Uses the existing task branch bank, which already associates real images and
simulator labels with the correct episode goal. Main dynamics replay is untouched.
"""

import hashlib
import math

import torch

from models.shared.goal_ranking import goal_ranking_loss
from models.shared.physical_state import PhysicalStateTargets


def goal_error(state, targets, geometry, tolerance):
    """Dimensionless distance: <=1 satisfies the task's geometric goal bounds."""
    state = torch.as_tensor(state, dtype=torch.float32)
    tolerance = torch.as_tensor(tolerance, dtype=torch.float32, device=state.device)
    if (not torch.isfinite(state).all() or not torch.isfinite(tolerance).all()
            or not torch.all(tolerance > 0)):
        raise ValueError("Goal-pair states and positive tolerances must be finite")
    relation = targets.goal_relation(state)
    if geometry == "box" and tolerance.numel() == 2:
        return (relation.abs() / tolerance).amax(-1)
    if geometry == "radial" and tolerance.numel() == 1:
        return relation.norm(dim=-1) / tolerance[0]
    raise ValueError("Goal-pair geometry/tolerance shape mismatch")


class GoalPairBank:
    """Uniform eligible-anchor sampling, then uniform good/bad frames per anchor.

Only actual successor frames are used. No future predictions or validation
labels select training examples. A private generator avoids changing native RNG.
"""

    def __init__(self, cases, config, *, split="train", pairs=32, seed=74_000_000, far=2.):
        if split not in ("train", "validation") or pairs < 1 or seed < 0 or not math.isfinite(far) or far <= 1:
            raise ValueError("Require TRAIN/validation split, positive pair count, seed >=0, and far >1")
        self.split, self.pairs, self.seed, self.far = split, int(pairs), int(seed), float(far)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.draws = 0
        self.entries, self.inventory = [], []
        targets = PhysicalStateTargets(config.state_head.task, config.state_head.fields)
        goal = config.jepa_model.goal
        if goal.get("alternatives", []):
            raise ValueError("Goal-pair training currently requires the native single goal image")
        seen = set()
        for case in cases:
            if case["split"] != split or case["id"] in seen:
                raise ValueError("Goal-pair split mismatch or duplicate anchor")
            seen.add(case["id"])
            images, states = case["image"], case["states"]
            if (images.ndim != 5 or images.shape[:2] != states.shape[:2]
                    or states.shape[-1] != len(targets.coordinates)
                    or images.dtype != torch.uint8 or case["goal_image"].dtype != torch.uint8
                    or tuple(case["goal_image"].shape) != tuple(images.shape[2:])):
                raise ValueError("Goal-pair image, physical-label, or episode-goal layout mismatch")
            errors = goal_error(states.flatten(0, 1), targets, goal.geometry, list(goal.tolerance))
            good = torch.where(errors <= 1.)[0]
            bad = torch.where(errors >= self.far)[0]
            self.inventory.append({"id": case["id"], "good_frames": len(good), "bad_frames": len(bad),
                                   "eligible": bool(len(good) and len(bad))})
            if len(good) and len(bad):
                self.entries.append((images.flatten(0, 1), case["goal_image"], good, bad))
        if not self.entries:
            raise ValueError(f"No eligible {split} goal pairs: need success and clearly failed frames at the same goal")

    def sample(self, count=None, *, seed=None):
        count = self.pairs if count is None else int(count)
        if count < 1:
            raise ValueError("Goal pair count must be positive")
        generator = self.generator if seed is None else torch.Generator().manual_seed(int(seed))
        rows = []
        for index in torch.randint(len(self.entries), (count,), generator=generator).tolist():
            images, goal, good, bad = self.entries[index]
            a = good[torch.randint(len(good), (), generator=generator)]
            b = bad[torch.randint(len(bad), (), generator=generator)]
            rows.append(torch.stack((images[a], images[b], goal)))
        if seed is None:
            self.draws += count
        return torch.stack(rows)

    def metadata(self):
        return {"split": self.split, "pairs_per_update": self.pairs, "seed": self.seed, "far_threshold": self.far,
                "good_threshold": 1., "eligible_anchors": len(self.entries), "inventory": self.inventory,
                "sampling": "uniform eligible anchors, uniform good/bad actual frames at same episode goal",
                "role": "additional physical supervision; independent of native offline-to-online taper"}

    def state_dict(self):
        return {"metadata": self.metadata(), "draws": self.draws, "generator_state": self.generator.get_state()}


@torch.no_grad()
def evaluate_goal_pairs(model, source, *, pairs=512, seed=75_000_000):
    """Identical sampled held-out pairs at each checkpoint; no training RNG advance."""
    if source.split != "validation":
        raise ValueError("Goal ranking evaluation requires validation pairs")
    images = source.sample(pairs, seed=seed)
    sums = {}
    for chunk in images.split(32):
        _, metrics = goal_ranking_loss(model, chunk, model.goal_ranking_margin)
        for key, value in metrics.items():
            if key != "goal_ranking/pairs":
                sums[key] = sums.get(key, 0.) + float(value) * len(chunk)
    return {"pairs": pairs, "seed": seed, "source": source.metadata(),
            "images_sha256": hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
            **{key: value / pairs for key, value in sums.items()}}
