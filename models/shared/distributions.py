import torch
from torch import distributions as torchd
from torch.nn import functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    has_rsample = True

    def __init__(self, logits, unimix_ratio=0.0):
        # (..., K)
        probs = F.softmax(logits.float(), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        # (..., K)
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode + (self.probs - self.probs.detach())

    def rsample(self, sample_shape=torch.Size()):
        # Hard categorical values with DreamerV3's probability straight-through gradient.
        sample = super().sample(sample_shape)
        return sample + (self.probs - self.probs.detach())


class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = logits.float()
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
            return self.unsquash(wavg)
        p1 = self.probs[..., : n // 2]
        p2 = self.probs[..., n // 2 :]
        b1 = self.bins[..., : n // 2]
        b2 = self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = (self.bins <= target_squashed.unsqueeze(-1)).int().sum(dim=-1) - 1
        above = len(self.bins) - (self.bins > target_squashed.unsqueeze(-1)).int().sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = F.one_hot(below, num_classes=len(self.bins)).float()
        oh_above = F.one_hot(above, num_classes=len(self.bins)).float()
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        # (..., D)
        self._mode = mode.float()
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class TanhNormal(torchd.TransformedDistribution):
    """Diagonal Gaussian squashed into the continuous action bounds."""

    def __init__(self, mean, std):
        normal = torchd.Independent(torchd.Normal(mean.float(), std.float()), 1)
        super().__init__(normal, [torchd.TanhTransform(cache_size=1)])

    @property
    def mode(self):
        # Deterministic policy action, not the mode of the transformed density.
        return self.base_dist.mode.tanh()

    def log_prob(self, action):
        # Expert actions and float32 tanh can reach +/-1, where atanh is infinite.
        return super().log_prob(action.float().clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    def entropy(self):
        # Reparameterized estimate: H(tanh(X)) = H(X) + E[log |tanh'(X)|].
        raw = self.base_dist.rsample()
        transform = self.transforms[0]
        correction = transform.log_abs_det_jacobian(raw, raw.tanh()).sum(-1)
        return self.base_dist.entropy() + correction


def tanh_normal(x, min_std, max_std):
    mean, std = torch.chunk(x.float(), 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    return TanhNormal(mean, std)


def binary(logits, **kwargs):
    return torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=logits.float()), 1)


def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(logits, bins)


def mse(logits, **kwargs):
    return MSEDist(logits)
