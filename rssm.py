import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_

MAMBA_CACHE_KEYS = (
    "mamba_angle_state",
    "mamba_ssm_state",
    "mamba_k_state",
    "mamba_v_state",
)

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = exc
else:
    _MAMBA3_IMPORT_ERROR = None


def _cfg_get(config, name, default):
    try:
        return getattr(config, name)
    except Exception:
        return default


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A)
        B = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        # (B, 3*U)
        x = torch.cat([x0, x1, x2], -1)
        # (B, G, 3*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        # (B, D)
        return update * cand + (1 - update) * deter


class Mamba3Layer(nn.Module):
    def __init__(
        self,
        deter,
        layer_idx,
        d_state,
        expand,
        headdim,
        is_mimo,
        mimo_rank,
        chunk_size,
        is_outproj_norm=False,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "RSSM core=mamba3 requires mamba_ssm.modules.mamba3.Mamba3. "
                "Install a local Mamba package that includes Mamba3."
            ) from _MAMBA3_IMPORT_ERROR
        self.mamba = Mamba3(
            d_model=deter,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=is_outproj_norm,
            layer_idx=layer_idx,
        )

    def initial_context(self, batch_size, device=None, dtype=None):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen=0, device=device, dtype=dtype)

    def step(self, x, angle_state, ssm_state, k_state, v_state):
        out, angle_state, ssm_state, k_state, v_state = self.mamba.step(
            x,
            angle_state,
            ssm_state,
            k_state,
            v_state,
        )
        return out, angle_state, ssm_state, k_state, v_state


class Mamba3Deter(nn.Module):
    def __init__(
        self,
        deter,
        stoch,
        act_dim,
        n_layers=1,
        d_state=32,
        expand=1,
        headdim=64,
        is_mimo=False,
        mimo_rank=1,
        chunk_size=16,
        is_outproj_norm=False,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "RSSM core=mamba3 requires mamba_ssm.modules.mamba3.Mamba3. "
                "Install a local Mamba package that includes Mamba3."
            ) from _MAMBA3_IMPORT_ERROR
        if int(n_layers) != 1:
            raise ValueError("The first real-cache Mamba3 RSSM implementation supports n_layers=1 only.")
        expand = int(expand)
        headdim = int(headdim)
        d_state = int(d_state)
        if expand <= 0 or headdim <= 0:
            raise ValueError(
                f"Mamba3 requires positive expand and headdim, got expand={expand}, headdim={headdim}."
            )
        if d_state not in (32, 64, 128):
            raise ValueError(
                f"Mamba3 step mode requires d_state in [32, 64, 128], got d_state={d_state}."
            )
        inner = int(deter) * expand
        if inner % headdim != 0:
            raise ValueError(
                f"Mamba3 requires deter * expand to be divisible by headdim, "
                f"got deter={deter}, expand={expand}, headdim={headdim}."
            )
        nheads = inner // headdim
        if nheads % 4 != 0:
            raise ValueError(
                "Mamba3 step mode requires the number of heads to be divisible by 4, "
                f"got nheads={nheads} from deter={deter}, expand={expand}, headdim={headdim}. "
                "Use a smaller headdim or larger expand."
            )
        self.deter = int(deter)
        self._token = nn.Linear(int(stoch) + int(act_dim), self.deter, bias=True)
        self.layer = Mamba3Layer(
            deter=self.deter,
            layer_idx=0,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=bool(is_mimo),
            mimo_rank=int(mimo_rank),
            chunk_size=max(1, int(chunk_size)),
            is_outproj_norm=bool(is_outproj_norm),
        )

    def initial_context(self, batch_size, device=None, dtype=None):
        return self.layer.initial_context(batch_size, device=device, dtype=dtype)

    def _norm_action(self, action):
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        return action

    def forward(self, stoch, action, angle_state=None, ssm_state=None, k_state=None, v_state=None):
        # (B, S, K), (B, A), optional official Mamba3 cache tensors
        B = action.shape[0]
        if angle_state is None or ssm_state is None or k_state is None or v_state is None:
            angle_state, ssm_state, k_state, v_state = self.initial_context(B, device=stoch.device)
        else:
            angle_state = angle_state.to(device=stoch.device)
            ssm_state = ssm_state.to(device=stoch.device)
            k_state = k_state.to(device=stoch.device)
            v_state = v_state.to(device=stoch.device)
        stoch = stoch.reshape(B, -1)
        action = self._norm_action(action)
        token = self._token(torch.cat([stoch, action], dim=-1))
        deter, angle_state, ssm_state, k_state, v_state = self.layer.step(
            token,
            angle_state,
            ssm_state,
            k_state,
            v_state,
        )
        return deter, angle_state, ssm_state, k_state, v_state

    def apply_weight_init(self):
        weight_init_(self._token)


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self._core = str(_cfg_get(config, "core", "block_gru"))
        self._warmup = int(_cfg_get(config, "warmup", 0))
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        if self._core == "block_gru":
            self._deter_net = Deter(
                self._deter,
                self.flat_stoch,
                act_dim,
                self._hidden,
                blocks=self._blocks,
                dynlayers=self._dyn_layers,
                act=config.act,
            )
        elif self._core == "mamba3":
            mcfg = _cfg_get(config, "mamba3", None)
            self._deter_net = Mamba3Deter(
                self._deter,
                self.flat_stoch,
                act_dim,
                n_layers=_cfg_get(mcfg, "n_layers", 1),
                d_state=_cfg_get(mcfg, "d_state", 32),
                expand=_cfg_get(mcfg, "expand", 1),
                headdim=_cfg_get(mcfg, "headdim", 64),
                is_mimo=_cfg_get(mcfg, "is_mimo", False),
                mimo_rank=_cfg_get(mcfg, "mimo_rank", 1),
                chunk_size=_cfg_get(mcfg, "chunk_size", 16),
                is_outproj_norm=_cfg_get(mcfg, "is_outproj_norm", False),
            )
        else:
            raise ValueError(f"Unsupported RSSM core: {self._core}")

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        if hasattr(self._deter_net, "apply_weight_init"):
            self._deter_net.apply_weight_init()
        else:
            self._deter_net.apply(weight_init_)
        self._obs_net.apply(weight_init_)
        self._img_net.apply(weight_init_)

    @property
    def uses_context(self):
        return self._core == "mamba3"

    def initial_context(self, batch_size, dtype=None):
        if not self.uses_context:
            return None
        return self._deter_net.initial_context(batch_size, device=self._device, dtype=dtype)

    def _unpack_state(self, state):
        if len(state) == 2:
            stoch, deter = state
            return stoch, deter, None
        if len(state) == 6:
            stoch, deter, angle_state, ssm_state, k_state, v_state = state
            return stoch, deter, (angle_state, ssm_state, k_state, v_state)
        raise ValueError(f"Expected RSSM state with 2 or 6 tensors, got {len(state)}.")

    def _ensure_cache(self, cache, batch_size, device):
        if not self.uses_context:
            return None
        if cache is None or any(tensor is None for tensor in cache):
            return self._deter_net.initial_context(batch_size, device=device)
        return tuple(tensor.to(device=device) for tensor in cache)

    def _reset_cache(self, cache, reset):
        if cache is None:
            return None
        return tuple(
            torch.where(
                rpad(reset, tensor.dim() - int(reset.dim())),
                torch.zeros_like(tensor),
                tensor,
            )
            for tensor in cache
        )

    def _stack_cache(self, caches):
        return tuple(torch.stack(items, dim=1) for items in zip(*caches))

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D), optional cache tensors), (B, T)
        L = action.shape[1]
        stoch, deter, cache = self._unpack_state(initial)
        if self.uses_context:
            cache = self._ensure_cache(cache, stoch.shape[0], deter.device)
        stochs, deters, logits = [], [], []
        caches = [] if self.uses_context else None
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            if self.uses_context:
                stoch, deter, logit, *cache = self.obs_step(
                    stoch, deter, action[:, i], embed[:, i], reset[:, i], *cache
                )
                cache = tuple(cache)
            else:
                stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            if caches is not None:
                caches.append(tuple(tensor.clone() for tensor in cache))
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        if caches is not None:
            return stochs, deters, logits, *self._stack_cache(caches)
        return stochs, deters, logits

    def obs_step(
        self,
        stoch,
        deter,
        prev_action,
        embed,
        reset,
        mamba_angle_state=None,
        mamba_ssm_state=None,
        mamba_k_state=None,
        mamba_v_state=None,
    ):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,), optional Mamba3 cache tensors
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        if self.uses_context:
            cache = self._ensure_cache(
                (mamba_angle_state, mamba_ssm_state, mamba_k_state, mamba_v_state),
                stoch.shape[0],
                deter.device,
            )
            cache = self._reset_cache(cache, reset)

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        if self.uses_context:
            deter, *cache = self._deter_net(stoch, prev_action, *cache)
            cache = tuple(cache)
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, D + E)
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        if self.uses_context:
            return stoch, deter, logit, *cache
        return stoch, deter, logit

    def img_step(
        self,
        stoch,
        deter,
        prev_action,
        mamba_angle_state=None,
        mamba_ssm_state=None,
        mamba_k_state=None,
        mamba_v_state=None,
    ):
        """Single prior step (no observation)."""

        # (B, D)
        if self.uses_context:
            cache = self._ensure_cache(
                (mamba_angle_state, mamba_ssm_state, mamba_k_state, mamba_v_state),
                stoch.shape[0],
                deter.device,
            )
            deter, *cache = self._deter_net(stoch, prev_action, *cache)
            cache = tuple(cache)
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        if self.uses_context:
            return stoch, deter, *cache
        return stoch, deter

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(
        self,
        stoch,
        deter,
        actions,
        mamba_angle_state=None,
        mamba_ssm_state=None,
        mamba_k_state=None,
        mamba_v_state=None,
    ):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A), optional Mamba3 cache tensors
        L = actions.shape[1]
        stochs, deters = [], []
        cache = (mamba_angle_state, mamba_ssm_state, mamba_k_state, mamba_v_state)
        caches = [] if self.uses_context else None
        if self.uses_context:
            cache = tuple(None if tensor is None else tensor.clone() for tensor in cache)
        for i in range(L):
            if self.uses_context:
                stoch, deter, *cache = self.img_step(stoch, deter, actions[:, i], *cache)
                cache = tuple(cache)
            else:
                stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
            if caches is not None:
                caches.append(tuple(tensor.clone() for tensor in cache))
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        if caches is not None:
            return stochs, deters, *self._stack_cache(caches)
        return stochs, deters

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
