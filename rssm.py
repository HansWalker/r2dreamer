import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_

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
        mlp_hidden,
        act="SiLU",
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "RSSM core=mamba3 requires mamba_ssm.modules.mamba3.Mamba3. "
                "Install a local Mamba package that includes Mamba3."
            ) from _MAMBA3_IMPORT_ERROR
        act = getattr(torch.nn, act)
        self.norm1 = nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32)
        self.mamba = Mamba3(
            d_model=deter,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        self.norm2 = nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32)
        self.mlp = nn.Sequential(
            nn.Linear(deter, mlp_hidden, bias=True),
            act(),
            nn.Linear(mlp_hidden, deter, bias=True),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def apply_weight_init(self):
        weight_init_(self.norm1)
        weight_init_(self.norm2)
        self.mlp.apply(weight_init_)


class Mamba3Deter(nn.Module):
    def __init__(
        self,
        deter,
        stoch,
        act_dim,
        hidden,
        context_len=16,
        n_layers=1,
        d_state=16,
        expand=1,
        headdim=128,
        is_mimo=False,
        mimo_rank=1,
        chunk_size=16,
        mlp_hidden_mult=1,
        act="SiLU",
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "RSSM core=mamba3 requires mamba_ssm.modules.mamba3.Mamba3. "
                "Install a local Mamba package that includes Mamba3."
            ) from _MAMBA3_IMPORT_ERROR
        inner = int(deter) * int(expand)
        if inner % int(headdim) != 0:
            raise ValueError(
                f"Mamba3 requires deter * expand to be divisible by headdim, "
                f"got deter={deter}, expand={expand}, headdim={headdim}."
            )
        act_cls = getattr(torch.nn, act)
        self.deter = int(deter)
        self.context_len = max(1, int(context_len))
        self.chunk_size = max(1, int(chunk_size))
        mlp_hidden = max(1, int(self.deter * float(mlp_hidden_mult)))
        self._token = nn.Sequential(
            nn.Linear(self.deter + int(stoch) + int(act_dim), int(hidden), bias=True),
            nn.RMSNorm(int(hidden), eps=1e-04, dtype=torch.float32),
            act_cls(),
            nn.Linear(int(hidden), self.deter, bias=True),
            nn.RMSNorm(self.deter, eps=1e-04, dtype=torch.float32),
        )
        self.layers = nn.ModuleList(
            [
                Mamba3Layer(
                    deter=self.deter,
                    layer_idx=i,
                    d_state=int(d_state),
                    expand=int(expand),
                    headdim=int(headdim),
                    is_mimo=bool(is_mimo),
                    mimo_rank=int(mimo_rank),
                    chunk_size=self.chunk_size,
                    mlp_hidden=mlp_hidden,
                    act=act,
                )
                for i in range(int(n_layers))
            ]
        )
        self.out_norm = nn.RMSNorm(self.deter, eps=1e-04, dtype=torch.float32)

    def initial_context(self, batch_size, device, dtype=torch.float32):
        return torch.zeros(
            batch_size,
            self.context_len,
            self.deter,
            dtype=dtype,
            device=device,
        )

    def _run_mamba(self, context):
        T = context.shape[1]
        pad = (-T) % self.chunk_size
        x = context
        if pad:
            zeros = torch.zeros(
                x.shape[0], pad, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, zeros], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return x[:, T - 1]

    def forward(self, stoch, deter, action, context=None):
        # (B, S, K), (B, D), (B, A), optional (B, C, D)
        B = action.shape[0]
        if context is None:
            context = self.initial_context(B, deter.device, deter.dtype)
        elif context.dtype != deter.dtype or context.device != deter.device:
            context = context.to(device=deter.device, dtype=deter.dtype)
        if context.shape[1] != self.context_len:
            if context.shape[1] > self.context_len:
                context = context[:, -self.context_len :]
            else:
                pad = self.initial_context(B, deter.device, deter.dtype)[
                    :, : self.context_len - context.shape[1]
                ]
                context = torch.cat([pad, context], dim=1)

        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        token = self._token(torch.cat([deter, stoch, action], dim=-1))
        if self.context_len == 1:
            context = token.unsqueeze(1)
        else:
            context = torch.cat([context[:, 1:], token.unsqueeze(1)], dim=1)
        deter = self._run_mamba(context)
        return deter, context

    def apply_weight_init(self):
        self._token.apply(weight_init_)
        for layer in self.layers:
            layer.apply_weight_init()
        weight_init_(self.out_norm)


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
                self._hidden,
                context_len=_cfg_get(mcfg, "context_len", 16),
                n_layers=_cfg_get(mcfg, "n_layers", 1),
                d_state=_cfg_get(mcfg, "d_state", 16),
                expand=_cfg_get(mcfg, "expand", 1),
                headdim=_cfg_get(mcfg, "headdim", 128),
                is_mimo=_cfg_get(mcfg, "is_mimo", False),
                mimo_rank=_cfg_get(mcfg, "mimo_rank", 1),
                chunk_size=_cfg_get(mcfg, "chunk_size", 16),
                mlp_hidden_mult=_cfg_get(mcfg, "mlp_hidden_mult", 1),
                act=config.act,
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

    def initial_context(self, batch_size, dtype=torch.float32):
        if not self.uses_context:
            return None
        return self._deter_net.initial_context(batch_size, self._device, dtype)

    def _unpack_state(self, state):
        if len(state) == 3:
            return state
        stoch, deter = state
        return stoch, deter, None

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D), optional context), (B, T)
        L = action.shape[1]
        stoch, deter, context = self._unpack_state(initial)
        stochs, deters, logits = [], [], []
        contexts = [] if self.uses_context else None
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit, context = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i], context)
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            if contexts is not None:
                contexts.append(context)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        contexts = torch.stack(contexts, dim=1) if contexts is not None else None
        return stochs, deters, logits, contexts

    def obs_step(self, stoch, deter, prev_action, embed, reset, context=None):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,), optional (B, C, D)
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        if self.uses_context:
            if context is None:
                context = self._deter_net.initial_context(stoch.shape[0], deter.device, deter.dtype)
            context = torch.where(
                rpad(reset, context.dim() - int(reset.dim())),
                torch.zeros_like(context),
                context,
            )

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        if self.uses_context:
            deter, context = self._deter_net(stoch, deter, prev_action, context)
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, D + E)
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit, context

    def img_step(self, stoch, deter, prev_action, context=None):
        """Single prior step (no observation)."""

        # (B, D)
        if self.uses_context:
            deter, context = self._deter_net(stoch, deter, prev_action, context)
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        return stoch, deter, context

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions, context=None):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A), optional (B, C, D)
        L = actions.shape[1]
        stochs, deters = [], []
        contexts = [] if self.uses_context else None
        for i in range(L):
            stoch, deter, context = self.img_step(stoch, deter, actions[:, i], context)
            stochs.append(stoch)
            deters.append(deter)
            if contexts is not None:
                contexts.append(context)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        contexts = torch.stack(contexts, dim=1) if contexts is not None else None
        return stochs, deters, contexts

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
