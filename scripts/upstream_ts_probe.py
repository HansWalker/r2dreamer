"""Pinned upstream TS predictor/rollout parity, with an explicit vision-only adapter."""

import ast
import copy
import hashlib
import io
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from inspect import signature
from types import SimpleNamespace
from urllib.request import urlopen

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import tools

from scripts.planner_recipe_support import NormalizedActionEncoder

COMMIT = "2c3c7666a69a730042590d548c6731259c4183ac"
BASE_URL = f"https://raw.githubusercontent.com/agentic-learning-ai-lab/temporal-straightening/{COMMIT}/models"
HASHES = {
    "vit.py": "60c77bdb03f3b2d565bfdb1695c9c193a802236763d33a03e79ed382b51f0670",
    "proprio.py": "3d50f7df7329985eceebd819effb37d2765a5734c921545208041df3e80c925b",
    "visual_world_model.py": "b1f20594248a6d4b08d8f452df77f68d0f894c0eea6faacdc466728da4c2904d",
    "objectives.py": "3d45b48728cdded0ff4bc75cc28f57e9a8db71b7d49681069216967f6f64d0c1",
}


def sources(cache):
    """Only fetch these small, immutable files; reject changed or corrupted cache entries."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, expected in HASHES.items():
        path = cache / name
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"{BASE_URL}/{name}" if name != "objectives.py" else f"{BASE_URL.rsplit('/', 1)[0]}/planning/{name}"
            with urlopen(url, timeout=30) as response:
                data = response.read(256_000)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Pinned upstream hash mismatch: {path}")
        if not path.exists():
            path.write_bytes(data)
        result[name] = data.decode("utf-8")
    return result


def load_reference(source, device):
    # Upstream hard-codes one mask to CUDA. Change only that device literal in the AST.
    tree = ast.parse(source["vit.py"])
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "cuda"]
    if len(matches) != 1:
        raise ValueError("Unexpected upstream mask device allocation.")
    matches[0].value = str(device)
    namespace = {}
    exec(compile(tree, f"upstream/{COMMIT}/vit.py", "exec"), namespace)
    proprio = {}
    exec(compile(source["proprio.py"], f"upstream/{COMMIT}/proprio.py", "exec"), proprio)
    # Extract actual upstream methods, avoiding unrelated torchvision/model constructors.
    methods = {"encode", "encode_act", "predict", "separate_emb", "replace_actions_from_z", "rollout",
               "forward", "visual_only", "_cos_curvature", "total_curvature"}
    world = next(n for n in ast.parse(source["visual_world_model.py"]).body
                 if isinstance(n, ast.ClassDef) and n.name == "VWorldModel")
    world.body = [n for n in world.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if {n.name for n in world.body} != methods:
        raise ValueError("Missing upstream rollout methods.")
    scope = {"torch": torch, "nn": nn, "F": torch.nn.functional, "rearrange": rearrange, "repeat": repeat}
    exec(compile(ast.Module(body=[world], type_ignores=[]), f"upstream/{COMMIT}/visual_world_model.py", "exec"), scope)
    return SimpleNamespace(predictor=namespace["ViTPredictor"], action=proprio["ProprioceptiveEmbedding"],
                           world=scope["VWorldModel"])


def predictor_key(key):
    if key == "position":
        return "pos_embedding"
    if key.startswith("norm."):
        return f"transformer.{key}"
    _, index, part, suffix = key.split(".", 3)
    return f"transformer.layers.{index}.{0 if part == 'attention' else 1}.{suffix}"


def reference_model(model, source):
    upstream = load_reference(source, model.device)
    local = model.predictor
    first = local.blocks[0]
    dim = local.position.shape[-1]
    reference = upstream.world()
    reference.predictor = upstream.predictor(
        num_patches=local._mask_patches, num_frames=model.history_size, dim=dim,
        depth=len(local.blocks), heads=first.attention.heads, dim_head=first.attention.dim_head,
        mlp_dim=first.feed_forward.net[1].out_features, dropout=0., emb_dropout=0.,
    ).to(model.device)
    translated = {predictor_key(key): value for key, value in local.state_dict().items()}
    reference.predictor.load_state_dict(translated, strict=True)
    wrapped = model.action_encoder
    encoder = wrapped.encoder if isinstance(wrapped, NormalizedActionEncoder) else wrapped
    with redirect_stdout(io.StringIO()):
        action = upstream.action(num_frames=model.history_size, tubelet_size=1,
                                 in_chans=model.action_dim, emb_dim=encoder.net[0].out_features).to(model.device)
    action.load_state_dict({"patch_embed.weight": encoder.net[0].weight[..., None],
                            "patch_embed.bias": encoder.net[0].bias,
                            "norm.weight": encoder.net[1].weight, "norm.bias": encoder.net[1].bias})
    reference.action_encoder = (NormalizedActionEncoder(action, wrapped.mean, wrapped.std, model.device)
                                if isinstance(wrapped, NormalizedActionEncoder) else action)
    reference.num_hist = model.history_size
    reference.concat_dim = 1
    reference.proprio_dim = 0
    reference.action_dim = encoder.net[0].out_features
    reference.num_action_repeat = reference.num_proprio_repeat = 1
    reference.num_pred = 1
    reference.decoder = None
    reference.train_decoder = False
    reference.stop_grad = True
    reference.emb_criterion = nn.MSELoss()
    reference.vcreg = False
    reference.straighten = model.curvature_weight > 0
    reference.straighten_scale = model.curvature_weight
    reference.curvature_mode = "cos"
    # Reuse upstream encode/replace/separate/rollout unchanged. Input is cached visual
    # tokens, not raw images; the deliberately removed proprioception has zero width.
    reference.encode_obs = lambda obs: {
        "visual": obs["visual"], "proprio": obs["visual"].new_empty(*obs["visual"].shape[:2], 0)}
    reference.eval()
    reference.requires_grad_(False)
    return reference


@contextmanager
def full_precision():
    """Autocast off is insufficient: Linear and Conv can otherwise use different TF32 paths."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@contextmanager
def frozen_precision(model, amp):
    previous = model.use_amp
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.use_amp = amp
    model.requires_grad_(False)
    try:
        with nullcontext() if amp else full_precision():
            yield
    finally:
        model.use_amp = previous
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def comparison(actual, expected, *, atol=2e-5, rtol=2e-4):
    actual, expected = actual.detach().float(), expected.detach().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Non-finite parity result.")
    return {"pass": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
            "max_abs_error": float((actual - expected).abs().max()),
            "rms_error": float((actual - expected).square().mean().sqrt()), "atol": atol, "rtol": rtol}


@tools.preserve_rng_state
def parity(model, source, history, past, actions):
    """One candidate, full history, five-step forward and action-gradient comparison."""
    reference = reference_model(model, source)
    history, past, actions = history[:1].float(), past[:1].float(), actions[:1, :1].float()
    with frozen_precision(model, False), torch.enable_grad():
        controls = actions.detach().requires_grad_()
        all_actions = torch.cat((past, controls[:, 0]), 1)
        one = all_actions[:, :model.history_size]
        local_one = model.predict(history, one)
        ref_one = reference.predict(reference.encode({"visual": history}, one))[..., :local_one.shape[-1]]
        local = model.rollout(history, past, controls)[:, 0]
        ref, _ = reference.rollout({"visual": history}, all_actions)
        ref = ref["visual"][:, model.history_size:]
        direction = torch.randn(local.shape, generator=torch.Generator().manual_seed(1701)).to(model.device)
        grad_local = torch.autograd.grad((local * direction).sum(), controls, retain_graph=True)[0]
        grad_ref = torch.autograd.grad((ref * direction).sum(), controls, retain_graph=True)[0]
        result = {"action_embedding": comparison(model.action_encoder(one), reference.encode_act(one)),
                  "teacher_forced": comparison(local_one, ref_one),
                  "rollout": comparison(local, ref),
                  "action_gradient": comparison(grad_local, grad_ref, atol=2e-4, rtol=2e-3)}
        from models.shared.latent_goal import latent_goal_cost
        namespace = {}
        exec(compile(source["objectives.py"], f"upstream/{COMMIT}/objectives.py", "exec"), namespace)
        objective = namespace["create_objective_fn"](alpha=0, base=2, mode="all")
        target = history[:, -1:].detach()
        # A nonempty dummy proprio channel avoids 0 * mean(empty) in upstream scoring.
        full = torch.cat((history, ref), dim=1)
        ref_cost = objective({"visual": full, "proprio": full.new_zeros(*full.shape[:2], 1)},
                             {"visual": target.expand_as(full), "proprio": full.new_zeros(*full.shape[:2], 1)})
        local_cost = latent_goal_cost(local[:, None], target[:, 0], reduction="mean", mode="ts_mpc", history=history)[:, 0]
        result["mpc_cost"] = comparison(local_cost, ref_cost)
        result["mpc_action_gradient"] = comparison(torch.autograd.grad(local_cost.sum(), controls, retain_graph=True)[0],
                                                    torch.autograd.grad(ref_cost.sum(), controls)[0], atol=2e-4, rtol=2e-3)
    return {"status": "PASS" if all(x["pass"] for x in result.values()) else "MISMATCH", "checks": result,
            "commit": COMMIT, "source_sha256": HASHES,
            "scope": "FP32 eval-mode predictor, action embedding and actual upstream rollout; matched weights, cached visual latents, no proprioception.",
            "compatibility": "Mask device literal changed; action Conv1d kernel=1 mapped from Linear; shared train action normalization.",
            "precision": "Autocast and TF32 disabled; math SDPA; previous backend settings restored.",
            "not_tested": "Upstream vision encoder, proprioception, training, loss and optimizer."}


@tools.preserve_rng_state
def training_parity(model, source, latent, actions):
    """Compare actual upstream visual loss/backprop and one matched optimizer step on copies."""
    local = copy.deepcopy(model)
    local.decoder = None
    local.use_amp = False
    local.eval()  # Dropout off on BOTH copies; stochastic mask equivalence is not assumed.
    reference = reference_model(local, source)
    reference.requires_grad_(True)
    groups = {}
    for name in ("predictor", "action_encoder"):
        module = getattr(local, name)
        ref_module = getattr(reference, name)
        reference_parameters = dict(ref_module.named_parameters())
        pairs = []
        for key, parameter in module.named_parameters():
            if name == "predictor":
                mapped = predictor_key(key)
            else:
                mapped = key.replace("net.0.weight", "patch_embed.weight").replace("net.0.bias", "patch_embed.bias")
                mapped = mapped.replace("net.1.", "norm.")
            pairs.append((parameter, reference_parameters[mapped]))
        groups[name] = pairs
    optimizers = []
    for name, pairs in groups.items():
        original = local.optimizers[name]
        state = copy.deepcopy(original.state_dict())
        for side in (0, 1):
            parameters = [pair[side] for pair in pairs]
            kwargs = {key: value for key, value in original.defaults.items() if key in signature(type(original)).parameters}
            optimizer = type(original)(parameters, **kwargs)
            optimizer.load_state_dict(copy.deepcopy(state))
            for parameter in parameters:
                for key, value in optimizer.state[parameter].items():
                    if isinstance(value, torch.Tensor) and value.numel() == parameter.numel():
                        optimizer.state[parameter][key] = value.reshape_as(parameter)
            optimizer.zero_grad(set_to_none=True)
            optimizers.append(optimizer)
    left = latent.detach().float().clone().requires_grad_()
    right = latent.detach().float().clone().requires_grad_()
    controls = actions.detach().float()
    with full_precision(), torch.enable_grad():
        loss, metrics = local.representation_loss({}, left, controls)
        padded = torch.cat((controls, torch.zeros_like(controls[:, :1])), dim=1)
        _, _, _, ref_loss, ref_metrics = reference({"visual": right}, padded)
        # Production retains the old visual channel coefficient; make this adaptation explicit.
        ref_loss = ref_loss + (local.prediction_weight - 1) * ref_metrics["z_loss"]
        loss.backward()
        ref_loss.backward()
        checks = {"loss": comparison(loss, ref_loss),
                  "prediction_loss": comparison(metrics["prediction_loss"], ref_metrics["z_visual_loss"]),
                  "latent_gradient": comparison(left.grad, right.grad)}
        pairs = [pair for group in groups.values() for pair in group]
        for side in (0, 1):
            torch.nn.utils.clip_grad_norm_([p[side] for p in pairs], local.grad_clip, error_if_nonfinite=True)
        for name, group in groups.items():
            checks[name + "_gradient"] = comparison(
                torch.cat([a.grad.flatten() for a, _ in group]), torch.cat([b.grad.flatten() for _, b in group]))
        for optimizer in optimizers:
            optimizer.step()
        for name, group in groups.items():
            checks[name + "_step"] = comparison(
                torch.cat([a.detach().flatten() for a, _ in group]), torch.cat([b.detach().flatten() for _, b in group]))
    return {"status": "PASS" if all(v["pass"] for v in checks.values()) else "MISMATCH", "checks": checks,
            "prediction_weight": local.prediction_weight,
            "scope": "Actual upstream visual loss, cached-latent gradients, predictor/action gradients and one optimizer step; copies only.",
            "not_tested": "Vision encoder/proprioception, stochastic dropout masks or full upstream training recipe. Encoder gradients are checked only at cached latent inputs.",
            "precision": "FP32, TF32 off, math SDPA, dropout off; optimizer moments and clipping matched."}
