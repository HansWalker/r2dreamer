"""Pinned upstream TS predictor/rollout parity, with an explicit vision-only adapter."""

import ast
import hashlib
import io
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import torch
from einops import rearrange, repeat
from torch import nn

from scripts.planner_recipe_support import NormalizedActionEncoder

COMMIT = "2c3c7666a69a730042590d548c6731259c4183ac"
BASE_URL = f"https://raw.githubusercontent.com/agentic-learning-ai-lab/temporal-straightening/{COMMIT}/models"
HASHES = {
    "vit.py": "60c77bdb03f3b2d565bfdb1695c9c193a802236763d33a03e79ed382b51f0670",
    "proprio.py": "3d50f7df7329985eceebd819effb37d2765a5734c921545208041df3e80c925b",
    "visual_world_model.py": "b1f20594248a6d4b08d8f452df77f68d0f894c0eea6faacdc466728da4c2904d",
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
            with urlopen(f"{BASE_URL}/{name}", timeout=30) as response:
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
    methods = {"encode", "encode_act", "predict", "separate_emb", "replace_actions_from_z", "rollout"}
    world = next(n for n in ast.parse(source["visual_world_model.py"]).body
                 if isinstance(n, ast.ClassDef) and n.name == "VWorldModel")
    world.body = [n for n in world.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if {n.name for n in world.body} != methods:
        raise ValueError("Missing upstream rollout methods.")
    scope = {"torch": torch, "nn": nn, "rearrange": rearrange, "repeat": repeat}
    exec(compile(ast.Module(body=[world], type_ignores=[]), f"upstream/{COMMIT}/visual_world_model.py", "exec"), scope)
    return SimpleNamespace(predictor=namespace["ViTPredictor"], action=proprio["ProprioceptiveEmbedding"],
                           world=scope["VWorldModel"])


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
    translated = {}
    for key, value in local.state_dict().items():
        if key == "position":
            target = "pos_embedding"
        elif key.startswith("norm."):
            target = f"transformer.{key}"
        else:
            _, index, part, suffix = key.split(".", 3)
            target = f"transformer.layers.{index}.{0 if part == 'attention' else 1}.{suffix}"
        translated[target] = value
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
    # Reuse upstream encode/replace/separate/rollout unchanged. Input is cached visual
    # tokens, not raw images; the deliberately removed proprioception has zero width.
    reference.encode_obs = lambda obs: {
        "visual": obs["visual"], "proprio": obs["visual"].new_empty(*obs["visual"].shape[:2], 0)}
    reference.eval()
    reference.requires_grad_(False)
    return reference


@contextmanager
def frozen_precision(model, amp):
    previous = model.use_amp
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.use_amp = amp
    model.requires_grad_(False)
    try:
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
        grad_ref = torch.autograd.grad((ref * direction).sum(), controls)[0]
        result = {"action_embedding": comparison(model.action_encoder(one), reference.encode_act(one)),
                  "teacher_forced": comparison(local_one, ref_one),
                  "rollout": comparison(local, ref),
                  "action_gradient": comparison(grad_local, grad_ref, atol=2e-4, rtol=2e-3)}
    return {"status": "PASS" if all(x["pass"] for x in result.values()) else "MISMATCH", "checks": result,
            "commit": COMMIT, "source_sha256": HASHES,
            "scope": "FP32 eval-mode predictor, action embedding and actual upstream rollout; matched weights, cached visual latents, no proprioception.",
            "compatibility": "Mask device literal changed; action Conv1d kernel=1 mapped from Linear; shared train action normalization.",
            "not_tested": "Upstream vision encoder, proprioception, training, loss, optimizer and planner objective."}
