"""Pinned upstream component checks. A PASS is NOT an original-task reproduction.

Only small, hash-verified official Python sources are downloaded. No datasets,
checkpoints, third-party packages, model-hub code, or training runs are fetched.
"""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import torch
from torch import nn

from scripts import upstream_ts_probe as ts

LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
LEWM_HASHES = {
    "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
    "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    "train.py": "5d5666a785148635f8bcb50b49f98dd4527c5aba8a02d90206eb5d46406079d2",
}
TS_HASHES = {**ts.HASHES, "dino.py": "559df99ad69fa4b56188ab683304276840abf7b8ce41b85756ac21bb23706cb1"}
REPOSITORIES = {"temporal_straightening": "agentic-learning-ai-lab/temporal-straightening",
                "leworldmodel": "lucas-maes/le-wm"}


class ReferenceUnavailable(RuntimeError):
    pass


def pinned_sources(family, cache, allow_download=True):
    """Return original source text and provenance; corruption is never silently repaired."""
    commit = ts.COMMIT if family == "temporal_straightening" else LEWM_COMMIT
    hashes = TS_HASHES if family == "temporal_straightening" else LEWM_HASHES
    root = Path(cache) / family / commit
    source, provenance = {}, {}
    for name, expected in hashes.items():
        relative = (f"planning/{name}" if name == "objectives.py" else f"models/{name}") if family == "temporal_straightening" else name
        url = f"https://raw.githubusercontent.com/{REPOSITORIES[family]}/{commit}/{relative}"
        path = root / name
        if path.exists():
            data = path.read_bytes()
        elif not allow_download:
            raise ReferenceUnavailable(f"Offline reference source missing: {path}")
        else:
            try:
                with urlopen(url, timeout=30) as response:
                    data = response.read(512_001)
            except OSError as error:
                raise ReferenceUnavailable(f"Could not retrieve {url}: {error}") from error
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned upstream hash mismatch: {path}; expected {expected}, got {actual}")
        if not path.exists():
            root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        source[name] = data.decode("utf-8")
        provenance[name] = {"url": url, "sha256": actual, "cache_path": str(path)}
    return source, {"repository": REPOSITORIES[family], "commit": commit, "files": provenance}


def _extract(source, name, methods=None):
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == name)
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        if {n.name for n in node.body} != set(methods):
            raise ValueError(f"Missing methods in pinned {name}")
    node.decorator_list = []
    return ast.Module(body=[node], type_ignores=[])


def _status(checks):
    return "PASS" if all(x["pass"] for x in checks.values()) else "MISMATCH"


def _optimizer_checks(groups, grad_clip, checks):
    """One fresh matched optimizer step; not the upstream trainer or scheduler."""
    pairs = [pair for _, group in groups for pair in group]
    for side in (0, 1):
        nn.utils.clip_grad_norm_([pair[side] for pair in pairs], grad_clip, error_if_nonfinite=True)
    for index, (original, group) in enumerate(groups):
        checks[f"parameter_gradient_group_{index}"] = ts.comparison(
            torch.cat([a.grad.flatten() for a, _ in group]),
            torch.cat([b.grad.flatten() for _, b in group]), atol=5e-5, rtol=5e-4)
        kwargs = {k: v for k, v in original.defaults.items() if k in inspect.signature(type(original)).parameters}
        for side in (0, 1):
            optimizer = type(original)([pair[side] for pair in group], **kwargs)
            optimizer.step()
        checks[f"optimizer_step_group_{index}"] = ts.comparison(
            torch.cat([a.detach().flatten() for a, _ in group]),
            torch.cat([b.detach().flatten() for _, b in group]))


def ts_aggregation_parity(model, source, latent, actions):
    """Execute actual upstream agg(), total_curvature(), forward() on matched modules."""
    local = copy.deepcopy(model).eval()
    local.decoder = None
    local.use_amp = False
    reference = ts.reference_model(local, source)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(_extract(source["dino.py"], "DinoV2Encoder", {"agg"}), "pinned/dino.py", "exec"), namespace)
    reference.encoder = namespace["DinoV2Encoder"]()
    reference.encoder.agg_type = "mlp"
    reference.encoder.emb_dim = latent.shape[-1]
    reference.encoder.agg_out_dim = local.encoder.agg_post_norm.normalized_shape[0]
    reference.encoder.agg_mlp_hidden_dim = local.encoder.agg_mlp[0].out_features
    # Execute the real upstream head-construction block without initializing DINO.
    # Its single hardcoded patch count is the sole architecture adaptation.
    encoder_ast = next(n for n in ast.parse(source["dino.py"]).body if getattr(n, "name", "") == "DinoV2Encoder")
    constructor = next(n for n in encoder_ast.body if getattr(n, "name", "") == "__init__")
    block = constructor.body[-1]
    if not isinstance(block, ast.If) or ast.unparse(block.test) != "self.agg_type == 'mlp'":
        raise ValueError("Unexpected pinned upstream aggregation constructor")
    patches = [n for n in ast.walk(block) if isinstance(n, ast.Constant) and n.value == 196]
    if len(patches) != 1:
        raise ValueError("Unexpected upstream aggregation patch-count adaptation")
    patches[0].value = latent.shape[-2]
    head_namespace = {"self": reference.encoder, "nn": nn}
    exec(compile(ast.Module(body=block.body, type_ignores=[]), "pinned/dino-head-constructor.py", "exec"), head_namespace)
    reference.encoder.to(local.device)
    reference.encoder.load_state_dict({k: v for k, v in local.encoder.state_dict().items() if k.startswith("agg_")}, strict=True)
    reference.curvature_mode = "aggcos"
    reference.requires_grad_(True)
    groups = []
    for name in ("predictor", "action_encoder"):
        ref_params = dict(getattr(reference, name).named_parameters())
        pairs = []
        for key, parameter in getattr(local, name).named_parameters():
            mapped = ts.predictor_key(key) if name == "predictor" else key.replace("net.0.weight", "patch_embed.weight").replace("net.0.bias", "patch_embed.bias").replace("net.1.", "norm.")
            pairs.append((parameter, ref_params[mapped]))
        groups.append((local.optimizers[name], pairs))
    groups.append((local.optimizers["encoder"], [
        (parameter, dict(reference.encoder.named_parameters())[name])
        for name, parameter in local.encoder.named_parameters() if name.startswith("agg_")]))
    left, right = [latent.detach().clone().requires_grad_() for _ in range(2)]
    with ts.full_precision(), torch.enable_grad():
        loss, metrics = local.representation_loss({}, left, actions)
        padded = torch.cat((actions, torch.zeros_like(actions[:, :1])), 1)
        _, _, _, ref_loss, ref_metrics = reference({"visual": right}, padded)
        ref_loss = ref_loss + (local.prediction_weight - 1) * ref_metrics["z_loss"]
        shape = latent.shape
        checks = {"aggregation_output": ts.comparison(local.encoder.agg(left.reshape(-1, *shape[-2:])), reference.encoder.agg(right.reshape(-1, *shape[-2:]))),
                  "curvature": ts.comparison(metrics["curvature_loss"], reference.total_curvature(right, mode="aggcos")),
                  "total_loss": ts.comparison(loss, ref_loss)}
        loss.backward()
        ref_loss.backward()
        checks["cached_feature_gradient"] = ts.comparison(left.grad, right.grad)
        _optimizer_checks(groups, local.grad_clip, checks)
    return {"status": _status(checks), "checks": checks,
            "head_constructor": {"source": "DinoV2Encoder.__init__ final agg_type=mlp block", "upstream_patches": 196,
                                 "adapted_patches": latent.shape[-2], "channels": latent.shape[-1],
                                 "hidden_dim": reference.encoder.agg_mlp_hidden_dim, "output_dim": reference.encoder.agg_out_dim},
            "scope": "Actual pinned upstream MLP aggregation method, aggregate curvature and visual loss; cached feature, predictor, action and aggregation-head gradients; one fresh matched optimizer step.",
            "adaptations": ["Head input shape follows local patch count, instead of upstream hardcoded 196 patches; matching Linear/ReLU/LayerNorm modules and copied weights.",
                            "Visual prediction coefficient follows supplied local configuration.",
                            "Moving random features only; upstream empty-motion curvature is NaN, local implementation deliberately returns zero."],
            "not_tested": ["Vision backbone", "Full upstream trainer/scheduler", "Dropout masks", "Original task performance"]}


def _lewm_key(key):
    if key == "position":
        return "pos_embedding"
    if key.startswith("norm."):
        return "transformer." + key
    _, index, part, suffix = key.split(".", 3)
    part = {"attention": "attn", "feed_forward": "mlp", "modulation": "adaLN_modulation"}.get(part, part)
    return f"transformer.layers.{index}.{part}.{suffix}"


def lewm_parity(model, source):
    """Actual official predictor, JEPA rollout, SIGReg, loss, backward and matched step."""
    modules, jepa, train = {}, {}, {"torch": torch}
    exec(compile(source["module.py"], "pinned/lewm/module.py", "exec"), modules)
    exec(compile(source["jepa.py"], "pinned/lewm/jepa.py", "exec"), jepa)
    exec(compile(_extract(source["train.py"], "lejepa_forward"), "pinned/lewm/train.py", "exec"), train)
    local = copy.deepcopy(model).eval()
    local.use_amp = False
    # Native AdaLN-zero initialization makes action gradients identically zero.
    # Probe a shared nonzero parameter state so parity cannot pass vacuously.
    with torch.no_grad():
        for block in local.predictor.blocks:
            block.modulation[-1].weight.normal_(std=.03)
            block.modulation[-1].bias.normal_(std=.03)
    predictor = local.predictor
    first = predictor.blocks[0]
    dim = predictor.position.shape[-1]
    ref_predictor = modules["ARPredictor"](
        num_frames=local.history_size, depth=len(predictor.blocks), heads=first.attention.heads,
        dim_head=first.attention.dim_head, mlp_dim=first.feed_forward.net[1].out_features,
        input_dim=dim, hidden_dim=dim, output_dim=dim, dropout=0, emb_dropout=0).to(local.device)
    ref_predictor.load_state_dict({_lewm_key(k): v for k, v in predictor.state_dict().items()}, strict=True)
    action = modules["Embedder"](input_dim=local.action_dim, smoothed_dim=local.action_dim, emb_dim=dim).to(local.device)
    action.load_state_dict({"patch_embed.weight": local.action_encoder.net[0].weight[..., None],
                            "patch_embed.bias": local.action_encoder.net[0].bias,
                            **{f"embed.{i}.{k}": v for i, li in ((0, 1), (2, 3)) for k, v in local.action_encoder.net[li].state_dict().items()}}, strict=True)
    def projector(original):
        result = modules["MLP"](dim, original.net[0].out_features, dim, norm_fn=nn.BatchNorm1d).to(local.device)
        result.load_state_dict(original.state_dict(), strict=True)
        return result
    class CachedVision(nn.Module):
        def forward(self, value, **kwargs):
            return SimpleNamespace(last_hidden_state=value[:, None])
    reference = jepa["JEPA"](CachedVision(), ref_predictor, action, projector(local.projector), projector(local.pred_projector)).eval()
    ref_sigreg = modules["SIGReg"](knots=local.sigreg.t.numel(), num_proj=local.sigreg.projections).to(local.device)
    shape = (4, local.history_size + 1, dim)
    left = torch.randn(shape, device=local.device, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    controls = torch.randn(4, local.history_size, local.action_dim, device=local.device)
    checks = {}
    with ts.full_precision(), torch.enable_grad():
        latent = local.projector(left)
        ref_encoded = reference.encode({"pixels": right, "action": controls})
        checks["projector_output"] = ts.comparison(latent, ref_encoded["emb"])
        checks["action_embedding"] = ts.comparison(local.action_encoder(controls), ref_encoded["act_emb"])
        checks["teacher_forced_prediction"] = ts.comparison(local.predict(latent[:, :-1], controls), reference.predict(ref_encoded["emb"][:, :-1], ref_encoded["act_emb"]))
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(local.device) if local.device.type == "cuda" else None
        loss, metrics = local.representation_loss({}, latent, controls)
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, local.device)
        cfg = SimpleNamespace(history_size=local.history_size, num_preds=1, loss=SimpleNamespace(sigreg=SimpleNamespace(weight=local.sigreg_weight)))
        owner = SimpleNamespace(model=reference, sigreg=ref_sigreg, log_dict=lambda *args, **kwargs: None)
        ref_loss = train["lejepa_forward"](owner, {"pixels": right, "action": controls}, "train", cfg)
        checks["prediction_loss"] = ts.comparison(metrics["prediction_loss"], ref_loss["pred_loss"])
        checks["sigreg_loss"] = ts.comparison(metrics["sigreg_loss"], ref_loss["sigreg_loss"])
        checks["total_loss"] = ts.comparison(loss, ref_loss["loss"])
        loss.backward()
        ref_loss["loss"].backward()
        checks["cached_feature_gradient"] = ts.comparison(left.grad, right.grad)
        groups = []
        pairs = []
        for name, ref_name in (("predictor", "predictor"), ("action_encoder", "action_encoder"), ("projector", "projector"), ("pred_projector", "pred_proj")):
            ref_params = dict(getattr(reference, ref_name).named_parameters())
            for key, parameter in getattr(local, name).named_parameters():
                if name == "predictor":
                    mapped = _lewm_key(key)
                elif name == "action_encoder":
                    mapped = key.replace("net.0.", "patch_embed.").replace("net.1.", "embed.0.").replace("net.3.", "embed.2.")
                else:
                    mapped = key
                pairs.append((parameter, ref_params[mapped]))
        _optimizer_checks([(local.optimizers["model"], pairs)], local.grad_clip, checks)
        # Use projected cached latents for both autoregressive planners; the upstream
        # rollout implementation itself is unchanged, only its input encoder is adapted.
        reference.encode = lambda info: {**info, "emb": info["pixels"], "act_emb": reference.action_encoder(info["action"])}
        history = latent[:1, :local.history_size].detach()
        past = controls[:1, :local.history_size - 1].detach()
        candidates = torch.randn(1, 2, 5, local.action_dim, device=local.device, requires_grad=True)
        all_actions = torch.cat((past[:, None].expand(-1, 2, -1, -1), candidates), 2)
        predicted = local.rollout(history, past, candidates)
        expected = reference.rollout({"pixels": history[:, None].expand(-1, 2, -1, -1)}, all_actions, history_size=local.history_size)["predicted_emb"][:, :, local.history_size:]
        checks["autoregressive_rollout"] = ts.comparison(predicted, expected)
        direction = torch.randn_like(predicted)
        grad = torch.autograd.grad((predicted * direction).sum(), candidates, retain_graph=True)[0]
        ref_grad = torch.autograd.grad((expected * direction).sum(), candidates, retain_graph=True)[0]
        checks["rollout_action_gradient"] = ts.comparison(grad, ref_grad, atol=2e-4, rtol=2e-3)
        checks["nonzero_action_gradient"] = {"pass": bool(grad.abs().max() > 1e-7), "max_abs": float(grad.abs().max())}
        from models.shared.latent_goal import latent_goal_cost
        goal = history[:, -1]
        cost = latent_goal_cost(predicted, goal, reduction="sum", mode="last")
        expected_cost = reference.criterion({"predicted_emb": expected, "goal_emb": goal[:, None, None]})
        checks["terminal_cost"] = ts.comparison(cost, expected_cost)
        checks["terminal_cost_action_gradient"] = ts.comparison(torch.autograd.grad(cost.sum(), candidates)[0], torch.autograd.grad(expected_cost.sum(), candidates)[0], atol=2e-4, rtol=2e-3)
    return {"status": _status(checks), "checks": checks,
            "scope": "Actual pinned upstream action encoder, conditional predictor, projector, SIGReg, training loss and JEPA rollout/terminal cost; cached-feature/parameter/action gradients; one fresh matched AdamW step.",
            "adaptations": ["Matched synthetic features and copied local weights/dimensions; image backbone replaced by an identity feature adapter.",
                            "AdaLN modulation weights set to common nonzero random values to exercise action conditioning.",
                            "FP32 eval mode, dropout disabled, BatchNorm running statistics copied, random SIGReg projections matched.",
                            "Fresh optimizer state and supplied local hyperparameters; full trainer and scheduler are not reproduced."],
            "not_tested": ["Vision encoder equivalence", "BatchNorm train-mode dynamics", "Dataset/action preprocessing", "Original task performance", "Mixed precision"]}


def run_reference_checks(configs, args=None, *, cache=None, allow_download=True):
    """Serializable component gate. Missing files fail closed as UNAVAILABLE.

    `configs` maps labels to resolved model configurations; labels may include both
    TS variants. Optional args fields: reference_cache, device, seed, reference_offline.
    No supplied model/checkpoint/configuration is mutated.
    """
    from omegaconf import OmegaConf
    from training import load_model_family
    args = args or SimpleNamespace()
    cache = Path(cache or getattr(args, "reference_cache", "local/reference_sources"))
    allow_download = allow_download and not getattr(args, "reference_offline", False) and not getattr(args, "no_reference_download", False)
    entries = configs.items() if hasattr(configs, "items") else ((str(c.model_family), c) for c in configs)
    report = {"schema_version": 1, "status": "UNAVAILABLE", "scope": "SYNTHETIC_COMPONENT_PARITY",
              "original_task": {"status": "NOT_RUN", "reason": "Requires separately installed official runtime, released checkpoint and original expert dataset; component parity is not task reproduction."},
              "versions": {"torch": torch.__version__}, "models": {}}
    completed = {}
    for label, configuration in entries:
        family = str(configuration.model_family)
        if family not in REPOSITORIES:
            report["models"][label] = {"status": "UNAVAILABLE", "reason": f"Unsupported family: {family}"}
            continue
        result = {"status": "UNAVAILABLE"}
        report["models"][label] = result
        try:
            config = copy.deepcopy(configuration)
            config.device = str(getattr(args, "device", config.device))
            config.jepa_model.use_amp = False
            result["configuration"] = OmegaConf.to_container(config, resolve=True)
            key = json.dumps({"family": family, "settings": result["configuration"]["jepa_model"],
                              "model_io": result["configuration"]["model_io"], "device": config.device}, sort_keys=True)
            if key in completed:
                previous = completed[key]
                result.update({k: copy.deepcopy(v) for k, v in report["models"][previous].items() if k != "configuration"})
                result["reused_component_check"] = previous
                continue
            source, provenance = pinned_sources(family, cache, allow_download)
            result["provenance"] = provenance
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if config.device.startswith("cuda") else []), ts.full_precision():
                torch.manual_seed(int(getattr(args, "seed", 17)))
                model = load_model_family(family).build_model(config).to(config.device).eval()
                if family == "leworldmodel":
                    result["components"] = {"native": lewm_parity(model, source)}
                else:
                    latent = torch.randn(4, model.history_size + 1, model.encoder.num_tokens, model.encoder.out_dim, device=model.device)
                    controls = torch.randn(4, model.history_size, model.action_dim, device=model.device)
                    actions = torch.randn(1, 1, 5, model.action_dim, device=model.device)
                    result["components"] = {"rollout": ts.parity(model, source, latent[:1, :-1], controls[:1, :-1], actions)}
                    function = ts_aggregation_parity if model.curvature_mode == "agg" else ts.training_parity
                    result["components"]["loss_and_step"] = function(model, source, latent, controls)
                result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["components"].values()) else "MISMATCH"
                completed[key] = label
                del model
        except ReferenceUnavailable as error:
            result.update(status="UNAVAILABLE", reason=str(error))
        except Exception as error:
            result.update(status="MISMATCH", reason=f"{type(error).__name__}: {error}")
    statuses = [r["status"] for r in report["models"].values()]
    report["status"] = "MISMATCH" if "MISMATCH" in statuses else "PASS" if statuses and all(s == "PASS" for s in statuses) else "UNAVAILABLE"
    paths = [Path(__file__), Path(ts.__file__), Path("models/leworldmodel/model.py"), Path("models/temporal_straightening/model.py"), Path("models/planning.py"), Path("models/shared/transformer.py"), Path("models/shared/latent_goal.py")]
    report["local_source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return report


def main():
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="Resolved YAML model configs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, default=Path("local/reference_sources"))
    parser.add_argument("--reference-offline", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = run_reference_checks({Path(p).stem: OmegaConf.load(p) for p in args.configs}, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "scope": report["scope"]}))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
