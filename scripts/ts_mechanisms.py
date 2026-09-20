"""Diagnostic-only TS adaptation controls, crossed readouts, and simulator action probes."""

import copy
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

import tools
from models.shared.physical_state import readout_mode
from scripts.diagnose_fixed_replay import MODES, batchnorm_state, representation_state
from scripts.diagnose_fresh_readout import physical_metrics, tensor_digest
from scripts.diagnose_planner_oracle import collect_cases, encode_images, preserved_native_state
from scripts.train_planner_check import checked_update, measure_errors, new_model
from scripts.train_ts_ablation import physical_checks, sampler_digest
from training import load_model_family
from training.progress import Progress, duration


@torch.no_grad()
def observed_bank(model, sources, settings):
    """Only real observation histories are decoded here, never recursive predictions."""
    bank = {}
    context = settings.context_length
    with readout_mode(model):
        for name, (dataset, windows) in sources.items():
            features, truths = [], []
            for start in range(0, len(windows), settings.batch_size):
                obs, _, labels = dataset.read_batch(windows[start:start + settings.batch_size], context + 1)
                # TS encodes frames independently in eval mode; retain complete head histories.
                frames = {key: value[:, context + 1 - model.state_head.history:].to(model.device)
                          for key, value in obs.items()}
                features.append(model.encode(frames))
                truths.append(labels[:, context].to(model.device))
            bank[name] = {"features": torch.cat(features), "truth": torch.cat(truths)}
    return bank


@torch.no_grad()
def crossed_readouts(model, old_head, old_bank, old_bn, sources, settings):
    current = observed_bank(model, sources, settings)
    with preserved_native_state(model):
        buffers = dict(model.named_buffers())
        for name, value in old_bn.items():
            buffers[name].copy_(value)
        old_statistics = observed_bank(model, sources, settings)
    result = {}
    for name, bank in current.items():
        truth = bank["truth"]
        torch.testing.assert_close(truth, old_bank[name]["truth"], rtol=0, atol=0)
        pairs = {
            "E0_D0": (old_bank[name]["features"], old_head),
            "Et_D0": (bank["features"], old_head),
            "E0_Dt": (old_bank[name]["features"], model.state_head),
            "Et_Dt": (bank["features"], model.state_head),
            "Et_BN0_Dt": (old_statistics[name]["features"], model.state_head),
        }
        result[name] = {key: physical_metrics(head(features)[:, 0], truth, head,
                                              model.goal_tolerance, model.goal_geometry)
                        for key, (features, head) in pairs.items()}
        result[name]["feature_drift_rms"] = (bank["features"] - old_bank[name]["features"]).square().mean().sqrt().item()
        result[name]["feature_drift_with_old_bn_rms"] = (
            old_statistics[name]["features"] - old_bank[name]["features"]
        ).square().mean().sqrt().item()
    return result


def response_metrics(predicted, actual):
    """Compare opposing controls from the same state; no cross-episode counterfactuals."""
    # collect_cases uses candidates 0=zero, 1=-1, 2=+1 for Cartpole.
    delta, target = predicted[2] - predicted[1], actual[2] - actual[1]
    norm, target_norm = delta.norm(), target.norm()
    return {
        "predicted_delta_rms": delta.square().mean().sqrt().item(),
        "actual_encoded_delta_rms": target.square().mean().sqrt().item(),
        "response_ratio": float(norm / target_norm) if target_norm > 1e-8 else None,
        "response_cosine": float((delta * target).sum() / (norm * target_norm))
        if min(norm, target_norm) > 1e-8 else None,
        "matched_action_rmse": (predicted - actual).square().mean().sqrt().item(),
        "wrong_action_rmse": (predicted.roll(1, dims=0) - actual).square().mean().sqrt().item(),
    }


def action_case(model, case):
    """Frozen eval-mode graph, including input/loss gradients; no .backward or optimizer step."""
    actions = case["action"].to(model.device)
    if len(actions) != 3 or actions.shape[-1] != 1:
        raise ValueError("Action mechanism probes require zero/-1/+1 Cartpole candidates.")
    with readout_mode(model):
        prefix = model.encode({"image": case["prefix"][None].to(model.device)})
        past = case["past_action"][None].to(model.device)
        actual = encode_images(model, case["image"].flatten(0, 1), 32).reshape(
            len(actions), actions.shape[1], *prefix.shape[2:])
        predicted = model.rollout(prefix, past, actions[None])[0]
        result = {str(h): response_metrics(predicted[:, h - 1], actual[:, h - 1])
                  for h in sorted({1, actions.shape[1]})}
        with torch.enable_grad():
            controls = torch.cat((past.expand(len(actions), -1, -1), actions[:, :1]), dim=1).detach().requires_grad_()
            prediction = model.predict(prefix.expand(len(actions), *prefix.shape[1:]), controls)[:, -1]
            loss = (prediction - actual[:, 0]).square().mean()
            parameters = list(model.action_encoder.parameters())
            gradients = (torch.autograd.grad(loss, [controls, *parameters], allow_unused=True, retain_graph=True)
                         if loss.requires_grad else [None] * (1 + len(parameters)))
            # A deterministic projection tests connectivity even if the loss gradient cancels.
            direction = torch.randn(prediction.shape, generator=torch.Generator().manual_seed(872)).to(model.device)
            projection = (prediction * direction).sum() / math.sqrt(prediction.numel())
            input_gradient = (torch.autograd.grad(projection, controls, allow_unused=True)[0]
                              if projection.requires_grad else None)
            finite = torch.isfinite(loss) and all(g is None or torch.isfinite(g).all()
                                                for g in [*gradients, input_gradient])
            if not finite:
                raise ValueError("Non-finite action-probe gradients.")
            result["gradients"] = {
                "input_connected": input_gradient is not None,
                "current_action_jacobian_projection_norm": float(input_gradient[:, -1].norm())
                if input_gradient is not None else 0.,
                "prediction_loss": float(loss.detach()),
                "current_action_loss_gradient_norm": float(gradients[0][:, -1].norm())
                if gradients[0] is not None else 0.,
                "action_encoder_parameters_with_gradient": sum(g is not None for g in gradients[1:]),
                "action_encoder_parameter_tensors": len(parameters),
                "action_encoder_loss_gradient_norm": math.sqrt(sum(float(g.double().square().sum())
                                                                    for g in gradients[1:] if g is not None)),
            }
    return result


@tools.preserve_rng_state
def action_probes(model, cases):
    original_amp = model.use_amp
    result = {}
    try:
        # On CUDA this distinguishes genuine insensitivity from BF16 quantization.
        for precision in (("native_bf16", "fp32") if model.device.type == "cuda" and original_amp else ("fp32",)):
            model.use_amp = precision != "fp32"
            result[precision] = [{"id": case["id"], **action_case(model, case)} for case in cases]
    finally:
        model.use_amp = original_amp
    return result


def case_metadata(cases):
    result = []
    for case in cases:
        delta = case["relation"][2, -1] - case["relation"][1, -1]
        delta[1] = torch.atan2(delta[1].sin(), delta[1].cos())
        result.append({
            "id": case["id"], "seed": case["seed"], "rollin_policy": case["rollin_policy"],
            "anchor_agent_step": case["anchor_agent_step"], "action": case["action"].tolist(),
            "true_goal_relations": case["relation"].tolist(),
            "terminal_physical_delta_plus_minus": delta.tolist(),
            "terminal_pixel_delta_rms_0_255": (case["image"][2, -1].float() - case["image"][1, -1].float()).square().mean().sqrt().item(),
            "sha256": tensor_digest({key: case[key] for key in ("prefix", "past_action", "action", "image", "relation")}),
        })
    return result


@torch.no_grad()
def parameter_changes(model, reference):
    result = {}
    for name in ("encoder", "predictor", "action_encoder"):
        values = [(value - reference[f"{name}.{key}"]).square().sum()
                  for key, value in getattr(model, name).named_parameters()]
        count = sum(value.numel() for value in getattr(model, name).parameters())
        result[name] = float((torch.stack(values).sum() / count).sqrt())
    return result


def snapshot(model, old_head, old_bank, old_bn, sources, settings, cases):
    return {"scores": measure_errors(model, sources, settings),
            "cross_readout": crossed_readouts(model, old_head, old_bank, old_bn, sources, settings),
            "action_probes": action_probes(model, cases)}


def compact_rmse(metric):
    values = metric["physical_rmse"]
    return f"{values['position[0]']:.4g}/{values['pole_angle']:.4g}"


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["TS mechanisms | one shared offline fit | identical starting weights/moments and replay per arm",
             "Physical pairs = cart position RMSE (m) / pole angle RMSE (rad); no nMSE guards."]
    baseline = report.get("baseline")
    if baseline:
        for cohort, scores in baseline["scores"].items():
            lines.append(f"BASELINE | {cohort} observed={compact_rmse(scores['all']['physical']['observed']['1'])}")
    for run in report["runs"]:
        lines.append(f"{run['status']} | {run['mode']} | updates={run.get('updates', 0)} | {duration(run.get('seconds', 0))}")
        if run["status"] in {"RUNNING", "FAIL"}:
            if "error" in run:
                lines.append(f"  {run['error']}")
            continue
        lines.append(f"  clipped={100 * run['clipped_fraction']:.1f}% | weight change RMS "
                     + ", ".join(f"{key}={value:.3g}" for key, value in run["parameter_delta_rms"].items()))
        last = run["snapshots"][-1]
        for cohort, scores in last["scores"].items():
            physical = scores["all"]["physical"]
            cross = last["cross_readout"][cohort]
            lines.append(f"  {cohort} observed={compact_rmse(physical['observed']['1'])} "
                         f"forecast h5={compact_rmse(physical['forecast']['5'])} h100={compact_rmse(physical['forecast']['100'])}")
            lines.append("  " + cohort + " " + " | ".join(f"{key}={compact_rmse(cross[key])}"
                         for key in ("E0_D0", "Et_D0", "E0_Dt", "Et_Dt", "Et_BN0_Dt")))
    if baseline:
        stages = [("offline", baseline), *[(run["mode"], run["snapshots"][-1]) for run in report["runs"]
                                            if run["status"] not in {"RUNNING", "FAIL"}]]
        lines.append("Actions | stage/precision | h5 predicted/actual latent response RMS | matched/wrong forecast RMSE | action-gradient norm")
        for name, stage in stages:
            for precision, cases in stage["action_probes"].items():
                mean = lambda key: sum(case["5"][key] for case in cases) / len(cases)
                grad = sum(case["gradients"]["current_action_jacobian_projection_norm"] for case in cases) / len(cases)
                lines.append(f"  {name}/{precision} | {mean('predicted_delta_rms'):.3g}/{mean('actual_encoded_delta_rms'):.3g} | "
                             f"{mean('matched_action_rmse'):.3g}/{mean('wrong_action_rmse'):.3g} | {grad:.3g}")
    lines += ["E0/Et = offline/current encoder; D0/Dt = offline/current readout; BN0 restores ONLY offline running statistics.",
              "Frozen BN also uses eval statistics during training; only the BN0 cross-check isolates buffers at fixed weights.",
              "Crossed heads test compatibility, not information loss by themselves. Freezing controls are diagnostic, not recommended recipes.",
              "REGRESSION: an intermediate/final coordinate RMSE exceeded 3 * max(offline RMSE, 0.01 original units).",
              "NO_REGRESSION is only this guard, not proof of control or learning quality. Action probes have no universal pass threshold.",
              "No planner optimization, policy evaluation, validation fitting, or checkpoint writes. Fixed replay is not own-policy learning."]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def pretrain_shared(config, dataset, args, report):
    """One fresh offline fit, reused by the diagnostic controls without disk checkpoints."""
    model = new_model(config, dataset)
    started = time.monotonic()
    progress = Progress("TS shared offline", args.expert_updates)
    with (args.output / "offline_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for step in range(1, args.expert_updates + 1):
            torch.manual_seed(args.seed + step)
            values = checked_update(model, lambda: model.update(dataset.sample_episode_batch()))
            log.write(json.dumps({"phase": "offline", "update": step, **values}, allow_nan=False) + "\n")
            progress.update(step, f"prediction={values['prediction_loss']:.3g} state={values['state/loss']:.3g}",
                            force=step == args.expert_updates)
    report["shared_offline"] = {"updates": args.expert_updates, "seconds": time.monotonic() - started,
                                "state_sha256": tensor_digest(model.state_dict()),
                                "parameters": sum(p.numel() for p in model.parameters())}
    return model


def train_controls(config, dataset, sampler_start, replay, plans, sources, settings, cases, args, report):
    """Pretrain once; branch every control from the complete same in-memory training state."""
    family = load_model_family("temporal_straightening")
    if len(plans) != args.online_updates:
        raise ValueError("Replay plans must match the declared adaptation budget.")
    dataset.load_state_dict(copy.deepcopy(sampler_start))
    model = pretrain_shared(config, dataset, args, report)
    shared = copy.deepcopy(family.checkpoint(model))
    post_offline_sampler = copy.deepcopy(dataset.state_dict())
    old_head = copy.deepcopy(model.state_head).eval().requires_grad_(False)
    old_bank = observed_bank(model, sources, settings)
    old_bn = {key: value.clone() for key, value in batchnorm_state(model).items()}
    report["baseline"] = snapshot(model, old_head, old_bank, old_bn, sources, settings, cases)
    write_report(args.output, report)
    for mode in MODES:
        started = time.monotonic()
        model.set_adaptation_mode("native")
        family.load_checkpoint(model, copy.deepcopy(shared), training=True)
        # Check this contract once per branch, never on the hot update path.
        torch.testing.assert_close(model.optimizer_state_dict(), shared["optimizer_state_dict"], rtol=0, atol=0)
        start_digest = tensor_digest(model.state_dict())
        if start_digest != report["shared_offline"]["state_sha256"]:
            raise RuntimeError("A control did not restore the complete offline state.")
        dataset.load_state_dict(copy.deepcopy(post_offline_sampler))
        model.set_adaptation_mode(mode)
        model._gradient_updates = model._clipped_updates = 0
        expert = {"batch": None}
        model.state_head.configure_online(lambda: model.readout_features(expert["batch"]))
        bn_digest, encoder_digest = tensor_digest(batchnorm_state(model)), tensor_digest(representation_state(model))
        run = {"mode": mode, "status": "RUNNING", "updates": 0, "snapshots": [],
               "initial_state_sha256": start_digest, "native_optimizer_moments": "restored from shared offline fit",
               "head_optimizer": "fresh online moments; same weights and warmup in every arm"}
        report["runs"].append(run)
        output = args.output / mode
        output.mkdir()
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        norms, clipped = [], 0
        try:
            with (output / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                progress = Progress(f"TS {mode} adaptation", args.online_updates)
                for step, plan in enumerate(plans, 1):
                    torch.manual_seed(args.seed + 1_000_000 + step)
                    batch = replay.batch(plan)
                    expert["batch"] = dataset.sample_episode_batch()
                    values = checked_update(model, lambda: model.update(batch))
                    run["updates"] = step
                    norms.append(values["grad_norm"])
                    clipped += int(values["grad_clipped"])
                    log.write(json.dumps({"phase": "adaptation", "update": step, **values}, allow_nan=False) + "\n")
                    progress.update(step, f"prediction={values['prediction_loss']:.3g} state={values['state/loss']:.3g}",
                                    force=step == args.online_updates)
                    if step in {max(1, args.online_updates // 4), args.online_updates}:
                        measured = snapshot(model, old_head, old_bank, old_bn, sources, settings, cases)
                        measured.update(updates=step, guards=physical_checks(report["baseline"]["scores"], measured["scores"]))
                        run["snapshots"].append(measured)
                        write_report(args.output, report)
            bn_same = bn_digest == tensor_digest(batchnorm_state(model))
            encoder_same = encoder_digest == tensor_digest(representation_state(model))
            if mode != "native" and not bn_same or mode == "frozen_encoder" and not encoder_same:
                raise RuntimeError("Frozen encoder parameters or BatchNorm statistics changed.")
            run.update(batchnorm_unchanged=bn_same, encoder_unchanged=encoder_same,
                       grad_norm_mean=sum(norms) / len(norms), grad_norm_max=max(norms),
                       clipped_fraction=clipped / len(norms), sampler_end_sha256=sampler_digest(dataset),
                       parameter_delta_rms=parameter_changes(model, shared["model_state_dict"]))
            regression = any(not coordinate["passed"] for snap in run["snapshots"]
                             for check in snap["guards"].values() for coordinate in check.values())
            run["status"] = "REGRESSION" if regression else "NO_REGRESSION"
            if model.device.type == "cuda":
                run["gpu_reserved_peak_gib"] = torch.cuda.max_memory_reserved(model.device) / 1024**3
        except Exception as error:
            run.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            model.state_head._expert_source = None
        run["seconds"] = time.monotonic() - started
        write_report(args.output, report)
    complete = [run for run in report["runs"] if run["status"] != "FAIL"]
    if len({run["sampler_end_sha256"] for run in complete}) > 1:
        raise RuntimeError("Expert sampling differed between controls.")


def run_mechanisms(config, dataset, sampler_start, replay, plans, sources, settings, args, report, started):
    probe_settings = SimpleNamespace(sim_seeds=[10_000_000 + args.seed], rollin_steps=[0, 64], candidates=3)
    print("Probe | collecting four fixed simulator anchors with zero/-1/+1 controls; no planning", flush=True)
    cases = collect_cases(config, probe_settings)
    report.update(experiment="ts_mechanisms", mechanisms_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  action_cases=case_metadata(cases))
    print(f"Train | shared offline={args.expert_updates} | each of 3 controls: replay updates={args.online_updates}", flush=True)
    train_controls(config, dataset, sampler_start, replay, plans, sources, settings, cases, args, report)
    report["elapsed_seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(run["status"] != "NO_REGRESSION" for run in report["runs"]))
