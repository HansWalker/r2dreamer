"""Matched predictor-only test of short autoregressive supervision, not a production recipe."""

import argparse
import copy
import hashlib
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_planner_oracle import (
    collect_cases,
    rank_correlation,
    ranking_summary,
)
from scripts.smoke_tiny_planners import FAMILIES, native_control
from scripts.train_planner_check import build_config, checked_update, new_model
from scripts.ts_fit_isolation import branch_bank, finite
from scripts.ts_mechanisms import case_metadata, response_metrics
from training import load_model_family
from training.progress import Progress, duration
from training.protocol import implementation_sha256


def cache_branches(model, cases):
    bank = branch_bank(model, cases)
    bank.update(prefix=torch.cat([c["prefix"] for c in bank["cases"]]),
                past_action=torch.cat([c["past_action"] for c in bank["cases"]]),
                future_action=torch.stack([c["action"] for c in bank["cases"]]),
                target=torch.stack([c["future"] for c in bank["cases"]]),
                reward=torch.stack([c["reward"] for c in cases]),
                relation=torch.stack([c["relation"] for c in cases]))
    with readout_mode(model):
        bank["goal"] = model.encode({"image": torch.stack([c["goal_image"] for c in cases])[:, None].to(model.device)})[:, 0]
    return bank


def fitting_loss(model, bank, rollout_weight):
    """Keep the native regularizer and total prediction coefficient unchanged."""
    if not 0 <= rollout_weight <= 1 or not math.isfinite(rollout_weight):
        raise ValueError("Rollout weight must be between zero and one.")
    loss, metrics = model.representation_loss({}, bank["latent"], bank["action"])
    if rollout_weight:
        # The planner's own recursion: no future observations in the input and no
        # detach between predicted steps. Simulator successors are targets only.
        prediction = model.rollout(bank["prefix"], bank["past_action"], bank["future_action"])
        rollout_loss = F.mse_loss(prediction, bank["target"].detach())
        coefficient = getattr(model, "prediction_weight", 1.0)
        loss = loss + coefficient * rollout_weight * (rollout_loss - metrics["prediction_loss"])
        metrics = {**metrics, "rollout_loss": rollout_loss}
    return loss, metrics


@tools.preserve_rng_state
@torch.no_grad()
def score_bank(model, bank):
    with readout_mode(model):
        prediction = native_control(model, lambda: model.rollout(
            bank["prefix"], bank["past_action"], bank["future_action"]))
        teacher = model.predict(bank["latent"][:, :-1], bank["action"])
        if not torch.isfinite(prediction).all() or not torch.isfinite(teacher).all():
            raise ValueError("Non-finite validation forecasts.")
        error = prediction - bank["target"]
        rmse = error.square().flatten(3).mean((0, 1, 3)).sqrt().tolist()
        cases = []
        for index, case in enumerate(bank["cases"]):
            response = response_metrics(prediction[index, :, 0], bank["target"][index, :, 0])
            actual = bank["target"][index, :, 0]
            blind = (actual - actual.mean(0, keepdim=True)).square().mean().item()
            response["mse_over_blind_floor"] = response["matched_action_rmse"] ** 2 / blind if blind > 1e-12 else None
            reduce = torch.sum if model.goal_reduction == "sum" else torch.mean
            predicted_cost = reduce((prediction[index, :, -1] - bank["goal"][index]).square().flatten(1), dim=1).tolist()
            oracle_cost = reduce((bank["target"][index, :, -1] - bank["goal"][index]).square().flatten(1), dim=1).tolist()
            physical_cost = (bank["relation"][index, :, -1] / model.goal_tolerance.cpu()).square().sum(-1).tolist()
            returns = bank["reward"][index].sum(-1).tolist()
            cases.append({"id": case["id"], "h1_response": response,
                          "predicted_cost": predicted_cost, "oracle_cost": oracle_cost, "returns": returns,
                          "physical_goal_cost": physical_cost,
                          "oracle_vs_physical_goal_rank": rank_correlation(oracle_cost, physical_cost),
                          **ranking_summary(predicted_cost, oracle_cost, returns)})
        return {"teacher_forced_mse": F.mse_loss(teacher, bank["latent"][:, 1:]).item(),
                "rollout_rmse_by_horizon": rmse, "cases": cases}


def protected_state(model):
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith(("predictor.", "action_encoder.", "pred_projector."))}


def pretrain(config, dataset, args, output, result, *, model=None):
    if model is None:
        model = new_model(config, dataset)
    initial = tensor_digest(model.state_dict())
    if hasattr(model, "configure_pretraining"):
        model.configure_pretraining(args.expert_updates)
    started = time.monotonic()
    progress = Progress(f"{config.model_family} {result.get('trial', 'offline')}", args.expert_updates)
    with (output / "offline_metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for step in range(1, args.expert_updates + 1):
            torch.manual_seed(args.seed + step)
            values = checked_update(model, lambda: model.update(dataset.sample_episode_batch()))
            log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
            progress.update(step, f"prediction={values['prediction_loss']:.4g}", force=step == args.expert_updates)
    result["offline"] = {"updates": args.expert_updates, "seconds": time.monotonic() - started,
                         "initial_state_sha256": initial,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "state_sha256": tensor_digest(model.state_dict())}
    return model


def fit_arm(model, config, shared, banks, weight, args, output, result, persist):
    model.set_adaptation_mode("native")
    family = load_model_family(config.model_family)
    family.load_checkpoint(model, copy.deepcopy(shared), training=True)
    for name, optimizer in model.optimizers.items():
        torch.testing.assert_close(optimizer.state_dict(), shared["optimizer_state_dict"][name], rtol=0, atol=0)
    initial = tensor_digest(model.state_dict())
    if initial != tensor_digest(shared["model_state_dict"]):
        raise RuntimeError("Arms did not start from identical offline weights.")
    if hasattr(model, "configure_online"):
        model.configure_online(args.fit_updates, resumed=False)
    model.set_adaptation_mode("frozen_encoder")
    for parameter in model.parameters():
        parameter.grad = None
    protected = tensor_digest(protected_state(model))
    parameters = [p for module in (model.predictor, model.action_encoder, model.pred_projector) for p in module.parameters()]
    arm = {"name": "one_step" if not weight else "short_rollout", "status": "RUNNING",
           "rollout_weight": weight, "initial_state_sha256": initial, "snapshots": []}
    result["arms"].append(arm)
    folder = output / arm["name"]
    folder.mkdir()
    started = time.monotonic()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    progress = Progress(f"{config.model_family} {arm['name']}", args.fit_updates)
    with (folder / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for step in range(1, args.fit_updates + 1):
            torch.manual_seed(args.seed + 2_000_000 + step)
            model.train()
            loss, metrics = native_control(model, lambda: fitting_loss(model, banks["train"], weight))
            for optimizer in model.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, model.grad_clip, error_if_nonfinite=True)
            learning_rates = {f"lr/{name}": opt.param_groups[0]["lr"] for name, opt in model.optimizers.items()}
            for optimizer in model.optimizers.values():
                optimizer.step()
            if getattr(model, "scheduler", None) is not None:
                model.scheduler.step()
            values = finite({"loss": loss.detach(), **{k: v.detach() for k, v in metrics.items()},
                             "grad_norm": norm, "grad_clipped": norm > model.grad_clip, **learning_rates})
            log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
            progress.update(step, f"prediction={values['prediction_loss']:.4g}", force=step == args.fit_updates)
            if step in {max(1, args.fit_updates // 4), args.fit_updates}:
                scores = {split: score_bank(model, bank) for split, bank in banks.items()}
                arm["snapshots"].append({"updates": step, "scores": scores})
                persist()
    if protected != tensor_digest(protected_state(model)):
        raise RuntimeError("Predictor fitting changed the encoder, head, or their buffers.")
    arm.update(status="COMPLETE", updates=args.fit_updates, protected_state_unchanged=True,
               seconds=time.monotonic() - started)
    if model.device.type == "cuda":
        arm["gpu_reserved_peak_gib"] = torch.cuda.max_memory_reserved(model.device) / 1024**3
    persist()


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["Short-rollout training candidate | predictor-only adaptation | no checkpoint writes",
             "Model/arm/split | teacher MSE | rollout RMSE h1/h5 | rank pred/oracle,oracle/physical | reward cases | regret learned/oracle"]
    for result in report["runs"]:
        stages = [("offline", result.get("baseline", {}))]
        stages += [(arm["name"], arm["snapshots"][-1]["scores"]) for arm in result["arms"] if arm["snapshots"]]
        for name, stage in stages:
            for split, score in stage.items():
                cases = score["cases"]
                ranks = []
                for key in ("forecast_vs_oracle_cost_rank", "oracle_vs_physical_goal_rank"):
                    values = [c[key] for c in cases if c[key] is not None]
                    ranks.append(f"{sum(values) / len(values):.3f}" if values else "n/a")
                rank = "/".join(ranks)
                informative = [c for c in cases if c["return_informative"]]
                regrets = "/".join(f"{sum(c[key]['regret'] for c in informative) / len(informative):.3g}"
                                   for key in ("learned_selection", "oracle_latent_selection")) if informative else "n/a"
                rmse = score["rollout_rmse_by_horizon"]
                lines.append(f"{result['model']}/{name}/{split} | {score['teacher_forced_mse']:.4g} | "
                             f"{rmse[0]:.4g}/{rmse[-1]:.4g} | {rank} | {len(informative)}/{len(cases)} | {regrets}")
        lines.append(f"{result['status']} | {result['model']}" + (f" | {result['error']}" if "error" in result else ""))
    lines += ["Candidate: 50% native one-step + 50% five-step autoregressive prediction; native regularizers retained.",
              "Same initialization, optimizer moments, data, dropout and update count per arm; the candidate costs more compute.",
              "Encoder/projector, BatchNorm statistics and head fixed during adaptation; production defaults unchanged.",
              "No physical-head planning. The rollout term supervises full-prefix futures; native windows also supervise earlier steps.",
              "Train and validation simulator seeds are disjoint. Future frames supervise losses only, never rollout inputs.",
              "Ranks compare predicted/true-encoded goal cost, then true-encoded/physical distance; constants yield n/a.",
              "Physical distance uses simulator goal relation/tolerance for scoring only; no physical labels or rewards in fitting.",
              "Regret uses short branch rewards, NOT full policy episodes. Zero reward spread gives n/a.",
              "COMPLETE means execution, not repair. Judge held-out rollouts/action ranking, not training loss alone.",
              "This is a fixed-data predictor test, not own-policy online training or a full-run stability test."]
    if "seconds" in report:
        lines.append(f"Time | {duration(report['seconds'])}")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--expert-updates", type=int, default=1000)
    parser.add_argument("--fit-updates", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("rollout_training_check_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if min(args.expert_updates, args.fit_updates) < 1 or args.seed < 0 or len(args.models) != len(set(args.models)):
        parser.error("Use positive update counts, a nonnegative seed and unique models.")
    args.scenario = "cartpole_balance_sparse"
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA test requires BF16 support.")
        torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"experiment": "short_rollout_training", "implementation_sha256": implementation_sha256(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "checkpoint_writes": False, "production_settings_changed": False, "runs": []}
    persist = lambda: write_report(args.output, report)
    # Collect once for both models; neither arm's policy can change this test's data.
    config = build_config(args.models[0], args)
    cases = {split: collect_cases(config, SimpleNamespace(sim_seeds=list(seeds), rollin_steps=[0, 64], candidates=7))
             for split, seeds in (("train", range(10_000_000 + args.seed, 10_000_004 + args.seed)),
                                  ("validation", range(11_000_000 + args.seed, 11_000_002 + args.seed)))}
    report["cases"] = {split: case_metadata(values) for split, values in cases.items()}
    print(f"Rollout check | models={len(args.models)} | offline={args.expert_updates} once/model | "
          f"fit={args.fit_updates} per arm | horizon=5 | fresh tiny models | no checkpoints", flush=True)
    for name in args.models:
        result = {"model": name, "status": "RUNNING", "arms": []}
        report["runs"].append(result)
        output = args.output / name
        output.mkdir()
        model = None
        try:
            config = build_config(name, args)
            if int(config.jepa_model.planner.horizon) != 5:
                raise ValueError("This test requires the five-step Cartpole planning horizon.")
            result["config"] = OmegaConf.to_container(config, resolve=True)
            family = load_model_family(name)
            with family.build_replay(config) as dataset:
                result["dataset_identity"] = dataset_identity(dataset.metadata)
                model = pretrain(config, dataset, args, output, result)
            shared = copy.deepcopy(family.checkpoint(model))
            banks = {split: cache_branches(model, values) for split, values in cases.items()}
            result["feature_banks"] = {split: {"windows": len(bank["latent"]),
                "native_prediction_targets": len(bank["latent"]) * (bank["latent"].shape[1] - 1),
                "rollout_prediction_targets": math.prod(bank["target"].shape[:3]),
                "sha256": tensor_digest({k: v for k, v in bank.items() if isinstance(v, torch.Tensor)})}
                for split, bank in banks.items()}
            result["baseline"] = {split: score_bank(model, bank) for split, bank in banks.items()}
            persist()
            for weight in (0., .5):
                fit_arm(model, config, shared, banks, weight, args, output, result, persist)
            result["status"] = "COMPLETE"
        except Exception as error:  # noqa: BLE001 - persist failures and continue the other model.
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            del model
        persist()
    report["seconds"] = time.monotonic() - started
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return int(any(result["status"] == "FAIL" for result in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
