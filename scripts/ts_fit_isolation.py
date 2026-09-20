"""Small action-branch fitting and head-only controls; never a production recipe."""

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import tools
from dmc_expert.storage import dataset_identity
from models.shared.physical_state import readout_mode
from scripts.diagnose_fresh_readout import physical_metrics, tensor_digest
from scripts.diagnose_planner_oracle import collect_cases, encode_images
from scripts.online_validation import TrajectoryDataset, collect_episode, episode_metadata
from scripts.ts_mechanisms import case_metadata, observed_bank, parameter_changes, pretrain_shared, response_metrics
from training import load_model_family
from training.evaluation import StateDataset
from training.progress import Progress, duration


def finite(values):
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Non-finite fitting metrics.")
    return {key: float(value) for key, value in values.items()}


@torch.no_grad()
def branch_bank(model, cases):
    """Cache native-length teacher-forced windows, including each branch's true successor."""
    bank, windows, controls = [], [], []
    with readout_mode(model):
        for case in cases:
            prefix = model.encode({"image": case["prefix"][None].to(model.device)})
            action = case["action"].to(model.device)
            count, horizon, _ = action.shape
            future = encode_images(model, case["image"].flatten(0, 1), 32).reshape(
                count, horizon, *prefix.shape[2:])
            past = case["past_action"][None].to(model.device)
            trajectory = torch.cat((prefix.expand(count, *prefix.shape[1:]), future), dim=1)
            outgoing = torch.cat((past.expand(count, -1, -1), action), dim=1)
            for step in range(horizon):
                windows.append(trajectory[:, step:step + model.sequence_length])
                controls.append(outgoing[:, step:step + model.history_size])
            bank.append({"id": case["id"], "prefix": prefix, "past_action": past,
                         "action": action, "future": future})
    return {"cases": bank, "latent": torch.cat(windows).detach(), "action": torch.cat(controls).detach()}


@tools.preserve_rng_state
@torch.no_grad()
def branch_scores(model, bank):
    # FP32 here changes the predictor arithmetic only: cached encoder targets remain identical.
    scores, amp = {}, model.use_amp
    try:
        for precision in (("native_bf16", "fp32_predictor") if amp and model.device.type == "cuda" else ("fp32",)):
            model.use_amp = precision == "native_bf16"
            with readout_mode(model):
                prediction = model.predict(bank["latent"][:, :-1], bank["action"])
                target = bank["latent"][:, 1:]
                cases = []
                for case in bank["cases"]:
                    predicted = model.rollout(case["prefix"], case["past_action"], case["action"][None])[0]
                    horizons = {}
                    for h in sorted({1, predicted.shape[1]}):
                        actual = case["future"][:, h - 1]
                        values = response_metrics(predicted[:, h - 1], actual)
                        blind = (actual - actual.mean(0, keepdim=True)).square().mean().item()
                        values.update(action_blind_mse_floor=blind,
                                      mse_over_blind_floor=values["matched_action_rmse"] ** 2 / blind if blind > 1e-12 else None)
                        horizons[str(h)] = values
                    cases.append({"id": case["id"], "horizons": horizons})
                scores[precision] = {
                    "teacher_forced_mse": F.mse_loss(prediction, target).item(),
                    "teacher_forced_last_mse": F.mse_loss(prediction[:, -1], target[:, -1]).item(),
                    "cases": cases,
                    "fits_train_criterion": all(c["horizons"]["1"]["mse_over_blind_floor"] is not None
                        and c["horizons"]["1"]["mse_over_blind_floor"] < .25
                        and c["horizons"]["1"]["matched_action_rmse"] < c["horizons"]["1"]["wrong_action_rmse"]
                        for c in cases),
                }
    finally:
        model.use_amp = amp
    return scores


def restore(model, shared):
    model.set_adaptation_mode("native")
    load_model_family("temporal_straightening").load_checkpoint(model, copy.deepcopy(shared), training=True)
    torch.testing.assert_close(model.optimizer_state_dict(), shared["optimizer_state_dict"], rtol=0, atol=0)
    if tensor_digest(model.state_dict()) != tensor_digest(shared["model_state_dict"]):
        raise RuntimeError("Fitting control did not restore the shared offline state.")
    model.set_adaptation_mode("frozen_encoder")


def action_controls(model, shared, banks, args, report):
    parameters = [*model.predictor.parameters(), *model.action_encoder.parameters()]
    optimizers = [model.optimizers[key] for key in ("predictor", "action_encoder")]
    for mode in ("native_dropout", "dropout_off"):
        restore(model, shared)
        protected = tensor_digest({k: v for k, v in model.state_dict().items()
                                   if not k.startswith(("predictor.", "action_encoder."))})
        run = {"mode": mode, "initial_state_sha256": tensor_digest(model.state_dict()), "snapshots": []}
        report["actions"].append(run)
        start = time.monotonic()
        folder = args.output / f"actions_{mode}"
        folder.mkdir()
        progress = Progress(f"Action fit {mode}", args.action_updates)
        with (folder / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for step in range(1, args.action_updates + 1):
                torch.manual_seed(args.seed + 2_000_000 + step)
                # eval disables nn.Dropout AND functional attention dropout, not gradients.
                model.train(mode == "native_dropout")
                loss, metrics = model.representation_loss({}, banks["train"]["latent"], banks["train"]["action"])
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, model.grad_clip, error_if_nonfinite=True)
                for optimizer in optimizers:
                    optimizer.step()
                values = finite({"loss": loss.detach(), "prediction_loss": metrics["prediction_loss"].detach(),
                                 "grad_norm": norm, "grad_clipped": norm > model.grad_clip})
                log.write(json.dumps({"update": step, **values}, allow_nan=False) + "\n")
                progress.update(step, f"prediction={values['prediction_loss']:.4g}", force=step == args.action_updates)
                if step in {max(1, args.action_updates // 4), args.action_updates}:
                    run["snapshots"].append({"updates": step, **{split: branch_scores(model, bank) for split, bank in banks.items()}})
                    write_report(args.output, report)
        after = tensor_digest({k: v for k, v in model.state_dict().items()
                               if not k.startswith(("predictor.", "action_encoder."))})
        if protected != after:
            raise RuntimeError("Action fitting changed encoder/readout weights or buffers.")
        run.update(seconds=time.monotonic() - start, protected_state_unchanged=True,
                   parameter_delta_rms=parameter_changes(model, shared["model_state_dict"]))


def head_windows(features, labels, history):
    """Retain complete histories and labels without crossing an episode or sampled clip."""
    def windows(value):
        return value.unfold(1, history, 1).movedim(-1, 2).flatten(0, 1).contiguous()
    if features.shape[:2] != labels.shape[:2] or features.shape[1] < history:
        raise ValueError("Head features and labels need aligned, complete histories.")
    return windows(features).detach(), windows(labels).detach()


def join_banks(parts):
    return tuple(torch.cat(values) for values in zip(*parts, strict=True))


@torch.no_grad()
def head_scores(model, banks):
    result = {}
    for name, (features, labels) in banks.items():
        prediction = torch.cat([model.state_head(chunk)[:, 0] for chunk in features.split(128)])
        truth = labels[:, -1]
        losses = F.smooth_l1_loss(prediction / model.state_head.loss_scale,
                                  truth / model.state_head.loss_scale, reduction="none")
        result[name] = {**physical_metrics(prediction, truth, model.state_head,
                                           model.goal_tolerance, model.goal_geometry),
                        "smooth_l1": losses.mean().item(),
                        "coordinate_loss": dict(zip(model.state_head.coordinates, losses.mean(0).tolist(), strict=True))}
    return result


def head_gradients(head, online, expert):
    """Exact source gradients for the production 50/50 mean loss, before Adam/clipping."""
    parameters = list(head.parameters())
    gradients, result = {}, {}
    count = min(head.samples_per_update, len(online[0]))
    if count < 2 or count % 2:
        raise ValueError("A 50/50 gradient probe requires an even batch with at least two examples.")
    for source, batch, size in (("online", online, count // 2), ("expert", expert, count - count // 2)):
        prediction, labels = head._examples(*batch, size)
        losses = F.smooth_l1_loss(prediction / head.loss_scale, labels / head.loss_scale, reduction="none").mean(0)
        rows = []
        for loss in losses:
            grad = torch.autograd.grad(loss, parameters, retain_graph=True)
            rows.append(torch.cat([g.detach().flatten().double() for g in grad]))
        gradients[source] = torch.stack(rows)
        result[source] = {"loss": losses.mean().item(), "coordinates": {
            key: {"loss": value.item(), "gradient_norm": row.norm().item()}
            for key, value, row in zip(head.coordinates, losses, rows, strict=True)}}
    left, right = gradients["online"].mean(0), gradients["expert"].mean(0)
    ln, rn = left.norm().item(), right.norm().item()
    dot = torch.dot(left, right).item()
    result.update(online_norm=ln, expert_norm=rn, norm_ratio=ln / rn if rn > 1e-12 else None,
                  cosine=dot / (ln * rn) if min(ln, rn) > 1e-12 else None,
                  mixed_dot_expert=torch.dot((left + right) / 2, right).item(),
                  note="Negative cosine indicates source conflict. Negative mixed_dot_expert is adverse for expert loss under plain gradient descent, not a prediction of Adam's actual step.")
    return result


def head_controls(model, shared, banks, args, report):
    generator = torch.Generator().manual_seed(args.seed + 3_000_000)
    count = model.state_head.samples_per_update
    if count < 2 or count % 2:
        raise ValueError("Head controls require an even samples_per_update of at least two.")
    plans = {source: torch.randint(len(banks[f"{source}_train"][0]), (args.online_updates, count), generator=generator)
             for source in ("expert", "simulator")}
    report["head_batch_plan_sha256"] = tensor_digest(plans)

    def batch(source, step):
        feature, labels = banks[f"{source}_train"]
        index = plans[source][step].to(feature.device)
        return feature[index], labels[index]

    for mode in ("expert_only", "mixed_50_50"):
        restore(model, shared)
        native = tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith("state_head.")})
        head = model.state_head
        head.expert_fraction = .5 if mode == "mixed_50_50" else 0.
        expert = {"batch": batch("expert", 0)}
        head.configure_online(lambda: expert["batch"])
        run = {"mode": mode, "initial_state_sha256": tensor_digest(model.state_dict()), "snapshots": [],
               "before": head_scores(model, banks),
               "gradient_before": head_gradients(head, batch("simulator", 0), batch("expert", 0))}
        report["heads"].append(run)
        folder = args.output / f"head_{mode}"
        folder.mkdir()
        start, initial_updates = time.monotonic(), int(head.updates)
        progress = Progress(f"Head fit {mode}", args.online_updates)
        try:
            with (folder / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                for step in range(args.online_updates):
                    expert["batch"] = batch("expert", step)
                    current = batch("simulator", step) if mode == "mixed_50_50" else expert["batch"]
                    values = finite(head.fit(*current))
                    log.write(json.dumps({"update": step + 1, **values}, allow_nan=False) + "\n")
                    progress.update(step + 1, f"loss={values['state/loss']:.4g}", force=step + 1 == args.online_updates)
                    if step + 1 in {max(1, args.online_updates // 4), args.online_updates}:
                        run["snapshots"].append({"updates": step + 1, "scores": head_scores(model, banks),
                                                 "gradients": head_gradients(head, batch("simulator", 0), batch("expert", 0))})
                        write_report(args.output, report)
        finally:
            head._expert_source = None
        after = tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith("state_head.")})
        if native != after or int(head.updates) - initial_updates != args.online_updates:
            raise RuntimeError("Head control changed native weights/buffers or used the wrong update count.")
        run.update(seconds=time.monotonic() - start, native_state_unchanged=True)


def cache_heads(model, dataset, config, args, report):
    train = []
    for _ in range(8):
        train.append(head_windows(*model.readout_features(dataset.sample_episode_batch()), model.state_head.history))
    banks = {"expert_train": join_banks(train)}
    heldout = StateDataset(dataset.h5, dataset.metadata, config.model_io, config.state_head.fields, model.state_head.targets)
    settings = SimpleNamespace(context_length=model.history_size, batch_size=8)
    windows = heldout.sample_windows(32, model.history_size + 1, 2_000_000 + args.seed, model.history_size, .5, 8)
    observed = observed_bank(model, {"expert": (heldout, windows)}, settings)["expert"]
    # Only the final label is consumed for these single-history evaluation examples.
    banks["expert_validation"] = (observed["features"], observed["truth"][:, None].expand(-1, model.state_head.history, -1))
    report["expert_validation_windows"] = [asdict(w) for w in windows]
    report["head_simulator_episodes"] = {}
    short = copy.deepcopy(config)
    short.env.time_limit = 200 * int(config.env.action_repeat)
    for split, seeds in (("train", range(7_000_000 + args.seed, 7_000_004 + args.seed)),
                         ("validation", range(8_000_000 + args.seed, 8_000_002 + args.seed))):
        episodes = [collect_episode(short, model, seed, "zero" if i % 2 == 0 else "random") for i, seed in enumerate(seeds)]
        TrajectoryDataset(episodes, forbidden_seeds=range(7_000_000 + args.seed, 7_000_004 + args.seed) if split == "validation" else ())
        parts = []
        with readout_mode(model):
            for episode in episodes:
                features = encode_images(model, episode["image"], 32)[None]
                parts.append(head_windows(features, episode["state"][None].to(model.device), model.state_head.history))
        banks[f"simulator_{split}"] = join_banks(parts)
        report["head_simulator_episodes"][split] = episode_metadata(episodes)
    report["head_feature_banks"] = {name: {"examples": len(features), "sha256": tensor_digest({"features": features, "labels": labels})}
                                    for name, (features, labels) in banks.items()}
    return banks


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    lines = ["TS fit isolation | one offline fit | frozen feature targets | no planner or checkpoint writes",
             "Actions: fitting measured separately from unseen-seed generalization; smaller MSE/blind is better.",
             "Control/split | h1 MSE/action-blind floor | h1 response ratio | h5 matched/wrong RMSE"]
    stages = [("offline", report.get("action_baseline", {}))]
    stages += [(run["mode"], run["snapshots"][-1]) for run in report["actions"] if run["snapshots"]]
    for name, stage in stages:
        for split in ("train", "validation"):
            if split not in stage:
                continue
            precision, score = next(iter(stage[split].items()))
            cases = score["cases"]
            def mean(h, key):
                values = [case["horizons"][h][key] for case in cases]
                return "n/a" if any(v is None for v in values) else f"{sum(values) / len(values):.4g}"
            lines.append(f"{name}/{split} ({precision}) | {mean('1', 'mse_over_blind_floor')} | "
                         f"{mean('1', 'response_ratio')} | {mean('5', 'matched_action_rmse')}/{mean('5', 'wrong_action_rmse')}")
            if split == "train":
                lines.append("  Training-branch fit criterion: " + ("MET" if score["fits_train_criterion"] else "NOT MET"))
    lines.append("Heads: held-out cart position / pole angle RMSE, m / rad; before -> after")
    for run in report["heads"]:
        if not run["snapshots"]:
            continue
        for source in ("expert_validation", "simulator_validation"):
            def pair(score):
                values = score["physical_rmse"]
                return f"{values['position[0]']:.4g}/{values['pole_angle']:.4g}"
            lines.append(f"{run['mode']}/{source} | {pair(run['before'][source])} -> {pair(run['snapshots'][-1]['scores'][source])}")
        for label, values in (("before", run["gradient_before"]), ("after", run["snapshots"][-1]["gradients"])):
            fmt = lambda value: "n/a" if value is None else f"{value:.4g}"
            lines.append(f"  50/50 gradient probe {label}: online/expert norm={fmt(values['norm_ratio'])} "
                         f"cosine={fmt(values['cosine'])} mixed_dot_expert={fmt(values['mixed_dot_expert'])}")
    lines += ["Action fit criterion: every TRAIN anchor's h1 MSE < 25% of its action-blind minimum, and matched RMSE < wrong-action RMSE.",
              "This is a fitting screen, not proof of generalization, h5 accuracy, or policy success. Dropout-off is diagnostic only.",
              "Head banks use real causal histories; encoder/predictor are unchanged. Expert-only is a retention control, not a proposed recipe.",
              "Per-coordinate gradient norms/losses, original-unit errors, intermediate scores and FP32 predictor probes are in JSON.",
              "FP32 probes retain the SAME cached native-precision encoder features; they are not separate FP32 training.",
              "COMPLETE means execution completed, not that either problem is repaired."]
    if "elapsed_seconds" in report:
        lines.append(f"Elapsed | {duration(report['elapsed_seconds'])}")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def run_fit_isolation(config, dataset, args, report, started):
    report.pop("runs")
    report.update(experiment="ts_fit_isolation", status="RUNNING", actions=[], heads=[],
                  fit_diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dataset_identity=dataset_identity(dataset.metadata),
                  budget={"expert_updates": args.expert_updates, "action_updates_per_arm": args.action_updates,
                          "head_updates_per_arm": args.online_updates})
    print(f"TS fitting | shared offline={args.expert_updates} | action arms=2x{args.action_updates} | head arms=2x{args.online_updates}", flush=True)
    model = pretrain_shared(config, dataset, args, report)
    shared = copy.deepcopy(load_model_family("temporal_straightening").checkpoint(model))
    banks = {}
    report["action_cases"] = {}
    for split, seed in (("train", 10_000_000 + args.seed), ("validation", 11_000_000 + args.seed)):
        print(f"Data | {split} action branches | fixed zero/random roll-ins, no planner", flush=True)
        cases = collect_cases(config, SimpleNamespace(sim_seeds=[seed], rollin_steps=[0, 64], candidates=3))
        banks[split] = branch_bank(model, cases)
        report["action_cases"][split] = case_metadata(cases)
    report["action_feature_banks"] = {split: {"windows": len(bank["latent"]),
        "sha256": tensor_digest({"latent": bank["latent"], "action": bank["action"]})} for split, bank in banks.items()}
    report["action_baseline"] = {split: branch_scores(model, bank) for split, bank in banks.items()}
    write_report(args.output, report)
    action_controls(model, shared, banks, args, report)
    restore(model, shared)
    print("Data | caching physical-head features once; four training and two validation simulator episodes", flush=True)
    head_banks = cache_heads(model, dataset, config, args, report)
    head_controls(model, shared, head_banks, args, report)
    report.update(status="COMPLETE", elapsed_seconds=time.monotonic() - started)
    print(write_report(args.output, report), end="")
    print(f"Reports | {args.output.resolve()}")
    return 0
