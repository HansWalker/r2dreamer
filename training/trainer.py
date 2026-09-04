"""Shared expert-pretraining and online-training lifecycle."""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

import tools
from envs import close_envs, make_envs, make_eval_envs
from training import load_model_family


@dataclass
class ExpertState:
    updates: int = 0
    last_eval_score: float | None = None
    best_eval_score: float | None = None


@dataclass
class OnlineState:
    env_steps: int = 0
    world_model_updates: int = 0
    last_eval_score: float | None = None
    best_eval_score: float | None = None
    last_eval_step: int | None = None


@dataclass(frozen=True)
class TrainingRun:
    config: Any
    logger: tools.Logger
    logdir: Path
    family: ModuleType
    model: torch.nn.Module


def save_checkpoint(run, path, phase, state, replay_state=None, expert_updates=0):
    payload = {
        **run.family.checkpoint(run.model),
        "phase": phase,
        "training_seed": int(run.config.seed),
        "rng_state": tools.get_rng_state(),
        "trainer_state": asdict(state),
        "expert_updates": int(expert_updates),
    }
    if replay_state is not None:
        payload["replay_state"] = replay_state
    temporary = Path(f"{path}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
@tools.preserve_rng_state
def evaluate(
    run,
    step,
    best_score=None,
    step_label="env_step",
):
    envs = make_eval_envs(run.config.env)
    tools.configure_randomness(int(run.config.env.eval_seed), bool(run.config.deterministic_run))
    try:
        print(f"Evaluation | running | {step_label}={int(step)}")
        score, length, extra = run.family.evaluate(run.config, run.model, envs)
    finally:
        close_envs(envs)
    improved = best_score is None or score > best_score
    best_score = score if improved else best_score
    scalars = {"episode/eval_score": score, "episode/eval_length": length}
    scalars.update({f"episode/eval_{name}": value for name, value in extra.items()})
    run.logger.write(
        step,
        scalars,
        console_message=(
            f"Evaluation | {step_label}={int(step)} | return={tools.format_scalar(score, 1)} | "
            f"length={tools.format_scalar(length, 0)} | best={tools.format_scalar(best_score, 1)}"
        ),
    )
    return score, best_score, improved


def pretrain(run, replay, checkpoint=None):
    settings = run.config.training.expert
    updates = math.ceil(float(settings.epochs) * replay.num_episodes / replay.batch_size)
    epochs = updates * replay.batch_size / replay.num_episodes
    log_every = int(settings.log_every)
    eval_every = int(settings.eval_every) if int(run.config.env.eval_episode_num) > 0 else 0
    save_every = int(settings.save_every)
    checkpoint = checkpoint or {}
    saved_state = checkpoint.get("trainer_state")
    state = (
        ExpertState(**saved_state)
        if saved_state is not None
        else ExpertState(updates=int(checkpoint.get("expert_updates", 0)))
    )
    if checkpoint.get("replay_state") is not None:
        replay.load_state_dict(checkpoint["replay_state"])
    started = time.perf_counter()
    start_update = state.updates
    if hasattr(run.model, "configure_pretraining"):
        run.model.configure_pretraining(updates)
    print(
        f"Expert | target_epochs={epochs:.2f} | updates={state.updates}/{updates} | "
        f"batch_size={replay.batch_size}"
    )

    for update in range(state.updates + 1, updates + 1):
        metrics = run.family.expert_update(run.model, replay.sample_episode_batch())
        state.updates = update

        if (log_every and update % log_every == 0) or update == updates:
            sec_per_update = (time.perf_counter() - started) / (update - start_update)
            epoch = update * replay.batch_size / replay.num_episodes
            detail = " | ".join(
                f"{label}={tools.format_scalar(metrics.get(key), 2)}"
                for label, key in run.family.EXPERT_METRICS.items()
            )
            scalars = {f"train/{name}": value for name, value in metrics.items()}
            scalars.update({
                "train/opt/updates": update,
                "train/expert/epochs": epoch,
                "train/timing/sec_per_update": sec_per_update,
            })
            run.logger.write(
                update,
                scalars,
                console_message=(
                    f"Expert | update={update}/{updates} ({100 * update / updates:.0f}%) | "
                    f"epoch={tools.format_scalar(epoch, 1)}/{tools.format_scalar(epochs, 1)} | "
                    f"speed={tools.format_scalar(sec_per_update, 2)}s/update | "
                    f"eta={tools.format_eta((updates - update) * sec_per_update)} | {detail} | "
                    f"eval={tools.format_scalar(state.last_eval_score, 1)} | "
                    f"best={tools.format_scalar(state.best_eval_score, 1)}"
                ),
            )

        if eval_every and (update % eval_every == 0 or update == updates):
            state.last_eval_score, state.best_eval_score, improved = evaluate(
                run,
                update,
                state.best_eval_score,
                step_label="expert_update",
            )
            if improved:
                save_checkpoint(
                    run,
                    run.logdir / "pretrained_best.pt",
                    phase="expert",
                    state=state,
                    replay_state=replay.state_dict(),
                    expert_updates=update,
                )

        if (save_every and update % save_every == 0) or update == updates:
            save_checkpoint(
                run,
                run.logdir / "pretrain_latest.pt",
                phase="expert",
                state=state,
                replay_state=replay.state_dict(),
                expert_updates=update,
            )

    save_checkpoint(
        run,
        run.logdir / "pretrained.pt",
        phase="expert",
        state=state,
        replay_state=replay.state_dict(),
        expert_updates=state.updates,
    )
    print(f"Checkpoint | saved=pretrained.pt | expert_updates={state.updates}")
    return state.updates


def train_online(run, session, checkpoint=None, expert_updates=0):
    settings = run.config.training.online
    checkpoint = checkpoint or {}
    resumed = checkpoint.get("phase") == "online"
    state = OnlineState(**checkpoint["trainer_state"]) if resumed else OnlineState()
    total_steps = int(settings.steps)
    if resumed:
        replay_state = checkpoint.get("replay_state")
        if replay_state is None:
            raise ValueError("Online training can only resume from latest.pt, which contains replay state.")
        session.replay.load_state_dict(replay_state)
    session.start(resumed=resumed)
    eval_every = int(settings.eval_every)
    save_every = int(settings.save_every)
    log_every = int(settings.log_every)
    eval_enabled = bool(eval_every and int(run.config.env.eval_episode_num) > 0)
    print(
        f"Online | steps={state.env_steps}/{total_steps} | replay={session.replay.count()} | "
        f"resumed={str(resumed).lower()}"
    )

    def save(name):
        replay_state = session.replay.state_dict() if name == "latest.pt" else None
        save_checkpoint(
            run,
            run.logdir / name,
            phase="online",
            state=state,
            replay_state=replay_state,
            expert_updates=expert_updates,
        )

    def run_evaluation():
        state.last_eval_score, state.best_eval_score, improved = evaluate(
            run,
            state.env_steps,
            state.best_eval_score,
        )
        state.last_eval_step = state.env_steps
        save("latest.pt")
        if improved:
            save("best.pt")

    if not resumed and eval_enabled:
        run_evaluation()

    next_eval = (state.last_eval_step if state.last_eval_step is not None else state.env_steps) + eval_every
    next_save = state.env_steps + save_every
    next_log = state.env_steps + log_every
    last_saved_step = state.env_steps if resumed or state.last_eval_step is not None else None
    last_log_step = state.env_steps
    last_log_time = time.perf_counter()
    metrics = {}

    while state.env_steps < total_steps:
        step_delta, episodes = session.collect()
        state.env_steps += step_delta
        for score, length in episodes:
            run.logger.write(state.env_steps, {"episode/score": score, "episode/length": length})

        update_metrics, update_count = session.update(state.env_steps)
        state.world_model_updates += update_count
        if update_metrics:
            metrics.update(update_metrics)

        if eval_enabled and state.env_steps >= next_eval:
            run_evaluation()
            last_saved_step = state.env_steps
            while next_eval <= state.env_steps:
                next_eval += eval_every

        if save_every and state.env_steps >= next_save:
            if last_saved_step != state.env_steps:
                save("latest.pt")
                last_saved_step = state.env_steps
            while next_save <= state.env_steps:
                next_save += save_every

        if metrics and log_every and state.env_steps >= next_log:
            now = time.perf_counter()
            fps = (state.env_steps - last_log_step) / max(now - last_log_time, 1e-6)
            last_log_step, last_log_time = state.env_steps, now
            detail = " | ".join(
                f"{label}={tools.format_scalar(metrics.get(key), 2)}"
                for label, key in run.family.ONLINE_METRICS.items()
            )
            scalars = {f"train/{name}": value for name, value in metrics.items()}
            scalars.update({
                "train/world_model_updates": state.world_model_updates,
                "replay/size": session.replay.count(),
                "fps/fps": fps,
            })
            run.logger.write(
                state.env_steps,
                scalars,
                console_message=(
                    f"Online | env_step={state.env_steps}/{total_steps} "
                    f"({100 * state.env_steps / total_steps:.0f}%) | "
                    f"wm_updates={state.world_model_updates} | "
                    f"speed={tools.format_scalar(fps, 0)}fps | "
                    f"eta={tools.format_eta((total_steps - state.env_steps) / max(fps, 1e-6))} | {detail} | "
                    f"eval={tools.format_scalar(state.last_eval_score, 1)} | "
                    f"best={tools.format_scalar(state.best_eval_score, 1)}"
                ),
            )
            while next_log <= state.env_steps:
                next_log += log_every

    if eval_enabled and state.last_eval_step != state.env_steps:
        run_evaluation()
    elif last_saved_step != state.env_steps:
        save("latest.pt")
    save("final.pt")
    print(f"Checkpoint | saved=final.pt | env_steps={state.env_steps}")
    return asdict(state)


def train(config, logger, logdir, checkpoint_path=None):
    family_name = str(config.model_family)
    if family_name == "dreamer":
        variant = str(config.model.rssm.core)
    elif family_name == "storm":
        variant = str(config.storm_model.sequence_core)
    else:
        variant = "default"
    print(f"Model | building | name={family_name}/{variant} | device={config.device}")
    family = load_model_family(family_name)
    model = family.build_model(config)
    run = TrainingRun(config, logger, Path(logdir), family, model)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model | ready | name={family_name}/{variant} | parameters={parameters:,}")
    checkpoint = None
    expert_updates = 0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("phase") not in {"expert", "online"}:
            raise ValueError("Checkpoint does not contain a supported training phase.")
        run.family.load_checkpoint(run.model, checkpoint, training=True)
        tools.set_rng_state(checkpoint.get("rng_state"))
        expert_updates = int(checkpoint.get("expert_updates", 0))
        print(
            f"Checkpoint | loaded={checkpoint_path} | phase={checkpoint['phase']} | "
            f"expert_updates={expert_updates}"
        )

    if bool(config.training.expert.enabled) and (checkpoint is None or checkpoint["phase"] == "expert"):
        with run.family.ExpertReplay(config) as replay:
            replay.validate_model_io(config.model_io)
            if hasattr(run.family, "configure_expert_replay"):
                run.family.configure_expert_replay(run.model, replay)
            print(f"Data | expert={replay.path} | episodes={replay.num_episodes}")
            expert_updates = pretrain(run, replay, checkpoint)
    elif not bool(config.training.expert.enabled):
        print("Expert | skipped | disabled")
    else:
        print(f"Expert | complete | updates={expert_updates}")

    if not int(config.training.online.steps):
        print("Online | skipped | target_steps=0")
        return {"expert_updates": expert_updates} if expert_updates else None

    train_envs = None
    try:
        train_envs = make_envs(config.env)
        session = run.family.OnlineSession(config, run.model, train_envs)
        return train_online(run, session, checkpoint, expert_updates)
    finally:
        close_envs(train_envs)
