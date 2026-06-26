import atexit
import json
import math
import pathlib
import sys
import time
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def save_checkpoint(agent, logdir, name, update=None, **metadata):
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    if update is not None:
        items_to_save["update"] = int(update)
    for key, value in metadata.items():
        if value is not None:
            items_to_save[key] = int(value) if isinstance(value, (int, float)) else value
    torch.save(items_to_save, logdir / name)


def latest_logged_update(logdir):
    path = logdir / "metrics.jsonl"
    if not path.exists():
        return 0
    latest = 0
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train/opt/updates" in row:
                latest = max(latest, int(row["train/opt/updates"]))
    return latest


def maybe_resume_offline(agent, config, logdir):
    if not bool(getattr(config.offline, "resume", False)):
        return 0
    checkpoint_path = logdir / str(getattr(config.offline, "resume_checkpoint", "latest.pt"))
    if not checkpoint_path.exists():
        print(f"Resume requested but checkpoint is missing: {checkpoint_path}. Starting from update 0.")
        return 0

    print(f"Resume offline training from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    try:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint.get("optims_state_dict", {}))
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not resume offline checkpoint. This usually means the checkpoint "
            "was created with different model dimensions/config. Use the same model "
            "config as the checkpoint or start a fresh logdir."
        ) from exc
    agent.clone_and_freeze()
    update = int(checkpoint.get("update", 0)) or latest_logged_update(logdir)
    print(f"Resumed at update {update}.")
    return update


def close_envs(envs):
    if envs is None:
        return
    close = getattr(envs, "close", None)
    if callable(close):
        close()
        return
    for env in getattr(envs, "envs", []):
        close = getattr(env, "close", None)
        if callable(close):
            close()


def make_eval_envs(config):
    train_envs, eval_envs, _, _ = make_envs(config.env)
    close_envs(train_envs)
    return eval_envs


def format_eval_console(step_label, step, score, length, best_score):
    return (
        f"phase=eval | {step_label}={int(step)} | "
        f"score={tools.format_scalar(score, 1)} | "
        f"length={tools.format_scalar(length, 0)} | "
        f"best={tools.format_scalar(best_score, 1)}"
    )


def format_expert_console(update, updates, episode_pass, total_passes, sec_per_update, metrics, eval_score, best_score):
    eta = (updates - update) * sec_per_update
    return (
        f"phase=expert | update={update}/{updates} ({tools.format_percent(update, updates)}) | "
        f"pass={tools.format_scalar(episode_pass, 1)}/{tools.format_scalar(total_passes, 1)} | "
        f"speed={tools.format_scalar(sec_per_update, 2)}s/update | "
        f"eta={tools.format_eta(eta)} | "
        f"loss={tools.format_scalar(metrics.get('opt/loss'), 2)} | "
        f"bc={tools.format_scalar(metrics.get('loss/bc'), 2)} | "
        f"eval={tools.format_scalar(eval_score, 1)} | "
        f"best={tools.format_scalar(best_score, 1)}"
    )


@torch.no_grad()
def evaluate_policy(agent, eval_envs, logger, step, step_label="step", best_score=None):
    if eval_envs is None or eval_envs.env_num == 0:
        return None
    print("Evaluating policy...")
    agent.eval()
    try:
        done = torch.ones(eval_envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(eval_envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(eval_envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(eval_envs.env_num, dtype=torch.float32, device=agent.device)
        agent_state = agent.get_initial_state(eval_envs.env_num)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            trans, step_done = eval_envs.step(act.detach(), done)
            trans = trans.to(agent.device, non_blocking=True)
            done = step_done.to(agent.device)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            if not torch.isfinite(act).all():
                raise RuntimeError("Policy produced non-finite actions during evaluation.")
            act = torch.clamp(act, -1.0, 1.0)
            returns += trans["reward"][:, 0] * ~once_done
            once_done |= done
        score = float(returns.mean())
        length = float(steps.to(torch.float32).mean())
        best = score if best_score is None else max(float(best_score), score)
        logger.scalar("episode/eval_score", score)
        logger.scalar("episode/eval_length", length)
        logger.write(step, console_message=format_eval_console(step_label, step, score, length, best))
        return score
    finally:
        agent.train()


def train_offline(config, logger, logdir):
    from offline_replay import DMCExpertReplay

    print(f"Load offline data: {config.offline.data_path}")
    replay = DMCExpertReplay(config.offline)
    eval_envs = None
    try:
        print(f"Offline data: {replay.path}")
        print(f"Offline episodes: {len(replay.episodes)}")
        print(f"Offline windows: {replay.num_windows}")

        if int(config.offline.eval_episode_num) > 0:
            print("Create eval envs.")
            eval_envs = make_eval_envs(config)

        print("Create agent.")
        agent = Dreamer(config.model, replay.obs_space(), replay.act_space()).to(config.device)
        start_update = maybe_resume_offline(agent, config, logdir)
        replay.skip_batches(start_update)

        best_score = None
        train_metrics = {}
        updates = int(config.offline.updates)
        eval_every = int(config.offline.eval_every)
        log_every = int(config.offline.log_every)
        save_every = int(config.offline.save_every)
        if start_update >= updates:
            print(f"Checkpoint is already at update {start_update}; target is {updates}. Nothing to train.")
            save_checkpoint(agent, logdir, "latest.pt", update=start_update)
            return

        last_eval_success_update = None
        train_start_time = time.perf_counter()
        for update in range(start_update + 1, updates + 1):
            warmup_data, data = replay.sample()
            if warmup_data is not None:
                warmup_data = warmup_data.to(config.device, non_blocking=True)
            data = data.to(config.device, non_blocking=True)
            train_metrics = agent.update_offline(warmup_data, data)

            if log_every and update % log_every == 0:
                completed_updates = update - start_update
                elapsed = time.perf_counter() - train_start_time
                sec_per_update = elapsed / max(completed_updates, 1)
                for name, value in train_metrics.items():
                    value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                    logger.scalar(f"train/{name}", value)
                logger.scalar("train/opt/updates", update)
                logger.scalar("train/timing/sec_per_update", sec_per_update)
                logger.write(update, fps=True)

            if eval_every and update % eval_every == 0:
                save_checkpoint(agent, logdir, "latest.pt", update=update)
                try:
                    score = evaluate_policy(agent, eval_envs, logger, update, step_label="update", best_score=best_score)
                except Exception as exc:
                    print(f"Evaluation failed at update {update}: {type(exc).__name__}: {exc}")
                    logger.scalar("episode/eval_error", 1.0)
                    logger.write(update)
                    close_envs(eval_envs)
                    eval_envs = make_eval_envs(config) if int(config.offline.eval_episode_num) > 0 else None
                else:
                    last_eval_success_update = update
                    if score is not None and (best_score is None or score > best_score):
                        best_score = score
                        save_checkpoint(agent, logdir, "best.pt", update=update)

            if save_every and update % save_every == 0:
                save_checkpoint(agent, logdir, "latest.pt", update=update)

        if train_metrics:
            completed_updates = updates - start_update
            elapsed = time.perf_counter() - train_start_time
            sec_per_update = elapsed / max(completed_updates, 1)
            for name, value in train_metrics.items():
                value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                logger.scalar(f"train/{name}", value)
            logger.scalar("train/opt/updates", updates)
            logger.scalar("train/timing/sec_per_update", sec_per_update)
            logger.write(updates, fps=True)
        save_checkpoint(agent, logdir, "latest.pt", update=updates)
        should_final_eval = int(config.offline.eval_episode_num) > 0 and last_eval_success_update != updates
        if should_final_eval:
            try:
                score = evaluate_policy(agent, eval_envs, logger, updates, step_label="update", best_score=best_score)
            except Exception as exc:
                print(f"Final evaluation failed: {type(exc).__name__}: {exc}")
                score = None
            if score is not None and (best_score is None or score > best_score):
                save_checkpoint(agent, logdir, "best.pt", update=updates)
    finally:
        replay.close()
        close_envs(eval_envs)


def train_expert_then_online(config, logger, logdir):
    from offline_replay import DMCExpertEpisodeReplay

    print(f"Load expert data: {config.expert_pretrain.data_path}")
    replay = DMCExpertEpisodeReplay(config.expert_pretrain)
    eval_envs = None
    try:
        print(f"Expert data: {replay.path}")
        print(f"Expert episodes: {replay.num_episodes}")

        eval_every = int(config.expert_pretrain.eval_every)
        eval_episode_num = int(config.env.eval_episode_num)
        if eval_every and eval_episode_num > 0:
            print("Create pretrain eval envs.")
            eval_envs = make_eval_envs(config)

        print("Create agent.")
        agent = Dreamer(config.model, replay.obs_space(), replay.act_space()).to(config.device)

        requested_updates = int(config.training.offline_steps)
        min_updates = math.ceil(2 * replay.num_episodes / replay.batch_size)
        updates = max(requested_updates, min_updates)
        passes = updates * replay.batch_size / replay.num_episodes
        print(
            "Offline expert pretrain: "
            f"{updates} updates, requested={requested_updates}, "
            f"batch_size={replay.batch_size}, about {passes:.1f} episode passes"
        )

        best_score = None
        last_eval_score = None
        train_metrics = {}
        log_every = int(config.expert_pretrain.log_every)
        save_every = int(config.expert_pretrain.save_every)
        train_start_time = time.perf_counter()
        for update in range(1, updates + 1):
            data = replay.sample_episode_batch().to(config.device, non_blocking=True)
            train_metrics = agent.update_expert_pretrain(data)

            if log_every and update % log_every == 0:
                elapsed = time.perf_counter() - train_start_time
                sec_per_update = elapsed / max(update, 1)
                for name, value in train_metrics.items():
                    value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                    logger.scalar(f"train/{name}", value)
                logger.scalar("train/opt/updates", update)
                logger.scalar("train/expert_pretrain/passes", update * replay.batch_size / replay.num_episodes)
                logger.scalar("train/timing/sec_per_update", sec_per_update)
                logger.write(
                    update,
                    console_message=format_expert_console(
                        update,
                        updates,
                        update * replay.batch_size / replay.num_episodes,
                        passes,
                        sec_per_update,
                        train_metrics,
                        last_eval_score,
                        best_score,
                    ),
                )

            if eval_every and update % eval_every == 0:
                save_checkpoint(agent, logdir, "pretrained_latest.pt", update=update)
                try:
                    score = evaluate_policy(agent, eval_envs, logger, update, step_label="update", best_score=best_score)
                except Exception as exc:
                    print(f"Pretrain evaluation failed at update {update}: {type(exc).__name__}: {exc}")
                    logger.scalar("episode/eval_error", 1.0)
                    logger.write(update)
                    close_envs(eval_envs)
                    eval_envs = make_eval_envs(config) if eval_episode_num > 0 else None
                else:
                    last_eval_score = score
                    if score is not None and (best_score is None or score > best_score):
                        best_score = score
                        save_checkpoint(agent, logdir, "pretrained_best.pt", update=update)

            if save_every and update % save_every == 0:
                save_checkpoint(agent, logdir, "pretrained_latest.pt", update=update)

        if train_metrics:
            elapsed = time.perf_counter() - train_start_time
            sec_per_update = elapsed / max(updates, 1)
            for name, value in train_metrics.items():
                value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                logger.scalar(f"train/{name}", value)
            logger.scalar("train/opt/updates", updates)
            logger.scalar("train/expert_pretrain/passes", passes)
            logger.scalar("train/timing/sec_per_update", sec_per_update)
            logger.write(
                updates,
                console_message=format_expert_console(
                    updates,
                    updates,
                    passes,
                    passes,
                    sec_per_update,
                    train_metrics,
                    last_eval_score,
                    best_score,
                ),
            )
        save_checkpoint(agent, logdir, "pretrained.pt", update=updates)
    finally:
        replay.close()
        close_envs(eval_envs)

    print("Create online envs.")
    replay_buffer = Buffer(config.buffer)
    train_envs, eval_envs, _, _ = make_envs(config.env)

    print("Start online Dreamer training.")
    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)

    def save_online(name, update, online_step):
        save_checkpoint(agent, logdir, name, update=update, online_step=online_step)

    online_state = policy_trainer.begin(agent, save_callback=save_online)
    save_checkpoint(
        agent,
        logdir,
        "latest.pt",
        update=online_state["updates"],
        online_step=online_state["step"],
    )


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    if "expert_pretrain" in config and bool(config.expert_pretrain.enabled):
        train_expert_then_online(config, logger, logdir)
        return

    if "offline" in config:
        train_offline(config, logger, logdir)
        return

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)

    def save_online(name, update, online_step):
        save_checkpoint(agent, logdir, name, update=update, online_step=online_step)

    online_state = policy_trainer.begin(agent, save_callback=save_online)
    save_checkpoint(
        agent,
        logdir,
        "latest.pt",
        update=online_state["updates"],
        online_step=online_state["step"],
    )


if __name__ == "__main__":
    main()
