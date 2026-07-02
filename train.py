import atexit
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


def format_expert_console(update, updates, epoch, total_epochs, sec_per_update, metrics, eval_score, best_score):
    eta = (updates - update) * sec_per_update
    return (
        f"phase=expert | update={update}/{updates} ({tools.format_percent(update, updates)}) | "
        f"epoch={tools.format_scalar(epoch, 1)}/{tools.format_scalar(total_epochs, 1)} | "
        f"speed={tools.format_scalar(sec_per_update, 2)}s/update | "
        f"eta={tools.format_eta(eta)} | "
        f"loss={tools.format_scalar(metrics.get('opt/loss'), 2)} | "
        f"bc={tools.format_scalar(metrics.get('loss/bc'), 2)} | "
        f"eval={tools.format_scalar(eval_score, 1)} | "
        f"best={tools.format_scalar(best_score, 1)}"
    )


def expert_pretrain_schedule(config, replay):
    """Convert user-facing expert epochs into optimizer updates.

    The combined expert-pretrain path is configured in dataset epochs because
    that is easier to reason about than raw update counts.
    """
    min_updates = math.ceil(2 * replay.num_episodes / replay.batch_size)
    requested_epochs = float(config.training.offline_epochs)
    requested_updates = math.ceil(requested_epochs * replay.num_episodes / replay.batch_size)
    updates = max(requested_updates, min_updates)
    epochs = updates * replay.batch_size / replay.num_episodes
    return requested_epochs, requested_updates, updates, epochs, min_updates


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

        requested_epochs, requested_updates, updates, epochs, min_updates = expert_pretrain_schedule(config, replay)
        print(
            "Offline expert pretrain: "
            f"{epochs:.2f} epochs, {updates} updates, "
            f"requested_epochs={requested_epochs:.2f}, requested_updates={requested_updates}, "
            f"minimum_updates={min_updates}, batch_size={replay.batch_size}"
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
                logger.scalar("train/expert_pretrain/epochs", update * replay.batch_size / replay.num_episodes)
                logger.scalar("train/timing/sec_per_update", sec_per_update)
                logger.write(
                    update,
                    console_message=format_expert_console(
                        update,
                        updates,
                        update * replay.batch_size / replay.num_episodes,
                        epochs,
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
            logger.scalar("train/expert_pretrain/epochs", epochs)
            logger.scalar("train/timing/sec_per_update", sec_per_update)
            logger.write(
                updates,
                console_message=format_expert_console(
                    updates,
                    updates,
                    epochs,
                    epochs,
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

    if int(config.training.online_steps) <= 0:
        save_checkpoint(agent, logdir, "latest.pt", update=updates, online_step=0)
        return

    print("Create online envs.")
    replay_buffer = Buffer(config.buffer)
    train_envs = eval_envs = None
    try:
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
    finally:
        close_envs(train_envs)
        close_envs(eval_envs)


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

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs = eval_envs = None
    try:
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
    finally:
        close_envs(train_envs)
        close_envs(eval_envs)


if __name__ == "__main__":
    main()
