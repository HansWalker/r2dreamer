import atexit
import json
import pathlib
import sys
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


def save_checkpoint(agent, logdir, name, update=None):
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    if update is not None:
        items_to_save["update"] = int(update)
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


@torch.no_grad()
def evaluate_policy(agent, eval_envs, logger, step):
    if eval_envs is None or eval_envs.env_num == 0:
        return None
    print("Evaluating the offline-trained policy...")
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
        score = returns.mean()
        logger.scalar("episode/eval_score", score)
        logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        logger.write(step)
        return float(score)
    finally:
        agent.train()


def train_offline(config, logger, logdir):
    from offline_replay import DMCExpertReplay

    replay = DMCExpertReplay(config.offline)
    print(f"Offline data: {replay.path}")
    print(f"Offline episodes: {len(replay.episodes)}")

    eval_envs = None
    if int(config.offline.eval_episode_num) > 0:
        print("Create eval envs.")
        eval_envs = make_eval_envs(config)

    print("Create agent.")
    agent = Dreamer(config.model, replay.obs_space(), replay.act_space()).to(config.device)
    start_update = maybe_resume_offline(agent, config, logdir)

    best_score = None
    train_metrics = {}
    updates = int(config.offline.updates)
    eval_every = int(config.offline.eval_every)
    log_every = int(config.offline.log_every)
    save_every = int(config.offline.save_every)
    if start_update >= updates:
        print(f"Checkpoint is already at update {start_update}; target is {updates}. Nothing to train.")
        save_checkpoint(agent, logdir, "latest.pt", update=start_update)
        close_envs(eval_envs)
        return

    last_eval_success_update = None
    for update in range(start_update + 1, updates + 1):
        warmup_data, data = replay.sample()
        train_metrics = agent.update_offline(warmup_data, data)

        if log_every and update % log_every == 0:
            for name, value in train_metrics.items():
                value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                logger.scalar(f"train/{name}", value)
            logger.scalar("train/opt/updates", update)
            logger.write(update, fps=True)

        if eval_every and update % eval_every == 0:
            save_checkpoint(agent, logdir, "latest.pt", update=update)
            try:
                score = evaluate_policy(agent, eval_envs, logger, update)
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
        for name, value in train_metrics.items():
            value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
            logger.scalar(f"train/{name}", value)
        logger.scalar("train/opt/updates", updates)
        logger.write(updates, fps=True)
    save_checkpoint(agent, logdir, "latest.pt", update=updates)
    should_final_eval = int(config.offline.eval_episode_num) > 0 and last_eval_success_update != updates
    if should_final_eval:
        try:
            score = evaluate_policy(agent, eval_envs, logger, updates)
        except Exception as exc:
            print(f"Final evaluation failed: {type(exc).__name__}: {exc}")
            score = None
        if score is not None and (best_score is None or score > best_score):
            save_checkpoint(agent, logdir, "best.pt", update=updates)
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
    policy_trainer.begin(agent)

    save_checkpoint(agent, logdir, "latest.pt")


if __name__ == "__main__":
    main()
