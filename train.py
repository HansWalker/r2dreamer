import atexit
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


def save_checkpoint(agent, logdir, name):
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / name)


@torch.no_grad()
def evaluate_policy(agent, eval_envs, logger, step):
    if eval_envs is None or eval_envs.env_num == 0:
        return None
    print("Evaluating the offline-trained policy...")
    agent.eval()
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
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done
    score = returns.mean()
    logger.scalar("episode/eval_score", score)
    logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
    logger.write(step)
    agent.train()
    return float(score)


def train_offline(config, logger, logdir):
    from offline_replay import DMCExpertReplay

    replay = DMCExpertReplay(config.offline)
    print(f"Offline data: {replay.path}")
    print(f"Offline episodes: {len(replay.episodes)}")

    print("Create eval envs.")
    train_envs, eval_envs, _, _ = make_envs(config.env)
    train_envs.close()

    print("Create agent.")
    agent = Dreamer(config.model, replay.obs_space(), replay.act_space()).to(config.device)

    best_score = None
    train_metrics = {}
    updates = int(config.offline.updates)
    eval_every = int(config.offline.eval_every)
    log_every = int(config.offline.log_every)
    save_every = int(config.offline.save_every)
    for update in range(1, updates + 1):
        warmup_data, data = replay.sample()
        train_metrics = agent.update_offline(warmup_data, data)

        if log_every and update % log_every == 0:
            for name, value in train_metrics.items():
                value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                logger.scalar(f"train/{name}", value)
            logger.scalar("train/opt/updates", update)
            logger.write(update, fps=True)

        if eval_every and update % eval_every == 0:
            score = evaluate_policy(agent, eval_envs, logger, update)
            save_checkpoint(agent, logdir, "latest.pt")
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                save_checkpoint(agent, logdir, "best.pt")

        if save_every and update % save_every == 0:
            save_checkpoint(agent, logdir, "latest.pt")

    if train_metrics:
        for name, value in train_metrics.items():
            value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
            logger.scalar(f"train/{name}", value)
        logger.scalar("train/opt/updates", updates)
        logger.write(updates, fps=True)
    score = evaluate_policy(agent, eval_envs, logger, updates) if int(config.offline.eval_episode_num) > 0 else None
    save_checkpoint(agent, logdir, "latest.pt")
    if score is not None and (best_score is None or score > best_score):
        save_checkpoint(agent, logdir, "best.pt")


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
