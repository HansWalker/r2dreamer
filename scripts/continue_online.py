#!/usr/bin/env python3
import argparse
import atexit
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from train import close_envs, save_checkpoint
from trainer import OnlineTrainer


def build_config(args):
    overrides = [
        f"logdir={args.logdir}",
        f"env.task={args.task}",
        f"training.online_steps={args.steps}",
        f"trainer.steps={args.steps}",
        f"env.eval_episode_num={args.eval_episodes}",
        f"trainer.eval_episode_num={args.eval_episodes}",
        f"trainer.eval_every={args.eval_every}",
        f"trainer.save_every={args.save_every}",
    ]
    if args.batch_length is not None:
        overrides.append(f"batch_length={args.batch_length}")
    if args.online_warmup_length is not None:
        overrides.append(f"online_warmup_length={args.online_warmup_length}")
    if args.train_ratio is not None:
        overrides.append(f"env.train_ratio={args.train_ratio}")
    if args.train_after is not None:
        overrides.append(f"++trainer.train_after={args.train_after}")

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=args.config, overrides=overrides)


def main():
    parser = argparse.ArgumentParser(description="Continue online Dreamer training from an existing checkpoint.")
    parser.add_argument("--config", default="offline_dmc_expert_mamba3")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--batch-length", type=int)
    parser.add_argument("--online-warmup-length", type=int)
    parser.add_argument("--train-ratio", type=int)
    parser.add_argument("--train-after", type=int)
    args = parser.parse_args()

    logdir = Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    console = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console.close())

    config = build_config(args)
    tools.set_seed_everywhere(int(config.seed))
    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    checkpoint_path = Path(args.checkpoint).expanduser()
    print(f"Logdir {logdir}")
    print(f"Load checkpoint {checkpoint_path}")
    print(f"Run online env steps {args.steps}")
    print(f"Task {args.task}")
    print(f"Replay warmup length {int(config.buffer.warmup_length)}")
    print(f"Batch length {int(config.buffer.batch_length)}")
    print(f"Train ratio {int(config.trainer.train_ratio)}")
    print(f"Train after env step {int(config.trainer.train_after)}")

    train_envs = eval_envs = None
    try:
        print("Create online envs.")
        train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
        print("Create agent.")
        agent = Dreamer(config.model, obs_space, act_space).to(config.device)

        checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint.get("optims_state_dict", {}))
        print(
            "Loaded source checkpoint: "
            f"update={checkpoint.get('update', '-')}, online_step={checkpoint.get('online_step', '-')}"
        )

        replay_buffer = Buffer(config.buffer)
        trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)

        def save_online(name, update, online_step):
            save_checkpoint(
                agent,
                logdir,
                name,
                update=update,
                online_step=online_step,
                source_checkpoint=str(checkpoint_path),
                source_update=checkpoint.get("update", -1),
                source_online_step=checkpoint.get("online_step", -1),
            )

        state = trainer.begin(agent, save_callback=save_online)
        save_online("latest.pt", state["updates"], state["step"])
        print(f"Finished online continuation: step={state['step']}, updates={state['updates']}")
    finally:
        close_envs(train_envs)
        close_envs(eval_envs)


if __name__ == "__main__":
    main()
