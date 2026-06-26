"""Helpers for evaluating one DMC expert training run."""

import json
import os
import sys
from pathlib import Path


def read_metrics(logdir):
    path = Path(logdir) / "metrics.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def logged_eval_summary(logdir):
    eval_rows = [row for row in read_metrics(logdir) if "episode/eval_score" in row]
    if not eval_rows:
        return None, None, None
    final = eval_rows[-1]
    best = max(eval_rows, key=lambda row: row["episode/eval_score"])
    return final.get("step"), final["episode/eval_score"], best["episode/eval_score"]


def eval_run(
    *,
    r2dreamer_dir,
    config,
    data,
    task,
    logdir,
    episodes=5,
    checkpoint="best.pt",
):
    import torch
    from hydra import compose, initialize_config_dir

    r2dreamer_dir = Path(r2dreamer_dir)
    os.chdir(r2dreamer_dir)
    if str(r2dreamer_dir) not in sys.path:
        sys.path.insert(0, str(r2dreamer_dir))

    from dreamer import Dreamer
    from offline_replay import DMCExpertReplay
    from train import close_envs, evaluate_policy, make_eval_envs
    import tools

    logdir = Path(logdir)
    checkpoint_path = logdir / checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        return {
            "model": config,
            "checkpoint": "missing",
            "fresh_eval_score": None,
            "logged_final_eval": None,
            "logged_best_eval": None,
            "logdir": str(logdir),
        }

    overrides = [
        f"eval_episode_num={int(episodes)}",
        f"logdir={logdir}",
        f"expert_data.data_path={Path(data)}",
        f"env.task={task}",
    ]
    with initialize_config_dir(config_dir=str(r2dreamer_dir / "configs"), version_base=None):
        cfg = compose(config_name=config, overrides=overrides)

    replay = DMCExpertReplay(cfg.offline)
    try:
        obs_space = replay.obs_space()
        act_space = replay.act_space()
    finally:
        replay.close()
    agent = Dreamer(cfg.model, obs_space, act_space).to(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.clone_and_freeze()
    agent.eval()

    eval_envs = make_eval_envs(cfg)
    eval_logdir = logdir / "notebook_eval"
    eval_logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(eval_logdir)
    try:
        fresh_score = evaluate_policy(agent, eval_envs, logger, int(checkpoint.get("update", 0)))
    finally:
        close_envs(eval_envs)

    logged_step, logged_final, logged_best = logged_eval_summary(logdir)
    result = {
        "model": config,
        "checkpoint": checkpoint_path.name,
        "checkpoint_update": int(checkpoint.get("update", 0)),
        "fresh_eval_score": fresh_score,
        "logged_eval_step": logged_step,
        "logged_final_eval": logged_final,
        "logged_best_eval": logged_best,
        "logdir": str(logdir),
    }
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
