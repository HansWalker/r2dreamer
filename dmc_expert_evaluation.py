"""Single-run notebook helpers for offline DMC expert evaluation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_metrics(logdir: Path):
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


def logged_eval_summary(logdir: Path):
    eval_rows = [row for row in load_metrics(logdir) if "episode/eval_score" in row]
    if not eval_rows:
        return None, None, None
    final = eval_rows[-1]
    best = max(eval_rows, key=lambda row: row["episode/eval_score"])
    return final.get("step"), final["episode/eval_score"], best["episode/eval_score"]


def newest_run(workdir: Path, pattern: str):
    runs = list((Path(workdir) / "runs").glob(pattern))
    if not runs:
        raise FileNotFoundError(f"No run folders match: {pattern}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def default_evaluation_runs(workdir: Path):
    from dmc_expert_training import DMC_EXPERT_MODELS, DMC_EXPERT_SCENARIOS

    data_dir = Path(workdir) / "data" / "dmc_expert"
    runs = []
    for scenario in DMC_EXPERT_SCENARIOS:
        for model in DMC_EXPERT_MODELS:
            runs.append(
                {
                    "name": f"{scenario['name']}_{model['name']}",
                    "scenario": scenario["name"],
                    "model": model["name"],
                    "config_name": model["config_name"],
                    "data_path": data_dir / scenario["data_rel"],
                    "env_task": scenario["env_task"],
                    "logdir": newest_run(
                        workdir,
                        f"offline_{scenario['name']}_{model['name']}_fresh_*",
                    ),
                }
            )
    return runs


def evaluate_training_run(
    run: dict,
    *,
    r2dreamer_dir: Path,
    train_store: Path | None = None,
    env_task: str | None = None,
    eval_episodes: int = 5,
    checkpoint_name: str = "best.pt",
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

    logdir = Path(run["logdir"])
    train_store = train_store if train_store is not None else run.get("data_path")
    env_task = env_task if env_task is not None else run.get("env_task")
    checkpoint_path = logdir / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        return {
            "model": run["name"],
            "checkpoint": "missing",
            "fresh_eval_score": None,
            "logged_final_eval": None,
            "logged_best_eval": None,
            "logdir": str(logdir),
        }

    overrides = [
        f"offline.eval_episode_num={int(eval_episodes)}",
        f"logdir={logdir}",
    ]
    if train_store is not None:
        overrides.append(f"offline.data_path={Path(train_store)}")
    if env_task is not None:
        overrides.append(f"env.task={env_task}")
    with initialize_config_dir(config_dir=str(r2dreamer_dir / "configs"), version_base=None):
        cfg = compose(config_name=run["config_name"], overrides=overrides)

    replay = DMCExpertReplay(cfg.offline)
    agent = Dreamer(cfg.model, replay.obs_space(), replay.act_space()).to(cfg.device)
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
        "model": run["name"],
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
