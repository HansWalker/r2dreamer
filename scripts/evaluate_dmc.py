"""Evaluate a DMC checkpoint with return, success, and physical prediction error."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Hydra training config used by the checkpoint.")
    parser.add_argument("--scenario", required=True, help="Scenario config name, such as point_mass.")
    parser.add_argument("--logdir", type=Path, required=True, help="Training run directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="Held-out HDF5 dataset for prediction metrics.")
    parser.add_argument("--allow-training-dataset", action="store_true", help="Allow an explicitly in-sample probe.")
    parser.add_argument("--device", help="Optional device override, such as cuda:0.")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override; repeat as needed.")
    parser.add_argument("--output", type=Path, help="JSON output path; defaults to <logdir>/evaluation.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import tools
    from envs import close_envs, make_eval_envs
    from training import load_model_family
    from training.evaluation import evaluate_state_prediction

    logdir = _path(args.logdir).resolve()
    dataset_path = _path(args.dataset).resolve()

    overrides = [
        *args.override,
        f"scenario={args.scenario}",
        f"logdir={logdir}",
    ]
    if args.device:
        overrides.append(f"device={args.device}")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(config)
    settings = config.evaluation.final
    config.env.eval_episode_num = int(settings.episodes)
    config.env.eval_seed = int(settings.seed)

    checkpoint_path = logdir / str(settings.checkpoint)
    horizons = tuple(map(int, settings.horizons))
    print(
        f"Evaluation | checkpoint={checkpoint_path.name} | episodes={settings.episodes} | "
        f"context={settings.context_length} | horizons={','.join(map(str, horizons))}"
    )
    seed = int(config.env.eval_seed)
    tools.configure_randomness(seed, bool(config.deterministic_run))
    family = load_model_family(str(config.model_family))
    model = family.build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    family.load_checkpoint(model, checkpoint, training=False)
    tools.configure_randomness(seed, bool(config.deterministic_run))

    goal_conditioned = bool(getattr(model, "goal_conditioned", False))
    goal_spec = OmegaConf.to_container(config.jepa_model.goal, resolve=True) if goal_conditioned else None
    print("Evaluation | policy_rollout=running")
    envs = make_eval_envs(config.env)
    try:
        episode_return, episode_length, extra = family.evaluate(config, model, envs)
    finally:
        close_envs(envs)

    training_dataset = _path(str(config.training.expert.data_path)).resolve()
    if dataset_path == training_dataset and not args.allow_training_dataset:
        raise ValueError("Prediction metrics require held-out data; pass a different --dataset path.")
    print(f"Evaluation | state_prediction=running | dataset={dataset_path}")
    tools.configure_randomness(int(settings.probe_seed), bool(config.deterministic_run))
    prediction = evaluate_state_prediction(
        model,
        config.model_io,
        dataset_path,
        proprio=config.env.proprio,
        context_length=int(settings.context_length),
        horizons=horizons,
        train_windows=int(settings.probe_train_windows),
        test_windows=int(settings.probe_test_windows),
        batch_size=int(settings.probe_batch_size),
        ridge=float(settings.probe_ridge),
        seed=int(settings.probe_seed),
    )

    trainer_state = checkpoint.get("trainer_state", {})
    result = {
        "config": args.config_name,
        "model_family": str(config.model_family),
        "scenario": args.scenario,
        "checkpoint": str(checkpoint_path),
        "checkpoint_phase": checkpoint.get("phase"),
        "expert_updates": int(checkpoint.get("expert_updates", 0)),
        "environment_steps": int(trainer_state.get("env_steps", 0)),
        "training_seed": int(checkpoint.get("training_seed", config.seed)),
        "evaluation_episodes": int(settings.episodes),
        "evaluation_seed_start": seed,
        "evaluation_seed_end": seed + int(settings.episodes) - 1,
        "mean_return": float(episode_return),
        "return_std": float(extra["return_std"]),
        "return_standard_error": float(extra["return_stderr"]),
        "mean_episode_length": float(episode_length),
        "task_success_rate": float(extra["success"]),
        "success_threshold": float(config.evaluation.success_threshold),
        "task_success_definition": "fraction of episodes that reach the normalized DMC reward threshold",
        "goal_conditioned": goal_conditioned,
        "goal_definition": (
            "fixed DMC task-relative success region predicted from latent state" if goal_conditioned else None
        ),
        "goal_spec": goal_spec,
        "physical_state_prediction": prediction,
        "dataset": str(dataset_path),
        "dataset_role": "training" if dataset_path == training_dataset else "held_out",
    }
    output = _path(args.output).resolve() if args.output else logdir / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        f"Result | return={result['mean_return']:.2f} +/- {result['return_standard_error']:.2f} | "
        f"success={100 * result['task_success_rate']:.1f}% | "
        f"state_nrmse={prediction['mean_nrmse']:.3f} | episode_length={result['mean_episode_length']:.1f}"
    )
    print(f"Output | evaluation={output}")


if __name__ == "__main__":
    main()
