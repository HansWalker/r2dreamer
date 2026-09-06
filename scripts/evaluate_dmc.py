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
    parser.add_argument("--scenario", required=True, help="Scenario config name, such as cartpole_balance_sparse.")
    parser.add_argument("--logdir", type=Path, required=True, help="Training run directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="Held-out HDF5 dataset for prediction metrics.")
    parser.add_argument("--device", help="Optional device override, such as cuda:0.")
    parser.add_argument("--checkpoint", help="Checkpoint filename; defaults to evaluation.final.checkpoint.")
    parser.add_argument("--skip-state-prediction", action="store_true", help="Evaluate policy metrics only.")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override; repeat as needed.")
    parser.add_argument("--output", type=Path, help="JSON output path; defaults to <logdir>/evaluation.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import tools
    from dmc_expert.storage import dataset_identity, validate_dataset
    from envs import close_envs, make_eval_envs
    from training import load_model_family
    from training.evaluation import evaluate_state_prediction
    from training.protocol import (
        completion_checkpoint,
        dynamics_targets_per_update,
        evaluation_identity,
        freeze_implementation,
        training_complete,
        validate_checkpoint,
        validate_training_recipe,
    )

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
    freeze_implementation()
    validate_training_recipe(config)
    settings = config.evaluation.final

    checkpoint_path = logdir / str(args.checkpoint or settings.checkpoint)
    horizons = tuple(map(int, settings.horizons))
    print(
        f"Evaluation | checkpoint={checkpoint_path.name} | episodes={settings.episodes} | "
        f"context={settings.context_length} | horizons={','.join(map(str, horizons))}"
    )
    seed = int(settings.seed)
    tools.configure_randomness(seed, bool(config.deterministic_run))
    family = load_model_family(str(config.model_family))
    model = family.build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = validate_checkpoint(checkpoint, config)
    if checkpoint_path.name == completion_checkpoint(config) and not training_complete(checkpoint, config):
        raise ValueError(f"{checkpoint_path.name} does not contain a completed training run.")
    if bool(config.training.expert.enabled) and int(checkpoint.get("expert_updates", -1)) < int(
        config.training.expert.updates
    ):
        raise ValueError("Checkpoint was saved before the configured expert-training budget completed.")
    if int(config.training.online.steps) and checkpoint.get("phase") != "online":
        raise ValueError("Online evaluation requires a checkpoint saved during online training.")
    checkpoint_protocol = identity["protocol"]

    metadata = validate_dataset(dataset_path, config, splits=("heldout",))
    evaluation_dataset = dataset_identity(metadata)
    training_dataset = checkpoint.get("dataset_identity")
    if bool(config.training.expert.enabled) and training_dataset != evaluation_dataset:
        raise ValueError("Evaluation dataset does not match the expert dataset recorded in the checkpoint.")

    family.load_checkpoint(model, checkpoint, training=False)
    config.env.eval_episode_num = int(settings.episodes)
    config.env.eval_seed = int(settings.seed)
    tools.configure_randomness(seed, bool(config.deterministic_run))

    goal_conditioned = bool(getattr(model, "goal_conditioned", False))
    goal_spec = OmegaConf.to_container(config.jepa_model.goal, resolve=True) if goal_conditioned else None
    print("Evaluation | policy_rollout=running")
    envs = make_eval_envs(config.env)
    try:
        episode_return, episode_length, extra = family.evaluate(config, model, envs)
    finally:
        close_envs(envs)

    prediction = None
    if not args.skip_state_prediction:
        print(f"Evaluation | state_prediction=running | dataset={dataset_path}")
        tools.configure_randomness(int(settings.probe_seed), bool(config.deterministic_run))
        prediction = evaluate_state_prediction(
            model,
            config,
            dataset_path,
            metadata,
        )

    trainer_state = checkpoint.get("trainer_state", {})
    expert_updates = int(checkpoint.get("expert_updates", 0))
    online_updates = int(trainer_state.get("world_model_updates", 0))
    world_model_observations_per_update = int(config.replay.batch_size) * int(config.replay.sequence_length)
    dynamics_targets = dynamics_targets_per_update(config)
    result = {
        "experiment_protocol": checkpoint_protocol,
        "evaluation_protocol": str(config.experiment_protocol),
        "run_identity": identity,
        "evaluation_identity": evaluation_identity(config, prediction is not None),
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "config": args.config_name,
        "model_family": str(config.model_family),
        "scenario": args.scenario,
        "checkpoint": str(checkpoint_path),
        "checkpoint_phase": checkpoint.get("phase"),
        "expert_updates": expert_updates,
        "expert_sampled_observations": expert_updates * world_model_observations_per_update,
        "expert_dynamics_targets": expert_updates * dynamics_targets,
        "environment_steps": int(trainer_state.get("env_steps", 0)),
        "online_updates": online_updates,
        "online_world_model_observations": online_updates * world_model_observations_per_update,
        "online_dynamics_targets": online_updates * dynamics_targets,
        "world_model_observations_per_update": world_model_observations_per_update,
        "dynamics_targets_per_update": dynamics_targets,
        "replay_capacity": int(config.replay.max_size),
        "online_warmup_transitions": int(config.training.online.warmup_transitions),
        "training_seed": int(checkpoint.get("training_seed", config.seed)),
        "evaluation_episodes": int(settings.episodes),
        "evaluation_seed_start": seed,
        "evaluation_seed_end": seed + int(settings.episodes) - 1,
        "mean_return": float(episode_return),
        "return_std": float(extra["return_std"]),
        "return_standard_error": float(extra["return_stderr"]),
        "mean_episode_length": float(episode_length),
        "task_success_rate": float(extra["success"]),
        "sustained_success_rate": float(extra["sustained_success"]),
        "success_threshold": float(config.evaluation.success_threshold),
        "sustained_success_steps": int(config.evaluation.sustained_success_steps),
        "task_success_definition": "fraction of episodes that reach the normalized DMC reward threshold once",
        "sustained_success_definition": "fraction that remain above the threshold for consecutive agent steps",
        "goal_conditioned": goal_conditioned,
        "goal_definition": (
            "fixed DMC task-relative success region predicted from latent state" if goal_conditioned else None
        ),
        "goal_spec": goal_spec,
        "physical_state_prediction": prediction,
        "state_prediction_evaluated": prediction is not None,
        "dataset": str(dataset_path),
        "dataset_identity": evaluation_dataset,
        "dataset_role": "held_out",
    }
    output = _path(args.output).resolve() if args.output else logdir / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    state_metric = "-" if prediction is None else f"{prediction['mean_nrmse']:.3f}"
    print(
        f"Result | return={result['mean_return']:.2f} +/- {result['return_standard_error']:.2f} | "
        f"success={100 * result['task_success_rate']:.1f}% | "
        f"sustained={100 * result['sustained_success_rate']:.1f}% | "
        f"state_nrmse={state_metric} | episode_length={result['mean_episode_length']:.1f}"
    )
    print(f"Output | evaluation={output}")


if __name__ == "__main__":
    main()
