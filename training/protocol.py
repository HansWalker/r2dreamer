"""Checkpoint identity and completion rules for DMC experiments."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ENV = "DMC_IMPLEMENTATION_SHA256"


def implementation_sha256():
    """Hash the source and configs used by collection, training, and evaluation."""
    paths = [ROOT / name for name in ("buffer.py", "main.py", "tools.py", "train.py")]
    for directory in ("dmc_expert", "envs", "models", "optim", "training"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend(ROOT / "scripts" / name for name in ("collect_dmc_expert_data.py", "evaluate_dmc.py"))
    paths.extend((ROOT / "configs").rglob("*.yaml"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_sha256():
    packages = (
        "torch",
        "tensordict",
        "numpy",
        "hydra-core",
        "omegaconf",
        "gymnasium",
        "tensorboard",
        "h5py",
        "cloudpickle",
        "huggingface-hub",
        "dm-control",
        "mujoco",
        "mamba-ssm",
        "apache-tvm-ffi",
        "cuda-bindings",
        "cuda-core",
        "cuda-pathfinder",
        "cuda-python",
        "einops",
        "ml-dtypes",
        "triton",
        "tilelang",
        "torch-c-dlpack-ext",
        "transformers",
        "quack-kernels",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "z3-solver",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            distribution = importlib.metadata.distribution(package)
            versions[package] = {
                "version": distribution.version,
                "direct_url": distribution.read_text("direct_url.json"),
            }
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    encoded = json.dumps(versions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze_implementation():
    """Keep one orchestrated experiment on one immutable source snapshot."""
    actual = implementation_sha256()
    expected = os.environ.setdefault(IMPLEMENTATION_ENV, actual)
    if actual != expected:
        raise RuntimeError(
            "Python source changed after this experiment started. Restore the original source or "
            "start a new experiment directory instead of mixing implementations in one result matrix."
        )
    return actual


def model_variant(config):
    family = str(config.model_family)
    if family == "dreamer":
        variant = str(config.model.rssm.core)
        return "gru" if variant == "block_gru" else variant
    if family == "storm":
        variant = str(config.storm_model.sequence_core)
        return "mamba3" if variant == "mamba" else variant
    return "default"


def dynamics_targets_per_update(config):
    """Count recurrent state-prediction targets in one world-model update."""
    batch_size = int(config.replay.batch_size)
    sequence_length = int(config.replay.sequence_length)
    if str(config.model_family) == "dreamer":
        return batch_size * sequence_length
    return batch_size * (sequence_length - 1)


def _training_config(config):
    """Return resolved settings that can change the trained model."""
    data = OmegaConf.to_container(config, resolve=True)
    for key in ("device", "hydra", "logdir", "resume_from"):
        data.pop(key, None)
    for key in ("dataset_root", "device"):
        data.get("env", {}).pop(key, None)
    for key in ("device", "storage_device"):
        data.get("replay", {}).pop(key, None)
    data.get("training", {}).get("expert", {}).pop("data_path", None)

    evaluation = data.get("evaluation", {})
    data["evaluation"] = {
        key: evaluation[key] for key in ("success_threshold", "sustained_success_steps") if key in evaluation
    }
    return data


def run_identity(config):
    training_config = _training_config(config)
    encoded = json.dumps(training_config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "protocol": str(config.experiment_protocol),
        "model_family": str(config.model_family),
        "model_variant": model_variant(config),
        "scenario": str(config.scenario.name),
        "task": str(config.scenario.collection_task),
        "seed": int(config.seed),
        "implementation_sha256": freeze_implementation(),
        "runtime_sha256": runtime_sha256(),
        "training_config_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def family_recipe_sha256(config):
    """Hash a family recipe after removing only its recurrent-core selector."""
    data = _training_config(config)
    family = str(config.model_family)
    if family == "dreamer":
        data["model"]["rssm"]["core"] = "<recurrent-core>"
    elif family == "storm":
        data["storm_model"]["sequence_core"] = "<recurrent-core>"
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_training_recipe(config):
    """Reject configuration combinations that change or break the comparison protocol."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    family = str(config.model_family)
    batch_size = int(config.replay.batch_size)
    sequence_length = int(config.replay.sequence_length)
    episodes_per_batch = int(config.replay.episodes_per_batch)
    train_episodes = int(config.expert_data.train_episodes)
    heldout_episodes = int(config.expert_data.heldout_episodes)
    image_size = tuple(map(int, config.env.size))
    require(batch_size > 0, "replay.batch_size must be positive")
    require(sequence_length >= 2, "replay.sequence_length must contain at least two observations")
    require(int(config.replay.max_size) >= sequence_length, "replay.max_size must fit one training sequence")
    require(
        1 <= episodes_per_batch <= batch_size,
        "replay.episodes_per_batch must be between one and replay.batch_size",
    )
    require(
        episodes_per_batch > 0 and batch_size % episodes_per_batch == 0,
        "replay.batch_size must divide evenly across replay.episodes_per_batch",
    )
    require(
        train_episodes >= episodes_per_batch,
        "expert_data.train_episodes must fit one update's source episodes",
    )
    require(heldout_episodes >= 2, "expert_data.heldout_episodes must support disjoint probe splits")
    require(len(image_size) == 2 and image_size[0] == image_size[1], "expert datasets require square images")
    require(str(config.expert_data.policy) == "tdmpc2", "expert_data.policy must be tdmpc2")
    require(str(config.expert_data.policy_mode) == "mpc", "expert_data.policy_mode must be mpc")

    expert = config.training.expert
    if bool(expert.enabled):
        require(int(expert.updates) > 0, "enabled expert training requires a positive update budget")
        require(int(expert.batch_size) == batch_size, "expert and replay batch sizes must match")

    online = config.training.online
    online_steps = int(online.steps)
    online_updates = int(online.updates)
    require(
        (online_steps == 0) == (online_updates == 0),
        "online steps and updates must either both be zero or both be positive",
    )
    if online_steps:
        action_repeat = int(config.env.action_repeat)
        env_num = int(config.env.env_num)
        time_limit = int(config.env.time_limit)
        warmup = int(online.warmup_transitions)
        require(action_repeat > 0 and env_num > 0, "env.action_repeat and env.env_num must be positive")
        require(time_limit % action_repeat == 0, "env.time_limit must be divisible by env.action_repeat")
        require(
            online_steps % (env_num * action_repeat) == 0,
            "online steps must be divisible by env.env_num * env.action_repeat",
        )
        transitions = online_steps // action_repeat
        require(0 <= warmup < transitions, "online warmup must be shorter than online collection")
        require(warmup % env_num == 0, "online warmup must align with the parallel environment count")
        require(
            time_limit // action_repeat >= sequence_length,
            "an online episode must be long enough to produce one replay sequence",
        )

    if family == "dreamer":
        imagine_batch = int(config.actor_critic.imagine_batch_size)
        require(int(config.model.imag_horizon) >= 1, "Dreamer imagination horizon must be positive")
        require(imagine_batch % batch_size == 0, "Dreamer imagination batch must divide evenly over replay batches")
        require(
            imagine_batch // batch_size >= 2,
            "Dreamer needs at least two replay imagination starts per sampled sequence",
        )
    elif family == "storm":
        settings = config.storm_train
        require(int(settings.imagine_context_length) >= 2, "STORM imagination context must be at least two")
        require(int(settings.imagine_horizon) >= 1, "STORM imagination horizon must be positive")
        require(
            int(settings.imagine_batch_size) == int(config.actor_critic.imagine_batch_size),
            "STORM trainer and actor-critic imagination batch sizes must match",
        )
        if str(config.storm_model.sequence_core) == "transformer":
            required = max(
                sequence_length,
                int(settings.context_length),
                int(settings.imagine_context_length),
            )
            require(
                int(config.storm_model.recurrent.transformer.max_length) >= required,
                "STORM Transformer max_length is shorter than a configured context",
            )
    elif family == "tdmpc2":
        require(
            sequence_length == int(config.tdmpc2_model.horizon) + 1,
            "TD-MPC2 replay length must equal model horizon + 1",
        )
        planner = config.tdmpc2_model.planner
        samples, elites = int(planner.samples), int(planner.elites)
        require(1 <= elites <= samples, "TD-MPC2 planner elites must be between one and samples")
        require(
            0 <= int(planner.policy_samples) <= samples,
            "TD-MPC2 policy samples must be between zero and planner samples",
        )
        require(int(planner.horizon) > 0 and int(planner.iterations) > 0, "TD-MPC2 planner settings must be positive")
    elif family in {"leworldmodel", "temporal_straightening"}:
        require(
            sequence_length == int(config.jepa_model.history_size) + 1,
            "planning-model replay length must equal history_size + 1",
        )
        planner = config.jepa_model.planner
        require(
            int(planner.horizon) > 0 and int(planner.iterations) > 0,
            "planner horizon and iterations must be positive",
        )
        require(int(planner.samples) > 0, "planner samples must be positive")
        if str(planner.type) == "cem":
            require(
                2 <= int(planner.elites) <= int(planner.samples),
                "CEM requires at least two elites and no more elites than samples",
            )
    else:
        errors.append(f"unknown model family {family!r}")

    final = config.evaluation.final
    horizons = tuple(map(int, final.horizons))
    require(int(final.episodes) > 0, "final evaluation requires at least one episode")
    require(int(final.context_length) > 0, "state-prediction context length must be positive")
    require(
        bool(horizons) and all(horizon > 0 for horizon in horizons),
        "state-prediction horizons must be nonempty and positive",
    )
    require(
        int(final.probe_train_windows) > 0 and int(final.probe_test_windows) > 0 and int(final.probe_batch_size) > 0,
        "state-prediction probe sizes must be positive",
    )
    require(0 <= float(config.evaluation.success_threshold) <= 1, "success threshold must be in [0, 1]")
    require(int(config.evaluation.sustained_success_steps) > 0, "sustained success steps must be positive")
    if errors:
        raise ValueError("Invalid training recipe:\n- " + "\n- ".join(errors))


def comparison_signature(config):
    """Return quantities that must match for runs in one comparison matrix."""
    batch_size = int(config.replay.batch_size)
    sequence_length = int(config.replay.sequence_length)
    return {
        "protocol": str(config.experiment_protocol),
        "deterministic_run": bool(config.deterministic_run),
        "expert_enabled": bool(config.training.expert.enabled),
        "expert_updates": int(config.training.expert.updates),
        "expert_shuffle": bool(config.training.expert.shuffle),
        "world_model_observations_per_update": batch_size * sequence_length,
        "source_episodes_per_update": int(config.replay.episodes_per_batch),
        "expert_policy": str(config.expert_data.policy),
        "expert_policy_mode": str(config.expert_data.policy_mode),
        "expert_dataset_train_episodes": int(config.expert_data.train_episodes),
        "expert_dataset_heldout_episodes": int(config.expert_data.heldout_episodes),
        "online_steps": int(config.training.online.steps),
        "online_updates": int(config.training.online.updates),
        "online_warmup_transitions": int(config.training.online.warmup_transitions),
        "replay_capacity": int(config.replay.max_size),
        "environment_count": int(config.env.env_num),
        "action_repeat": int(config.env.action_repeat),
        "episode_time_limit": int(config.env.time_limit),
        "checkpoint_eval_episodes": int(config.env.eval_episode_num),
        "checkpoint_eval_seed": int(config.env.eval_seed),
        "expert_eval_every": int(config.training.expert.eval_every),
        "online_eval_every": int(config.training.online.eval_every),
        "success_threshold": float(config.evaluation.success_threshold),
        "sustained_success_steps": int(config.evaluation.sustained_success_steps),
        "final_eval_episodes": int(config.evaluation.final.episodes),
        "final_eval_seed": int(config.evaluation.final.seed),
        "prediction_context": int(config.evaluation.final.context_length),
        "prediction_horizons": tuple(map(int, config.evaluation.final.horizons)),
        "probe_train_windows": int(config.evaluation.final.probe_train_windows),
        "probe_test_windows": int(config.evaluation.final.probe_test_windows),
        "probe_batch_size": int(config.evaluation.final.probe_batch_size),
        "probe_ridge": float(config.evaluation.final.probe_ridge),
        "probe_seed": int(config.evaluation.final.probe_seed),
    }


def validate_checkpoint(checkpoint, config):
    expected = run_identity(config)
    if checkpoint.get("experiment_protocol") != expected["protocol"]:
        raise ValueError(
            f"Checkpoint protocol {checkpoint.get('experiment_protocol', 'legacy_unversioned')} "
            f"does not match {expected['protocol']}."
        )
    if checkpoint.get("checkpoint_id") is None:
        raise ValueError("Checkpoint has no unique identity. Start a fresh run with the current protocol.")
    actual = checkpoint.get("run_identity")
    if actual is None:
        raise ValueError("Checkpoint has no run identity. Start a fresh run with the current experiment protocol.")
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        details = ", ".join(f"{key}={actual.get(key)!r} (expected {expected[key]!r})" for key in mismatches)
        raise ValueError(f"Checkpoint belongs to a different experiment: {details}.")
    return expected


def training_complete(checkpoint, config):
    validate_checkpoint(checkpoint, config)
    state = checkpoint.get("trainer_state", {})
    expert_complete = not bool(config.training.expert.enabled) or int(checkpoint.get("expert_updates", -1)) >= int(
        config.training.expert.updates
    )
    online_steps = int(config.training.online.steps)
    if online_steps:
        return (
            expert_complete
            and checkpoint.get("phase") == "online"
            and int(state.get("env_steps", -1)) >= online_steps
            and int(state.get("world_model_updates", -1)) >= int(config.training.online.updates)
        )
    if bool(config.training.expert.enabled):
        return checkpoint.get("phase") == "expert" and expert_complete
    return False


def completion_checkpoint(config):
    if not int(config.training.online.steps) and not bool(config.training.expert.enabled):
        raise ValueError("The training recipe disables both expert and online training.")
    return "final.pt" if int(config.training.online.steps) else "pretrained.pt"


def evaluation_identity(config, state_prediction):
    settings = OmegaConf.to_container(config.evaluation.final, resolve=True)
    return {
        "protocol": str(config.experiment_protocol),
        "success_threshold": float(config.evaluation.success_threshold),
        "sustained_success_steps": int(config.evaluation.sustained_success_steps),
        "settings": settings,
        "state_prediction": bool(state_prediction),
    }
