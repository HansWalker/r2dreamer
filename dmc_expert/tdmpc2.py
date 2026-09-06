"""TD-MPC2 checkpoint and task integration for DMC expert collection."""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_REPO = "nicklashansen/tdmpc2"


def source_sha256(root: Path) -> str:
    """Hash the external TD-MPC2 source that interprets expert checkpoints."""
    package_root = Path(root).expanduser().resolve() / "tdmpc2"
    config_path = package_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing TD-MPC2 config: {config_path}")
    paths = sorted(
        path for path in package_root.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    slug: str

    @property
    def dmc_name(self) -> str:
        return f"{self.domain}/{self.task}"

    @property
    def store_name(self) -> str:
        return f"{self.domain}_{self.task}"


def tdmpc2_slug(domain: str, task: str) -> str:
    if domain == "ball_in_cup" and task == "catch":
        return "cup-catch"
    if domain == "point_mass":
        return f"pointmass-{task.replace('_', '-')}"
    return f"{domain}/{task}".replace("/", "-").replace("_", "-")


def discover_tasks() -> list[TaskSpec]:
    from dm_control import suite

    return [TaskSpec(domain, task, tdmpc2_slug(domain, task)) for domain, task in sorted(suite.ALL_TASKS)]


def select_tasks(all_tasks: list[TaskSpec], requested: list[str]) -> list[TaskSpec]:
    if requested == ["all"]:
        return all_tasks

    aliases = {}
    for task in all_tasks:
        aliases[task.dmc_name] = task
        aliases[f"{task.domain}_{task.task}"] = task
        aliases[task.slug] = task

    selected = []
    for name in requested:
        if name not in aliases:
            raise ValueError(f"Unknown DMC task: {name}")
        selected.append(aliases[name])
    return selected


def checkpoint_name(task: TaskSpec, seed: int) -> str:
    return f"dmcontrol/{task.slug}-{seed}.pt"


def download_checkpoint(task: TaskSpec, seed: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            repo_type="model",
            filename=checkpoint_name(task, seed),
        )
    )


def resolve_checkpoint(task: TaskSpec, seed: int, checkpoints=None) -> Path:
    checkpoints = checkpoints or {}
    if task.dmc_name not in checkpoints:
        return download_checkpoint(task, seed)
    custom = checkpoints[task.dmc_name]
    if not custom:
        raise ValueError(f"No local TD-MPC2 checkpoint is configured for {task.dmc_name}.")
    path = Path(custom).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing TD-MPC2 checkpoint for {task.dmc_name}: {path}")
    return path


def _add_to_path(root: Path) -> Path:
    package_root = Path(root).expanduser().resolve() / "tdmpc2"
    config_path = package_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing TD-MPC2 config: {config_path}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    return config_path


def _make_config(
    config_path: Path,
    task: TaskSpec,
    obs_dim: int,
    action_dim: int,
    collection,
):
    from common import MODEL_SIZE, TASK_SET
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    cfg.task = task.slug
    cfg.obs = "state"
    cfg.model_size = 5
    cfg.compile = False
    cfg.seed = collection.seed
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.checkpoint = ""
    cfg.work_dir = ""
    cfg.exp_name = "expert_collect"

    for key, value in MODEL_SIZE[cfg.model_size].items():
        cfg[key] = value

    cfg.multitask = cfg.task in TASK_SET
    cfg.task_dim = 0
    cfg.tasks = [cfg.task]
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.obs_shape = {"state": (obs_dim,)}
    cfg.action_dim = action_dim
    cfg.episode_length = collection.max_episode_steps
    cfg.obs_shapes = None
    cfg.action_dims = None
    cfg.episode_lengths = None

    cfg.mpc = bool(collection.expert["mpc"])
    cfg.iterations = int(collection.expert["iterations"])
    cfg.num_samples = int(collection.expert["num_samples"])
    cfg.num_elites = min(int(cfg.num_elites), int(cfg.num_samples))
    return cfg_to_dataclass(cfg)


def _move_key(state_dict, target_state_dict, old_key, new_key):
    if old_key in state_dict and new_key in target_state_dict:
        state_dict[new_key] = state_dict.pop(old_key)


def _move_mlp_layer(state_dict, target_state_dict, old_prefix, new_prefix, linear_idx, norm_idx):
    norm_weight = f"{old_prefix}.{norm_idx}.weight"
    mappings = (
        (f"{old_prefix}.{linear_idx}.weight", f"{new_prefix}.weight"),
        (f"{old_prefix}.{linear_idx}.bias", f"{new_prefix}.bias"),
        (norm_weight, f"{new_prefix}.ln.weight"),
        (f"{old_prefix}.{norm_idx}.bias", f"{new_prefix}.ln.bias"),
    )
    for old_key, new_key in mappings:
        _move_key(state_dict, target_state_dict, old_key, new_key)


def _convert_old_mlp(target_state_dict, source_state_dict):
    converted = dict(source_state_dict)

    if "_encoder.state.1.weight" in converted and "_encoder.state.1.ln.weight" not in converted:
        for idx in range(16):
            _move_mlp_layer(
                converted,
                target_state_dict,
                "_encoder.state",
                f"_encoder.state.{idx}",
                1 + 3 * idx,
                2 + 3 * idx,
            )

    if "_dynamics.0.0.weight" in converted:
        final_norm_weight = converted.pop("_dynamics.1.weight", None)
        final_norm_bias = converted.pop("_dynamics.1.bias", None)
        for idx in range(16):
            new_prefix = f"_dynamics.{idx}"
            if f"{new_prefix}.weight" not in target_state_dict:
                continue
            old_linear = 3 * idx
            old_norm = 3 * idx + 1
            has_old_norm = f"_dynamics.0.{old_norm}.weight" in converted
            _move_mlp_layer(
                converted,
                target_state_dict,
                "_dynamics.0",
                new_prefix,
                old_linear,
                old_norm,
            )
            if not has_old_norm and f"{new_prefix}.ln.weight" in target_state_dict:
                if final_norm_weight is not None:
                    converted[f"{new_prefix}.ln.weight"] = final_norm_weight
                if final_norm_bias is not None:
                    converted[f"{new_prefix}.ln.bias"] = final_norm_bias

    for prefix in ("_reward", "_pi", "_termination"):
        if f"{prefix}.1.weight" not in converted or f"{prefix}.0.ln.weight" in converted:
            continue
        for idx in range(16):
            new_prefix = f"{prefix}.{idx}"
            if f"{new_prefix}.weight" not in target_state_dict:
                continue
            _move_mlp_layer(
                converted,
                target_state_dict,
                prefix,
                new_prefix,
                3 * idx,
                3 * idx + 1,
            )
    return converted


def convert_state_dict(target_state_dict, source_state_dict):
    source_state_dict = _convert_old_mlp(target_state_dict, source_state_dict)
    if "_detach_Qs_params.0.weight" not in source_state_dict:
        names = ["weight", "bias", "ln.weight", "ln.bias"]
        converted = dict(source_state_dict)
        for key, value in list(source_state_dict.items()):
            if key.startswith("_Qs.params."):
                num = int(key[len("_Qs.params.") :])
                new_key = f"{num // 4}.{names[num % 4]}"
                converted.pop(key, None)
                converted[f"_Qs.params.{new_key}"] = value
                converted[f"_detach_Qs_params.{new_key}"] = value
            elif key.startswith("_target_Qs.params."):
                num = int(key[len("_target_Qs.params.") :])
                new_key = f"{num // 4}.{names[num % 4]}"
                converted.pop(key, None)
                converted[f"_target_Qs_params.{new_key}"] = value

        for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
            for key in ("__batch_size", "__device"):
                meta_key = f"{prefix}params.{key}"
                if meta_key in target_state_dict:
                    converted[meta_key] = target_state_dict[meta_key]
        for key in ("log_std_min", "log_std_dif", "_action_masks"):
            if key in target_state_dict:
                converted[key] = target_state_dict[key]
        source_state_dict = converted

    return {key: value for key, value in source_state_dict.items() if key in target_state_dict}


def _load_checkpoint(agent, checkpoint_path: Path):
    import torch

    state_dict = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    state_dict = state_dict.get("model", state_dict)
    target_state_dict = agent.model.state_dict()
    state_dict = convert_state_dict(target_state_dict, state_dict)
    incompatible = agent.model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.endswith((".__batch_size", ".__device"))]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Could not load TD-MPC2 checkpoint cleanly. "
            f"Missing keys: {missing}. Unexpected keys: {list(incompatible.unexpected_keys)}."
        )


def load_agent(
    collection,
    checkpoint_path: Path,
    task: TaskSpec,
    obs_dim: int,
    action_dim: int,
):
    cfg = _make_config(
        _add_to_path(collection.tdmpc2_root),
        task,
        obs_dim=obs_dim,
        action_dim=action_dim,
        collection=collection,
    )

    from tdmpc2 import TDMPC2

    agent = TDMPC2(cfg)
    _load_checkpoint(agent, checkpoint_path)
    agent.eval()
    return agent
