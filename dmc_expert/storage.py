"""Dense HDF5 storage for DMC expert episodes."""

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DATA_FORMAT = "dmc_expert_hdf5_dense_v1"
METADATA_KEYS = (
    "domain_name",
    "task_name",
    "policy",
    "policy_mode",
    "expert",
    "checkpoint_repo",
    "checkpoint_path",
    "checkpoint_sha256",
    "tdmpc2_source_sha256",
    "collector_sha256",
    "collector_runtime",
    "checkpoint_seed",
    "obs_dim",
    "action_dim",
    "observation_keys",
    "observation_shapes",
    "max_episode_steps",
    "image_size",
    "action_repeat",
    "time_limit",
    "seed",
    "episode_seed_rule",
    "episode_splits",
    "action_min",
    "action_max",
    "raw_action_min",
    "raw_action_max",
    "goal_relation",
)


def dataset_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Identify the task and collection recipe without machine-local paths."""
    values = {
        "format": metadata.get("format"),
        "num_episodes": metadata.get("num_episodes"),
        **{key: metadata.get(key) for key in METADATA_KEYS if key != "checkpoint_path"},
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return {
        "task": f"{metadata.get('domain_name')}/{metadata.get('task_name')}",
        "dataset_id": metadata.get("dataset_id", "legacy"),
        "collection_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_dataset_task(metadata: dict[str, Any], expected_task: str):
    actual = f"{metadata.get('domain_name')}/{metadata.get('task_name')}"
    if actual != str(expected_task):
        raise ValueError(f"Dataset contains {actual!r}, but this run requires {str(expected_task)!r}.")


def validate_dataset_splits(metadata: dict[str, Any], train_episodes: int, heldout_episodes: int):
    expected = {
        "train": [0, int(train_episodes)],
        "heldout": [int(train_episodes), int(train_episodes) + int(heldout_episodes)],
    }
    actual = metadata.get("episode_splits")
    total = int(metadata.get("num_episodes", -1))
    if actual != expected or total != expected["heldout"][1]:
        raise ValueError(
            f"Dataset has num_episodes={total} and episode_splits={actual!r}, "
            f"but this experiment requires {expected!r}."
        )


def validate_dataset_protocol(
    metadata: dict[str, Any],
    *,
    task: str,
    train_episodes: int,
    heldout_episodes: int,
    action_repeat: int,
    max_episode_steps: int,
    image_size: int,
    policy: str = "tdmpc2",
    policy_mode: str = "mpc",
):
    """Reject expert data collected under a different experiment protocol."""
    if metadata.get("format") != DATA_FORMAT:
        raise ValueError(f"Dataset uses format={metadata.get('format')!r}; expected {DATA_FORMAT!r}.")
    validate_dataset_task(metadata, task)
    validate_dataset_splits(metadata, train_episodes, heldout_episodes)

    expected = {
        "policy": str(policy),
        "policy_mode": str(policy_mode),
        "action_repeat": int(action_repeat),
        "max_episode_steps": int(max_episode_steps),
        "image_size": int(image_size),
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    action_dim = int(metadata.get("action_dim", 0))
    action_min = np.asarray(metadata.get("action_min", []), dtype=np.float32).reshape(-1)
    action_max = np.asarray(metadata.get("action_max", []), dtype=np.float32).reshape(-1)
    normalized_actions = (
        action_dim > 0
        and action_min.size in (1, action_dim)
        and action_max.size in (1, action_dim)
        and np.allclose(action_min, -1.0)
        and np.allclose(action_max, 1.0)
    )
    if mismatches or not normalized_actions:
        details = [f"{key}={actual!r} (expected {value!r})" for key, (actual, value) in mismatches.items()]
        if not normalized_actions:
            details.append("stored actions are not normalized to [-1, 1]")
        raise ValueError("Dataset collection protocol mismatch: " + ", ".join(details) + ".")


def validate_dataset_storage(path: Path, metadata: dict[str, Any], splits=("train", "heldout")):
    """Verify that requested episodes are complete and match the dense HDF5 schema."""
    path = Path(path).expanduser()
    data_path = path / "data.hdf5"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing expert dataset: {data_path}")

    episodes = int(metadata["num_episodes"])
    steps = int(metadata["max_episode_steps"])
    image_size = int(metadata["image_size"])
    expected = {
        "observations": ((episodes, steps + 1, int(metadata["obs_dim"])), np.float32),
        "actions": ((episodes, steps, int(metadata["action_dim"])), np.float32),
        "rewards": ((episodes, steps, 1), np.float32),
        "discounts": ((episodes, steps, 1), np.float32),
        "terminations": ((episodes, steps, 1), np.uint8),
        "truncations": ((episodes, steps, 1), np.uint8),
        "images": ((episodes, steps + 1, image_size, image_size, 3), np.uint8),
        "lengths": ((episodes,), np.int32),
        "returns": ((episodes,), np.float32),
        "complete": ((episodes,), np.uint8),
    }
    goal = metadata.get("goal_relation")
    if goal:
        expected["goal_relations"] = (
            (episodes, steps + 1, *map(int, goal["shape"])),
            np.float32,
        )

    with h5py.File(data_path, "r") as h5:
        for name, (shape, dtype) in expected.items():
            if name not in h5:
                raise ValueError(f"{data_path} is missing required array {name!r}.")
            if h5[name].shape != tuple(shape) or h5[name].dtype != np.dtype(dtype):
                raise ValueError(
                    f"{data_path}:{name} has shape/dtype {h5[name].shape}/{h5[name].dtype}, "
                    f"expected {tuple(shape)}/{np.dtype(dtype)}."
                )

        selected = np.concatenate([split_episode_indices(metadata, name, episodes) for name in splits])
        complete = np.asarray(h5["complete"][selected])
        if np.any(complete != 1):
            raise ValueError(f"{path} is missing {int(np.count_nonzero(complete != 1))} requested episodes.")

        lengths = np.asarray(h5["lengths"][selected], dtype=np.int64)
        invalid = (lengths < 1) | (lengths > steps)
        if np.any(invalid):
            raise ValueError(f"{path} has {int(invalid.sum())} invalid requested episode lengths.")
        if not np.isfinite(np.asarray(h5["returns"][selected], dtype=np.float32)).all():
            raise ValueError(f"{path} contains a non-finite requested episode return.")

        terminal = np.asarray(h5["terminations"][selected, :, 0], dtype=bool)
        truncated = np.asarray(h5["truncations"][selected, :, 0], dtype=bool)
        ending = terminal | truncated
        valid_steps = np.arange(steps)[None] < lengths[:, None]
        final = ending[np.arange(len(selected)), lengths - 1]
        if np.any(terminal & truncated) or np.any((ending & valid_steps).sum(axis=1) != 1) or not final.all():
            raise ValueError(f"{path} has invalid termination/truncation markers in the requested episodes.")


def validate_dataset(path: Path, config, splits=("train", "heldout")) -> dict[str, Any]:
    """Load and validate one dataset against a resolved training config."""
    path = Path(path).expanduser()
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing expert dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_dataset_protocol(
        metadata,
        task=config.scenario.collection_task,
        train_episodes=config.expert_data.train_episodes,
        heldout_episodes=config.expert_data.heldout_episodes,
        action_repeat=config.env.action_repeat,
        max_episode_steps=int(config.env.time_limit) // int(config.env.action_repeat),
        image_size=config.env.size[0],
        policy=config.expert_data.policy,
        policy_mode=config.expert_data.policy_mode,
    )
    validate_dataset_storage(path, metadata, splits)
    return metadata


def observation_indices(metadata: dict[str, Any], selection) -> np.ndarray:
    """Map named DMC observation coordinates onto the stored flat state."""
    layout = {}
    offset = 0
    for key in metadata["observation_keys"]:
        size = int(np.prod(metadata["observation_shapes"][key]))
        layout[key] = (offset, size)
        offset += size

    selected = []
    for key, indices in selection.items():
        start, size = layout[str(key)]
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any((indices < 0) | (indices >= size)):
            raise ValueError(f"Observation selection {key}={indices.tolist()} exceeds size {size}.")
        selected.extend((start + indices).tolist())
    return np.asarray(selected, dtype=np.int64)


def read_image_window(images, episode: int, start: int, length: int, frame_stack: int = 1) -> np.ndarray:
    """Read one causal image window, repeating the first frame for missing history."""
    if frame_stack == 1:
        return np.asarray(images[episode, start : start + length])
    prefix_start = max(0, start - frame_stack + 1)
    frames = np.asarray(images[episode, prefix_start : start + length])
    missing = frame_stack - 1 - (start - prefix_start)
    if missing:
        frames = np.concatenate((np.repeat(frames[:1], missing, axis=0), frames))
    return np.concatenate(
        [frames[offset : offset + length] for offset in range(frame_stack)],
        axis=-1,
    )


def _check_metadata(existing: dict[str, Any], requested: dict[str, Any]):
    missing = [key for key in METADATA_KEYS if key in requested and key not in existing]
    if missing:
        raise RuntimeError(
            "Existing DMC expert metadata cannot be resumed safely because it is missing "
            f"{', '.join(missing)}. Collect a fresh dataset instead."
        )
    if int(existing.get("num_episodes", -1)) != int(requested["num_episodes"]):
        raise RuntimeError(
            f"Existing DMC expert data has {existing.get('num_episodes')!r} episodes, "
            f"but this collection requests {requested['num_episodes']}. Collect a fresh dataset instead."
        )
    mismatches = [
        key for key in METADATA_KEYS if key in existing and key in requested and existing[key] != requested[key]
    ]
    if mismatches:
        details = ", ".join(f"{key}={existing[key]!r} (requested {requested[key]!r})" for key in mismatches)
        raise RuntimeError(f"Existing DMC expert metadata is incompatible: {details}.")


def open_dataset(path: Path, metadata: dict[str, Any], resume: bool):
    path = Path(path)
    metadata = dict(metadata)
    if path.exists() and not resume:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "metadata.json"
    data_path = path / "data.hdf5"
    if resume and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("format") != DATA_FORMAT:
            raise RuntimeError(
                f"{path} uses format={existing.get('format')!r}, expected {DATA_FORMAT!r}. "
                "Convert or delete the old dataset before collecting with the dense HDF5 format."
            )
        _check_metadata(existing, metadata)
        if "dataset_id" not in existing:
            existing["dataset_id"] = uuid.uuid4().hex
            metadata_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        metadata = existing
    else:
        metadata["dataset_id"] = uuid.uuid4().hex
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    h5 = h5py.File(data_path, "a")
    ensure_arrays(h5, metadata)
    return h5, metadata_path, data_path


def split_episode_indices(metadata: dict[str, Any], name: str, total: int) -> np.ndarray:
    """Return one configured episode range."""
    splits = metadata.get("episode_splits", {})
    if name not in splits or len(splits[name]) != 2:
        raise ValueError(f"Dataset metadata has no valid {name!r} episode split.")
    start, stop = map(int, splits[name])
    if not 0 <= start < stop <= total:
        raise ValueError(f"Dataset {name!r} split [{start}, {stop}) exceeds {total} episodes.")
    return np.arange(start, stop, dtype=np.int64)


def _require_array(h5, name: str, shape: tuple[int, ...], dtype, chunks: tuple[int, ...], **options):
    if name not in h5:
        return h5.create_dataset(
            name,
            shape=shape,
            maxshape=(None, *shape[1:]),
            chunks=chunks,
            dtype=dtype,
            **options,
        )
    dataset = h5[name]
    if dataset.shape != shape:
        raise RuntimeError(
            f"Existing HDF5 array {name!r} has shape {dataset.shape}, expected {shape}. "
            "Start a fresh dataset or convert the existing data."
        )
    if dataset.dtype != np.dtype(dtype):
        raise RuntimeError(f"Existing HDF5 array {name!r} has dtype {dataset.dtype}, expected {np.dtype(dtype)}.")
    return dataset


def ensure_arrays(h5, metadata: dict[str, Any]):
    episodes = int(metadata["num_episodes"])
    steps = int(metadata["max_episode_steps"])
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    obs_chunk = min(steps + 1, 128)
    step_chunk = min(steps, 128)

    _require_array(
        h5,
        "observations",
        (episodes, steps + 1, obs_dim),
        np.float32,
        (1, obs_chunk, obs_dim),
    )
    _require_array(
        h5,
        "actions",
        (episodes, steps, action_dim),
        np.float32,
        (1, step_chunk, action_dim),
    )
    for name, dtype in (
        ("rewards", np.float32),
        ("discounts", np.float32),
        ("terminations", np.uint8),
        ("truncations", np.uint8),
    ):
        _require_array(h5, name, (episodes, steps, 1), dtype, (1, step_chunk, 1))
    image_size = int(metadata["image_size"])
    _require_array(
        h5,
        "images",
        (episodes, steps + 1, image_size, image_size, 3),
        np.uint8,
        (1, min(8, steps + 1), image_size, image_size, 3),
        compression="lzf",
        shuffle=True,
    )
    goal_relation = metadata.get("goal_relation")
    if goal_relation:
        shape = tuple(map(int, goal_relation["shape"]))
        _require_array(
            h5,
            "goal_relations",
            (episodes, steps + 1, *shape),
            np.float32,
            (1, obs_chunk, *shape),
        )
    _require_array(h5, "lengths", (episodes,), np.int32, (min(1024, episodes),))
    _require_array(h5, "returns", (episodes,), np.float32, (min(1024, episodes),))
    _require_array(h5, "complete", (episodes,), np.uint8, (min(1024, episodes),))


def completed_episodes(h5) -> int:
    complete = np.asarray(h5["complete"], dtype=bool)
    missing = np.flatnonzero(~complete)
    return int(missing[0]) if missing.size else int(complete.shape[0])


def write_progress(path: Path, episodes: int, rows: int, target: int):
    payload = {
        "episodes": int(episodes),
        "rows": int(rows),
        "target_episodes": int(target),
    }
    (Path(path) / "progress.json").write_text(json.dumps(payload), encoding="utf-8")


def append_episode(h5, episode_idx: int, episode: dict[str, np.ndarray], episode_return: float):
    h5["complete"][episode_idx] = 0
    length = int(episode["actions"].shape[0])
    h5["observations"][episode_idx, : length + 1] = episode["observations"]
    h5["actions"][episode_idx, :length] = episode["actions"]
    h5["rewards"][episode_idx, :length] = episode["rewards"]
    h5["discounts"][episode_idx, :length] = episode["discounts"]
    h5["terminations"][episode_idx, :length] = episode["terminations"]
    h5["truncations"][episode_idx, :length] = episode["truncations"]
    h5["images"][episode_idx, : length + 1] = episode["images"]
    if "goal_relations" in h5:
        h5["goal_relations"][episode_idx, : length + 1] = episode["goal_relations"]
    h5["lengths"][episode_idx] = length
    h5["returns"][episode_idx] = float(episode_return)
    h5.flush()
    h5["complete"][episode_idx] = 1
    h5.flush()
