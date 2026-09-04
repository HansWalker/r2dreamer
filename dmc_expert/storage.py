"""Dense HDF5 storage for DMC expert episodes."""

import json
import shutil
from pathlib import Path
from typing import Any

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


def _check_metadata(existing: dict[str, Any], requested: dict[str, Any]):
    mismatches = [
        key for key in METADATA_KEYS if key in existing and key in requested and existing[key] != requested[key]
    ]
    if mismatches:
        details = ", ".join(f"{key}={existing[key]!r} (requested {requested[key]!r})" for key in mismatches)
        raise RuntimeError(f"Existing DMC expert metadata is incompatible: {details}.")
    if requested.get("goal_relation") != existing.get("goal_relation"):
        raise RuntimeError(
            "Existing DMC expert data does not contain the requested goal-relation labels. "
            "Collect a fresh dataset for task-relative planning."
        )


def open_dataset(path: Path, metadata: dict[str, Any], resume: bool):
    import h5py

    path = Path(path)
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
        changed = False
        if int(metadata["num_episodes"]) > int(existing.get("num_episodes", 0)):
            existing["num_episodes"] = int(metadata["num_episodes"])
            changed = True
        if "episode_splits" not in existing and "episode_splits" in metadata:
            existing["episode_splits"] = metadata["episode_splits"]
            changed = True
        if changed:
            metadata_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        metadata = existing
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    h5 = h5py.File(data_path, "a")
    ensure_arrays(h5, metadata)
    return h5, metadata_path, data_path


def split_episode_indices(metadata: dict[str, Any], name: str, total: int) -> np.ndarray:
    """Return the configured episode range, with legacy datasets treated as training-only."""
    splits = metadata.get("episode_splits")
    if not splits:
        if name == "train":
            return np.arange(total, dtype=np.int64)
        raise ValueError("Dataset metadata has no held-out episode split.")
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
    if dataset.shape[1:] != shape[1:]:
        raise RuntimeError(
            f"Existing HDF5 array {name!r} has shape {dataset.shape}, expected (*, {shape[1:]}). "
            "Start a fresh dataset or convert the existing data."
        )
    if dataset.dtype != np.dtype(dtype):
        raise RuntimeError(f"Existing HDF5 array {name!r} has dtype {dataset.dtype}, expected {np.dtype(dtype)}.")
    if dataset.shape[0] < shape[0]:
        dataset.resize((shape[0], *dataset.shape[1:]))
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
    h5["complete"][episode_idx] = 1
