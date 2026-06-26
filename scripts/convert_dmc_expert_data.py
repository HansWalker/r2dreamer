"""Convert old zarr DMC expert datasets to the dense HDF5 format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_dmc_expert_data import (  # noqa: E402
    DATA_FORMAT,
    append_episode,
    ensure_arrays,
    write_progress,
)


OLD_ZARR_FORMAT = "dmc_expert_interleaved_episodes_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert old zarr DMC expert datasets to dense HDF5."
    )
    parser.add_argument("--input", required=True, type=Path, help="Old zarr dataset path.")
    parser.add_argument("--output", required=True, type=Path, help="New dataset path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output path before conversion if it already exists.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Convert at most this many complete episodes.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def open_zarr(path: Path):
    import zarr

    root = zarr.open(str(path), mode="r")
    if "obs" not in root or "episode_start" not in root or "episode_length" not in root:
        raise RuntimeError(f"{path} is not an old DMC expert zarr dataset.")
    if root.attrs.get("format") not in (None, OLD_ZARR_FORMAT):
        raise RuntimeError(
            f"{path} uses format={root.attrs.get('format')!r}; expected {OLD_ZARR_FORMAT!r}."
        )
    return root


def dense_metadata(metadata: dict[str, Any], episode_count: int, max_steps: int) -> dict[str, Any]:
    out = dict(metadata)
    out["format"] = DATA_FORMAT
    out["num_episodes"] = int(episode_count)
    out["max_episode_steps"] = int(max_steps)
    out["layout"] = (
        "episode-major dense arrays; observations[e, t], "
        "actions[e, t] -> observations[e, t + 1]"
    )
    out["data_file"] = "data.hdf5"
    out.setdefault("image_size", None)
    return out


def prepare_output(path: Path, metadata: dict[str, Any], overwrite: bool):
    import h5py

    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output already exists: {path}. Use --overwrite to replace it.")
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True)
    write_json(path / "metadata.json", metadata)
    h5 = h5py.File(path / "data.hdf5", "w")
    ensure_arrays(h5, metadata)
    return h5


def zarr_metadata(root, episode_count: int) -> tuple[dict[str, Any], int]:
    lengths = np.asarray(root["episode_length"][:episode_count], dtype=np.int64)
    max_steps = int(max(lengths.max() - 1, 1))
    observation_shapes = root.attrs.get("observation_shapes", {})
    if isinstance(observation_shapes, str):
        observation_shapes = json.loads(observation_shapes)
    metadata = dense_metadata(dict(root.attrs), episode_count, max_steps)
    metadata["observation_shapes"] = observation_shapes
    metadata["obs_dim"] = int(root.attrs.get("obs_dim", root["obs"].shape[1]))
    metadata["action_dim"] = int(root.attrs.get("action_dim", root["action"].shape[1]))
    if "image" in root:
        metadata["image_size"] = int(root.attrs.get("image_size") or root["image"].shape[1])
    else:
        metadata["image_size"] = None
    return metadata, max_steps


def zarr_episode(root, index: int) -> tuple[dict[str, np.ndarray], float]:
    start = int(root["episode_start"][index])
    stored_length = int(root["episode_length"][index])
    length = stored_length - 1
    obs_slice = slice(start, start + stored_length)
    step_slice = slice(start, start + length)

    episode = {
        "observations": np.asarray(root["obs"][obs_slice], dtype=np.float32),
        "actions": np.asarray(root["action"][step_slice], dtype=np.float32),
        "rewards": np.asarray(root["reward"][step_slice], dtype=np.float32).reshape(length, 1),
        "discounts": np.asarray(root["discount"][step_slice], dtype=np.float32).reshape(length, 1),
        "terminations": np.asarray(root["terminated"][step_slice], dtype=np.uint8).reshape(length, 1),
        "truncations": np.asarray(root["timeout"][step_slice], dtype=np.uint8).reshape(length, 1),
    }
    if "image" in root:
        episode["images"] = np.asarray(root["image"][obs_slice], dtype=np.uint8)
    return episode, float(episode["rewards"].sum())


def convert_zarr(root, output: Path, overwrite: bool, limit: int | None):
    total = int(root["episode_start"].shape[0])
    if limit is not None:
        total = min(total, int(limit))
    metadata, _ = zarr_metadata(root, total)
    dst = prepare_output(output, metadata, overwrite)

    rows = 0
    converted = 0
    try:
        for idx in range(total):
            if int(root["episode_length"][idx]) < 2:
                continue
            episode, episode_return = zarr_episode(root, idx)
            append_episode(dst, converted, episode, episode_return)
            rows += int(episode["actions"].shape[0])
            converted += 1
    finally:
        dst.close()

    if converted != total:
        metadata["num_episodes"] = converted
        write_json(output / "metadata.json", metadata)
    write_progress(output, converted, rows, converted)
    print(
        f"Converted {converted} zarr episodes, {rows} transitions -> {output}\n"
        "Note: zarr conversion uses one fewer transition per episode because the "
        "old zarr schema does not store the final next observation."
    )


def main():
    args = parse_args()
    src = args.input.expanduser().resolve()
    dst = args.output.expanduser().resolve()
    if src == dst:
        raise RuntimeError("Input and output paths must be different.")
    if src.is_relative_to(dst) or dst.is_relative_to(src):
        raise RuntimeError(
            "Input and output paths must not contain each other. "
            "Write converted datasets to a separate output root."
        )

    root = open_zarr(src)
    if root.attrs.get("format") == DATA_FORMAT:
        raise RuntimeError(f"{src} is already in the dense HDF5 format.")
    convert_zarr(root, dst, args.overwrite, args.limit_episodes)


if __name__ == "__main__":
    main()
