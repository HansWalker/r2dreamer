"""Read-only audit of every scenario's expert dataset before training."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from dmc_expert.storage import dataset_identity, observation_indices, split_episode_indices, validate_dataset
from models.shared.physical_state import PhysicalStateTargets

ROOT = Path(__file__).resolve().parents[1]


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(
        zip(
            ("min", "p05", "median", "p95", "max", "mean", "std", "variance"),
            [
                *np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist(),
                float(values.mean()),
                float(values.std()),
                float(values.var()),
            ],
            strict=True,
        )
    )


def pooled_moments(rows, name, counts):
    """Pool within/between-episode variation, weighted by valid timesteps, not episode count."""
    means = np.asarray([row[f"{name}_mean"] for row in rows])
    mean = np.average(means, axis=0, weights=counts)
    variance = np.average(
        np.asarray([row[f"{name}_variance"] for row in rows]) + (means - mean) ** 2,
        axis=0,
        weights=counts,
    )
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "variance": variance.tolist(),
        "min": np.min([row[f"{name}_min"] for row in rows], axis=0).tolist(),
        "max": np.max([row[f"{name}_max"] for row in rows], axis=0).tolist(),
    }


def scan_numeric(h5, metadata, config, errors, warnings):
    targets = PhysicalStateTargets(config.state_head.task, config.state_head.fields)
    indices = observation_indices(metadata, config.state_head.fields)
    if sum(np.prod(metadata["observation_shapes"][key]) for key in metadata["observation_keys"]) != metadata["obs_dim"]:
        raise ValueError("Observation field layout does not add up to obs_dim.")
    if int(metadata["action_dim"]) != int(np.prod(config.model_io.action.shape)):
        raise ValueError("Dataset action dimension does not match the scenario config.")
    if metadata.get("goal_relation"):
        goal = metadata["goal_relation"]
        if goal["geometry"] != config.scenario.goal.geometry or not np.allclose(
            goal["tolerance"], config.scenario.goal.tolerance, rtol=1e-5, atol=1e-7
        ):
            raise ValueError("Stored goal geometry/tolerance does not match the scenario config.")
    else:
        warnings.append("No stored goal_relations; the observation-to-goal consistency check was skipped.")

    splits = {name: split_episode_indices(metadata, name, len(h5["complete"])) for name in ("train", "heldout")}
    lengths = np.asarray(h5["lengths"], dtype=np.int64)
    returns = np.asarray(h5["returns"], dtype=np.float64)
    fields = ["observations", "actions", "rewards", "discounts", "terminations", "truncations"]
    if "goal_relations" in h5:
        fields.append("goal_relations")
    failures, seen, duplicate_examples = {}, {}, []
    duplicate_counts = {"train": 0, "heldout": 0, "cross_split": 0}
    records = {name: [] for name in splits}
    context = int(config.evaluation.final.context_length)
    repeat = int(config.env.action_repeat)
    started = last_print = time.monotonic()

    def fail(message, episode):
        item = failures.setdefault(message, {"episodes": 0, "examples": []})
        item["episodes"] += 1
        if len(item["examples"]) < 5:
            item["examples"].append(int(episode))

    for begin in range(0, len(lengths), 64):
        end = min(begin + 64, len(lengths))
        arrays = {key: np.asarray(h5[key][begin:end]) for key in fields}
        for offset, episode in enumerate(range(begin, end)):
            length = int(lengths[episode])
            row = {
                key: value[offset, : length + int(key in {"observations", "goal_relations"})]
                for key, value in arrays.items()
            }
            finite = True
            for key, value in row.items():
                if not np.isfinite(value).all():
                    fail(f"Non-finite {key}", episode)
                    finite = False
            if not finite:
                continue
            action, reward = row["actions"], row["rewards"][:, 0]
            terminal, truncated = row["terminations"][:, 0], row["truncations"][:, 0]
            if np.any(np.abs(action) > 1 + 1e-5):
                fail("Actions outside [-1, 1]", episode)
            if np.any((reward < -1e-5) | (reward > repeat + 1e-5)):
                fail("Rewards outside [0, action_repeat]", episode)
            if np.any(terminal > 1) or np.any(truncated > 1):
                fail("Non-binary termination/truncation flags", episode)
            if not np.allclose(row["discounts"][:, 0], 1 - terminal.astype(np.float32), atol=1e-6):
                fail("Discounts inconsistent with termination (timeouts must bootstrap)", episode)
            if not np.isclose(reward.sum(dtype=np.float64), returns[episode], rtol=1e-5, atol=1e-3):
                fail("Stored return does not equal the sum of rewards", episode)

            state = targets.encode(row["observations"][:, indices])
            if targets.task == "dmc_cartpole_balance_sparse":
                orientation = state[:, targets.positions[1:]]
                if not np.allclose(np.square(orientation).sum(-1), 1, atol=1e-4):
                    fail("Cartpole cosine/sine orientation is not a unit vector", episode)
            if "goal_relations" in row:
                relation = targets.goal_relation(torch.from_numpy(state)).numpy()
                difference = relation - row["goal_relations"]
                if targets.task == "dmc_cartpole_balance_sparse":
                    difference[:, 1] = np.arctan2(np.sin(difference[:, 1]), np.cos(difference[:, 1]))
                if np.any(np.abs(difference) > 5e-5):
                    fail("Goal relations disagree with physical observations", episode)

            split = "train" if episode < int(splits["heldout"][0]) else "heldout"
            digest = hashlib.sha256(np.int64(length).tobytes())
            for key in fields:
                digest.update(row[key].tobytes())
            digest = digest.digest()
            if digest in seen:
                previous_split, previous_episode = seen[digest]
                category = split if previous_split == split else "cross_split"
                duplicate_counts[category] += 1
                if category == "cross_split" and len(duplicate_examples) < 10:
                    duplicate_examples.append({"train_episode": previous_episode, "heldout_episode": episode})
            else:
                seen[digest] = (split, episode)

            qualifies = reward / repeat >= float(config.evaluation.success_threshold)
            tail = max(1, int(np.ceil(length * float(config.evaluation.maintenance_fraction))))
            hits = np.flatnonzero(qualifies)
            delta = np.diff(state[:, targets.positions].astype(np.float64), axis=0)[context - 1 :]
            records[split].append(
                {
                    "episode": episode,
                    "length": length,
                    "return": returns[episode],
                    "normalized_reward": reward.mean(dtype=np.float64) / repeat,
                    "late_success": qualifies[-tail:].mean() >= float(config.evaluation.maintenance_occupancy),
                    "first_success": int(hits[0] + 1) if hits.size else np.nan,
                    "success_steps": int(qualifies.sum()),
                    "zero_reward_steps": int(np.count_nonzero(np.abs(reward) <= 1e-6)),
                    "full_reward_steps": int(np.count_nonzero(np.abs(reward - repeat) <= 1e-6)),
                    "saturated_action_fraction": np.mean(np.abs(action) >= 0.999),
                    "initial_physical_state": state[0],
                    "motion_square": np.square(delta).sum(0),
                    "motion_steps": len(delta),
                    "static_steps": int(np.count_nonzero(np.max(np.abs(delta), axis=-1) <= 1e-5)),
                }
            )
            for name, values in (("physical", state), ("action", action), ("step_reward", reward)):
                records[split][-1].update(
                    {
                        f"{name}_mean": values.mean(0, dtype=np.float64),
                        f"{name}_variance": values.var(0, dtype=np.float64),
                        f"{name}_min": values.min(0),
                        f"{name}_max": values.max(0),
                    }
                )
        now = time.monotonic()
        if now - last_print >= 10 or end == len(lengths):
            print(f"  Numeric | episodes={end:,}/{len(lengths):,} | elapsed={now - started:.0f}s", flush=True)
            last_print = now

    errors.extend(
        f"{name}: {item['episodes']} episodes; examples={item['examples']}" for name, item in failures.items()
    )
    if duplicate_counts["cross_split"]:
        errors.append(f"Exact numeric trajectories repeated across train/heldout: {duplicate_counts['cross_split']}")
    for split in splits:
        if duplicate_counts[split]:
            warnings.append(f"{split}: {duplicate_counts[split]} exact duplicate numeric episodes within the split.")

    statistics = {}
    for split, rows in records.items():
        if not rows:
            statistics[split] = {"valid_numeric_episodes": 0}
            continue
        lengths_in_split = np.asarray([row["length"] for row in rows])
        transitions = int(lengths_in_split.sum())
        steps = sum(row["motion_steps"] for row in rows)
        static = sum(row["static_steps"] for row in rows) / steps if steps else None
        stats = {
            "valid_numeric_episodes": len(rows),
            "transitions": transitions,
            **{
                key: summary([row[key] for row in rows])
                for key in ("length", "return", "normalized_reward", "first_success", "saturated_action_fraction")
            },
            "late_success_rate": float(np.mean([row["late_success"] for row in rows])),
            "any_success_rate": float(np.mean([np.isfinite(row["first_success"]) for row in rows])),
            "zero_return_episode_rate": float(np.mean([abs(row["return"]) <= 1e-6 for row in rows])),
            "perfect_reward_episode_rate": float(np.mean([row["full_reward_steps"] == row["length"] for row in rows])),
            "low_return_episode_rate": float(np.mean([row["normalized_reward"] < 0.8 for row in rows])),
            "step_reward": pooled_moments(rows, "step_reward", lengths_in_split),
            **{
                name + "_fraction": sum(row[name + "_steps"] for row in rows) / transitions
                for name in ("success", "zero_reward", "full_reward")
            },
            "actions": pooled_moments(rows, "action", lengths_in_split),
            "action_saturation_fraction": float(
                np.average(
                    [row["saturated_action_fraction"] for row in rows],
                    weights=lengths_in_split,
                )
            ),
            "physical_coordinates": targets.coordinates,
            **{
                f"physical_{key}": value
                for key, value in pooled_moments(rows, "physical", lengths_in_split + 1).items()
            },
            "initial_physical_std": np.std(
                [row["initial_physical_state"] for row in rows], axis=0, dtype=np.float64
            ).tolist(),
            "post_context_position_coordinates": [targets.coordinates[index] for index in targets.positions],
            "post_context_position_rms_step": (
                np.sqrt(np.sum([row["motion_square"] for row in rows], axis=0) / steps).tolist() if steps else None
            ),
            "post_context_near_static_fraction": static,
        }
        # Nonoverlapping chronological blocks expose collection drift without storing per-step data.
        blocks = []
        for indices_in_block in np.array_split(np.arange(len(rows)), min(10, len(rows))):
            group = [rows[index] for index in indices_in_block]
            blocks.append(
                {
                    "first_episode": group[0]["episode"],
                    "last_episode": group[-1]["episode"],
                    "episodes": len(group),
                    "return": summary([row["return"] for row in group]),
                    "mean_normalized_reward": float(np.mean([row["normalized_reward"] for row in group])),
                    "late_success_rate": float(np.mean([row["late_success"] for row in group])),
                }
            )
        stats["collection_blocks"] = blocks
        stats["last_minus_first_block_normalized_reward"] = (
            (blocks[-1]["mean_normalized_reward"] - blocks[0]["mean_normalized_reward"]) if len(blocks) > 1 else None
        )
        statistics[split] = stats
        if stats["normalized_reward"]["mean"] < 0.8:
            warnings.append(f"{split}: mean normalized reward is below 0.8; inspect expert quality.")
        if static is not None and static > 0.9:
            warnings.append(f"{split}: over 90% of transitions after the observed prefix are nearly static.")
        if stats["late_success_rate"] < 0.9:
            warnings.append(f"{split}: fewer than 90% of episodes meet the late-success criterion.")
        drift = stats["last_minus_first_block_normalized_reward"]
        if len(rows) >= 100 and drift is not None and abs(drift) > 0.1:
            warnings.append(f"{split}: first/last collection blocks differ by over 0.1 in normalized reward.")

    split_gap = None
    if all(records.values()):
        train, heldout = statistics["train"], statistics["heldout"]
        split_gap = {
            "mean_return": heldout["return"]["mean"] - train["return"]["mean"],
            "mean_normalized_reward": heldout["normalized_reward"]["mean"] - train["normalized_reward"]["mean"],
            "late_success_rate": heldout["late_success_rate"] - train["late_success_rate"],
        }
        if min(len(rows) for rows in records.values()) >= 100 and (
            abs(split_gap["mean_normalized_reward"]) > 0.05 or abs(split_gap["late_success_rate"]) > 0.1
        ):
            warnings.append("Train/held-out expert performance differs noticeably; inspect the split summaries.")

    eligible_train = int(np.count_nonzero(lengths[splits["train"]] >= int(config.replay.sequence_length) - 1))
    required = int(config.evaluation.final.context_length) + max(map(int, config.evaluation.final.horizons)) - 1
    eligible_heldout = int(np.count_nonzero(lengths[splits["heldout"]] >= required))
    if eligible_train < int(config.replay.episodes_per_batch):
        errors.append("Too few training episodes are long enough for a replay batch.")
    if not eligible_heldout:
        errors.append(f"No held-out episode can supply the final evaluation's {required} transitions.")
    return {
        "splits": statistics,
        "issues": failures,
        "duplicate_numeric_episodes": duplicate_counts,
        "cross_split_duplicate_examples": duplicate_examples,
        "usable_training_episodes": eligible_train,
        "usable_forecast_episodes": eligible_heldout,
        "motion_definition": "Position changes after the observed prefix; near-static means all changes <= 1e-5.",
        "context_length": context,
        "heldout_minus_train": split_gap,
        "quality_definitions": {
            "scope": "Collected expert policy performance, not trained world-model performance.",
            "variance": "Population variance (ddof=0); return statistics weight episodes equally.",
            "step_reward": f"Raw reward per agent transition, summed over action_repeat={repeat}; valid steps only.",
            "normalized_reward": "Episode return divided by episode length and action_repeat.",
            "first_success": "One-based agent step of first reward-threshold hit; summary excludes never-hit episodes.",
            "low_return_threshold": 0.8,
            "success_threshold": float(config.evaluation.success_threshold),
            "maintenance_fraction": float(config.evaluation.maintenance_fraction),
            "maintenance_occupancy": float(config.evaluation.maintenance_occupancy),
            "drift_warnings": "Heuristics, not significance tests; drift/split warnings require at least 100 episodes.",
        },
    }


def scan_images(h5, metadata, output, *, image_episodes, full_images, seed, context, errors, warnings):
    rng = np.random.default_rng(seed)
    splits = {name: split_episode_indices(metadata, name, len(h5["complete"])) for name in ("train", "heldout")}
    sampled = {
        name: sorted(rng.choice(episodes, min(image_episodes, len(episodes)), replace=False).tolist())
        for name, episodes in splits.items()
    }
    previews = {episode for episodes in sampled.values() for episode in episodes[:4]}
    selected = (
        list(range(len(h5["complete"]))) if full_images else sorted(ep for group in sampled.values() for ep in group)
    )
    frames_checked = pairs = unchanged = solid_frames = 0
    solid_examples, frozen, rows = [], [], []
    started = last_print = time.monotonic()
    for number, episode in enumerate(selected, 1):
        length = int(h5["lengths"][episode])
        frames = np.asarray(h5["images"][episode, : length + 1])
        solid = np.ptp(frames.reshape(len(frames), -1, 3), axis=1).max(axis=-1) == 0
        solid_frames += int(solid.sum())
        solid_examples.extend(
            (episode, int(index)) for index in np.flatnonzero(solid)[: max(0, 5 - len(solid_examples))]
        )
        changed = np.any(frames[1:] != frames[:-1], axis=(1, 2, 3))
        if not changed.any():
            frozen.append(episode)
        pairs += len(changed)
        unchanged += int(np.count_nonzero(~changed))
        frames_checked += len(frames)
        if episode in previews:
            split = "train" if episode < int(splits["heldout"][0]) else "heldout"
            indices = sorted({0, min(1, length), min(16, length), min(context - 1, length), length // 2, length})
            rows.append(
                (split, episode, float(h5["returns"][episode]), [(index, frames[index].copy()) for index in indices])
            )
        now = time.monotonic()
        if now - last_print >= 10 or number == len(selected):
            print(
                f"  Images  | episodes={number:,}/{len(selected):,} | frames={frames_checked:,}"
                f" | elapsed={now - started:.0f}s",
                flush=True,
            )
            last_print = now
    if solid_frames:
        errors.append(f"Solid-color images: {solid_frames} frames; (episode, timestep) examples={solid_examples}")
    if frozen:
        warnings.append(f"Identical images throughout {len(frozen)} checked episodes; examples={frozen[:5]}.")

    cell, height = 136, 164
    sheet = Image.new("RGB", (6 * cell, 28 + len(rows) * height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{metadata['domain_name']}/{metadata['task_name']} | sampled stored RGB frames", fill="black")
    for row, (split, episode, reward, images) in enumerate(rows):
        top = 28 + row * height
        draw.text((8, top), f"{split} | episode {episode} | return {reward:.2f}", fill="black")
        for column, (index, frame) in enumerate(images):
            left = column * cell + 4
            sheet.paste(Image.fromarray(frame).resize((128, 128), Image.Resampling.NEAREST), (left, top + 16))
            draw.text((left, top + 147), f"t={index}", fill="black")
    sheet.save(output)
    return {
        "mode": "all" if full_images else "sampled_episodes",
        "episodes_checked": len(selected),
        "frames_checked": frames_checked,
        "episode_indices": selected if not full_images else "all",
        "solid_frames": solid_frames,
        "unchanged_adjacent_frame_fraction": unchanged / pairs if pairs else None,
        "frozen_episodes": len(frozen),
        "contact_sheet": str(output),
        "seed": seed,
    }


def audit_dataset(config, path, output, *, image_episodes=16, full_images=False, seed=0):
    result = {"scenario": str(config.scenario.name), "dataset": str(path), "errors": [], "warnings": []}
    metadata = validate_dataset(path, config)
    result["identity"] = dataset_identity(metadata)
    result["episode_splits"] = metadata["episode_splits"]
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(path / "data.hdf5", "r") as h5:
        transitions = int(np.asarray(h5["lengths"], dtype=np.int64).sum())
        result["numeric"] = scan_numeric(h5, metadata, config, result["errors"], result["warnings"])
        result["images"] = scan_images(
            h5,
            metadata,
            output / f"{config.scenario.name}_frames.png",
            image_episodes=image_episodes,
            full_images=full_images,
            seed=seed,
            context=int(config.evaluation.final.context_length),
            errors=result["errors"],
            warnings=result["warnings"],
        )
    progress_path = path / "progress.json"
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text())
            expected = {
                "episodes": int(metadata["num_episodes"]),
                "target_episodes": int(metadata["num_episodes"]),
                "rows": transitions,
            }
            if progress != expected:
                result["warnings"].append("progress.json disagrees with HDF5; the audit uses HDF5, not progress.json.")
        except (ValueError, OSError):
            result["warnings"].append("progress.json could not be read; the audit uses HDF5 instead.")
    else:
        result["warnings"].append("progress.json is absent; HDF5 completeness was checked directly.")
    result["status"] = "FAIL" if result["errors"] else "WARN" if result["warnings"] else "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="dmc_benchmark",
        help="Hydra experiment matrix; defaults to all three production scenarios.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, help="Override the matrix dataset root (not an individual task directory)."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "dataset_audit", help="Report and contact-sheet directory."
    )
    parser.add_argument(
        "--image-episodes",
        type=int,
        default=16,
        help="Complete image episodes checked per split; numeric data are always fully scanned.",
    )
    parser.add_argument(
        "--full-images", action="store_true", help="Read every valid image frame; substantially more disk I/O."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the reproducible image sample.")
    parser.add_argument("--override", action="append", default=[], help="Hydra matrix override; repeat as needed.")
    args = parser.parse_args()
    if args.image_episodes < 1:
        parser.error("--image-episodes must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output = args.output.expanduser().resolve()
    configs = []
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        matrix = compose(config_name=args.config_name, overrides=args.override)
        OmegaConf.resolve(matrix)
        root = args.dataset_root or Path(str(matrix.evaluation.dataset_root))
        root = root.expanduser()
        root = (root if root.is_absolute() else ROOT / root).resolve()
        # Dataset budgets and shapes are shared by all families in the matrix.
        entry = next(iter(next(iter(matrix.models.values())).values()))
        name = entry if isinstance(entry, str) else entry.config
        extra = [] if isinstance(entry, str) else list(entry.get("overrides", []))
        for scenario in matrix.scenarios:
            config = compose(
                config_name=name, overrides=[*matrix.training.overrides, *extra, f"scenario={scenario}", "device=cpu"]
            )
            OmegaConf.resolve(config)
            configs.append(config)
    if not configs:
        parser.error("The experiment matrix has no scenarios")
    print(
        f"Dataset audit | scenarios={len(configs)} | numeric=all"
        f" | images={'all' if args.full_images else 'sampled'} | root={root}",
        flush=True,
    )
    print("Quality summaries describe the collected expert trajectories, not the world models.", flush=True)
    report = {
        "audit_version": "dmc_dataset_audit_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config_name,
        "dataset_root": str(root),
        "datasets": [],
        "limitations": [
            "Image checks are sampled unless --full-images is used; coverage is recorded for each dataset.",
            "No simulator replay: checks cannot prove image/state/action timing or physics. Inspect contact sheets.",
            "Exact numeric duplicate detection does not detect near-duplicates or image-only duplication.",
            "Reward and near-static thresholds are quality warnings, not model-performance guarantees.",
            "High returns and low return variance do not establish trajectory diversity or correct reward labels.",
        ],
    }
    for config in configs:
        path = root / str(config.scenario.dataset)
        print(f"\nAudit | scenario={config.scenario.name} | dataset={path}", flush=True)
        try:
            result = audit_dataset(
                config, path, output, image_episodes=args.image_episodes, full_images=args.full_images, seed=args.seed
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result = {
                "scenario": str(config.scenario.name),
                "dataset": str(path),
                "status": "FAIL",
                "errors": [str(error)],
                "warnings": [],
            }
        report["datasets"].append(result)
        print(f"{result['status']} | {config.scenario.name}")
        for split, stats in result.get("numeric", {}).get("splits", {}).items():
            if stats["valid_numeric_episodes"]:
                returns, rewards = stats["return"], stats["step_reward"]
                blocks = stats["collection_blocks"]
                print(
                    f"  {split} | episodes={stats['valid_numeric_episodes']:,} | transitions={stats['transitions']:,}"
                )
                print(
                    f"    Return | mean={returns['mean']:.2f} | std={returns['std']:.2f}"
                    f" | variance={returns['variance']:.2f}"
                )
                print(
                    f"           | min={returns['min']:.2f} | p05={returns['p05']:.2f} | median={returns['median']:.2f}"
                    f" | p95={returns['p95']:.2f} | max={returns['max']:.2f}"
                )
                print(
                    f"    Reward | per-agent-step mean={rewards['mean']:.4f} | std={rewards['std']:.4f}"
                    f" | variance={rewards['variance']:.4f} | zero={stats['zero_reward_fraction']:.1%}"
                    f" | full={stats['full_reward_fraction']:.1%}"
                )
                print(
                    f"    Episodes | any_success={stats['any_success_rate']:.1%}"
                    f" | late_success={stats['late_success_rate']:.1%}"
                    f" | zero_return={stats['zero_return_episode_rate']:.1%}"
                )
                first_hit = (
                    f"{stats['first_success']['median']:.1f} agent steps" if stats["first_success"] else "no hits"
                )
                print(
                    f"             | perfect_reward={stats['perfect_reward_episode_rate']:.1%}"
                    f" | normalized_return<0.8={stats['low_return_episode_rate']:.1%} | first_hit_median={first_hit}"
                )
                action_std = ", ".join(f"{value:.4f}" for value in stats["actions"]["std"])
                print(f"    Actions | std=[{action_std}] | saturated={stats['action_saturation_fraction']:.1%}")
                if stats["post_context_near_static_fraction"] is not None:
                    print(f"    Motion | near_static_after_prefix={stats['post_context_near_static_fraction']:.1%}")
                if len(blocks) > 1:
                    print(
                        f"    Collection | {len(blocks)} chronological blocks"
                        f" | first/last return={blocks[0]['return']['mean']:.2f}/{blocks[-1]['return']['mean']:.2f}"
                    )
        gap = result.get("numeric", {}).get("heldout_minus_train")
        if gap is not None:
            print(
                f"  Heldout - train | mean return={gap['mean_return']:+.2f}"
                f" | normalized reward={gap['mean_normalized_reward']:+.4f}"
                f" | late success={100 * gap['late_success_rate']:+.1f} percentage points"
            )
        for label in ("errors", "warnings"):
            for message in result[label]:
                print(f"  {label[:-1].upper()} | {message}")
        if "images" in result:
            print(f"  Frames | {result['images']['contact_sheet']}")
    report["status"] = (
        "FAIL"
        if any(row["errors"] for row in report["datasets"])
        else "WARN"
        if any(row["warnings"] for row in report["datasets"])
        else "PASS"
    )
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\n{report['status']} | report={report_path}", flush=True)
    raise SystemExit(1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
