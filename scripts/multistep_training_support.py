"""Offline adaptation for the opt-in recursive TS/LeWM experiment."""

import copy
import json
import time

from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from scripts.paper_faithful_duration_support import TaskBranchReplay
from scripts.train_paper_faithful_check import update
from scripts.train_paper_faithful_duration import atomic_save, file_hash
from training.progress import Progress


# Exact implementation used by the downloaded duration checkpoints. The new
# loss adds no parameters and changes no inference path. Only the explicit
# multi-step experiment accepts this source version; other evaluators stay strict.
ONE_STEP_SOURCE_IMPLEMENTATION = "e5a56aa48a102fa73c5eaddea14252f7811b50f32ebf65ea77fb22e0039c6636"


def offline_settings(config, record, bank, updates, seed):
    fraction = record["row"].get("coverage_fraction")
    rows, sources = int(config.replay.batch_size), int(config.replay.episodes_per_batch)
    if fraction != .5 or rows % 2 or sources % 2:
        raise ValueError("Offline adaptation requires the source's 50/50 expert/TRAIN-branch recipe and even rows/sources.")
    # Constructing this sampler checks TRAIN split, distinct anchors and lengths
    # before the expensive first evaluation.
    TaskBranchReplay(bank, batch_size=rows // 2, sequence_length=int(config.replay.sequence_length),
                     episodes_per_batch=sources // 2, seed=seed + 12345)
    return {"updates": updates, "seed": seed, "branch_fraction": fraction,
            "batch_sequences": rows, "sequence_length": int(config.replay.sequence_length),
            "initialization": "completed one-step offline checkpoint; additional offline adaptation, not a fresh fit",
            "optimizer": "retain moments; restart offline learning-rate schedule, then original full online schedule",
            "readout_labels_per_update": int(config.state_head.samples_per_update)}


def adapt_offline(config, model, family, record, bank, folder, row, persist_report):
    settings = row["budget"]["offline_adaptation"]
    updates, seed = settings["updates"], settings["seed"]
    replay_config = copy.deepcopy(config)
    replay_config.replay.seed = seed
    replay_config.training.expert.batch_size = settings["batch_sequences"]
    branches = TaskBranchReplay(bank, batch_size=settings["batch_sequences"] // 2,
                               sequence_length=int(config.replay.sequence_length),
                               episodes_per_batch=int(config.replay.episodes_per_batch) // 2, seed=seed + 12345)
    tools.configure_randomness(seed, bool(config.deterministic_run))
    # A finished LeWM source schedule has LR=0. Start a new offline schedule,
    # retaining Adam moments; loading the finished scheduler would train nothing.
    if hasattr(model, "configure_pretraining"):
        model.configure_pretraining(updates, resumed=False)
    initial_native, initial_head = model._gradient_updates, int(model.state_head.updates)
    initial_examples = int(model.state_head.examples)
    started = time.monotonic()

    def save():
        path = folder / "offline_latest.pt"
        atomic_save({**family.checkpoint(model), "format": "multistep_offline_adaptation_v1",
                     "resume_supported": False, "phase": "offline_adaptation",
                     "training_config": OmegaConf.to_container(config, resolve=True),
                     "source_checkpoint_sha256": record["file_sha256"],
                     "dataset_identity": record["dataset_identity"], "bank_sha256": record["bank_sha256"],
                     "updates": row["offline_updates"], "settings": settings, "rng_state": tools.get_rng_state(),
                     "goal_ranking_sampler": (model._goal_ranking_source.state_dict()
                                              if model._goal_ranking_source is not None else None),
                     "counters": {name: getattr(model, name) for name in ("_gradient_updates", "_clipped_updates")}}, path)
        row["offline_checkpoint"] = {"file": path.name, "updates": row["offline_updates"], "sha256": file_hash(path)}
        persist_report()

    with family.build_replay(replay_config) as expert:
        if dataset_identity(expert.metadata) != record["dataset_identity"]:
            raise ValueError("Offline adaptation dataset differs from the source checkpoint.")
        progress = Progress(f"Offline adaptation {record['model']}", updates)
        with (folder / "offline_metrics.jsonl").open("w", buffering=1) as log:
            try:
                for step in range(1, updates + 1):
                    metrics = update(model, expert, branches, .5, step, seed)
                    row["offline_updates"] = step
                    if (model._gradient_updates != initial_native + step
                            or int(model.state_head.updates) != initial_head + step
                            or int(model.state_head.examples) != initial_examples + step * settings["readout_labels_per_update"]):
                        raise RuntimeError("Offline native/readout updates or label budget differ from the declared counts.")
                    log.write(json.dumps({"updates": step, "metrics": metrics}, allow_nan=False) + "\n")
                    progress.update(step, f"loss={metrics['loss']:.5f}", force=step == updates)
                    if step % 500 == 0:
                        save()
            finally:
                if row["offline_updates"]:
                    save()
    row["offline"] = {"updates": row["offline_updates"], "seconds": time.monotonic() - started,
                      "head_updates_before": initial_head, "head_updates_after": int(model.state_head.updates)}
