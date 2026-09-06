#!/usr/bin/env python3
"""Run DMC collection, training, and evaluation matrices from one Hydra config."""

import argparse
import contextlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity, validate_dataset
from training.protocol import (
    comparison_signature,
    completion_checkpoint,
    evaluation_identity,
    family_recipe_sha256,
    freeze_implementation,
    model_variant,
    training_complete,
    validate_checkpoint,
    validate_training_recipe,
)

ROOT = Path(__file__).resolve().parent
ACTIVE_PROCESSES = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()
CONSOLE_PREFIXES = (
    "Run |",
    "Model |",
    "Data |",
    "Checkpoint |",
    "Collection |",
    "Expert |",
    "Online |",
    "Evaluation |",
    "Result |",
    "Output |",
)


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    scenario: str
    family: str
    variant: str
    dataset: str
    config: str
    overrides: tuple[str, ...]
    seed: int
    logdir: Path


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(name, overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(config)
    return config


def build_runs(config, scenario_name, scenario):
    output_dir = resolve_path(config["output_dir"])
    runs = []
    for family, variants in config["models"].items():
        for variant, model in variants.items():
            entry = {"config": model, "overrides": []} if isinstance(model, str) else model
            for seed in config["seeds"]:
                runs.append(
                    ExperimentRun(
                        name=f"{scenario_name}/{family}/{variant}/seed_{seed}",
                        scenario=scenario_name,
                        family=str(family),
                        variant=str(variant),
                        dataset=scenario["dataset"],
                        config=entry["config"],
                        overrides=tuple(entry.get("overrides", [])),
                        seed=int(seed),
                        logdir=output_dir / scenario_name / family / variant / f"seed_{seed}",
                    )
                )
    return runs


def compose_run_config(config, run):
    overrides = (*config["training"].get("overrides", []), *run.overrides)
    run_overrides = (
        f"scenario={run.scenario}",
        f"seed={run.seed}",
        f"device={config['device']}",
        f"logdir={run.logdir}",
        *overrides,
    )
    return load_config(run.config, run_overrides), overrides, run_overrides


def validate_matrix(config, scenario_runs, stages):
    """Validate every child config before collection or model construction starts."""
    if not scenario_runs:
        raise ValueError("The experiment matrix contains no scenarios.")
    if not any(bool(enabled) for enabled in stages.values()):
        raise ValueError("The experiment has no enabled stages.")
    if bool(stages["evaluate"]) and not config["evaluation"].get("checkpoints"):
        raise ValueError("Evaluation is enabled, but no checkpoints are configured.")
    if int(config["collection"]["parallelism"]) < 1:
        raise ValueError("Collection parallelism must be positive.")

    resolved = {}
    signatures = {}
    family_recipes = {}
    for scenario, runs in scenario_runs.items():
        if not runs:
            raise ValueError(f"The experiment matrix contains no model runs for {scenario}.")
        for run in runs:
            run_config, _, _ = compose_run_config(config, run)
            validate_training_recipe(run_config)
            if str(run_config.model_family) != run.family:
                raise ValueError(f"Matrix entry {run.name} loads model_family={run_config.model_family!s}.")
            if model_variant(run_config) != run.variant:
                raise ValueError(f"Matrix entry {run.name} loads model variant {model_variant(run_config)!r}.")
            if str(run_config.scenario.dataset) != run.dataset:
                raise ValueError(
                    f"Matrix entry {run.name} uses dataset={run.dataset!r}, but its training config "
                    f"uses {str(run_config.scenario.dataset)!r}."
                )
            if int(run_config.replay.seed) != run.seed or int(run_config.env.seed) != run.seed:
                raise ValueError(f"Matrix entry {run.name} must use seed={run.seed} for both replay and environment.")
            if resolve_path(run_config.env.dataset_root) != resolve_path(config["evaluation"]["dataset_root"]):
                raise ValueError(f"Matrix entry {run.name} uses a different training and evaluation dataset root.")
            if bool(stages["evaluate"]):
                for spec in config["evaluation"]["checkpoints"]:
                    checkpoint = str(spec["checkpoint"])
                    available = {
                        "pretrained.pt": bool(run_config.training.expert.enabled),
                        "pretrained_best.pt": bool(
                            run_config.training.expert.enabled
                            and int(run_config.training.expert.eval_every)
                            and int(run_config.env.eval_episode_num)
                        ),
                        "final.pt": bool(int(run_config.training.online.steps)),
                        "best.pt": bool(
                            int(run_config.training.online.steps)
                            and int(run_config.training.online.eval_every)
                            and int(run_config.env.eval_episode_num)
                        ),
                    }.get(checkpoint, True)
                    if not available:
                        raise ValueError(
                            f"Matrix entry {run.name} requests {checkpoint}, but its training recipe "
                            "cannot create that checkpoint."
                        )
            resolved[run.name] = run_config
            signatures[run.name] = comparison_signature(run_config)
            if run.family in {"dreamer", "storm"}:
                key = (run.scenario, run.family, run.seed)
                recipe = family_recipe_sha256(run_config)
                previous = family_recipes.setdefault(key, (run.name, recipe))
                if recipe != previous[1]:
                    raise ValueError(
                        f"Recurrent variants {previous[0]} and {run.name} change settings beyond the core selector."
                    )

    reference = signatures[next(iter(signatures))]
    for name, signature in signatures.items():
        differences = {key: (reference[key], value) for key, value in signature.items() if value != reference[key]}
        if differences:
            details = ", ".join(
                f"{key}={actual!r} (reference {expected!r})" for key, (expected, actual) in differences.items()
            )
            raise ValueError(f"Comparison budget mismatch for {name}: {details}.")

    expected_root = resolve_path(config["evaluation"]["dataset_root"])
    if not stages["collect"]:
        return resolved

    collection = load_config(str(config["collection"]["config"]))
    if resolve_path(collection.output_dir) != expected_root:
        raise ValueError("Collection output and training/evaluation dataset roots do not match.")
    for scenario, runs in scenario_runs.items():
        first_run = runs[0]
        run_config = resolved[first_run.name]
        task = str(run_config.scenario.collection_task)
        image_size = tuple(map(int, run_config.env.size))
        checks = (
            (task in collection.tasks, f"collection config does not include {task}"),
            (first_run.dataset == str(run_config.scenario.dataset), "matrix and model dataset names differ"),
            (str(run_config.scenario.dataset) == task.replace("/", "_"), "scenario dataset name does not match task"),
            (image_size == (int(collection.image_size),) * 2, "collection and model image sizes differ"),
            (int(collection.action_repeat) == int(run_config.env.action_repeat), "collection action repeat differs"),
            (
                int(collection.max_episode_steps)
                == int(run_config.env.time_limit) // int(run_config.env.action_repeat),
                "collection and online episode lengths differ",
            ),
            (
                resolve_path(run_config.env.dataset_root) == expected_root,
                "training and evaluation dataset roots differ",
            ),
            (
                int(collection.episodes.train) == int(run_config.expert_data.train_episodes),
                "collection and training episode budgets differ",
            ),
            (
                int(collection.episodes.heldout) == int(run_config.expert_data.heldout_episodes),
                "collection and held-out episode budgets differ",
            ),
            (str(run_config.expert_data.policy) == "tdmpc2", "training expects a non-TD-MPC2 expert"),
            (
                ("mpc" if bool(collection.expert.mpc) else "actor") == str(run_config.expert_data.policy_mode),
                "collection and training expert policy modes differ",
            ),
        )
        failures = [message for passed, message in checks if not passed]
        if failures:
            raise ValueError(f"Dataset protocol mismatch for {scenario}: " + "; ".join(failures) + ".")
    return resolved


def validate_datasets(config, scenario_runs, resolved):
    root = resolve_path(config["evaluation"]["dataset_root"])
    identities = {}
    for runs in scenario_runs.values():
        run = runs[0]
        run_config = resolved[run.name]
        path = root / run.dataset
        metadata = validate_dataset(path, run_config)
        identities[run.dataset] = dataset_identity(metadata)
        print(f"Data | validated={path} | episodes={metadata['num_episodes']}", flush=True)
    return identities


def stop_process(process):
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def stop_active_processes():
    with ACTIVE_PROCESSES_LOCK:
        processes = tuple(ACTIVE_PROCESSES)
    for process in processes:
        stop_process(process)


def execute(label, command, log_path, *, dry_run=False):
    rendered = shlex.join(map(str, command))
    if dry_run:
        print(f"PLAN | {label}", flush=True)
        return

    print(f"START | {label}", flush=True)
    started = time.perf_counter()
    output_tail = deque(maxlen=120)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {rendered}\n\n")
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES.add(process)
        try:
            for line in process.stdout:
                log.write(line)
                message = line.rstrip()
                output_tail.append(message)
                if message.startswith(CONSOLE_PREFIXES):
                    print(f"  {message}", flush=True)
            returncode = process.wait()
        except BaseException:
            stop_process(process)
            raise
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.discard(process)
    if returncode:
        elapsed = timedelta(seconds=round(time.perf_counter() - started))
        print(f"FAILED | {label} | elapsed={elapsed}", flush=True)
        lines = list(output_tail)
        starts = [index for index, line in enumerate(lines) if line.startswith("Traceback (most recent call last):")]
        lines = lines[starts[-1] :] if starts else lines[-20:]
        if len(lines) > 30:
            lines = [lines[0], "    ...", *lines[-28:]]
        for line in lines:
            print(line, flush=True)
        print(f"LOG | {log_path}", flush=True)
        raise SystemExit(returncode)
    elapsed = timedelta(seconds=round(time.perf_counter() - started))
    print(f"DONE | {label} | elapsed={elapsed}", flush=True)


def execute_parallel(jobs, parallelism, *, dry_run=False):
    if dry_run or parallelism <= 1:
        for job in jobs:
            execute(*job, dry_run=dry_run)
        return

    executor = ThreadPoolExecutor(max_workers=min(parallelism, len(jobs)))
    futures = [executor.submit(execute, *job) for job in jobs]
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        stop_active_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown()


def collect_scenarios(config, scenarios, *, dry_run=False):
    config_name = config["collection"]["config"]
    logdir = resolve_path(config["output_dir"]) / "orchestrator"
    jobs = [
        (
            f"collect | {name} | config={config_name}",
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.collect_dmc_expert_data",
                "--config-name",
                config_name,
                "--task",
                scenario["collection_task"],
            ],
            logdir / name / f"{config_name}.log",
        )
        for name, scenario in scenarios.items()
    ]
    parallelism = min(int(config["collection"]["parallelism"]), len(jobs))
    print(f"\nCollection | scenarios={len(jobs)} | parallelism={parallelism}", flush=True)
    started = time.perf_counter()
    execute_parallel(jobs, parallelism, dry_run=dry_run)
    if not dry_run:
        elapsed = timedelta(seconds=round(time.perf_counter() - started))
        print(f"Collection | complete | scenarios={len(jobs)} | elapsed={elapsed}", flush=True)


def current_evaluation(output, spec, config, dataset_path, expected_dataset, checkpoint_path):
    if not output.exists() or not checkpoint_path.exists():
        return False
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        identity = validate_checkpoint(checkpoint, config)
        if checkpoint_path.name == completion_checkpoint(config) and not training_complete(checkpoint, config):
            return False
        prediction_expected = bool(spec.get("state_prediction", True))
        metrics = (
            result.get("mean_return"),
            result.get("task_success_rate"),
            result.get("sustained_success_rate"),
        )
        prediction = result.get("physical_state_prediction")
        if prediction_expected:
            metrics += (None if prediction is None else prediction.get("mean_nrmse"),)
        return all((
            result.get("run_identity") == identity,
            result.get("evaluation_identity") == evaluation_identity(config, prediction_expected),
            checkpoint.get("checkpoint_id") is not None,
            result.get("checkpoint_id") == checkpoint.get("checkpoint_id"),
            Path(result.get("checkpoint", "")).resolve() == checkpoint_path.resolve(),
            Path(result.get("dataset", "")).resolve() == dataset_path.resolve(),
            result.get("dataset_identity") == expected_dataset,
            result.get("dataset_role") == "held_out",
            not bool(config.training.expert.enabled) or checkpoint.get("dataset_identity") == expected_dataset,
            bool(result.get("state_prediction_evaluated")) == prediction_expected,
            all(value is not None and math.isfinite(float(value)) for value in metrics),
        ))
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_models(config, runs, stages, dataset_identities, *, dry_run=False):
    training = config["training"]
    device = config["device"]
    resume = bool(training.get("resume", False))
    overwrite = bool(training["overwrite"])
    evaluation = config["evaluation"]
    dataset_root = resolve_path(evaluation["dataset_root"]) if stages["evaluate"] else None
    evaluation_specs = tuple(evaluation.get("checkpoints", ()))
    for index, run in enumerate(runs, 1):
        run_label = f"{index}/{len(runs)} | {run.name}"
        run_config, overrides, run_overrides = compose_run_config(config, run)
        completed_checkpoint = run.logdir / completion_checkpoint(run_config)
        dataset_path = None if dataset_root is None else dataset_root / run.dataset
        reset = bool(stages["train"] and overwrite and run.logdir.exists())
        if reset:
            print(f"Overwrite | {run_label}", flush=True)
            if not dry_run:
                shutil.rmtree(run.logdir)

        train_run = bool(stages["train"])
        checkpoint = None
        if train_run and run.logdir.exists() and not reset:
            if not resume:
                raise FileExistsError(f"Training directory already exists: {run.logdir}")
            if completed_checkpoint.exists():
                saved = torch.load(completed_checkpoint, map_location="cpu", weights_only=False)
                if not training_complete(saved, run_config):
                    raise ValueError(f"Incomplete completion checkpoint: {completed_checkpoint}")
                train_run = False
                print(f"Resume | {run_label} | training complete", flush=True)
            else:
                checkpoint = next(
                    (
                        run.logdir / name
                        for name in ("latest.pt", "pretrain_latest.pt", "pretrained.pt")
                        if (run.logdir / name).exists()
                    ),
                    None,
                )
                if checkpoint is None:
                    print(f"Restart | {run_label} | no usable checkpoint", flush=True)
                    if not dry_run:
                        shutil.rmtree(run.logdir)
                else:
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    validate_checkpoint(saved, run_config)
                    print(f"Resume | {run_label} | checkpoint={checkpoint.name}", flush=True)

        if train_run:
            resume_override = (f"resume_from={checkpoint}",) if checkpoint else ()
            execute(
                f"train {run_label} | config={run.config}",
                [
                    sys.executable,
                    "-u",
                    "train.py",
                    "--config-name",
                    run.config,
                    *run_overrides,
                    *resume_override,
                ],
                run.logdir / "stdout.log",
                dry_run=dry_run,
            )
            if not dry_run:
                saved = torch.load(completed_checkpoint, map_location="cpu", weights_only=False)
                if not training_complete(saved, run_config):
                    raise RuntimeError(f"Training did not produce a complete checkpoint: {completed_checkpoint}")

        if stages["evaluate"]:
            for spec in evaluation_specs:
                output = run.logdir / str(spec["output"])
                checkpoint_path = run.logdir / str(spec["checkpoint"])
                if (
                    not dry_run
                    and resume
                    and not reset
                    and current_evaluation(
                        output,
                        spec,
                        run_config,
                        dataset_path,
                        dataset_identities[run.dataset],
                        checkpoint_path,
                    )
                ):
                    print(f"Skip | {run_label} | evaluation={spec['name']} complete", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.evaluate_dmc",
                    "--config-name",
                    run.config,
                    "--scenario",
                    run.scenario,
                    "--logdir",
                    run.logdir,
                    "--dataset",
                    dataset_root / run.dataset,
                    "--device",
                    device,
                    "--checkpoint",
                    checkpoint_path.name,
                    "--output",
                    output,
                ]
                for override in (f"seed={run.seed}", *overrides):
                    command.extend(("--override", override))
                if not bool(spec.get("state_prediction", True)):
                    command.append("--skip-state-prediction")
                name = str(spec["name"])
                execute(
                    f"evaluate-{name} {run_label}",
                    command,
                    run.logdir / str(spec.get("log", f"evaluation_{name}.log")),
                    dry_run=dry_run,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="dmc_benchmark")
    parser.add_argument("--override", action="append", default=[], help="Hydra override; repeat as needed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete matrix without running it.")
    args = parser.parse_args()

    implementation = freeze_implementation()
    config = load_config(args.config_name, args.override)
    scenarios = {name: config["scenario_configs"][name] for name in config["scenarios"]}
    stages = config["stages"]
    scenario_runs = {name: build_runs(config, name, scenario) for name, scenario in scenarios.items()}
    resolved = validate_matrix(config, scenario_runs, stages)
    runs_per_scenario = sum(len(variants) for variants in config["models"].values()) * len(config["seeds"])
    active_stages = ",".join(name for name, enabled in stages.items() if enabled)
    started = time.perf_counter()
    print(
        f"Experiment | scenarios={len(scenarios)} | runs={len(scenarios) * runs_per_scenario} | "
        f"stages={active_stages} | device={config['device']} | implementation={implementation[:12]} | "
        f"output={resolve_path(config['output_dir'])}",
        flush=True,
    )

    if stages["collect"]:
        collect_scenarios(config, scenarios, dry_run=args.dry_run)

    if stages["train"] or stages["evaluate"]:
        dataset_identities = {} if args.dry_run else validate_datasets(config, scenario_runs, resolved)
        for index, scenario_name in enumerate(scenarios, 1):
            print(f"\nScenario {index}/{len(scenarios)} | {scenario_name} | runs={runs_per_scenario}", flush=True)
            run_models(config, scenario_runs[scenario_name], stages, dataset_identities, dry_run=args.dry_run)
    elapsed = timedelta(seconds=round(time.perf_counter() - started))
    print(f"\nCOMPLETE | runs={len(scenarios) * runs_per_scenario} | elapsed={elapsed}")


if __name__ == "__main__":
    main()
