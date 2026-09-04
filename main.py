#!/usr/bin/env python3
"""Run DMC collection, training, and evaluation matrices from one Hydra config."""

import argparse
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
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
                        dataset=scenario["dataset"],
                        config=entry["config"],
                        overrides=tuple(entry.get("overrides", [])),
                        seed=int(seed),
                        logdir=output_dir / scenario_name / family / variant / f"seed_{seed}",
                    )
                )
    return runs


def execute(label, command, log_path, *, dry_run=False):
    rendered = shlex.join(map(str, command))
    if dry_run:
        print(f"PLAN | {label}", flush=True)
        return

    print(f"START | {label}", flush=True)
    started = time.perf_counter()
    output_tail = deque(maxlen=120)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
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
        try:
            for line in process.stdout:
                log.write(line)
                message = line.rstrip()
                output_tail.append(message)
                if message.startswith(CONSOLE_PREFIXES):
                    print(f"  {message}", flush=True)
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
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


def collect(config, scenario_name, scenario, *, dry_run=False):
    logdir = resolve_path(config["output_dir"]) / "orchestrator" / scenario_name
    configs = config["collection"]["configs"]
    for index, config_name in enumerate(configs, 1):
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.collect_dmc_expert_data",
            "--config-name",
            config_name,
            "--task",
            scenario["collection_task"],
        ]
        execute(
            f"collect {index}/{len(configs)} | {scenario_name} | config={config_name}",
            command,
            logdir / f"{config_name}.log",
            dry_run=dry_run,
        )


def run_models(config, runs, stages, *, dry_run=False):
    training = config["training"]
    device = config["device"]
    resume = bool(training.get("resume", False))
    overwrite = bool(training["overwrite"])
    dataset_root = resolve_path(config["evaluation"]["dataset_root"]) if stages["evaluate"] else None
    for index, run in enumerate(runs, 1):
        run_label = f"{index}/{len(runs)} | {run.name}"
        overrides = (*training.get("overrides", []), *run.overrides)
        evaluation_output = run.logdir / "evaluation.json"
        reset = bool(stages["train"] and overwrite and run.logdir.exists())
        evaluation_complete = bool(resume and evaluation_output.exists() and not reset)
        if reset:
            print(f"Overwrite | {run_label}", flush=True)
            if not dry_run:
                shutil.rmtree(run.logdir)

        train_run = bool(stages["train"])
        checkpoint = None
        if train_run and run.logdir.exists() and not reset:
            if not resume:
                raise FileExistsError(f"Training directory already exists: {run.logdir}")
            if (run.logdir / "final.pt").exists():
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
                    f"scenario={run.scenario}",
                    f"seed={run.seed}",
                    f"device={device}",
                    f"logdir={run.logdir}",
                    *overrides,
                    *resume_override,
                ],
                run.logdir / "stdout.log",
                dry_run=dry_run,
            )
            evaluation_complete = False

        if stages["evaluate"]:
            if evaluation_complete:
                print(f"Skip | {run_label} | evaluation complete", flush=True)
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
                "--output",
                evaluation_output,
            ]
            for override in (f"seed={run.seed}", *overrides):
                command.extend(("--override", override))
            execute(
                f"evaluate {run_label}",
                command,
                run.logdir / "evaluation.log",
                dry_run=dry_run,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="dmc_benchmark")
    parser.add_argument("--override", action="append", default=[], help="Hydra override; repeat as needed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete matrix without running it.")
    args = parser.parse_args()

    config = load_config(args.config_name, args.override)
    scenarios = {name: config["scenario_configs"][name] for name in config["scenarios"]}
    stages = config["stages"]
    runs_per_scenario = sum(len(variants) for variants in config["models"].values()) * len(config["seeds"])
    active_stages = ",".join(name for name, enabled in stages.items() if enabled)
    started = time.perf_counter()
    print(
        f"Experiment | scenarios={len(scenarios)} | runs={len(scenarios) * runs_per_scenario} | "
        f"stages={active_stages} | device={config['device']} | output={resolve_path(config['output_dir'])}",
        flush=True,
    )

    for index, (scenario_name, scenario) in enumerate(scenarios.items(), 1):
        print(f"\nScenario {index}/{len(scenarios)} | {scenario_name} | runs={runs_per_scenario}", flush=True)
        if stages["collect"]:
            collect(config, scenario_name, scenario, dry_run=args.dry_run)
        if stages["train"] or stages["evaluate"]:
            scenario_runs = build_runs(config, scenario_name, scenario)
            run_models(config, scenario_runs, stages, dry_run=args.dry_run)
    elapsed = timedelta(seconds=round(time.perf_counter() - started))
    print(f"\nCOMPLETE | runs={len(scenarios) * runs_per_scenario} | elapsed={elapsed}")


if __name__ == "__main__":
    main()
