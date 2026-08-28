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
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent


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


def execute(label, command, log_path, *, dry_run=False, echo=False):
    rendered = shlex.join(map(str, command))
    print(f"[{label}] {rendered}", flush=True)
    if dry_run:
        return

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
                if echo:
                    print(line, end="", flush=True)
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
        raise RuntimeError(f"{label} exited with code {returncode}; see {log_path}")
    print(f"[{label}] complete", flush=True)


def collect(config, scenario_name, scenario, *, dry_run=False):
    logdir = resolve_path(config["output_dir"]) / "orchestrator" / scenario_name
    for config_name in config["collection"]["configs"]:
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
            f"collect:{scenario_name}/{config_name}",
            command,
            logdir / f"{config_name}.log",
            dry_run=dry_run,
            echo=True,
        )


def run_models(config, runs, stages, *, dry_run=False):
    training = config["training"]
    device = config["device"]
    resume = bool(training.get("resume", False))
    overwrite = bool(training["overwrite"])
    dataset_root = resolve_path(config["evaluation"]["dataset_root"]) if stages["evaluate"] else None
    for run in runs:
        overrides = (*training.get("overrides", []), *run.overrides)
        evaluation_output = run.logdir / "evaluation.json"
        reset = bool(stages["train"] and overwrite and run.logdir.exists())
        evaluation_complete = bool(resume and evaluation_output.exists() and not reset)
        if reset:
            print(f"[overwrite:{run.name}] {run.logdir}", flush=True)
            if not dry_run:
                shutil.rmtree(run.logdir)

        train_run = bool(stages["train"])
        checkpoint = None
        if train_run and run.logdir.exists() and not reset:
            if not resume:
                raise FileExistsError(f"Training directory already exists: {run.logdir}")
            if (run.logdir / "final.pt").exists():
                train_run = False
                print(f"[resume:{run.name}] training already complete", flush=True)
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
                    print(f"[restart:{run.name}] no checkpoint found", flush=True)
                    if not dry_run:
                        shutil.rmtree(run.logdir)
                else:
                    print(f"[resume:{run.name}] {checkpoint.name}", flush=True)

        if train_run:
            resume_override = (f"resume_from={checkpoint}",) if checkpoint else ()
            execute(
                f"train:{run.name}",
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
                print(f"[skip:{run.name}] evaluation already complete", flush=True)
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
                f"evaluate:{run.name}",
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
    print(
        f"Experiment matrix: {len(scenarios)} scenarios, {len(scenarios) * runs_per_scenario} model runs",
        flush=True,
    )

    for scenario_name, scenario in scenarios.items():
        print(f"\nScenario: {scenario_name}", flush=True)
        if stages["collect"]:
            collect(config, scenario_name, scenario, dry_run=args.dry_run)
        if stages["train"] or stages["evaluate"]:
            scenario_runs = build_runs(config, scenario_name, scenario)
            run_models(config, scenario_runs, stages, dry_run=args.dry_run)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
