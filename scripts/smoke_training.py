"""Full-size, checkpoint-free training/resource smoke check on existing expert data."""

import argparse
import csv
import io
import itertools
import json
import math
import multiprocessing
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def collect_gpu_samples(stop, result):
    """NVML via nvidia-smi includes CUDA allocations outside PyTorch's allocator."""
    memory, utilization = [], []
    try:
        while True:
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            pids = {str(os.getpid()), *(str(child.pid) for child in multiprocessing.active_children())}
            mine = [
                row
                for row in csv.reader(io.StringIO(apps.stdout), skipinitialspace=True)
                if len(row) == 3 and row[0].strip() in pids
            ]
            if mine:
                memory.append(sum(float(row[2]) for row in mine))
                uuids = {row[1] for row in mine}
                gpus = subprocess.run(
                    ["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                utilization.extend(
                    float(row[1])
                    for row in csv.reader(io.StringIO(gpus.stdout), skipinitialspace=True)
                    if len(row) == 2 and row[0] in uuids
                )
            if stop.wait(0.5):
                break
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["sampling_warning"] = str(error)
    result.update(
        {
            "process_gpu_peak_mib": max(memory, default=None),
            "gpu_samples": len(memory),
            "gpu_scope": "Worker and active child processes reported by nvidia-smi compute-apps",
            "device_utilization_mean_percent": sum(utilization) / len(utilization) if utilization else None,
            "device_utilization_max_percent": max(utilization, default=None),
        }
    )


def pin_batch(value):
    if hasattr(value, "pin_memory"):
        return value.pin_memory()
    if isinstance(value, dict):
        return {key: pin_batch(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(pin_batch(item) for item in value)
    return value


def seed_online_replay(config, dataset, replay):
    """Translate complete expert episodes to the native online replay layouts, in RAM only."""
    import numpy as np
    import torch
    from tensordict import TensorDict

    from dmc_expert.storage import read_image_window, read_physical_state
    from models.shared.physical_state import STATE_KEY

    selected = np.random.default_rng(int(config.seed)).choice(
        dataset.episodes,
        int(config.replay.episodes_per_batch),
        replace=False,
    )
    episodes = []
    dreamer = config.model_family == "dreamer"
    for index in selected:
        length = int(dataset.lengths[index])
        frames = length + int(dreamer)
        data = {
            "image": torch.from_numpy(read_image_window(dataset.images, index, 0, frames, dataset.frame_stack)),
            STATE_KEY: torch.from_numpy(
                read_physical_state(
                    dataset.h5,
                    index,
                    slice(0, frames),
                    dataset.state_indices,
                    dataset.state_targets,
                )
            ),
        }
        action = torch.from_numpy(dataset.actions[index, :length])
        reward = torch.from_numpy(dataset.rewards[index, :length])
        terminal = torch.from_numpy(dataset.terminations[index, :length]).bool()
        if dreamer:
            # Buffer stores outgoing actions but incoming rewards; its sampler shifts actions once.
            data.update(
                {
                    "action": torch.cat((action, torch.zeros_like(action[:1]))),
                    "reward": torch.cat((torch.zeros_like(reward[:1]), reward)),
                    "is_terminal": torch.cat((torch.zeros_like(terminal[:1]), terminal)),
                    "is_first": torch.zeros(frames, 1, dtype=torch.bool),
                    "is_last": torch.zeros(frames, 1, dtype=torch.bool),
                }
            )
            data["is_first"][0] = data["is_last"][-1] = True
        else:
            data.update(action=action, reward=reward, terminal=terminal.float())
        episodes.append(TensorDict(data, batch_size=(frames,)))
    state = {"completed": episodes}
    replay.load_state_dict(state if dreamer else {"obs_keys": ("image", STATE_KEY), "replay": state})
    if not replay.ready():
        raise ValueError("The production replay capacity cannot retain enough source episodes for this smoke batch.")
    return list(map(int, selected))


def run_worker(job):
    import torch

    import tools
    from envs import close_envs, make_envs
    from training import load_model_family
    from training.protocol import validate_training_recipe

    config = OmegaConf.create(job["config"])
    result = {
        "name": job["name"],
        "status": "FAIL",
        "config_yaml": OmegaConf.to_yaml(config),
        "phases": [],
        "resources": {},
        "seed_runs": job.get("seed_runs", 1),
    }
    device = torch.device(config.device)
    monitor = None
    envs = None
    stop = threading.Event()
    started = time.perf_counter()

    def measure(name, operation):
        stage = {"name": name}
        result["phases"].append(stage)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            value = operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage["status"] = "PASS"
            return value
        finally:
            stage["seconds"] = time.perf_counter() - start
            stage.setdefault("status", "FAIL")
            if device.type == "cuda":
                stage.update(
                    {
                        "allocated_peak_mib": torch.cuda.max_memory_allocated(device) / MIB,
                        "reserved_peak_mib": torch.cuda.max_memory_reserved(device) / MIB,
                        "allocated_end_mib": torch.cuda.memory_allocated(device) / MIB,
                        "reserved_end_mib": torch.cuda.memory_reserved(device) / MIB,
                    }
                )
            print(
                f"Phase | {name} | {stage['status']} | seconds={stage['seconds']:.2f}"
                f" | reserved_peak_mib={stage.get('reserved_peak_mib', 'CPU')}",
                flush=True,
            )

    try:
        validate_training_recipe(config)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            free, total = torch.cuda.mem_get_info(device)
            result["hardware"] = {
                "gpu": torch.cuda.get_device_name(device),
                "total_mib": total / MIB,
                "free_before_build_mib": free / MIB,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
            monitor = threading.Thread(target=collect_gpu_samples, args=(stop, result["resources"]), daemon=True)
            monitor.start()
        tools.configure_randomness(config.seed, bool(config.deterministic_run))
        family = load_model_family(config.model_family)
        model = measure("build", lambda: family.build_model(config))
        result["parameters"] = {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        }
        with measure("dataset_open", lambda: family.build_replay(config)) as dataset:
            model.state_head.set_stats(dataset.state_mean, dataset.state_std)
            if hasattr(model, "configure_pretraining"):
                model.configure_pretraining(int(config.training.expert.updates))

            def update(name, operation):
                before = int(model.state_head.updates.item())
                model.train()
                metrics = measure(name, operation)
                if int(model.state_head.updates.item()) != before + 1:
                    raise RuntimeError(f"{name} did not complete one physical-state optimizer update.")
                values = {key: float(torch.as_tensor(value).detach()) for key, value in metrics.items()}
                if not values or not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"{name} returned missing/non-finite training metrics.")
                result["phases"][-1]["metrics"] = values

            def sample():
                batch = dataset.sample_episode_batch()
                return pin_batch(batch) if device.type == "cuda" else batch

            for step in range(job["updates"]):
                batch = measure(f"sample/{step + 1}", sample)
                update(f"expert/{step + 1}", lambda batch=batch: family.expert_update(model, batch))
                del batch

            session = family.OnlineSession(config, model, None)
            episodes = measure("seed_replay", lambda: seed_online_replay(config, dataset, session.replay))
            raw_replay = sum(
                tensor.numel() * tensor.element_size()
                for episode in session.replay.completed
                for tensor in episode.values()
            )
            result["replay"] = {
                "source": "training-split expert episodes, not freshly collected online experience",
                "episodes": episodes,
                "rows": session.replay.count(),
                "capacity": session.replay.max_size,
                "storage_device": str(session.replay.storage_device),
                "raw_tensor_mib": raw_replay / MIB,
                "full_capacity_raw_tensor_mib": raw_replay / session.replay.count() * session.replay.max_size / MIB,
            }
            if hasattr(model, "configure_online"):
                model.configure_online(int(config.training.online.updates), resumed=False)
            for step in range(job["updates"]):
                update(f"online/{step + 1}", lambda: session.update(1))
            if job["rollout_steps"]:
                envs = measure("env_build", lambda: make_envs(config.env))
                session.envs = envs
                measure("env_reset", session.start)
                if config.model_family == "dreamer":
                    measure("env_prime", session.collect)  # Dreamer's first collect only resets the environments.
                for step in range(job["rollout_steps"]):
                    delta, _ = measure(f"collect/{step + 1}", session.collect)
                    result["phases"][-1]["environment_steps"] = delta
                    if not all(
                        torch.isfinite(value).all()
                        for rows in session.replay.current
                        if rows
                        for value in rows[-1].values()
                    ):
                        raise RuntimeError("Non-finite transition after collection.")
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError("Non-finite parameters after training.")
        result["status"] = "PASS"
    except Exception as error:  # noqa: BLE001 - Record each worker failure without losing the rest of the matrix.
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        close_envs(envs)
        stop.set()
        if monitor is not None:
            monitor.join()
        result["elapsed_seconds"] = time.perf_counter() - started
        # Linux ru_maxrss is KiB; macOS reports bytes.
        result["resources"]["host_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            MIB if sys.platform == "darwin" else 1024
        )
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["resources"]["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        result["timing_projection"] = project_time(result)
        path = Path(job["result_path"])
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result["status"] == "PASS"


def project_time(result):
    """Project measured training work, not unmeasured evaluation/checkpoint costs."""
    if result["status"] != "PASS":
        return None
    config = OmegaConf.create(result["config_yaml"])
    rates = {}
    for name in ("sample", "expert", "online", "collect"):
        times = [phase["seconds"] for phase in result["phases"] if phase["name"].startswith(name + "/")]
        if times:
            warm = times[1:] or times
            rates[name] = {
                "samples": len(times),
                "cold_seconds": times[0],
                "seconds_per_call": statistics.mean(warm),
                "min_seconds": min(warm),
                "max_seconds": max(warm),
                "basis": "after first call" if len(times) > 1 else "cold call only; not steady state",
            }
    expert = config["training"]["expert"]
    online = config["training"]["online"]
    expert_updates = int(expert["updates"]) if expert["enabled"] else 0
    online_updates = int(online["updates"]) if online["steps"] else 0
    batch_steps = int(config["env"]["env_num"]) * int(config["env"]["action_repeat"])
    for name in ("expert", "online"):
        rates[name]["updates_per_second"] = 1 / rates[name]["seconds_per_call"]
    if "collect" in rates:
        rates["collect"]["environment_steps_per_second"] = batch_steps / rates["collect"]["seconds_per_call"]
    calls = math.ceil(int(online["steps"]) / batch_steps)
    read = rates["sample"]["seconds_per_call"]
    compute = rates["expert"]["seconds_per_call"]
    # The real trainer overlaps one HDF5 sample with the current update. Bound
    # that overlap rather than assuming that disk I/O is either free or fully serial.
    expert_range = [expert_updates * max(read, compute), expert_updates * (read + compute)]
    update_seconds = online_updates * rates["online"]["seconds_per_call"]
    collect_seconds = calls * rates["collect"]["seconds_per_call"] if "collect" in rates else None
    startup = sum(
        phase["seconds"]
        for phase in result["phases"]
        if phase["name"]
        in {
            "build",
            "dataset_open",
            "env_build",
            "env_reset",
            "env_prime",
        }
    )
    cold_overhead = sum(max(0, rate["cold_seconds"] - rate["seconds_per_call"]) for rate in rates.values())
    return {
        "rates": rates,
        "expert_updates": expert_updates,
        "online_updates": online_updates,
        "online_environment_steps": int(online["steps"]),
        "online_vector_calls": calls,
        "expert_seconds_range": expert_range,
        "online_update_seconds": update_seconds,
        "online_collection_seconds": collect_seconds,
        "startup_seconds": startup,
        "cold_overhead_seconds": cold_overhead,
        "training_hours_range": [
            (startup + cold_overhead + seconds + update_seconds + (collect_seconds or 0)) / 3600
            for seconds in expert_range
        ],
        "collection_included": collect_seconds is not None or calls == 0,
        "steady_samples": all(rate["samples"] >= 3 for rate in rates.values()),
    }


def timing_summary(results, jobs):
    projections = [row["timing_projection"] for row in results if row.get("timing_projection")]
    return {
        "projected_runs": len(projections),
        "requested_runs": len(jobs),
        "serial_training_hours_range": [
            sum(
                row["timing_projection"]["training_hours_range"][index] * row.get("seed_runs", 1)
                for row in results
                if row.get("timing_projection")
            )
            for index in (0, 1)
        ],
        "complete": len(projections) == len(jobs) and all(item["collection_included"] for item in projections),
        "limitations": [
            "A training subtotal, not full experiment wall time; failed/unmeasured runs are excluded.",
            "GPU measurements should be taken on an otherwise idle GPU; this tool does not stop other jobs.",
            "Expert-data collection, periodic/final evaluation, checkpoint I/O, and logging overhead are excluded.",
            "The range bounds expert data-prefetch overlap, not statistical uncertainty.",
            "Use --updates 3 or more: one cold update includes compilation and optimizer initialization.",
            "Short rollouts start near reset; later context growth, planner convergence, and resets can change costs.",
            "Online replay is seeded from experts: rollout-length distribution and data contention may differ later.",
            "Subsequent seeds are extrapolated using the first seed's measurements, not separately benchmarked.",
        ],
    }


def memory_candidates(results):
    """Conservative memory arithmetic only; no concurrent training is launched."""
    groups = {}
    for row in results:
        groups.setdefault("/".join(row["name"].split("/")[1:]), []).append(row)
    estimates = {}
    for name, rows in groups.items():
        if any(row["status"] != "PASS" or not row.get("hardware") for row in rows):
            continue
        peaks = []
        for row in rows:
            reserved = max(phase.get("reserved_peak_mib", 0) for phase in row["phases"])
            process = row["resources"].get("process_gpu_peak_mib")
            peaks.append(max(reserved, process if process is not None else reserved + 1024))
        estimates[name] = 1.25 * max(peaks)
    total = min((row["hardware"]["total_mib"] for row in results if row.get("hardware")), default=0)
    # Never assume VRAM occupied by other jobs is available to this experiment.
    free = min((row["hardware"]["free_before_build_mib"] for row in results if row.get("hardware")), default=0)
    budget = min(0.9 * total, free)
    pairs = [
        {"models": [left, right], "estimated_mib": estimates[left] + estimates[right]}
        for left, right in itertools.combinations(estimates, 2)
        if estimates[left] + estimates[right] <= budget
    ]
    return {
        "budget_mib": budget,
        "estimated_per_run_mib": estimates,
        "pairs": sorted(pairs, key=lambda pair: pair["estimated_mib"]),
        "rule": "Worst measured scenario per variant +25%; reserve 10% of total VRAM."
        " If process sampling is unavailable, add 1 GiB to PyTorch reserved memory before the margin.",
        "limitations": [
            "Candidates require a concurrent trial; memory fit does not establish a speedup or stability.",
            "First updates include compilation/optimizer initialization and are not steady-state timings.",
            "One sampled batch is not a worst-case guarantee; use --updates 3 or more for stronger evidence.",
            "Final evaluation, checkpoint saving, and full-capacity host replay are not profiled.",
            "GPU utilization is device-wide, sampled over the worker lifetime; other processes can affect it.",
            "Collection begins at episode start; later live caches and resets may cost more.",
            "Compute-app samples may omit graphics-only renderer allocations and miss brief peaks.",
        ],
    }


def run_job(job):
    from main import execute

    path = Path(job["result_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    log_path = path.with_name("stdout.log")
    started = time.perf_counter()
    returncode = 0
    try:
        execute(
            job["name"],
            [sys.executable, "-u", "-m", "scripts.smoke_training", "--worker"],
            log_path,
            input_text=json.dumps(job),
        )
    except SystemExit as error:
        returncode = error.code
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {"name": job["name"], "status": "FAIL", "error": f"Worker exited {returncode} without a report"}
    )
    if returncode:
        result["status"] = "FAIL"
    result.update(log=str(log_path), wall_seconds=time.perf_counter() - started)
    return result


def compare_parallel(jobs, output):
    from main import run_jobs

    report = {}
    for name, parallelism in (("serial", 1), ("parallel", 2)):
        results = [None] * len(jobs)

        def measure(item, name=name, results=results):
            index, job = item
            path = output / name / job["name"] / "resources.json"
            results[index] = run_job(dict(job, result_path=str(path)))

        started = time.perf_counter()
        run_jobs(measure, list(enumerate(jobs)), parallelism)
        report[name] = {"wall_seconds": time.perf_counter() - started, "runs": results}
        if any(result["status"] != "PASS" for result in results):
            break
    passed = len(report) == 2 and all(row["status"] == "PASS" for group in report.values() for row in group["runs"])
    report["passed"] = passed
    if passed:
        report["combined_wall_speedup"] = report["serial"]["wall_seconds"] / report["parallel"]["wall_seconds"]
        report["warm_phase_slowdown"] = {
            before["name"]: {
                phase: after["timing_projection"]["rates"][phase]["seconds_per_call"] / rate["seconds_per_call"]
                for phase, rate in before["timing_projection"]["rates"].items()
            }
            for before, after in zip(report["serial"]["runs"], report["parallel"]["runs"], strict=True)
        }
        print(f"Pair | combined wall-time speedup={report['combined_wall_speedup']:.2f}x", flush=True)
        print("Pair | includes process startup; inspect warm-phase rates and repeat before selecting concurrency.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "parallel_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pair | {'PASS' if passed else 'FAIL'} | report={path}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="dmc_benchmark")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "training_smoke")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--models", nargs="+", help="Family/variant names, e.g. dreamer/gru storm/mamba3.")
    parser.add_argument("--device", help="Defaults to the matrix device; CPU is for local functional checks only.")
    parser.add_argument(
        "--updates", type=int, default=1, help="Native updates per phase; keep production schedule lengths."
    )
    parser.add_argument(
        "--rollout-steps", type=int, help="Real batched collection steps (default: --updates); 0 skips DMC."
    )
    parser.add_argument("--override", action="append", default=[], help="Hydra matrix override.")
    parser.add_argument("--model-override", action="append", default=[], help="Hydra override for each selected model.")
    parser.add_argument(
        "--compare-parallel", action="store_true", help="Benchmark exactly two runs serially, then together."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(0 if run_worker(json.load(sys.stdin)) else 1)
    if args.updates < 1:
        parser.error("--updates must be positive")
    if args.rollout_steps is not None and args.rollout_steps < 0:
        parser.error("--rollout-steps cannot be negative")
    output = args.output.expanduser().resolve()
    jobs = []
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        matrix = compose(config_name=args.config_name, overrides=args.override)
        OmegaConf.resolve(matrix)
        entries = {
            f"{family}/{variant}": entry
            for family, variants in matrix.models.items()
            for variant, entry in variants.items()
        }
        scenarios, models = args.scenarios or list(matrix.scenarios), args.models or list(entries)
        if set(scenarios) - set(matrix.scenarios) or set(models) - set(entries):
            parser.error("Unknown scenario or family/variant for this matrix")
        root = args.dataset_root or Path(str(matrix.evaluation.dataset_root))
        root = root.expanduser()
        root = (root if root.is_absolute() else ROOT / root).resolve()
        for scenario in scenarios:
            for name in models:
                entry = entries[name]
                config_name = entry if isinstance(entry, str) else entry.config
                extra = [] if isinstance(entry, str) else list(entry.get("overrides", []))
                config = compose(
                    config_name=config_name,
                    overrides=[
                        *matrix.training.overrides,
                        *extra,
                        *args.model_override,
                        f"scenario={scenario}",
                        f"seed={matrix.seeds[0]}",
                        f"device={args.device or matrix.device}",
                    ],
                )
                config.training.expert.data_path = str(root / config.scenario.dataset)
                OmegaConf.resolve(config)
                jobs.append(
                    {
                        "name": f"{scenario}/{name}",
                        "config": OmegaConf.to_container(config),
                        "updates": args.updates,
                        "rollout_steps": args.updates if args.rollout_steps is None else args.rollout_steps,
                        "seed_runs": len(matrix.seeds),
                        "result_path": str(output / scenario / name / "resources.json"),
                    }
                )
    if not jobs:
        parser.error("The selected matrix contains no runs")
    if args.compare_parallel and (len(jobs) != 2 or len(set(scenarios)) != 1 or len(set(models)) != 2):
        parser.error("--compare-parallel requires one scenario and two distinct models")
    print(
        f"Training smoke | runs={len(jobs)} | updates={args.updates} expert + {args.updates} online"
        f" | collection_steps={jobs[0]['rollout_steps']}"
        " | checkpoints=disabled",
        flush=True,
    )
    results = []
    if args.compare_parallel and not args.dry_run:
        raise SystemExit(0 if compare_parallel(jobs, output) else 1)
    for index, job in enumerate(jobs, 1):
        config = job["config"]
        print(
            f"START | {index}/{len(jobs)} | {job['name']}"
            f" | batch={config['replay']['batch_size']} | sequence={config['replay']['sequence_length']}",
            flush=True,
        )
        if args.dry_run:
            continue
        result = run_job(job)
        log_path = Path(result["log"])
        results.append(result)
        print(f"{result['status']} | {job['name']} | log={log_path}", flush=True)
        if result["status"] == "FAIL":
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-16:]), flush=True)
        else:
            params = result["parameters"]
            print(f"  Model | trainable={params['trainable']:,} | frozen={params['frozen']:,}", flush=True)
            for name, rate in result["timing_projection"]["rates"].items():
                peaks = [
                    phase["reserved_peak_mib"]
                    for phase in result["phases"]
                    if phase["name"].startswith(name + "/") and "reserved_peak_mib" in phase
                ]
                peak = max(peaks, default=None)
                memory = f"{peak / 1024:.2f} GiB" if peak is not None else "CPU"
                print(
                    f"  {name:8} | first={rate['cold_seconds']:.2f}s | estimate/call={rate['seconds_per_call']:.2f}s"
                    f" | samples={rate['samples']} | GPU reserved peak={memory}",
                    flush=True,
                )
            resources = result["resources"]
            print(
                f"  Resources | process GPU peak MiB={resources.get('process_gpu_peak_mib')}"
                f" | host RAM peak={resources['host_peak_rss_mib'] / 1024:.2f} GiB",
                flush=True,
            )
            print(
                f"  GPU use | mean={resources.get('device_utilization_mean_percent')}%"
                f" | peak={resources.get('device_utilization_max_percent')}% (device-wide)",
                flush=True,
            )
            print(
                f"  Replay | full-capacity tensors~{result['replay']['full_capacity_raw_tensor_mib'] / 1024:.2f} GiB"
                " (excludes Python/allocator overhead)",
                flush=True,
            )
            lower, upper = result["timing_projection"]["training_hours_range"]
            print(
                f"  Training estimate | {lower:.2f}-{upper:.2f} hours/seed | excludes evaluation/checkpoint I/O",
                flush=True,
            )
        report = {
            "runs": results,
            "memory_only_candidates": memory_candidates(results),
            "timing": timing_summary(results, jobs),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.dry_run:
        failures = sum(row["status"] != "PASS" for row in results)
        print(f"\nSmoke | passed={len(results) - failures}/{len(jobs)} | failed={failures} | checkpoints=disabled")
        pairs = report["memory_only_candidates"]["pairs"]
        print(f"\nMemory-only candidate pairs: {len(pairs)}; concurrent testing is still required.")
        for pair in pairs[:5]:
            print(f"  {' + '.join(pair['models'])}: estimated {pair['estimated_mib'] / 1024:.2f} GiB")
        lower, upper = report["timing"]["serial_training_hours_range"]
        print(
            f"Serial training estimate | {lower:.1f}-{upper:.1f} hours"
            f" | projected={report['timing']['projected_runs']}/{len(jobs)} configurations"
            " | excludes evaluation/checkpoint I/O"
        )
        if args.updates < 3:
            print("Timing caution | cold-start dominated; rerun with --updates 3 for a better estimate.")
        print(f"Report | {output / 'report.json'}")
        raise SystemExit(int(bool(failures)))


if __name__ == "__main__":
    main()
