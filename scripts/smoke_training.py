"""Full-size, checkpoint-free training/resource smoke check on existing expert data."""

import argparse
import copy
import csv
import hashlib
import heapq
import importlib.metadata
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
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def check_cuda_compile_dependencies():
    """The eager Mamba runtime's Triton override is not an Inductor-compatible stack."""
    for text in importlib.metadata.requires("torch") or ():
        requirement = Requirement(text)
        if requirement.name != "triton" or (requirement.marker and not requirement.marker.evaluate()):
            continue
        try:
            installed = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or not requirement.specifier.contains(installed, prereleases=True):
            torch_version = importlib.metadata.version("torch")
            raise RuntimeError(
                f"torch {torch_version} requires triton{requirement.specifier} for CUDA compilation;"
                f" installed: {installed or 'missing'}. The pinned Mamba stack overrides this dependency"
                " for eager kernels, not torch.compile. Omit --compare-compile and keep model.compile=false."
                " Test compilation in a separate environment with matching Torch/Triton versions;"
                " do not downgrade Triton in the working Mamba environment."
            )


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


def batch_digest(batch):
    """Compare the actual CPU samples across storage/compilation benchmark cases."""
    import torch

    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            value = value.detach().contiguous()
            digest.update(f"{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(value.numpy().tobytes())
        elif hasattr(value, "items"):
            for key, item in sorted(value.items()):
                digest.update(str(key).encode())
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(value).encode())

    visit(batch)
    return digest.hexdigest()


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
    from envs import close_envs, make_envs, make_eval_envs
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
        "warmup_updates": job.get("warmup_updates", 1),
        "sample_sha256": [],
        "online_mode": "interleaved" if job.get("online_burst") else "isolated",
    }
    device = torch.device(config.device)
    monitor = None
    envs = None
    stop = threading.Event()
    started = time.perf_counter()
    compilation_counters = None
    if config.model_family == "dreamer" and config.model.compile:
        from torch._dynamo.utils import counters

        compilation_counters = counters

    def measure(name, operation):
        stage = {"name": name}
        result["phases"].append(stage)
        graphs_before = compilation_counters["stats"]["unique_graphs"] if compilation_counters is not None else 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            value = operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage["status"] = "PASS"
            return value
        finally:
            stage["seconds"] = time.perf_counter() - start
            stage["cpu_seconds"] = time.process_time() - cpu_start
            stage.setdefault("status", "FAIL")
            if compilation_counters is not None:
                stage["compiled_graphs"] = compilation_counters["stats"]["unique_graphs"] - graphs_before
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
        result["resources"].update(
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        )
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

            def update(name, operation, count=1):
                before = int(model.state_head.updates.item())
                model.train()
                metrics = measure(name, operation)
                if int(model.state_head.updates.item()) != before + count:
                    raise RuntimeError(f"{name} did not complete {count} physical-state optimizer updates.")
                values = {key: float(torch.as_tensor(value).detach()) for key, value in metrics.items()}
                if not values or not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"{name} returned missing/non-finite training metrics.")
                result["phases"][-1]["metrics"] = values
                result["phases"][-1]["updates"] = count

            def sample():
                batch = dataset.sample_episode_batch()
                return pin_batch(batch) if device.type == "cuda" else batch

            for step in range(job["updates"]):
                batch = measure(f"sample/{step + 1}", sample)
                if job.get("verify_samples"):
                    result["sample_sha256"].append(batch_digest(batch))
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
            burst = job.get("online_burst", 0)
            if not burst:
                for step in range(job["updates"]):
                    update(f"online/{step + 1}", lambda: session.update(1))
            if job["rollout_steps"]:
                envs = measure("env_build", lambda: make_envs(config.env))
                session.envs = envs
                measure("env_reset", session.start)
                if config.model_family == "dreamer":
                    measure("env_prime", session.collect)  # Dreamer's first collect only resets the environments.
                online_updates = 0
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
                    if burst and online_updates < job["updates"]:
                        count = min(burst, job["updates"] - online_updates)
                        update(f"online/{online_updates + 1}", lambda count=count: session.update(count), count)
                        online_updates += count
            close_envs(envs)
            envs = None
            for batch_size in job.get("eval_batches", []):
                eval_config = copy.deepcopy(config)
                eval_config.env.eval_episode_num = batch_size
                eval_config.env.time_limit = job["eval_steps"] * int(config.env.action_repeat)
                eval_config.env.eval_seed = int(config.evaluation.final.seed)
                envs = measure(f"eval_build/{batch_size}", lambda cfg=eval_config: make_eval_envs(cfg.env))
                measure(f"evaluation/{batch_size}", lambda cfg=eval_config, group=envs: family.evaluate(cfg, model, group))
                result["phases"][-1].update(batch_size=batch_size, vector_steps=job["eval_steps"])
                close_envs(envs)
                envs = None
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
        if compilation_counters is not None:
            result["compilation"] = {
                "unique_graphs": compilation_counters["stats"]["unique_graphs"],
                "graph_breaks": dict(compilation_counters["graph_break"]),
            }
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
    cold_overhead = 0.0
    for name in ("sample", "expert", "online", "collect"):
        phases = [phase for phase in result["phases"] if phase["name"].startswith(name + "/")]
        times = [phase["seconds"] / phase.get("updates", 1) for phase in phases]
        if times:
            discard = 1 if name == "collect" else result.get("warmup_updates", 1)
            if name == "online":
                completed, discard = 0, 0
                for phase in phases:
                    if completed >= result.get("warmup_updates", 1):
                        break
                    completed += phase.get("updates", 1)
                    discard += 1
            warm = times[discard:] or times[-1:]
            rates[name] = {
                "samples": len(times),
                "measured_samples": len(warm),
                "compilation_in_measured_calls": sum(
                    phase.get("compiled_graphs", 0) for phase in (phases[discard:] or phases[-1:])
                ),
                "cold_seconds": times[0],
                "seconds_per_call": statistics.mean(warm),
                "min_seconds": min(warm),
                "max_seconds": max(warm),
                "basis": f"after {discard} warmup calls" if len(times) > discard else "insufficient warmup",
            }
            warm_phases = phases[discard:] or phases[-1:]
            if all("cpu_seconds" in phase for phase in warm_phases):
                cpu = statistics.mean(phase["cpu_seconds"] / phase.get("updates", 1) for phase in warm_phases)
                rates[name]["cpu_seconds_per_call"] = cpu
                rates[name]["cpu_core_equivalents"] = cpu / statistics.mean(warm)
            cold_overhead += sum(
                max(0, seconds - statistics.mean(warm)) * phase.get("updates", 1)
                for phase, seconds in zip(phases[:discard], times[:discard], strict=True)
            )
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
    return {
        "rates": rates,
        "evaluation_seconds_per_vector_step": {
            str(phase["batch_size"]): phase["seconds"] / phase["vector_steps"]
            for phase in result["phases"] if phase["name"].startswith("evaluation/")
        },
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
        "steady_samples": all(rate["measured_samples"] >= 2 for rate in rates.values()),
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
    label = f"{job.get('case', 'serial')} | {job['name']}"
    print(f"START | {label}", flush=True)
    try:
        execute(
            label,
            [sys.executable, "-u", "-m", "scripts.smoke_training", "--worker"],
            log_path,
            input_text=json.dumps(job),
            quiet=True,
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
    result.update(case=job.get("case", "serial"), log=str(log_path), wall_seconds=time.perf_counter() - started)
    print(f"{result['status']} | {label} | {result['wall_seconds']:.1f}s", flush=True)
    return result


def write_summary(results, output, comparisons=()):
    """Keep terminal/paste output short; full phase details remain in JSON and worker logs."""
    if comparisons and all("model" in case for case in comparisons):
        write_scenario_summary(results, output, comparisons)
        return

    def number(value, digits=2):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        "Training smoke summary | no checkpoints",
        "Seconds/call: sample/expert/online/collect (warm). Memory: process GPU / PyTorch reserved / host RAM, GiB.",
        ("Case | Run | BxT | Trainable/frozen M | Seconds/call | Memory GiB | GPU avg/max % | Hours/seed"
         " | Threads/online CPU cores | Eval s/step(batch)"),
    ]
    for row in results:
        label = f"{row.get('case', 'serial')} | {row['name']}"
        if row["status"] != "PASS":
            error = " ".join(row.get("error", "see log").split())
            lines.append(f"FAIL | {label} | {error} | {row.get('log', '')}")
            continue
        config = OmegaConf.create(row["config_yaml"])
        projection = row["timing_projection"]
        rates = projection["rates"]
        seconds = "/".join(
            number(rates.get(name, {}).get("seconds_per_call"), 3) for name in ("sample", "expert", "online", "collect")
        )
        resources = row["resources"]
        peak = resources.get("process_gpu_peak_mib")
        reserved = max((phase.get("reserved_peak_mib", 0) for phase in row["phases"]), default=0)
        memory = "/".join(
            number(value / 1024 if value is not None else None)
            for value in (peak, reserved, resources.get("host_peak_rss_mib"))
        )
        utilization = "/".join(
            number(resources.get(f"device_utilization_{name}_percent"), 0) for name in ("mean", "max")
        )
        params = row["parameters"]
        hours = "-".join(number(value) for value in projection["training_hours_range"])
        cpu = number(rates.get("online", {}).get("cpu_core_equivalents"))
        evaluation = "/".join(
            f"{batch}:{seconds:.3f}"
            for batch, seconds in projection.get("evaluation_seconds_per_vector_step", {}).items()
        ) or "-"
        lines.append(
            f"{label} | {config.replay.batch_size}x{config.replay.sequence_length}"
            f" | {params['trainable'] / 1e6:.3f}/{params['frozen'] / 1e6:.3f}"
            f" | {seconds} | {memory} | {utilization} | {hours}"
            f" | {resources.get('torch_threads', '-')}/{cpu} | {evaluation}"
        )
        if "compilation" in row:
            compilation = row["compilation"]
            lines.append(
                f"Compilation | {row['name']} | graphs={compilation['unique_graphs']}"
                f" | graph_breaks={sum(compilation['graph_breaks'].values())}"
            )
    if comparisons:
        lines.append("Comparison | Wall speedup | Estimated warm speedup: sample/expert/online/collect | Eval speedup(batch)")
        for case in comparisons:
            if not case["passed"]:
                lines.append(f"FAIL | {case['name']} | {case.get('error', 'worker failure')}")
                continue
            ratios = "/".join(
                number(case["warm_speedup"].get(name)) for name in ("sample", "expert", "online", "collect")
            )
            eval_ratios = "/".join(
                f"{batch}:{ratio:.2f}x" for batch, ratio in case.get("evaluation_speedup", {}).items()
            ) or "-"
            lines.append(f"{case['name']} | {case['wall_speedup']:.2f}x | {ratios} | {eval_ratios}")
    passed = sum(row["status"] == "PASS" for row in results)
    lines.append(f"Workers: {passed}/{len(results)} passed. Full diagnostics: {output / 'report.json'}")
    if not comparisons:
        timing = timing_summary(results, results)
        lower, upper = timing["serial_training_hours_range"]
        lines.append(
            f"Serial training subtotal: {lower:.1f}-{upper:.1f} hours"
            f" ({timing['projected_runs']}/{len(results)} runs projected)."
        )
    if comparisons:
        lines.append("Warm speedups estimate phase-only queues at the worker limit; phases are not synchronized.")
        lines.append("Wall ratios include startup. Cases run baseline first; OS/compiler caches are not cleared.")
    lines.append("Hours exclude evaluation/checkpoint I/O; ranges bound prefetch overlap, not uncertainty.")
    if any(not row["timing_projection"]["steady_samples"] for row in results if row.get("timing_projection")):
        lines.append("Timing caution: fewer than two post-warmup calls in some phases; increase updates/rollout steps.")
    if any(
        rate.get("compilation_in_measured_calls", 0)
        for row in results
        if row.get("timing_projection")
        for rate in row["timing_projection"]["rates"].values()
    ):
        lines.append("Timing caution: compilation occurred after warmup; inspect phase timings and increase warmup.")
    if any(
        not row.get("timing_projection", {}).get("collection_included", True)
        for row in results
        if row.get("timing_projection")
    ):
        lines.append("Collection was skipped for some runs; their projected hours omit environment interaction.")
    lines.append("Short rollouts do not bound late-episode memory; GPU utilization is device-wide.")
    if any(row.get("online_mode") == "interleaved" for row in results):
        lines.append("Online burst times are normalized per optimizer update; replay includes newly collected prefixes.")
    if any(row.get("timing_projection", {}).get("evaluation_seconds_per_vector_step") for row in results if row.get("timing_projection")):
        lines.append("Evaluation timings use short episodes at the requested batch sizes, include reset, and are not quality metrics.")
    text = "\n".join(lines) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="", flush=True)
    print(f"Pasteable summary | {output / 'summary.txt'}", flush=True)


def write_scenario_summary(results, output, comparisons):
    lines = [
        "Scenario concurrency | same model/variant across scenarios | no checkpoints",
        "Model | Workers | Serial/concurrent s | Speedup | S/E/O/C | GPU peak-sum GiB | Overlap | Result",
    ]
    for case in comparisons:
        wall = case.get("wall_seconds")
        elapsed = f"{case['serial_wall_seconds']:.1f}/" + (f"{wall:.1f}" if wall is not None else "-")
        speedup = f"{case['wall_speedup']:.2f}x" if case["passed"] else "-"
        rates = case.get("warm_speedup", {})
        phases = "/".join(
            f"{rates[name]:.2f}" if name in rates else "-" for name in ("sample", "expert", "online", "collect")
        )
        peak = case.get("process_gpu_peak_sum_mib")
        memory = f"{peak / 1024:.2f}" if peak is not None else "-"
        overlap = case.get("overlap_fraction")
        overlap = f"{overlap:.0%}" if overlap is not None else "-"
        status = "PASS" if case["passed"] else "FAIL"
        lines.append(
            f"{case['model']} | {case['parallelism']} | {elapsed} | {speedup} | {phases}"
            f" | {memory} | {overlap} | {status}"
        )
    failed = [row for row in results if row["status"] != "PASS"]
    for row in failed:
        error = " ".join(row.get("error", "see log").split())
        lines.append(f"FAIL | {row.get('case', 'serial')} | {row['name']} | {error[:160]} | {row.get('log', '')}")
    lines.extend(
        [
            f"Workers: {len(results) - len(failed)}/{len(results)} passed. Full diagnostics: {output / 'report.json'}",
            "S/E/O/C = sample/expert/online/collect. Phase speedups simulate a queue at the tested worker limit.",
            "Each trial runs all selected scenarios, longest serial job first; levels differ only in concurrency.",
            "Wall time includes startup and warmup. Overlap = fraction with at least two worker processes alive.",
            "GPU peak-sum = largest N measured process peaks, not a simultaneous measurement or a full-run bound.",
            "Failed trials have no speedup. OOMs do not stop later trials; final exit status remains nonzero.",
            "OS/compiler caches are not cleared; phases are not synchronized; evaluation/checkpoint I/O is untested.",
        ]
    )
    if any(not row["timing_projection"]["steady_samples"] for row in results if row.get("timing_projection")):
        lines.append("Timing caution: fewer than two post-warmup calls in some phases; increase updates/rollout steps.")
    if any(not row["timing_projection"]["collection_included"] for row in results if row.get("timing_projection")):
        lines.append("Collection was skipped for some runs; those comparisons measure training only.")
    output.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="", flush=True)
    print(f"Pasteable summary | {output / 'summary.txt'}", flush=True)


def queue_seconds(durations, workers):
    """Estimate phase-only makespan with the same bounded FIFO scheduling as run_jobs."""
    slots = [0.0] * min(workers, len(durations))
    for seconds in durations:
        heapq.heapreplace(slots, slots[0] + seconds)
    return max(slots, default=0.0)


def compare_runs(jobs, output, *, pairs=(), compile_dreamer=False, storage=None, scenario_workers=(), gradient_batches=()):
    """Reuse serial baselines, then change only concurrency, compilation, or storage."""
    from main import run_jobs

    jobs = copy.deepcopy(jobs)
    if compile_dreamer:
        for job in jobs:
            if job["config"]["model_family"] == "dreamer":
                job["config"]["model"]["compile"] = False
    for job in jobs:
        job["verify_samples"] = True
        if gradient_batches and job["config"]["model_family"] == "temporal_straightening":
            job["config"]["jepa_model"]["planner"]["gradient_batch_size"] = gradient_batches[0]
    report = {"runs": [], "comparisons": []}

    def run_case(name, selected, parallelism=1):
        results = [None] * len(selected)

        def measure(item):
            index, job = item
            path = output / name / job["name"] / "resources.json"
            start = time.perf_counter() - started
            results[index] = run_job(dict(job, case=name, result_path=str(path)))
            results[index].update(case_start_seconds=start, case_end_seconds=time.perf_counter() - started)

        started = time.perf_counter()
        run_jobs(measure, list(enumerate(selected)), parallelism)
        wall = time.perf_counter() - started
        report["runs"].extend(results)
        return results, wall

    baseline, _ = run_case("serial", jobs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    references = {row["name"]: row for row in baseline}
    cases = [
        (f"pair_{index}", [job for job in jobs if job["name"].split("/", 1)[1] in pair], 2)
        for index, pair in enumerate(pairs, 1)
    ]
    if scenario_workers:
        groups = {}
        for job in jobs:
            groups.setdefault(job["name"].split("/", 1)[1], []).append(job)
        for selected in groups.values():
            selected.sort(key=lambda job: references[job["name"]]["wall_seconds"], reverse=True)
            cases.extend((f"scenarios_{workers}", selected, workers) for workers in scenario_workers)
    for job in jobs:
        if gradient_batches and job["config"]["model_family"] == "temporal_straightening":
            for batch_size in gradient_batches[1:]:
                candidate = copy.deepcopy(job)
                candidate["config"]["jepa_model"]["planner"]["gradient_batch_size"] = batch_size
                cases.append((f"gradient_{batch_size}", [candidate], 1))
        if compile_dreamer and job["config"]["model_family"] == "dreamer":
            compiled = copy.deepcopy(job)
            compiled["config"]["model"]["compile"] = True
            cases.append(("compiled", [compiled], 1))
        if storage is not None:
            local = copy.deepcopy(job)
            local["config"]["training"]["expert"]["data_path"] = str(storage / job["config"]["scenario"]["dataset"])
            cases.append(("local_data", [local], 1))

    for name, selected, parallelism in cases:
        before = [references[job["name"]] for job in selected]
        case = {
            "name": f"{name}: {', '.join(job['name'] for job in selected)}",
            "passed": False,
            "parallelism": parallelism,
            "runs": [job["name"] for job in selected],
            "serial_wall_seconds": sum(row["wall_seconds"] for row in before),
        }
        if scenario_workers:
            case["model"] = selected[0]["name"].split("/", 1)[1]
        if any(row["status"] != "PASS" for row in before):
            case["error"] = "serial baseline failed"
            report["comparisons"].append(case)
            continue
        after, wall = run_case(name, selected, parallelism)
        case["wall_seconds"] = wall
        peaks = [row.get("resources", {}).get("process_gpu_peak_mib") for row in after]
        if all(peak is not None for peak in peaks):
            case["process_gpu_peak_sum_mib"] = sum(sorted(peaks, reverse=True)[:parallelism])
        events = sorted(
            event for row in after for event in ((row["case_start_seconds"], 1), (row["case_end_seconds"], -1))
        )
        active, previous, overlap = 0, 0.0, 0.0
        for at, change in events:
            if active >= 2:
                overlap += at - previous
            active += change
            previous = at
        case["overlap_fraction"] = overlap / wall
        if all(row["status"] == "PASS" for row in after):
            same_samples = all(a["sample_sha256"] == b["sample_sha256"] for a, b in zip(before, after, strict=True))
            case.update(passed=same_samples, identical_samples=same_samples)
            if not same_samples:
                case["error"] = "sampled batches differ from the serial baseline"
        else:
            case["error"] = "worker failure"
        if case["passed"]:
            case["wall_speedup"] = case["serial_wall_seconds"] / wall
            rates = [row["timing_projection"]["rates"] for row in before]
            case["warm_speedup"] = {
                phase: sum(rate[phase]["seconds_per_call"] for rate in rates)
                / queue_seconds(
                    [row["timing_projection"]["rates"][phase]["seconds_per_call"] for row in after], parallelism
                )
                for phase in rates[0]
            }
            evaluation = [row["timing_projection"].get("evaluation_seconds_per_vector_step", {}) for row in before + after]
            batches = set.intersection(*(set(rate) for rate in evaluation))
            case["evaluation_speedup"] = {
                batch: sum(rate[batch] for rate in evaluation[:len(before)])
                / queue_seconds([rate[batch] for rate in evaluation[len(before):]], parallelism)
                for batch in sorted(batches, key=int)
            }
        report["comparisons"].append(case)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["passed"] = all(row["status"] == "PASS" for row in report["runs"]) and all(
        case["passed"] for case in report["comparisons"]
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_summary(report["runs"], output, report["comparisons"])
    return report["passed"]


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
        "--warmup-updates", type=int, default=1, help="Initial updates excluded from warm timing means."
    )
    parser.add_argument(
        "--rollout-steps", type=int, help="Real batched collection steps (default: --updates); 0 skips DMC."
    )
    parser.add_argument(
        "--online-burst", type=int, default=0,
        help="Interleave online updates after collection in bursts of this size; 0 measures isolated phases.",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=0,
        help="Profile short deterministic evaluation episodes, without saving checkpoints or accuracy metrics.",
    )
    parser.add_argument(
        "--eval-batches", type=int, nargs="+", default=[5, 50],
        help="Evaluation batch sizes profiled when --eval-steps is positive.",
    )
    parser.add_argument(
        "--compare-gradient-batch", type=int, nargs="+", default=[],
        help="Compare TS planner microbatches, e.g. 128 256; the first is the serial baseline.",
    )
    parser.add_argument("--override", action="append", default=[], help="Hydra matrix override.")
    parser.add_argument("--model-override", action="append", default=[], help="Hydra override for each selected model.")
    parser.add_argument(
        "--compare-parallel", action="store_true", help="Benchmark exactly two runs serially, then together."
    )
    parser.add_argument(
        "--scenario-workers",
        type=int,
        nargs="+",
        choices=(2, 3),
        default=[],
        help="Compare each model across scenarios: shared serial baseline, then worker limits 2 and/or 3.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("MODEL_A", "MODEL_B"),
        help="Repeat for specific concurrent pairs; serial baselines are shared.",
    )
    parser.add_argument("--compare-compile", action="store_true", help="Compare eager and compiled Dreamer modules.")
    parser.add_argument(
        "--compare-storage",
        type=Path,
        help="Compare with an existing copy of the same datasets at this root (no automatic copy).",
    )
    parser.add_argument(
        "--stage-storage", type=Path,
        help="Copy selected datasets here with SHA-256 verification, then compare source/local reads. Requires enough local disk space.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(0 if run_worker(json.load(sys.stdin)) else 1)
    if args.updates < 1:
        parser.error("--updates must be positive")
    if args.warmup_updates < 0:
        parser.error("--warmup-updates cannot be negative")
    if args.rollout_steps is not None and args.rollout_steps < 0:
        parser.error("--rollout-steps cannot be negative")
    rollout_steps = args.updates if args.rollout_steps is None else args.rollout_steps
    if args.online_burst < 0 or (args.online_burst and rollout_steps * args.online_burst < args.updates):
        parser.error("--online-burst must be nonnegative and collection must provide enough bursts for --updates")
    if args.eval_steps < 0 or any(batch < 1 for batch in args.eval_batches):
        parser.error("Evaluation steps must be nonnegative and batch sizes positive")
    if args.compare_gradient_batch and (
        len(args.compare_gradient_batch) < 2
        or min(args.compare_gradient_batch) < 1
        or len(set(args.compare_gradient_batch)) != len(args.compare_gradient_batch)
    ):
        parser.error("--compare-gradient-batch needs at least two distinct positive sizes")
    if args.stage_storage and args.compare_storage:
        parser.error("Use --stage-storage to copy, or --compare-storage for existing copies")
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
        paired_models = list(dict.fromkeys(name for pair in args.pair for name in pair))
        scenarios = args.scenarios or list(matrix.scenarios)
        models = args.models or paired_models or list(entries)
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
                        "warmup_updates": args.warmup_updates,
                        "rollout_steps": rollout_steps,
                        "online_burst": args.online_burst,
                        "eval_steps": args.eval_steps,
                        "eval_batches": list(dict.fromkeys(args.eval_batches)) if args.eval_steps else [],
                        "seed_runs": len(matrix.seeds),
                        "result_path": str(output / scenario / name / "resources.json"),
                    }
                )
    if not jobs:
        parser.error("The selected matrix contains no runs")
    if len({job["name"] for job in jobs}) != len(jobs):
        parser.error("Duplicate scenarios/models would write the same diagnostic files")
    if args.scenario_workers:
        if any((args.pair, args.compare_parallel, args.compare_compile, args.compare_storage,
                args.stage_storage, args.compare_gradient_batch)):
            parser.error("--scenario-workers is separate from pair/compile/storage comparisons")
        if len(set(args.scenario_workers)) != len(args.scenario_workers):
            parser.error("Duplicate scenario worker limits would overwrite reports")
        if max(args.scenario_workers) > len(scenarios):
            parser.error("Select at least as many scenarios as the largest --scenario-workers limit")
    if args.pair and args.compare_parallel:
        parser.error("Use --pair or --compare-parallel, not both")
    if args.compare_parallel and (len(jobs) != 2 or len(set(scenarios)) != 1 or len(set(models)) != 2):
        parser.error("--compare-parallel requires one scenario and two distinct models")
    pairs = args.pair or ([models] if args.compare_parallel else [])
    if pairs and (len(scenarios) != 1 or any(len(set(pair)) != 2 or set(pair) - set(models) for pair in pairs)):
        parser.error("Pairs require one scenario and two distinct selected models per pair")
    if args.compare_compile and not any(job["config"]["model_family"] == "dreamer" for job in jobs):
        parser.error("--compare-compile requires at least one Dreamer model")
    if args.compare_gradient_batch and "temporal_straightening/default" not in models:
        parser.error("--compare-gradient-batch requires Temporal Straightening")
    if args.compare_gradient_batch and not rollout_steps and not args.eval_steps:
        parser.error("--compare-gradient-batch needs collection or evaluation to exercise the planner")
    if not args.dry_run and any(
        job["config"]["model_family"] == "dreamer"
        and (args.compare_compile or job["config"]["model"]["compile"])
        and job["config"]["device"].split(":", 1)[0] == "cuda"
        for job in jobs
    ):
        try:
            check_cuda_compile_dependencies()
        except RuntimeError as error:
            parser.exit(2, f"Compile preflight | {error}\n")
    storage = args.stage_storage or args.compare_storage
    storage = storage.expanduser().resolve() if storage else None
    staging = []
    if args.stage_storage and not args.dry_run:
        from scripts.stage_dmc_data import stage_dataset

        for dataset in sorted({job["config"]["scenario"]["dataset"] for job in jobs}):
            staging.append(stage_dataset(root / dataset, storage / dataset))
        output.mkdir(parents=True, exist_ok=True)
        (output / "staging.json").write_text(json.dumps(staging, indent=2) + "\n", encoding="utf-8")
    if storage is not None and not args.dry_run:
        # Validate before spending GPU time; measured sample hashes also check the data actually consumed.
        for dataset in {job["config"]["scenario"]["dataset"] for job in jobs}:
            source, target = root / dataset, storage / dataset
            if source.resolve() == target.resolve():
                parser.error("--compare-storage must name a different dataset location")
            for filename in ("metadata.json", "data.hdf5"):
                if not (source / filename).is_file() or not (target / filename).is_file():
                    parser.error(f"Storage comparison requires both copies of {dataset}/{filename}")
            if json.loads((source / "metadata.json").read_text()) != json.loads((target / "metadata.json").read_text()):
                parser.error(f"Dataset metadata differs between storage locations: {dataset}")
            if (source / "data.hdf5").stat().st_size != (target / "data.hdf5").stat().st_size:
                parser.error(f"Dataset file size differs between storage locations: {dataset}")
        print("Storage | no cache flushing; sampled batch equality checked; staging time is separate from worker timing", flush=True)
    print(
        f"Training smoke | runs={len(jobs)} | updates={args.updates} expert + {args.updates} online"
        f" | collection_steps={jobs[0]['rollout_steps']}"
        f" | online_burst={args.online_burst} | eval_steps={args.eval_steps} | checkpoints=disabled",
        flush=True,
    )
    results = []
    if args.compare_gradient_batch:
        print(f"Planning | gradient_batches={args.compare_gradient_batch} | serial baseline={args.compare_gradient_batch[0]}", flush=True)
    if args.scenario_workers:
        print(
            f"Scenario concurrency | variants={len(models)} | levels={[1, *args.scenario_workers]}"
            f" | worker_runs={len(jobs) * (1 + len(args.scenario_workers))} | no memory prefilter",
            flush=True,
        )
    if pairs or args.compare_compile or storage is not None or args.scenario_workers or args.compare_gradient_batch:
        if args.dry_run:
            print(
                f"Comparison plan | serial={len(jobs)} | pairs={pairs}"
                f" | compile_dreamer={args.compare_compile} | storage={storage}"
                f" | scenario_workers={args.scenario_workers}"
                f" | gradient_batches={args.compare_gradient_batch} | stage_storage={bool(args.stage_storage)}"
            )
        else:
            raise SystemExit(
                0
                if compare_runs(
                    jobs,
                    output,
                    pairs=pairs,
                    compile_dreamer=args.compare_compile,
                    storage=storage,
                    scenario_workers=args.scenario_workers,
                    gradient_batches=args.compare_gradient_batch,
                )
                else 1
            )
    for index, job in enumerate(jobs, 1):
        if args.dry_run:
            config = job["config"]
            print(
                f"PLAN | {index}/{len(jobs)} | {job['name']}"
                f" | batch={config['replay']['batch_size']} | sequence={config['replay']['sequence_length']}"
            )
            continue
        result = run_job(job)
        results.append(result)
        report = {
            "runs": results,
            "memory_only_candidates": memory_candidates(results),
            "timing": timing_summary(results, jobs),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.dry_run:
        failures = sum(row["status"] != "PASS" for row in results)
        write_summary(results, output)
        pairs = report["memory_only_candidates"]["pairs"]
        print(f"\nMemory-only candidate pairs: {len(pairs)}; concurrent testing is still required.")
        for pair in pairs[:5]:
            print(f"  {' + '.join(pair['models'])}: estimated {pair['estimated_mib'] / 1024:.2f} GiB")
        raise SystemExit(int(bool(failures)))


if __name__ == "__main__":
    main()
