#!/usr/bin/env python3
"""Verify dependencies, DMC rendering, TD-MPC2, and production Mamba3 kernels."""

import argparse
import ctypes.util
import importlib
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_DIR = Path(__file__).resolve().parents[1]
DEPENDENCY_OVERRIDES = {
    ("mamba-ssm", "apache-tvm-ffi"),
    ("quack-kernels", "nvidia-cutlass-dsl"),
    ("torch", "triton"),
}


def expected_versions():
    expected = {
        "mamba-ssm": "2.3.2.post1",
        "torch": "2.8.0",
    }
    for line in (REPO_DIR / "requirements/mamba3-cu128.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, version = line.split("==", maxsplit=1)
            expected[package] = version
    return expected


def check_dependencies():
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError(f"Expected Python 3.10 or 3.11, got {sys.version.split()[0]}")
    if ctypes.util.find_library("z3") is None:
        raise RuntimeError("libz3.so is unavailable; install libz3-dev")

    expected = expected_versions()
    version_errors = []
    for package, version in expected.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version_errors.append(f"missing {package}=={version}")
            continue
        comparable = installed.split("+", maxsplit=1)[0] if package == "torch" else installed
        if comparable != version:
            version_errors.append(f"expected {package}=={version}, got {installed}")
    if version_errors:
        raise RuntimeError("Pinned package check failed:\n  " + "\n  ".join(version_errors))

    pending = [("r2dreamer", ("", "dmc")), *[(name, ("",)) for name in expected]]
    checked = set()
    errors = []
    overrides = set()
    while pending:
        package, extras = pending.pop()
        package = canonicalize_name(package)
        if package in checked:
            continue
        checked.add(package)
        distribution = importlib.metadata.distribution(package)
        for text in distribution.requires or ():
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in extras):
                continue
            dependency = canonicalize_name(requirement.name)
            try:
                installed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"{package} requires missing package {requirement}")
                continue
            if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
                pair = (package, dependency)
                if pair in DEPENDENCY_OVERRIDES:
                    overrides.add(pair)
                else:
                    errors.append(f"{package} requires {requirement}, got {installed}")
            pending.append((dependency, ("",)))

    if errors:
        raise RuntimeError("Dependency check failed:\n  " + "\n  ".join(errors))
    print(f"Python dependency graph: passed ({len(overrides)} documented overrides)")


def check_runtime():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cloudpickle
    import h5py
    import numpy as np
    import torch
    from dm_control import suite
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter

    for module in ("gymnasium", "huggingface_hub", "mujoco", "tensorboard"):
        importlib.import_module(module)

    payload = {"value": np.arange(4, dtype=np.float32)}
    if cloudpickle.loads(cloudpickle.dumps(payload))["value"].dtype != np.float32:
        raise RuntimeError("cloudpickle round trip failed")
    TensorDict({"value": torch.ones(2, 1)}, batch_size=[2])

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        with h5py.File(path / "check.hdf5", "w") as handle:
            handle.create_dataset("value", data=payload["value"])
        with h5py.File(path / "check.hdf5", "r") as handle:
            if not np.array_equal(handle["value"][:], payload["value"]):
                raise RuntimeError("HDF5 round trip failed")
        writer = SummaryWriter(path / "tensorboard")
        writer.add_scalar("check/value", 1.0, 0)
        writer.close()

    environment = suite.load("ball_in_cup", "catch", task_kwargs={"random": 0})
    try:
        environment.reset()
        spec = environment.action_spec()
        environment.step(np.zeros(spec.shape, dtype=spec.dtype))
        image = environment.physics.render(height=16, width=16, camera_id=0)
    finally:
        environment.physics.free()
    if image.shape != (16, 16, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"Unexpected DMC render: shape={image.shape}, dtype={image.dtype}")

    print(f"Core Python, HDF5, TensorDict, TensorBoard, and DMC {os.environ['MUJOCO_GL']} render: passed")


def check_tdmpc2():
    root = os.environ.get("TDMPC2_DIR")
    if not root:
        print("TD-MPC2 checkout: skipped (TDMPC2_DIR is unset)")
        return

    package_root = Path(root).expanduser().resolve() / "tdmpc2"
    config_path = package_root / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"TDMPC2_DIR does not contain tdmpc2/config.yaml: {root}")
    sys.path.insert(0, str(package_root))
    common = importlib.import_module("common")
    tdmpc2 = importlib.import_module("tdmpc2")
    if not hasattr(tdmpc2, "TDMPC2") or not hasattr(common, "MODEL_SIZE"):
        raise RuntimeError(f"Incompatible TD-MPC2 checkout: {root}")
    print(f"TD-MPC2 checkout: passed ({Path(root).expanduser().resolve()})")


def check_mamba3():
    import torch
    from hydra import compose, initialize_config_dir

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8"))
    nvcc = cuda_home / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler not found: {nvcc}")
    if "release 12.8," not in subprocess.check_output([nvcc, "--version"], text=True):
        raise RuntimeError(f"Expected CUDA 12.8 toolkit at {cuda_home}")
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ["PATH"] = f"{nvcc.parent}:{os.environ['PATH']}"

    from models.shared.mamba3 import Mamba3Layer

    for module in (
        "cuda.bindings.driver",
        "cuda.pathfinder",
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
    ):
        importlib.import_module(module)

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected PyTorch CUDA 12.8, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU")

    with initialize_config_dir(config_dir=str(REPO_DIR / "configs"), version_base=None):
        dreamer = compose(config_name="offline_dmc_expert_mamba3_vision")
        storm = compose(config_name="storm_dmc_mamba_vision")

    configs = (
        ("Dreamer", dreamer.model.deter, dreamer.model.rssm.mamba3, torch.float16, dreamer.replay.batch_size),
        (
            "STORM",
            storm.storm_model.hidden_dim,
            storm.storm_model.recurrent.mamba3,
            torch.bfloat16,
            storm.replay.batch_size,
        ),
    )
    device = torch.device("cuda")
    for name, d_model, config, amp_dtype, batch_size in configs:
        batch_size = int(batch_size)
        layer = Mamba3Layer(
            d_model=int(d_model),
            config=config,
            layer_idx=0,
        ).to(device)

        sequence = torch.randn(
            batch_size,
            int(config.chunk_size),
            int(d_model),
            device=device,
            requires_grad=True,
        )
        with torch.autocast("cuda", dtype=amp_dtype):
            output = layer(sequence)
            output.float().square().mean().backward()
        if not torch.isfinite(output).all() or layer.mamba.in_proj.weight.grad is None:
            raise RuntimeError(f"{name} Mamba3 sequence training failed")

        layer.zero_grad(set_to_none=True)
        token = sequence[:, 0].detach().contiguous()
        with torch.autocast("cuda", dtype=amp_dtype):
            recurrent_cache = layer.initial_context(batch_size, device=device, dtype=amp_dtype)
            recurrent, *recurrent_cache = layer.recurrent_step(token, *recurrent_cache)
            recurrent.float().square().mean().backward()
        if not torch.isfinite(recurrent).all() or layer.mamba.in_proj.weight.grad is None:
            raise RuntimeError(f"{name} Mamba3 recurrent training failed")
        if not torch.allclose(output[:, 0], recurrent, rtol=2e-2, atol=2e-2):
            error = float((output[:, 0] - recurrent).abs().max())
            raise RuntimeError(f"{name} Mamba3 sequence and recurrent steps differ (max error {error:.3g})")

        layer.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            fast_cache = layer.initial_context(batch_size, device=device, dtype=amp_dtype)
            step, *fast_cache = layer.fast_step(token, *fast_cache)
        if not torch.isfinite(step).all() or not all(value.is_contiguous() for value in fast_cache):
            raise RuntimeError(f"{name} Mamba3 cached step failed")
        if not torch.allclose(recurrent, step, rtol=2e-2, atol=2e-2):
            error = float((recurrent - step).abs().max())
            raise RuntimeError(f"{name} Mamba3 recurrent and fused steps differ (max error {error:.3g})")
        for recurrent_state, fast_state in zip(recurrent_cache, fast_cache, strict=True):
            if not torch.allclose(recurrent_state, fast_state, rtol=2e-2, atol=2e-2):
                error = float((recurrent_state - fast_state).abs().max())
                raise RuntimeError(f"{name} Mamba3 recurrent and fused caches differ (max error {error:.3g})")
        torch.cuda.synchronize()
        print(f"{name:7} {str(amp_dtype).removeprefix('torch.')} Mamba3 sequence, recurrent, and cached steps: passed")

    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_dependencies()
    check_mamba3()
    check_runtime()
    check_tdmpc2()
    print("DMC setup check passed.")


if __name__ == "__main__":
    main()
