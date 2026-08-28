#!/usr/bin/env python3
"""Verify the DMC environment and production Mamba3 kernels."""

import argparse
import ctypes.util
import importlib
import importlib.metadata
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]


def expected_versions():
    expected = {
        "mamba-ssm": "2.3.2.post1",
        "torch": "2.8.0",
        "triton": "3.4.0",
    }
    for line in (REPO_DIR / "requirements/mamba3-cu128.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, version = line.split("==", maxsplit=1)
            expected[package] = version
    return expected


def check_versions():
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError(f"Expected Python 3.10 or 3.11, got {sys.version.split()[0]}")
    if ctypes.util.find_library("z3") is None:
        raise RuntimeError("libz3.so is unavailable; install libz3-dev")

    for package, expected in expected_versions().items():
        installed = importlib.metadata.version(package)
        comparable = installed.split("+", maxsplit=1)[0] if package == "torch" else installed
        if comparable != expected:
            raise RuntimeError(f"Expected {package}=={expected}, got {installed}")


def check_mamba3():
    import torch
    from hydra import compose, initialize_config_dir

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
    check_versions()
    check_mamba3()
    print("DMC setup check passed.")


if __name__ == "__main__":
    main()
