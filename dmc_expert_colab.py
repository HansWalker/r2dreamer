"""Small Colab setup helper for the DMC expert notebook."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PACKAGES = """
numpy==2.0.2 pyyaml zarr<3 huggingface_hub dm_control==1.0.28 mujoco==3.3.0
omegaconf hydra-core tensorboard>=2.20,<3 gymnasium==1.2.0 tensordict torchrl
kornia termcolor tqdm pandas moviepy imageio imageio-ffmpeg h5py wheel ninja packaging einops
""".split()


def run(label: str, cmd, *, cwd=None, env=None):
    print(f"\n== {label} ==")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def sync_tdmpc2(tdmpc2_dir: Path):
    repo = "https://github.com/nicklashansen/tdmpc2.git"
    if tdmpc2_dir.exists():
        run("Update TD-MPC2", ["git", "remote", "set-url", "origin", repo], cwd=tdmpc2_dir)
        run("Pull TD-MPC2", ["git", "pull", "--ff-only"], cwd=tdmpc2_dir)
    else:
        run("Clone TD-MPC2", ["git", "clone", repo, str(tdmpc2_dir)])


def clear_mamba_modules():
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]


def install_mamba3():
    run("Remove optional TileLang kernels", [sys.executable, "-m", "pip", "uninstall", "-y", "tilelang"])
    try:
        clear_mamba_modules()
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3
    except Exception:
        env = dict(os.environ, MAMBA_FORCE_BUILD="TRUE")
        run(
            "Install Mamba3 from source",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "git+https://github.com/state-spaces/mamba.git",
                "--no-build-isolation",
            ],
            env=env,
        )
        run("Remove optional TileLang kernels", [sys.executable, "-m", "pip", "uninstall", "-y", "tilelang"])
        clear_mamba_modules()
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3


def setup_colab(workdir: Path, r2dreamer_dir: Path):
    tdmpc2_dir = Path(workdir) / "tdmpc2"
    data_dir = Path(workdir) / "data" / "dmc_expert"
    data_dir.mkdir(parents=True, exist_ok=True)

    sync_tdmpc2(tdmpc2_dir)
    run("Install Python packages", [sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    mamba3 = install_mamba3()

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TDMPC2_DIR"] = str(tdmpc2_dir)
    os.environ["DMC_EXPERT_DATA_DIR"] = str(data_dir)

    import torch

    print("\nReady")
    print("R2Dreamer:", r2dreamer_dir)
    print("TD-MPC2:", tdmpc2_dir)
    print("Data:", data_dir)
    print("Mamba3:", mamba3)
    print("CUDA:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("Use a Colab GPU runtime before running this notebook.")

    return tdmpc2_dir, data_dir, mamba3
