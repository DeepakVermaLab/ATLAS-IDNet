from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from atlas_idnet.config import write_json


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_runtime_provenance() -> dict:
    payload = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": {
            "cwd": os.getcwd(),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "packages": {
            name: _package_version(name)
            for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "torch", "pyarrow"]
        },
        "gpu": {
            "nvidia_smi": _command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        },
    }
    try:
        import psutil

        payload["hardware"] = {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
        }
    except Exception:
        payload["hardware"] = {
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_count": None,
            "ram_bytes": None,
        }
    try:
        import torch

        payload["torch_runtime"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda_version": torch.version.cuda,
        }
    except Exception:
        payload["torch_runtime"] = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "torch_cuda_version": None,
        }
    return payload


def write_runtime_provenance(output_dir: Path) -> dict:
    payload = collect_runtime_provenance()
    write_json(output_dir / "runtime_provenance.json", payload)
    return payload
