from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVE_FOLDER_ID = "1llQSxhyHiks9lOTX8ZZRMtpn1YnGx8AR"


@dataclass
class DatasetConfig:
    name: str
    local_roots: list[str]
    drive_roots: list[str]
    label_candidates: list[str] = field(default_factory=lambda: ["label", "Label", "Attack", "Attack_type", "Class"])
    max_files: int | None = None
    max_rows: int = 250_000
    chunk_size: int = 100_000


@dataclass
class ExperimentConfig:
    project_name: str = "ATLAS-IDNet"
    seed: int = 42
    target_drive_folder_id: str = DRIVE_FOLDER_ID
    results_root: str | None = None
    work_root: str | None = None
    test_size: float = 0.2
    val_size: float = 0.1
    batch_size: int = 1024
    epochs: int = 35
    learning_rate: float = 8e-4
    hidden_dim: int = 256
    dropout: float = 0.15
    patience: int = 8
    explain_samples: int = 512
    latency_samples: int = 4096
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)


def default_config() -> ExperimentConfig:
    return ExperimentConfig(
        datasets={
            "CICIoT2023": DatasetConfig(
                name="CICIoT2023",
                local_roots=[
                    "/Volumes/LocalDrive1/Dataset/Security/CICIoT2023/raw/wataiData",
                    "/Volumes/LocalDrive1/Dataset/Security/CICIoT2023",
                ],
                drive_roots=[
                    "/content/drive/MyDrive/projects/Dataset/Security/CICIoT2023",
                    "/content/drive/MyDrive/projects/Dataset/Security/CICIoT2023/wataiData",
                ],
                max_rows=350_000,
            ),
            "NF-UQ-NIDS-v2": DatasetConfig(
                name="NF-UQ-NIDS-v2",
                local_roots=[
                    "/Volumes/LocalDrive1/Dataset/Security/NF-UQ-NIDS-v2/raw/NF-UQ-NIDS-v2.csv",
                    "/Volumes/LocalDrive1/Dataset/Security/NF-UQ-NIDS-v2",
                ],
                drive_roots=[
                    "/content/drive/MyDrive/projects/Dataset/Security/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv",
                    "/content/drive/MyDrive/projects/Dataset/Security/NF-UQ-NIDS-v2",
                    "/kaggle/input/nfuqnidsv2-network-intrusion-detection-dataset/NF-UQ-NIDS-v2.csv",
                ],
                label_candidates=["Attack", "Label", "label", "class"],
                max_rows=350_000,
            ),
        }
    )


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def in_colab() -> bool:
    return Path("/content").exists() and "COLAB_RELEASE_TAG" in os.environ


def mount_google_drive_if_available() -> bool:
    if not in_colab():
        return False
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        return True
    except Exception:
        return False


def resolve_results_root(cfg: ExperimentConfig, run_name: str | None = None) -> Path:
    if cfg.results_root:
        root = Path(cfg.results_root)
    elif Path("/content/drive/MyDrive").exists() or mount_google_drive_if_available():
        root = Path("/content/drive/MyDrive/projects/ATLAS-IDNet/results")
    else:
        root = repo_root() / "implementation" / "results"
    if run_name:
        root = root / run_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_work_root(cfg: ExperimentConfig) -> Path:
    if cfg.work_root:
        root = Path(cfg.work_root)
    elif Path("/content").exists():
        root = Path("/content/atlas_idnet_work")
    else:
        root = repo_root() / "implementation" / "work"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    serializable = asdict(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    datasets_payload = payload.get("datasets", {})
    datasets = {
        name: DatasetConfig(**dataset_payload)
        for name, dataset_payload in datasets_payload.items()
    }
    values = {k: v for k, v in payload.items() if k != "datasets"}
    return ExperimentConfig(**values, datasets=datasets)


def load_config(path: Path | str | None = None) -> ExperimentConfig:
    if path is None:
        return default_config()
    return config_from_dict(read_json(Path(path)))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
