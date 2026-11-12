from pathlib import Path


DRIVE_REMOTE_ROOT = "gdrive:projects/Dataset/Security"

DATASETS = {
    "CICIoT2023": {
        "local_source": Path("/Volumes/LocalDrive1/Dataset/Security/CICIoT2023/raw/wataiData"),
        "drive_destination": f"{DRIVE_REMOTE_ROOT}/CICIoT2023",
        "expected_min_csv_count": 169,
    },
    "NF-UQ-NIDS-v2": {
        "local_source": Path("/Volumes/LocalDrive1/Dataset/Security/NF-UQ-NIDS-v2/raw/NF-UQ-NIDS-v2.csv"),
        "drive_destination": f"{DRIVE_REMOTE_ROOT}/NF-UQ-NIDS-v2",
        "expected_min_csv_count": 1,
    },
}

