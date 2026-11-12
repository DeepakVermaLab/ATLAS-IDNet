from pathlib import Path


PROJECT_ROOT = Path("/Volumes/LocalDrive1/onlyscholar-projects/ATLAS-IDNet")
DATASET_ROOT = Path("/Volumes/LocalDrive1/Dataset/Security")

DATASETS = {
    "CICIoT2023": {
        "target_dir": DATASET_ROOT / "CICIoT2023",
        "official_page": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        "official_download_form": "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/",
        "kaggle_slug": "madhavmalhotra/unb-cic-iot-dataset",
        "kaggle_slugs": [
            "madhavmalhotra/unb-cic-iot-dataset",
            "himadri07/ciciot2023",
            "akashdogra/cic-iot-2023",
        ],
    },
    "NF-UQ-NIDS-v2": {
        "target_dir": DATASET_ROOT / "NF-UQ-NIDS-v2",
        "official_page": "https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets",
        "official_rdm_file": "https://rdm.uq.edu.au/files/e2412450-ef9c-11ed-827d-e762de186848",
        "kaggle_slug": "aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset",
        "kaggle_slugs": [
            "aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset",
            "dhoogla/nfuqnidsv2",
        ],
    },
}
