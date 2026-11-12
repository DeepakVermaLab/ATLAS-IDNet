# ATLAS-IDNet Dataset Download Notebooks

This folder contains two dataset-wise download notebooks for ATLAS-IDNet. They are independent, so both notebooks can be run in parallel.

- `notebooks/01_download_ciciot2023.ipynb`
- `notebooks/02_download_nf_uq_nids_v2.ipynb`

Shared Python code is kept under:

- `src/atlas_idnet_download/download_config.py`

Corrected local dataset root:

```text
/Volumes/LocalDrive1/Dataset/Security
```

The requested path spelling `LocalDrive1/Dataset/Securty` was corrected to `LocalDrive1/Dataset/Security`.

Dataset folders created by the notebooks:

```text
/Volumes/LocalDrive1/Dataset/Security/CICIoT2023
/Volumes/LocalDrive1/Dataset/Security/NF-UQ-NIDS-v2
```

Notes:

- CICIoT2023's official CIC source currently uses a registration/download form, so the notebook supports Kaggle API download and also accepts a manually downloaded archive placed in the dataset folder.
- NF-UQ-NIDS-v2 is listed by University of Queensland; if the public RDM endpoint blocks scripted downloads, the notebook also supports Kaggle API download or a manually placed archive/CSV.
- Configure Kaggle credentials before running the Kaggle path: place `kaggle.json` at `~/.kaggle/kaggle.json` or export `KAGGLE_USERNAME` and `KAGGLE_KEY`.
