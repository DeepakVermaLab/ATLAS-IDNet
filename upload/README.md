# ATLAS-IDNet Google Drive Upload

Upload destination:

```text
gdrive:projects/Dataset/Security
```

Dataset destinations:

```text
gdrive:projects/Dataset/Security/CICIoT2023
gdrive:projects/Dataset/Security/NF-UQ-NIDS-v2
```

Run the notebooks independently, in parallel if needed:

- `notebooks/01_upload_ciciot2023_to_drive.ipynb`
- `notebooks/02_upload_nf_uq_nids_v2_to_drive.ipynb`

The upload uses `rclone copy --copy-links` because the local dataset folders point into the KaggleHub cache by symlink. AppleDouble sidecars, `.DS_Store`, and partial files are excluded.

