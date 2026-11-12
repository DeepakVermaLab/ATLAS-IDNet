# Local Dataset Status

Checked local root:

```text
/Volumes/LocalDrive1/Dataset/Security
```

## CICIoT2023

Local source:

```text
/Volumes/LocalDrive1/Dataset/Security/CICIoT2023/raw/wataiData
```

Status:

- Exists: yes
- Source type: symlink into KaggleHub cache
- CSV files: 169
- Data size through symlink: about 13 GB
- First CSV header begins with `flow_duration,Header_Length,Protocol Type,...`

Upload destination:

```text
gdrive:projects/Dataset/Security/CICIoT2023
```

## NF-UQ-NIDS-v2

Local source:

```text
/Volumes/LocalDrive1/Dataset/Security/NF-UQ-NIDS-v2/raw/NF-UQ-NIDS-v2.csv
```

Status:

- Exists: yes
- Source type: symlink into KaggleHub cache
- CSV files: 1
- File size: 13,729,330,230 bytes
- First CSV header begins with `IPV4_SRC_ADDR,L4_SRC_PORT,IPV4_DST_ADDR,...`

Upload destination:

```text
gdrive:projects/Dataset/Security/NF-UQ-NIDS-v2
```

Note: upload must use `rclone copy --copy-links` so the symlink targets are uploaded instead of only symlink metadata.

