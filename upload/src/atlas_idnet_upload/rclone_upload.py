from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from .upload_config import DATASETS


EXCLUDES = ["._*", ".DS_Store", "*.part"]


def find_rclone() -> str:
    env_path = os.environ.get("RCLONE_BIN")
    if env_path and Path(env_path).exists():
        return env_path
    found = shutil.which("rclone")
    if found:
        return found
    candidates = ["/opt/homebrew/bin/rclone", "/usr/local/bin/rclone", "/opt/local/bin/rclone"]
    candidates.extend(str(p) for p in Path("/opt/homebrew/Cellar/rclone").glob("*/bin/rclone"))
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    for shell in ["/bin/zsh", "/usr/bin/zsh", "/bin/bash", "/usr/bin/bash", "/bin/sh", "/usr/bin/sh"]:
        if not Path(shell).exists():
            continue
        try:
            shell_found = subprocess.run([shell, "-lc", "command -v rclone"], capture_output=True, text=True)
        except OSError:
            continue
        if shell_found.returncode == 0 and shell_found.stdout.strip():
            return shell_found.stdout.strip()
    checked = ", ".join(candidates)
    raise RuntimeError(f"rclone is not available in this runtime. Checked: {checked}")


def csv_count(path: Path) -> int:
    if path.is_file():
        return int(path.name.endswith(".csv") and not path.name.startswith("._"))
    return sum(1 for p in path.rglob("*.csv") if p.is_file() and not p.name.startswith("._"))


def build_rclone_copy_command(dataset_name: str, dry_run: bool = False) -> list[str]:
    cfg = DATASETS[dataset_name]
    source = cfg["local_source"]
    if not source.exists():
        raise FileNotFoundError(f"Missing local source: {source}")
    if csv_count(source) < cfg["expected_min_csv_count"]:
        raise RuntimeError(f"CSV count is lower than expected for {dataset_name}: {csv_count(source)}")

    cmd = [
        find_rclone(),
        "copy",
        str(source),
        cfg["drive_destination"],
        "--copy-links",
        "--progress",
        "--transfers",
        "4",
        "--checkers",
        "8",
        "--drive-chunk-size",
        "256M",
        "--stats",
        "30s",
    ]
    for pattern in EXCLUDES:
        cmd.extend(["--exclude", pattern])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def build_rclone_verify_command(dataset_name: str) -> list[str]:
    cfg = DATASETS[dataset_name]
    return [
        find_rclone(),
        "size",
        cfg["drive_destination"],
        "--exclude",
        "._*",
        "--exclude",
        ".DS_Store",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    copy_cmd = build_rclone_copy_command(args.dataset, dry_run=args.dry_run)
    print(" ".join(copy_cmd))
    subprocess.check_call(copy_cmd)
    if not args.dry_run:
        subprocess.check_call(build_rclone_verify_command(args.dataset))


if __name__ == "__main__":
    main()
