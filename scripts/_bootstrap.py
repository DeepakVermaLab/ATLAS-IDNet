from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "implementation" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Optional JSON config path. Defaults to built-in ATLAS-IDNet config.")
    parser.add_argument("--run-dir", default=None, help="Existing run directory. If omitted, a new run directory is created.")
    parser.add_argument("--run-name", default=None, help="Name for a new run directory.")
    parser.add_argument("--dataset", default="all", help="Dataset name or all.")
    parser.add_argument("--task", default="multiclass", choices=["binary", "multiclass"], help="Detection task.")
    parser.add_argument("--max-rows", default=None, type=int, help="Optional row cap for quick runs.")
    return parser


def load_cfg_and_run_dir(args):
    add_src_to_path()
    from atlas_idnet.config import load_config
    from atlas_idnet.pipeline import create_run_dir

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else create_run_dir(cfg, args.run_name)
    return cfg, run_dir
