from __future__ import annotations

from _bootstrap import add_src_to_path, base_parser, load_cfg_and_run_dir


def main() -> None:
    add_src_to_path()
    parser = base_parser("Check ATLAS-IDNet runtime and dataset visibility.")
    args = parser.parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)

    from atlas_idnet.pipeline import run_environment_check

    result = run_environment_check(cfg, run_dir)
    print("Run directory:", run_dir)
    for name, payload in result["datasets"].items():
        print(f"{name}: files={payload['file_count']} bytes={payload['total_bytes']}")


if __name__ == "__main__":
    main()
