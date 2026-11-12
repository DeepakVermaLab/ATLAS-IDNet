from __future__ import annotations

from _bootstrap import add_src_to_path, base_parser, load_cfg_and_run_dir


def main() -> None:
    add_src_to_path()
    parser = base_parser("Run ATLAS-IDNet EDA.")
    args = parser.parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)

    from atlas_idnet.pipeline import run_eda_stage

    run_eda_stage(cfg, run_dir, dataset=args.dataset, max_rows=args.max_rows)
    print("EDA artifacts:", run_dir)


if __name__ == "__main__":
    main()
