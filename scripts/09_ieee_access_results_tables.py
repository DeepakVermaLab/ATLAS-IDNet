from __future__ import annotations

from _bootstrap import add_src_to_path, base_parser, load_cfg_and_run_dir


def main() -> None:
    add_src_to_path()
    parser = base_parser("Build IEEE Access formatted result tables from paper_summary artifacts.")
    parser.add_argument("--top-n-features", default=10, type=int, help="Number of top explainability features to include per dataset.")
    args = parser.parse_args()
    _, run_dir = load_cfg_and_run_dir(args)

    from atlas_idnet.reporting.ieee_access_tables import build_ieee_access_tables

    inventory = build_ieee_access_tables(run_dir, top_n_features=args.top_n_features)
    print("IEEE Access table artifacts:", inventory["output_dir"])
    print(inventory)


if __name__ == "__main__":
    main()

