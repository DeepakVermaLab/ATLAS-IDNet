from __future__ import annotations

from _bootstrap import add_src_to_path, base_parser, load_cfg_and_run_dir


def main() -> None:
    add_src_to_path()
    parser = base_parser("Build paper-ready ATLAS-IDNet result tables and figures from existing artifacts.")
    parser.add_argument("--top-n-features", default=10, type=int, help="Number of top explainability features to report per dataset.")
    args = parser.parse_args()
    _, run_dir = load_cfg_and_run_dir(args)

    from atlas_idnet.reporting.paper_summary import build_paper_summary

    summary = build_paper_summary(run_dir, task=args.task, top_n_features=args.top_n_features)
    print("Paper summary artifacts:", summary["output_dir"])
    print(summary)


if __name__ == "__main__":
    main()

