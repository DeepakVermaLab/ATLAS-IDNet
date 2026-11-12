from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path


def main() -> None:
    repo_root = add_src_to_path()
    parser = argparse.ArgumentParser(
        description="Generate XFig-style manuscript diagrams and data-backed result figures for ATLAS-IDNet."
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "paper" / "v1" / "source" / "figures" / "generated"),
        help="Directory where PNG, SVG, PDF, snippets, and inventory files are written.",
    )
    parser.add_argument(
        "--literature-csv",
        default=str(repo_root / "paper" / "ls" / "literature_matrix.csv"),
        help="Verified literature matrix CSV used for the literature evidence heatmap.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional completed ATLAS-IDNet run directory. If it contains paper_summary tables, result figures are generated.",
    )
    args = parser.parse_args()

    from atlas_idnet.reporting.xfig_style_figures import build_xfig_style_figures

    inventory = build_xfig_style_figures(
        output_dir=Path(args.output_dir),
        literature_csv=Path(args.literature_csv) if args.literature_csv else None,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print("Figure output directory:", inventory["output_dir"])
    print("Generated figures:", len(inventory["outputs"]))
    if inventory["skipped"]:
        print("Skipped data-backed figures:")
        for stem, reason in inventory["skipped"].items():
            print(f"  - {stem}: {reason}")
    print("LaTeX snippets:", inventory["latex_snippets"])
    print("Inventory:", Path(inventory["output_dir"]) / "figure_inventory.json")


if __name__ == "__main__":
    main()

