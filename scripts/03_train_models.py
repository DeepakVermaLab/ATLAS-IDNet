from __future__ import annotations

from _bootstrap import add_src_to_path, base_parser, load_cfg_and_run_dir


def main() -> None:
    add_src_to_path()
    parser = base_parser("Train ATLAS-IDNet and baselines.")
    parser.add_argument("--skip-baselines", action="store_true", help="Do not train logistic regression and random forest baselines.")
    parser.add_argument("--skip-atlas", action="store_true", help="Do not train the ATLAS-IDNet neural model.")
    args = parser.parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)

    from atlas_idnet.pipeline import run_training_stage

    run_training_stage(
        cfg,
        run_dir,
        dataset=args.dataset,
        task=args.task,
        max_rows=args.max_rows,
        train_baselines=not args.skip_baselines,
        train_atlas=not args.skip_atlas,
    )
    print("Training artifacts:", run_dir)


if __name__ == "__main__":
    main()
