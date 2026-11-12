from __future__ import annotations

from _bootstrap import add_src_to_path, base_parser, load_cfg_and_run_dir


def main() -> None:
    add_src_to_path()
    parser = base_parser("Run ATLAS-IDNet explainability and agentic report stages.")
    parser.add_argument("--agent-only", action="store_true", help="Only rebuild the grounded agentic report from existing artifacts.")
    args = parser.parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)

    from atlas_idnet.pipeline import run_agentic_stage, run_explainability_stage

    if args.agent_only:
        run_agentic_stage(cfg, run_dir, dataset=args.dataset, task=args.task)
    else:
        run_explainability_stage(cfg, run_dir, dataset=args.dataset, task=args.task, max_rows=args.max_rows)
        run_agentic_stage(cfg, run_dir, dataset=args.dataset, task=args.task)
    print("Explainability/agent artifacts:", run_dir)


if __name__ == "__main__":
    main()
