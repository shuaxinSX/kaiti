"""
生成 reference 参考解与 comparison 产物。
"""

import argparse
import json

from src.eval.reference_eval import solve_reference_from_config, solve_reference_from_run_dir
from src.train.runner import configure_logging


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Solve and export reference evaluation artifacts.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run-dir",
        default=None,
        help="Existing run directory containing config_merged.yaml and optional model_state.pt.",
    )
    source_group.add_argument(
        "--config",
        default=None,
        help="Base config path for standalone reference generation.",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional overlay config path used only in config mode.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--velocity-model",
        default=None,
        help="Optional medium override used only in config mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Required in config mode. Ignored in run-dir mode.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.run_dir is not None:
        if args.overlay is not None or args.velocity_model is not None or args.output_dir is not None:
            parser.error("--overlay, --velocity-model, and --output-dir are only valid with --config.")
        configure_logging("INFO")
        summary = solve_reference_from_run_dir(args.run_dir, device=args.device)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required when using --config.")
        configure_logging("INFO")
        summary = solve_reference_from_config(
            args.config,
            overlay_path=args.overlay,
            device=args.device,
            velocity_model=args.velocity_model,
            output_dir=args.output_dir,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
