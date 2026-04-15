"""
为已有 run 目录补充评估 CSV 和热力图。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.runner import evaluate_saved_run


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate an existing run directory in-place.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing config_merged.yaml and model_state.pt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:0, ...",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    summary = evaluate_saved_run(args.run_dir, device=args.device)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
