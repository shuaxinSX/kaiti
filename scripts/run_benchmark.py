"""
运行 A6 小型实验矩阵，并汇总训练与 reference 评估结果。
"""

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import load_config
from src.eval import compute_prediction_envelope, export_reference_artifacts
from src.train.runner import (
    configure_logging,
    export_evaluation_artifacts,
    save_config_snapshot,
    save_losses,
    save_model,
    save_summary,
    save_wavefield,
)
from src.train.trainer import Trainer


DEFAULT_VELOCITY_MODELS = ("homogeneous", "smooth_lens", "layered")
DEFAULT_OMEGAS = (10.0, 20.0, 30.0)
DEFAULT_GRID_SIZES = (32, 64)


def parse_csv_list(raw_value, cast):
    return [cast(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def resolve_output_root(output_root):
    resolved = Path(output_root)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def scale_pml_width(base_grid_size, base_pml_width, target_grid_size):
    if base_grid_size <= 0:
        return int(base_pml_width)
    scaled = int(round(float(base_pml_width) * float(target_grid_size) / float(base_grid_size)))
    return max(1, scaled)


def case_name(velocity_model, omega, grid_size):
    omega_label = ("%g" % float(omega)).replace(".", "p")
    return f"{velocity_model}__omega{omega_label}__grid{int(grid_size)}"


def build_case_table(base_cfg, velocity_models, omegas, grid_sizes, epochs_override):
    base_grid_size = int(base_cfg.grid.nx)
    base_pml_width = int(base_cfg.pml.width)
    cases = []

    for velocity_model in velocity_models:
        for omega in omegas:
            for grid_size in grid_sizes:
                cases.append(
                    {
                        "name": case_name(velocity_model, omega, grid_size),
                        "velocity_model": str(velocity_model),
                        "omega": float(omega),
                        "grid_size": int(grid_size),
                        "pml_width": scale_pml_width(base_grid_size, base_pml_width, int(grid_size)),
                        "epochs": int(epochs_override)
                        if epochs_override is not None
                        else int(base_cfg.training.epochs),
                    }
                )
    return cases


def write_matrix_plan(output_root, payload):
    with open(output_root / "matrix_plan.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_matrix_summary(output_root, rows):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_root / "benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_root / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def run_case(case_spec, args):
    cfg = load_config(args.config, args.overlay)
    cfg.physics.omega = float(case_spec["omega"])
    cfg.grid.nx = int(case_spec["grid_size"])
    cfg.grid.ny = int(case_spec["grid_size"])
    cfg.pml.width = int(case_spec["pml_width"])
    cfg.medium.velocity_model = str(case_spec["velocity_model"])
    cfg.training.epochs = int(case_spec["epochs"])

    configure_logging(cfg.logging.level)
    output_dir = args.output_root / case_spec["name"]
    output_dir.mkdir(parents=True, exist_ok=False)
    save_config_snapshot(cfg, output_dir)

    trainer = Trainer(cfg, device=args.device)
    start_time = time.perf_counter()
    losses = trainer.train(epochs=cfg.training.epochs)
    runtime_sec = time.perf_counter() - start_time

    u_total = trainer.reconstruct_wavefield()
    save_losses(losses, output_dir)
    save_wavefield(u_total, output_dir)
    save_model(trainer.model, output_dir)

    metrics = export_evaluation_artifacts(trainer, losses, output_dir)
    reference_metrics = export_reference_artifacts(
        trainer,
        output_dir,
        predicted_envelope=compute_prediction_envelope(trainer),
    )

    summary = {
        "status": "ok",
        "case": case_spec["name"],
        "output_dir": str(output_dir),
        "device": str(trainer.device),
        "velocity_model": cfg.medium.velocity_model,
        "omega": float(cfg.physics.omega),
        "grid_nx": int(cfg.grid.nx),
        "grid_ny": int(cfg.grid.ny),
        "pml_width": int(cfg.pml.width),
        "epochs": int(cfg.training.epochs),
        "runtime_sec": float(runtime_sec),
        "final_loss": float(losses[-1]),
        "best_loss": float(min(losses)),
        "loss_reduction_ratio": float(losses[-1] / losses[0]) if losses and losses[0] != 0.0 else 0.0,
    }
    summary.update(metrics)
    summary.update(reference_metrics)
    save_summary(summary, output_dir)
    return summary


def record_failure(case_spec, args, exc):
    output_dir = args.output_root / case_spec["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    error_payload = {
        "status": "failed",
        "case": case_spec["name"],
        "output_dir": str(output_dir),
        "velocity_model": case_spec["velocity_model"],
        "omega": float(case_spec["omega"]),
        "grid_nx": int(case_spec["grid_size"]),
        "grid_ny": int(case_spec["grid_size"]),
        "pml_width": int(case_spec["pml_width"]),
        "epochs": int(case_spec["epochs"]),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    save_summary(error_payload, output_dir)
    with open(output_dir / "traceback.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    return error_payload


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the A6 benchmark matrix.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Base config path.",
    )
    parser.add_argument(
        "--overlay",
        default="configs/debug.yaml",
        help="Overlay config path. Defaults to the debug-sized profile.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Benchmark root directory. Defaults to outputs/benchmark_matrix_TIMESTAMP.",
    )
    parser.add_argument(
        "--velocity-models",
        default=",".join(DEFAULT_VELOCITY_MODELS),
        help="Comma-separated velocity models.",
    )
    parser.add_argument(
        "--omegas",
        default=",".join(str(value) for value in DEFAULT_OMEGAS),
        help="Comma-separated angular frequencies.",
    )
    parser.add_argument(
        "--grid-sizes",
        default=",".join(str(value) for value in DEFAULT_GRID_SIZES),
        help="Comma-separated physical grid sizes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override applied to every case.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later cases after a failure and record failed rows.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    velocity_models = parse_csv_list(args.velocity_models, str)
    omegas = parse_csv_list(args.omegas, float)
    grid_sizes = parse_csv_list(args.grid_sizes, int)

    if args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = resolve_output_root(f"outputs/benchmark_matrix_{timestamp}")
    else:
        args.output_root = resolve_output_root(args.output_root)

    base_cfg = load_config(args.config, args.overlay)
    cases = build_case_table(
        base_cfg,
        velocity_models=velocity_models,
        omegas=omegas,
        grid_sizes=grid_sizes,
        epochs_override=args.epochs,
    )
    write_matrix_plan(
        args.output_root,
        {
            "config": str(args.config),
            "overlay": str(args.overlay) if args.overlay is not None else None,
            "device": args.device,
            "case_count": len(cases),
            "cases": cases,
        },
    )

    rows = []
    for index, case_spec in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case_spec['name']} "
            f"(omega={case_spec['omega']}, grid={case_spec['grid_size']}, "
            f"velocity={case_spec['velocity_model']})",
            flush=True,
        )
        try:
            rows.append(run_case(case_spec, args))
        except Exception as exc:
            rows.append(record_failure(case_spec, args, exc))
            write_matrix_summary(args.output_root, rows)
            if not args.continue_on_error:
                raise

        write_matrix_summary(args.output_root, rows)

    aggregate = {
        "output_root": str(args.output_root),
        "case_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "failed_count": sum(1 for row in rows if row.get("status") != "ok"),
    }
    with open(args.output_root / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
