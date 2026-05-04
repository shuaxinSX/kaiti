"""
训练运行器
==========

提供可复现的训练入口，并将结果保存到项目输出目录。
"""

import argparse
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

from src.config import load_config
from src.eval import compute_prediction_envelope, export_reference_artifacts
from src.train.trainer import Trainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
_NUMERIC_EPS = 1.0e-12


def configure_logging(level="INFO"):
    """Configure root logging once for CLI execution."""
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    root_logger.setLevel(resolved_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def resolve_output_dir(save_dir, output_dir=None):
    """Resolve the final output directory under the current project folder."""
    if output_dir is not None:
        final_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = Path(save_dir) / f"run_{timestamp}"

    if not final_dir.is_absolute():
        final_dir = Path.cwd() / final_dir

    final_dir.mkdir(parents=True, exist_ok=False)
    return final_dir


def save_losses(losses, output_dir):
    """Persist loss history as both numeric data and a plot."""
    losses_array = np.asarray(losses, dtype=np.float64)
    np.save(output_dir / "losses.npy", losses_array)

    with open(output_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow([epoch, f"{loss:.16e}"])

    tail_window = min(len(losses), 500)
    with open(output_dir / "loss_tail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch in range(len(losses) - tail_window, len(losses)):
            writer.writerow([epoch + 1, f"{losses_array[epoch]:.16e}"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses_array, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses_array, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Training Loss (Log Scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve_log.png", dpi=160)
    plt.close(fig)


def save_wavefield(u_total, output_dir):
    """Save the reconstructed wavefield and a quick-look visualization."""
    np.save(output_dir / "wavefield.npy", u_total)

    magnitude = np.abs(u_total)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_specs = (
        ("Real", u_total.real),
        ("Imag", u_total.imag),
        ("Magnitude", magnitude),
    )
    for ax, (title, values) in zip(axes, plot_specs):
        image = ax.imshow(values, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_dir / "wavefield.png", dpi=160)
    plt.close(fig)


def save_model(model, output_dir):
    """Save a CPU copy of the trained model weights."""
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, output_dir / "model_state.pt")


def save_config_snapshot(cfg, output_dir):
    """Save the merged runtime configuration."""
    with open(output_dir / "config_merged.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)


def save_summary(summary, output_dir):
    """Write the run summary as JSON."""
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def dual_channel_to_complex(a_scat_dual):
    """Convert [1, 2, H, W] dual-channel output into a complex numpy array."""
    a_r = a_scat_dual[0, 0].detach().cpu().numpy().astype(np.float64)
    a_i = a_scat_dual[0, 1].detach().cpu().numpy().astype(np.float64)
    return a_r + 1j * a_i


def compute_model_diagnostics(trainer):
    """Compute residual and field diagnostics for the current model state."""
    with torch.no_grad():
        a_scat_dual = trainer.model(trainer.net_input)
        residual = trainer.residual_computer.compute(a_scat_dual)

    a_scat = dual_channel_to_complex(a_scat_dual)
    u_total = trainer.reconstruct_wavefield()
    residual_real = residual["residual_real"].detach().cpu().numpy().astype(np.float64)
    residual_imag = residual["residual_imag"].detach().cpu().numpy().astype(np.float64)
    residual_mag = np.hypot(residual_real, residual_imag)

    physical_y, physical_x = trainer.grid.physical_slice()
    physical_mask = ~trainer.grid.pml_mask()
    evaluation_mask = physical_mask & trainer.loss_mask.astype(bool)
    slowness_physical = trainer.medium.slowness[physical_y, physical_x]
    slowness_sq_contrast = float(np.max(np.abs(slowness_physical ** 2 - trainer.medium.s0 ** 2)))

    return {
        "a_scat": a_scat,
        "a_scat_mag": np.abs(a_scat),
        "u_total": u_total,
        "wavefield_mag": np.abs(u_total),
        "residual_real": residual_real,
        "residual_imag": residual_imag,
        "residual_mag": residual_mag,
        "velocity": trainer.medium.velocity,
        "slowness": trainer.medium.slowness,
        "rhs_mag": np.abs(trainer.rhs),
        "loss_mask": trainer.loss_mask.astype(np.float64),
        "physical_slice": (physical_y, physical_x),
        "physical_mask": physical_mask,
        "evaluation_mask": evaluation_mask,
        "omega": float(trainer.omega),
        "grid_h": float(trainer.grid.h),
        "s0": float(trainer.medium.s0),
        "domain_diameter": float(
            math.hypot(
                trainer.grid.x_max - trainer.grid.x_min,
                trainer.grid.y_max - trainer.grid.y_min,
            )
        ),
        "slowness_sq_contrast": slowness_sq_contrast,
        "nsno_blocks": int(trainer.cfg.model.nsno_blocks),
        "fno_modes": int(trainer.cfg.model.fno_modes),
    }


def save_vector_csv(values, output_path):
    """Persist a 1D numeric vector."""
    np.savetxt(output_path, np.asarray(values, dtype=np.float64), delimiter=",", fmt="%.16e")


def save_matrix_csv(matrix, output_path):
    """Persist a 2D numeric matrix."""
    np.savetxt(output_path, np.asarray(matrix, dtype=np.float64), delimiter=",", fmt="%.16e")


def estimate_phase_reconstruction_budget(omega, h, scattering_mean, scattering_p95):
    """Estimate first-order wavefield reconstruction sensitivity to tau errors.

    The A3 bound is |du| <= |du0| + |dA| + omega * |A| * |dtau|.
    We report two tau-error scales: O(h^2) for smooth media and O(h) for
    kink/order-loss regions.
    """
    phase_h2 = float(omega * h ** 2)
    phase_h = float(omega * h)
    return {
        "phase_tau_error_budget_h2": phase_h2,
        "phase_tau_error_budget_h": phase_h,
        "wavefield_phase_error_budget_mean_h2": float(phase_h2 * scattering_mean),
        "wavefield_phase_error_budget_p95_h2": float(phase_h2 * scattering_p95),
        "wavefield_phase_error_budget_mean_h": float(phase_h * scattering_mean),
        "wavefield_phase_error_budget_p95_h": float(phase_h * scattering_p95),
    }


def estimate_neumann_capacity_budget(
    omega,
    s0,
    domain_diameter,
    slowness_sq_contrast,
    nsno_blocks,
    fno_modes,
):
    """Estimate A4 spectral and Neumann-series capacity diagnostics."""
    carrier_mode_floor = float(omega * s0 * domain_diameter / (2.0 * math.pi))
    mode_ratio = float(fno_modes / max(carrier_mode_floor, _NUMERIC_EPS))
    scattering_strength = float(
        omega * domain_diameter * slowness_sq_contrast / max(abs(s0), _NUMERIC_EPS)
    )
    neumann_convergent = bool(scattering_strength < 1.0)
    tail_proxy = (
        float(scattering_strength ** (int(nsno_blocks) + 1))
        if neumann_convergent
        else None
    )
    return {
        "full_wave_nyquist_mode_floor": carrier_mode_floor,
        "fno_mode_to_full_wave_nyquist": mode_ratio,
        "scattering_strength_proxy": scattering_strength,
        "neumann_proxy_convergent": neumann_convergent,
        "neumann_depth_tail_proxy": tail_proxy,
    }


def compute_metric_bundle(losses, diagnostics):
    """Assemble scalar metrics that summarize convergence and final field quality."""
    losses_array = np.asarray(losses, dtype=np.float64)
    tail_window = min(len(losses_array), 100)
    evaluation_values = diagnostics["residual_mag"][diagnostics["evaluation_mask"]]
    physical_values = diagnostics["residual_mag"][diagnostics["physical_mask"]]
    wavefield_values = diagnostics["wavefield_mag"][diagnostics["physical_mask"]]
    scattering_values = diagnostics["a_scat_mag"][diagnostics["physical_mask"]]
    scattering_mean = float(np.mean(scattering_values))
    scattering_p95 = float(np.quantile(scattering_values, 0.95))

    initial_loss = float(losses_array[0])
    final_loss = float(losses_array[-1])
    metrics = {
        "initial_loss": initial_loss,
        "best_loss": float(np.min(losses_array)),
        "final_loss": final_loss,
        "loss_reduction_ratio": float(final_loss / initial_loss) if initial_loss != 0.0 else 0.0,
        "mean_last_100_loss": float(np.mean(losses_array[-tail_window:])),
        "std_last_100_loss": float(np.std(losses_array[-tail_window:])),
        "residual_mean_evaluation": float(np.mean(evaluation_values)),
        "residual_rmse_evaluation": float(np.sqrt(np.mean(evaluation_values ** 2))),
        "residual_p95_evaluation": float(np.quantile(evaluation_values, 0.95)),
        "residual_max_evaluation": float(np.max(evaluation_values)),
        "residual_mean_physical": float(np.mean(physical_values)),
        "residual_p95_physical": float(np.quantile(physical_values, 0.95)),
        "residual_max_physical": float(np.max(physical_values)),
        "wavefield_mag_mean_physical": float(np.mean(wavefield_values)),
        "wavefield_mag_p95_physical": float(np.quantile(wavefield_values, 0.95)),
        "wavefield_mag_max_physical": float(np.max(wavefield_values)),
        "scattering_mag_mean_physical": scattering_mean,
        "scattering_mag_p95_physical": scattering_p95,
        "scattering_mag_max_physical": float(np.max(scattering_values)),
    }
    metrics.update(
        estimate_phase_reconstruction_budget(
            diagnostics["omega"],
            diagnostics["grid_h"],
            scattering_mean,
            scattering_p95,
        )
    )
    metrics.update(
        estimate_neumann_capacity_budget(
            diagnostics["omega"],
            diagnostics["s0"],
            diagnostics["domain_diameter"],
            diagnostics["slowness_sq_contrast"],
            diagnostics["nsno_blocks"],
            diagnostics["fno_modes"],
        )
    )
    return metrics


def save_metrics_csv(metrics, output_dir):
    """Write one-row metric summaries for spreadsheet-style inspection."""
    with open(output_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_quantiles_csv(diagnostics, output_dir):
    """Write distribution quantiles for the main diagnostic fields."""
    quantiles = (0.50, 0.90, 0.95, 0.99, 1.00)
    regions = {
        "physical": diagnostics["physical_mask"],
        "evaluation": diagnostics["evaluation_mask"],
    }
    fields = {
        "residual_magnitude": diagnostics["residual_mag"],
        "wavefield_magnitude": diagnostics["wavefield_mag"],
        "scattering_magnitude": diagnostics["a_scat_mag"],
    }

    with open(output_dir / "quantiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "region", "q50", "q90", "q95", "q99", "q100"])
        for field_name, values in fields.items():
            for region_name, mask in regions.items():
                subset = values[mask]
                writer.writerow(
                    [field_name, region_name]
                    + [f"{np.quantile(subset, q):.16e}" for q in quantiles]
                )


def save_centerline_profiles(trainer, diagnostics, output_dir):
    """Export central x/y profiles that are useful for quick spreadsheet plots."""
    physical_y, physical_x = diagnostics["physical_slice"]
    source_i = trainer.source.i_s
    source_j = trainer.source.j_s

    rows = []
    for x_index, coord in enumerate(trainer.grid.x_coords[physical_x]):
        j = x_index + trainer.grid.pml_width
        rows.append({
            "axis": "x",
            "coord": coord,
            "velocity": trainer.medium.velocity[source_i, j],
            "wavefield_mag": diagnostics["wavefield_mag"][source_i, j],
            "scattering_mag": diagnostics["a_scat_mag"][source_i, j],
            "residual_mag": diagnostics["residual_mag"][source_i, j],
            "loss_mask": trainer.loss_mask[source_i, j],
        })

    for y_index, coord in enumerate(trainer.grid.y_coords[physical_y]):
        i = y_index + trainer.grid.pml_width
        rows.append({
            "axis": "y",
            "coord": coord,
            "velocity": trainer.medium.velocity[i, source_j],
            "wavefield_mag": diagnostics["wavefield_mag"][i, source_j],
            "scattering_mag": diagnostics["a_scat_mag"][i, source_j],
            "residual_mag": diagnostics["residual_mag"][i, source_j],
            "loss_mask": trainer.loss_mask[i, source_j],
        })

    with open(output_dir / "centerline_profiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "axis",
                "coord",
                "velocity",
                "wavefield_mag",
                "scattering_mag",
                "residual_mag",
                "loss_mask",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_diagnostic_csvs(trainer, diagnostics, output_dir):
    """Persist grid-aligned CSVs that are convenient for external plotting."""
    physical_y, physical_x = diagnostics["physical_slice"]

    save_vector_csv(trainer.grid.x_coords[physical_x], output_dir / "x_coords_physical.csv")
    save_vector_csv(trainer.grid.y_coords[physical_y], output_dir / "y_coords_physical.csv")
    save_matrix_csv(trainer.medium.velocity[physical_y, physical_x], output_dir / "velocity_physical.csv")
    save_matrix_csv(diagnostics["loss_mask"][physical_y, physical_x], output_dir / "loss_mask_physical.csv")
    save_matrix_csv(diagnostics["a_scat_mag"][physical_y, physical_x], output_dir / "scattering_magnitude_physical.csv")
    save_matrix_csv(diagnostics["wavefield_mag"][physical_y, physical_x], output_dir / "wavefield_magnitude_physical.csv")
    save_matrix_csv(diagnostics["residual_mag"][physical_y, physical_x], output_dir / "residual_magnitude_physical.csv")


def save_heatmap_figure(fields, output_path, source_xy):
    """Render a list of heatmaps with a consistent layout."""
    n_fields = len(fields)
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax, (title, values, cmap) in zip(axes.flat, fields):
        image = ax.imshow(values, origin="lower", cmap=cmap)
        ax.scatter(source_xy[0], source_xy[1], c="cyan", s=18, marker="x", linewidths=1.5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[n_fields:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_diagnostic_heatmaps(trainer, diagnostics, output_dir):
    """Save field and residual heatmaps cropped to the physical domain."""
    physical_y, physical_x = diagnostics["physical_slice"]
    source_xy = (
        trainer.source.j_s - trainer.grid.pml_width,
        trainer.source.i_s - trainer.grid.pml_width,
    )

    velocity = trainer.medium.velocity[physical_y, physical_x]
    scattering_mag = diagnostics["a_scat_mag"][physical_y, physical_x]
    wavefield_mag = diagnostics["wavefield_mag"][physical_y, physical_x]
    residual_real = diagnostics["residual_real"][physical_y, physical_x]
    residual_imag = diagnostics["residual_imag"][physical_y, physical_x]
    residual_mag = diagnostics["residual_mag"][physical_y, physical_x]
    residual_log = np.log10(np.maximum(residual_mag, 1.0e-12))
    rhs_log = np.log10(np.maximum(diagnostics["rhs_mag"][physical_y, physical_x], 1.0e-12))
    mask = diagnostics["loss_mask"][physical_y, physical_x]

    save_heatmap_figure(
        [
            ("Velocity", velocity, "viridis"),
            ("|A_scat|", scattering_mag, "magma"),
            ("|u_total|", wavefield_mag, "magma"),
        ],
        output_dir / "field_heatmaps.png",
        source_xy,
    )

    save_heatmap_figure(
        [
            ("Residual Real", residual_real, "coolwarm"),
            ("Residual Imag", residual_imag, "coolwarm"),
            ("|Residual|", residual_mag, "inferno"),
            ("log10 |Residual|", residual_log, "inferno"),
            ("log10 |RHS_eq|", rhs_log, "plasma"),
            ("Loss Mask", mask, "gray"),
        ],
        output_dir / "residual_heatmaps.png",
        source_xy,
    )


def export_evaluation_artifacts(trainer, losses, output_dir):
    """Compute and persist diagnostics for the current trainer state."""
    diagnostics = compute_model_diagnostics(trainer)
    metrics = compute_metric_bundle(losses, diagnostics)

    save_metrics_csv(metrics, output_dir)
    save_quantiles_csv(diagnostics, output_dir)
    save_centerline_profiles(trainer, diagnostics, output_dir)
    save_diagnostic_csvs(trainer, diagnostics, output_dir)
    save_diagnostic_heatmaps(trainer, diagnostics, output_dir)

    return metrics


def evaluate_saved_run(run_dir, device="auto"):
    """Load an existing run directory and regenerate diagnostics in-place."""
    run_dir = Path(run_dir).resolve()
    cfg_path = run_dir / "config_merged.yaml"
    model_path = run_dir / "model_state.pt"
    losses_path = run_dir / "losses.npy"

    cfg = load_config(cfg_path)
    configure_logging(cfg.logging.level)
    trainer = Trainer(cfg, device=device)
    state_dict = torch.load(model_path, map_location=trainer.device)
    trainer.model.load_state_dict(state_dict)

    losses = np.load(losses_path).astype(np.float64).tolist()
    save_losses(losses, run_dir)
    metrics = export_evaluation_artifacts(trainer, losses, run_dir)
    reference_metrics = export_reference_artifacts(
        trainer,
        run_dir,
        predicted_envelope=compute_prediction_envelope(trainer),
    )

    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    summary.update(metrics)
    summary.update(reference_metrics)
    summary["output_dir"] = str(run_dir)
    summary["device"] = str(trainer.device)
    save_summary(summary, run_dir)
    return summary


def run_training(
    base_config,
    overlay_config=None,
    device="auto",
    epochs=None,
    output_dir=None,
    velocity_model=None,
):
    """Run one end-to-end training job and persist artifacts."""
    cfg = load_config(base_config, overlay_config)
    if velocity_model is not None:
        cfg.medium.velocity_model = velocity_model
    if epochs is not None:
        cfg.training.epochs = int(epochs)
    configure_logging(cfg.logging.level)
    final_output_dir = resolve_output_dir(cfg.logging.save_dir, output_dir)
    save_config_snapshot(cfg, final_output_dir)

    trainer = Trainer(cfg, device=device)
    actual_epochs = cfg.training.epochs

    logger.info("运行输出目录: %s", final_output_dir)
    start_time = time.perf_counter()
    losses = trainer.train(epochs=actual_epochs)
    runtime_sec = time.perf_counter() - start_time
    u_total = trainer.reconstruct_wavefield()

    save_losses(losses, final_output_dir)
    save_wavefield(u_total, final_output_dir)
    save_model(trainer.model, final_output_dir)
    metrics = export_evaluation_artifacts(trainer, losses, final_output_dir)

    summary = {
        "output_dir": str(final_output_dir),
        "device": str(trainer.device),
        "epochs": int(actual_epochs),
        "final_loss": float(losses[-1]),
        "runtime_sec": runtime_sec,
        "velocity_model": cfg.medium.velocity_model,
        "grid_shape": [int(trainer.grid.ny_total), int(trainer.grid.nx_total)],
    }
    summary.update(metrics)
    save_summary(summary, final_output_dir)

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run one training job and save artifacts.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Base config path.",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional overlay config path, e.g. configs/debug.yaml.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to logging.save_dir/run_TIMESTAMP.",
    )
    parser.add_argument(
        "--velocity-model",
        default=None,
        help="Optional medium override: homogeneous, smooth_lens, layered.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    summary = run_training(
        base_config=args.config,
        overlay_config=args.overlay,
        device=args.device,
        epochs=args.epochs,
        output_dir=args.output_dir,
        velocity_model=args.velocity_model,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
