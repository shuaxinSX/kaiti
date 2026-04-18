"""
reference 评估辅助工具
======================

把 reference solver 升级为标准化的评估与导出链路。
"""

import csv
import json
import logging
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

from src.config import load_config
from src.eval.reference_solver import solve_reference_scattering
from src.train.trainer import Trainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

REFERENCE_SUMMARY_KEYS = (
    "reference_available",
    "rel_l2_to_reference",
    "amp_mae_to_reference",
    "phase_mae_to_reference",
    "reference_residual_rmse",
    "reference_residual_max",
)

_COMPARISON_PHASE_EPS = 1.0e-8
_COMPARISON_NORM_EPS = 1.0e-12


def _complex_to_dual_channel(field, device):
    channels = np.stack([field.real, field.imag], axis=0).astype(np.float32)
    return torch.from_numpy(channels).unsqueeze(0).to(device)


def _physical_masks(trainer):
    physical_y, physical_x = trainer.grid.physical_slice()
    physical_mask = ~trainer.grid.pml_mask()
    evaluation_mask = physical_mask & trainer.loss_mask.astype(bool)
    return {
        "physical_slice": (physical_y, physical_x),
        "physical_mask": physical_mask,
        "evaluation_mask": evaluation_mask,
    }


def _wrapped_phase_difference(predicted, reference):
    return np.angle(np.exp(1j * (np.angle(predicted) - np.angle(reference))))


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_heatmap_figure(fields, output_path, source_xy):
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


def compute_prediction_envelope(trainer):
    """Run the current model once and return a complex scattering envelope."""
    with torch.no_grad():
        predicted = trainer.model(trainer.net_input)
    real = predicted[0, 0].detach().cpu().numpy().astype(np.float64)
    imag = predicted[0, 1].detach().cpu().numpy().astype(np.float64)
    return real + 1j * imag


def reconstruct_wavefield_from_envelope(trainer, envelope):
    """Reconstruct u = u0 + A * exp(i(omega * tau + pi/4)) from a complex envelope."""
    tau_safe = np.where(np.isfinite(trainer.eik.tau), trainer.eik.tau, 0.0)
    phase_restorer = np.exp(1j * (trainer.omega * tau_safe + math.pi / 4.0))
    return trainer.bg.u0 + envelope * phase_restorer


def compute_reference_residual(trainer, reference_envelope):
    """Evaluate the current discrete residual on the reference envelope."""
    reference_dual = _complex_to_dual_channel(reference_envelope, trainer.device)
    with torch.no_grad():
        residual = trainer.residual_computer.compute(reference_dual)

    residual_real = residual["residual_real"].detach().cpu().numpy().astype(np.float64)
    residual_imag = residual["residual_imag"].detach().cpu().numpy().astype(np.float64)
    return np.hypot(residual_real, residual_imag)


def compute_reference_metrics(trainer, reference_envelope, predicted_envelope=None):
    """Compute residual and optional comparison metrics for the reference solution."""
    masks = _physical_masks(trainer)
    evaluation_mask = masks["evaluation_mask"]
    reference_residual_mag = compute_reference_residual(trainer, reference_envelope)
    evaluation_values = reference_residual_mag[evaluation_mask]

    summary_metrics = {
        "reference_available": True,
        "reference_residual_rmse": float(np.sqrt(np.mean(evaluation_values ** 2)))
        if evaluation_values.size
        else 0.0,
        "reference_residual_max": float(np.max(evaluation_values)) if evaluation_values.size else 0.0,
    }
    reference_metrics = {
        "reference_available": True,
        "comparison_available": predicted_envelope is not None,
        "evaluation_point_count": int(np.count_nonzero(evaluation_mask)),
        "phase_metric_points": 0,
        "reference_residual_rmse": summary_metrics["reference_residual_rmse"],
        "reference_residual_max": summary_metrics["reference_residual_max"],
        "rel_l2_to_reference": None,
        "amp_mae_to_reference": None,
        "phase_mae_to_reference": None,
    }

    error_fields = None
    if predicted_envelope is None:
        return summary_metrics, reference_metrics, reference_residual_mag, error_fields

    predicted_values = predicted_envelope[evaluation_mask]
    reference_values = reference_envelope[evaluation_mask]
    diff_values = predicted_values - reference_values
    reference_norm = max(float(np.linalg.norm(reference_values)), _COMPARISON_NORM_EPS)

    rel_l2 = float(np.linalg.norm(diff_values) / reference_norm)
    amp_mae = (
        float(np.mean(np.abs(np.abs(predicted_values) - np.abs(reference_values))))
        if predicted_values.size
        else 0.0
    )

    phase_mask = evaluation_mask & (np.abs(reference_envelope) > _COMPARISON_PHASE_EPS)
    phase_metric_points = int(np.count_nonzero(phase_mask))
    phase_error = _wrapped_phase_difference(predicted_envelope, reference_envelope)
    phase_mae = (
        float(np.mean(np.abs(phase_error[phase_mask])))
        if phase_metric_points
        else 0.0
    )

    comparison_metrics = {
        "rel_l2_to_reference": rel_l2,
        "amp_mae_to_reference": amp_mae,
        "phase_mae_to_reference": phase_mae,
    }
    summary_metrics.update(comparison_metrics)
    reference_metrics.update(comparison_metrics)
    reference_metrics["phase_metric_points"] = phase_metric_points
    error_fields = {
        "absolute_error": np.abs(predicted_envelope - reference_envelope),
        "phase_error": phase_error,
    }
    return summary_metrics, reference_metrics, reference_residual_mag, error_fields


def save_reference_comparison_csv(reference_metrics, output_dir):
    """Persist scalar comparison metrics as a single-row CSV."""
    fieldnames = [
        "reference_available",
        "comparison_available",
        "rel_l2_to_reference",
        "amp_mae_to_reference",
        "phase_mae_to_reference",
        "phase_metric_points",
        "reference_residual_rmse",
        "reference_residual_max",
    ]
    with open(output_dir / "reference_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: reference_metrics.get(name) for name in fieldnames})


def save_reference_error_heatmaps(
    trainer,
    reference_envelope,
    predicted_envelope,
    error_fields,
    output_dir,
):
    """Render cropped reference/prediction/error heatmaps for the physical domain."""
    masks = _physical_masks(trainer)
    physical_y, physical_x = masks["physical_slice"]
    source_xy = (
        trainer.source.j_s - trainer.grid.pml_width,
        trainer.source.i_s - trainer.grid.pml_width,
    )

    fields = [
        ("|A_ref|", np.abs(reference_envelope[physical_y, physical_x]), "magma"),
        ("|A_pred|", np.abs(predicted_envelope[physical_y, physical_x]), "magma"),
        ("|A_pred - A_ref|", error_fields["absolute_error"][physical_y, physical_x], "inferno"),
        ("phase(A_ref)", np.angle(reference_envelope[physical_y, physical_x]), "twilight"),
        ("phase(A_pred)", np.angle(predicted_envelope[physical_y, physical_x]), "twilight"),
        ("phase error", error_fields["phase_error"][physical_y, physical_x], "coolwarm"),
    ]
    _save_heatmap_figure(fields, output_dir / "reference_error_heatmaps.png", source_xy)


def export_reference_artifacts(trainer, output_dir, predicted_envelope=None):
    """Solve, score, and persist the reference solution for the current trainer state."""
    output_dir = Path(output_dir)
    logger.info("生成 reference 评估产物: %s", output_dir)

    reference_envelope = solve_reference_scattering(trainer).astype(np.complex128, copy=False)
    reference_wavefield = reconstruct_wavefield_from_envelope(trainer, reference_envelope)
    summary_metrics, reference_metrics, _, error_fields = compute_reference_metrics(
        trainer,
        reference_envelope,
        predicted_envelope=predicted_envelope,
    )

    masks = _physical_masks(trainer)
    reference_metrics.update(
        {
            "grid_shape": [int(trainer.grid.ny_total), int(trainer.grid.nx_total)],
            "physical_grid_shape": [
                int(reference_envelope[masks["physical_slice"][0], masks["physical_slice"][1]].shape[0]),
                int(reference_envelope[masks["physical_slice"][0], masks["physical_slice"][1]].shape[1]),
            ],
        }
    )

    np.save(output_dir / "reference_envelope.npy", reference_envelope)
    np.save(output_dir / "reference_wavefield.npy", reference_wavefield)
    _save_json(output_dir / "reference_metrics.json", reference_metrics)

    if predicted_envelope is not None:
        save_reference_comparison_csv(reference_metrics, output_dir)
        save_reference_error_heatmaps(
            trainer,
            reference_envelope,
            predicted_envelope,
            error_fields,
            output_dir,
        )

    return summary_metrics


def _resolve_output_dir(output_dir):
    resolved = Path(output_dir)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def solve_reference_from_config(
    config_path,
    overlay_path=None,
    device="auto",
    velocity_model=None,
    output_dir=None,
):
    """Generate reference artifacts directly from config inputs."""
    if output_dir is None:
        raise ValueError("Config mode requires --output-dir.")

    cfg = load_config(config_path, overlay_path)
    if velocity_model is not None:
        cfg.medium.velocity_model = velocity_model

    resolved_output_dir = _resolve_output_dir(output_dir)
    trainer = Trainer(cfg, device=device)
    summary = export_reference_artifacts(trainer, resolved_output_dir)
    summary["output_dir"] = str(resolved_output_dir)
    summary["device"] = str(trainer.device)
    return summary


def solve_reference_from_run_dir(run_dir, device="auto"):
    """Generate reference artifacts for an existing run directory and update summary.json."""
    run_dir = Path(run_dir).resolve()
    cfg = load_config(run_dir / "config_merged.yaml")
    trainer = Trainer(cfg, device=device)

    model_path = run_dir / "model_state.pt"
    predicted_envelope = None
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)
        predicted_envelope = compute_prediction_envelope(trainer)

    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    summary.update(export_reference_artifacts(trainer, run_dir, predicted_envelope=predicted_envelope))
    summary["output_dir"] = str(run_dir)
    summary["device"] = str(trainer.device)
    _save_json(summary_path, summary)
    return summary
