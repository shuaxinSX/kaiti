"""
训练运行器测试
==============
"""

import csv
import yaml

from pathlib import Path

from src.train.runner import (
    estimate_neumann_capacity_budget,
    evaluate_saved_run,
    run_training,
)


def test_neumann_capacity_budget_for_homogeneous_medium():
    metrics = estimate_neumann_capacity_budget(
        omega=30.0,
        s0=1.0,
        domain_diameter=2.0 ** 0.5,
        slowness_sq_contrast=0.0,
        nsno_blocks=4,
        fno_modes=16,
    )

    assert metrics["full_wave_nyquist_mode_floor"] > 0.0
    assert metrics["fno_mode_to_full_wave_nyquist"] > 0.0
    assert metrics["scattering_strength_proxy"] == 0.0
    assert metrics["neumann_proxy_convergent"] is True
    assert metrics["neumann_depth_tail_proxy"] == 0.0


def test_neumann_capacity_budget_marks_strong_scattering():
    metrics = estimate_neumann_capacity_budget(
        omega=30.0,
        s0=1.0,
        domain_diameter=2.0 ** 0.5,
        slowness_sq_contrast=0.5,
        nsno_blocks=4,
        fno_modes=16,
    )

    assert metrics["scattering_strength_proxy"] > 1.0
    assert metrics["neumann_proxy_convergent"] is False
    assert metrics["neumann_depth_tail_proxy"] is None


def test_run_training_saves_artifacts(tmp_path):
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    output_dir = tmp_path / "artifacts"

    summary = run_training(
        base_config=base,
        overlay_config=debug,
        device="cpu",
        epochs=1,
        output_dir=output_dir,
        velocity_model="smooth_lens",
    )

    assert summary["device"] == "cpu"
    assert summary["epochs"] == 1
    assert summary["velocity_model"] == "smooth_lens"
    expected_h = 1.0 / 32.0
    assert abs(summary["phase_tau_error_budget_h2"] - 10.0 * expected_h ** 2) < 1.0e-12
    assert abs(summary["phase_tau_error_budget_h"] - 10.0 * expected_h) < 1.0e-12
    assert summary["phase_tau_error_budget_h"] > summary["phase_tau_error_budget_h2"]
    assert summary["full_wave_nyquist_mode_floor"] > 0.0
    assert summary["fno_mode_to_full_wave_nyquist"] > 0.0
    assert summary["scattering_strength_proxy"] >= 0.0
    assert isinstance(summary["neumann_proxy_convergent"], bool)
    assert output_dir.exists()
    assert (output_dir / "config_merged.yaml").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "losses.npy").exists()
    assert (output_dir / "losses.csv").exists()
    assert (output_dir / "loss_tail.csv").exists()
    assert (output_dir / "loss_curve.png").exists()
    assert (output_dir / "loss_curve_log.png").exists()
    assert (output_dir / "wavefield.npy").exists()
    assert (output_dir / "wavefield.png").exists()
    assert (output_dir / "model_state.pt").exists()
    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "quantiles.csv").exists()
    assert (output_dir / "centerline_profiles.csv").exists()
    assert (output_dir / "x_coords_physical.csv").exists()
    assert (output_dir / "y_coords_physical.csv").exists()
    assert (output_dir / "velocity_physical.csv").exists()
    assert (output_dir / "wavefield_magnitude_physical.csv").exists()
    assert (output_dir / "residual_magnitude_physical.csv").exists()
    assert (output_dir / "field_heatmaps.png").exists()
    assert (output_dir / "residual_heatmaps.png").exists()

    with open(output_dir / "config_merged.yaml", "r", encoding="utf-8") as f:
        merged = yaml.safe_load(f)
    assert merged["training"]["epochs"] == 1
    assert merged["medium"]["velocity_model"] == "smooth_lens"

    with open(output_dir / "metrics_summary.csv", "r", encoding="utf-8") as f:
        metrics_row = next(csv.DictReader(f))
    assert float(metrics_row["phase_tau_error_budget_h2"]) == summary["phase_tau_error_budget_h2"]
    assert float(metrics_row["wavefield_phase_error_budget_p95_h"]) >= 0.0
    assert float(metrics_row["full_wave_nyquist_mode_floor"]) == summary["full_wave_nyquist_mode_floor"]
    assert float(metrics_row["scattering_strength_proxy"]) == summary["scattering_strength_proxy"]

    refreshed = evaluate_saved_run(output_dir, device="cpu")
    assert refreshed["output_dir"] == str(output_dir)
    assert refreshed["residual_max_evaluation"] >= 0.0
    assert refreshed["phase_tau_error_budget_h"] == summary["phase_tau_error_budget_h"]
    assert refreshed["scattering_strength_proxy"] == summary["scattering_strength_proxy"]
