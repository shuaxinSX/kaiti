"""
训练运行器测试
==============
"""

import csv
import yaml

from pathlib import Path

from src.train.runner import evaluate_saved_run, run_training


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

    refreshed = evaluate_saved_run(output_dir, device="cpu")
    assert refreshed["output_dir"] == str(output_dir)
    assert refreshed["residual_max_evaluation"] >= 0.0
    assert refreshed["phase_tau_error_budget_h"] == summary["phase_tau_error_budget_h"]
