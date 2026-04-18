import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.config import load_config
from src.eval.reference_eval import (
    compute_prediction_envelope,
    compute_reference_metrics,
)
from src.eval.reference_solver import solve_reference_scattering
from src.train.runner import evaluate_saved_run, run_training
from src.train.trainer import Trainer


def _build_trainer(velocity_model):
    root = Path(__file__).parent.parent
    cfg = load_config(root / "configs" / "base.yaml", root / "configs" / "debug.yaml")
    cfg.medium.velocity_model = velocity_model
    return Trainer(cfg, device="cpu")


def test_reference_homogeneous_zero_scattering():
    trainer = _build_trainer("homogeneous")
    reference_envelope = solve_reference_scattering(trainer)
    metrics, _, _, _ = compute_reference_metrics(trainer, reference_envelope)

    physical_mask = ~trainer.grid.pml_mask()
    assert np.max(np.abs(reference_envelope[physical_mask])) <= 1.0e-12
    assert metrics["reference_residual_rmse"] <= 1.0e-12
    assert metrics["reference_residual_max"] <= 1.0e-12


def test_reference_smooth_lens_residual_is_more_trustworthy_than_untrained_prediction():
    trainer = _build_trainer("smooth_lens")
    predicted_envelope = compute_prediction_envelope(trainer)
    reference_envelope = solve_reference_scattering(trainer)

    reference_metrics, _, _, _ = compute_reference_metrics(
        trainer,
        reference_envelope,
        predicted_envelope=predicted_envelope,
    )
    predicted_metrics, _, _, _ = compute_reference_metrics(trainer, predicted_envelope)

    assert reference_metrics["reference_residual_rmse"] < 1.0e-5
    assert reference_metrics["reference_residual_rmse"] < predicted_metrics["reference_residual_rmse"]
    assert reference_metrics["reference_residual_max"] < predicted_metrics["reference_residual_max"]


def test_solve_reference_cli_from_config_writes_core_artifacts(tmp_path):
    root = Path(__file__).parent.parent
    output_dir = tmp_path / "reference_from_config"

    subprocess.run(
        [
            sys.executable,
            "scripts/solve_reference.py",
            "--config",
            "configs/base.yaml",
            "--overlay",
            "configs/debug.yaml",
            "--device",
            "cpu",
            "--velocity-model",
            "homogeneous",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        check=True,
    )

    assert (output_dir / "reference_envelope.npy").exists()
    assert (output_dir / "reference_wavefield.npy").exists()
    assert (output_dir / "reference_metrics.json").exists()
    assert not (output_dir / "reference_comparison.csv").exists()
    assert not (output_dir / "reference_error_heatmaps.png").exists()

    with open(output_dir / "reference_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["reference_available"] is True
    assert metrics["comparison_available"] is False


def test_evaluate_saved_run_writes_reference_artifacts_and_summary_keys(tmp_path):
    root = Path(__file__).parent.parent
    output_dir = tmp_path / "run_artifacts"

    run_training(
        base_config=root / "configs" / "base.yaml",
        overlay_config=root / "configs" / "debug.yaml",
        device="cpu",
        epochs=1,
        output_dir=output_dir,
        velocity_model="smooth_lens",
    )

    refreshed = evaluate_saved_run(output_dir, device="cpu")

    assert (output_dir / "reference_envelope.npy").exists()
    assert (output_dir / "reference_wavefield.npy").exists()
    assert (output_dir / "reference_metrics.json").exists()
    assert (output_dir / "reference_comparison.csv").exists()
    assert (output_dir / "reference_error_heatmaps.png").exists()

    for key in (
        "reference_available",
        "rel_l2_to_reference",
        "amp_mae_to_reference",
        "phase_mae_to_reference",
        "reference_residual_rmse",
        "reference_residual_max",
    ):
        assert key in refreshed

    with open(output_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["reference_available"] is True
