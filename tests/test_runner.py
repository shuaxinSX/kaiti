"""
训练运行器测试
==============
"""

import yaml

from pathlib import Path

from src.train.runner import run_training


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
    assert output_dir.exists()
    assert (output_dir / "config_merged.yaml").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "losses.npy").exists()
    assert (output_dir / "losses.csv").exists()
    assert (output_dir / "loss_curve.png").exists()
    assert (output_dir / "wavefield.npy").exists()
    assert (output_dir / "wavefield.png").exists()
    assert (output_dir / "model_state.pt").exists()

    with open(output_dir / "config_merged.yaml", "r", encoding="utf-8") as f:
        merged = yaml.safe_load(f)
    assert merged["training"]["epochs"] == 1
    assert merged["medium"]["velocity_model"] == "smooth_lens"
