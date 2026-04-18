from pathlib import Path

import pytest
import yaml

from src.config import load_config


def _write_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def test_load_config_base_only():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    cfg = load_config(base)

    assert cfg.physics.omega == 30.0
    assert cfg.grid.nx == 128


def test_load_config_single_overlay():
    root = Path(__file__).parent.parent
    cfg = load_config(root / "configs" / "base.yaml", root / "configs" / "debug.yaml")

    assert cfg.physics.omega == 10.0
    assert cfg.grid.nx == 32
    assert cfg.medium.c_background == 1.0


def test_load_config_merges_multiple_overlays_and_skips_none(tmp_path):
    base = tmp_path / "base.yaml"
    overlay_1 = tmp_path / "overlay_1.yaml"
    overlay_2 = tmp_path / "overlay_2.yaml"

    _write_yaml(
        base,
        {
            "training": {"epochs": 500, "lr": 1.0e-3},
            "medium": {"velocity_model": "homogeneous", "contrast": 0.1},
            "logging": {"save_dir": "outputs/base"},
        },
    )
    _write_yaml(
        overlay_1,
        {
            "training": {"epochs": 5},
            "logging": {"save_dir": "outputs/debug"},
        },
    )
    _write_yaml(
        overlay_2,
        {
            "medium": {"velocity_model": "smooth_lens"},
            "logging": {"level": "DEBUG"},
        },
    )

    cfg = load_config(base, None, overlay_1, overlay_2)

    assert cfg.training.epochs == 5
    assert cfg.training.lr == pytest.approx(1.0e-3)
    assert cfg.medium.velocity_model == "smooth_lens"
    assert cfg.medium.contrast == pytest.approx(0.1)
    assert cfg.logging.save_dir == "outputs/debug"
    assert cfg.logging.level == "DEBUG"


def test_load_config_missing_overlay_raises(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base, {"training": {"epochs": 1}})

    with pytest.raises(FileNotFoundError):
        load_config(base, tmp_path / "missing.yaml")


def test_load_config_deep_merge_preserves_nested_keys(tmp_path):
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"

    _write_yaml(
        base,
        {
            "loss": {
                "weights": {"data": 1.0, "pde": 0.5},
                "mask": {"exclude_pml": False, "pad": 2},
            }
        },
    )
    _write_yaml(
        overlay,
        {
            "loss": {
                "weights": {"pde": 2.0},
                "mask": {"exclude_pml": True},
            }
        },
    )

    cfg = load_config(base, overlay)

    assert cfg.loss.weights.data == pytest.approx(1.0)
    assert cfg.loss.weights.pde == pytest.approx(2.0)
    assert cfg.loss.mask.exclude_pml is True
    assert cfg.loss.mask.pad == 2
