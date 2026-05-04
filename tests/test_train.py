"""
tests/test_train.py — 训练管线集成测试
"""

from pathlib import Path
import copy

import numpy as np
import torch
import pytest

from src.config import load_config
from src.train.trainer import Trainer, build_network_input, resolve_residual_config
from src.train.losses import loss_data, loss_total


@pytest.fixture
def cfg_homogeneous():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    return load_config(base, debug)


@pytest.fixture
def cfg_lens():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    cfg = load_config(base, debug)
    cfg.medium.velocity_model = "smooth_lens"
    return cfg


def _grid_shape(cfg):
    return (
        cfg.grid.ny + 2 * cfg.pml.width,
        cfg.grid.nx + 2 * cfg.pml.width,
    )


def _write_reference_envelope(path, shape):
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float64),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float64),
        indexing="ij",
    )
    reference = (0.25 + 0.05 * xx) + 1j * (-0.1 + 0.03 * yy)
    np.save(path, reference.astype(np.complex128))
    return reference


class TestLossFunctions:

    def test_loss_data(self):
        """数据损失计算正确。"""
        pred = torch.zeros(1, 2, 10, 10)
        true = torch.ones(1, 2, 10, 10)
        assert loss_data(pred, true).item() == pytest.approx(1.0)

    def test_loss_total_pde_only(self):
        """lambda_data=0 时只有 PDE 损失。"""
        l_pde = torch.tensor(0.5)
        total = loss_total(l_pde, lambda_pde=2.0, lambda_data=0.0)
        assert total.item() == pytest.approx(1.0)

    def test_loss_total_with_data(self):
        """PDE + 数据损失组合。"""
        l_pde = torch.tensor(1.0)
        l_data = torch.tensor(2.0)
        total = loss_total(l_pde, l_data, lambda_pde=1.0, lambda_data=0.5)
        assert total.item() == pytest.approx(2.0)


class TestTrainer:

    def test_homogeneous_zero_loss(self, cfg_homogeneous):
        """均匀介质 + 零初始化 → loss=0。"""
        trainer = Trainer(cfg_homogeneous)
        losses = trainer.train(epochs=1)
        assert losses[0] == pytest.approx(0.0, abs=1e-15)

    def test_lens_loss_decreases(self, cfg_lens):
        """变速介质 loss 能下降。"""
        trainer = Trainer(cfg_lens)
        losses = trainer.train(epochs=30)
        assert losses[-1] < losses[0]

    def test_reconstruct_wavefield(self, cfg_homogeneous):
        """波场重构无 NaN。"""
        trainer = Trainer(cfg_homogeneous)
        trainer.train(epochs=1)
        u = trainer.reconstruct_wavefield()
        assert u.shape == (trainer.grid.ny_total, trainer.grid.nx_total)
        assert np.all(np.isfinite(u))

    def test_network_input_shape(self, cfg_homogeneous):
        """网络输入 shape 为 [1, 8, H, W]。"""
        trainer = Trainer(cfg_homogeneous)
        assert trainer.net_input.shape == (
            1, 8, trainer.grid.ny_total, trainer.grid.nx_total
        )

    def test_network_input_no_nan(self, cfg_homogeneous):
        """网络输入不含 NaN/Inf。"""
        trainer = Trainer(cfg_homogeneous)
        assert torch.all(torch.isfinite(trainer.net_input))

    def test_trainer_respects_explicit_device(self, cfg_homogeneous):
        """Trainer 会将模型、输入和残差缓存放到指定设备。"""
        trainer = Trainer(cfg_homogeneous, device="cpu")
        assert next(trainer.model.parameters()).device.type == "cpu"
        assert trainer.net_input.device.type == "cpu"
        assert trainer.residual_computer.device.type == "cpu"

    def test_trainer_reads_residual_lap_tau_mode(self, cfg_homogeneous):
        """Trainer 从配置读取 residual.lap_tau_mode。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        assert resolve_residual_config(cfg)["lap_tau_mode"] == "stretched_divergence"

        cfg.residual.lap_tau_mode = "mixed_legacy"
        trainer = Trainer(cfg, device="cpu")

        assert trainer.residual_computer.lap_tau_mode == "mixed_legacy"

    def test_pde_only_ignores_supervision_when_lambda_data_zero(self, cfg_homogeneous, tmp_path):
        """lambda_data=0 时即使 supervision 打开，也保持 PDE-only。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        cfg.training.lambda_data = 0.0
        cfg.training.supervision = {
            "enabled": True,
            "reference_path": str(tmp_path / "missing_reference.npy"),
        }

        trainer = Trainer(cfg)
        losses = trainer.train(epochs=1)

        assert isinstance(losses, list)
        assert trainer.supervision_enabled is False
        assert trainer.reference_target is None
        assert trainer.loss_history_data == [0.0]

    def test_data_only_training_uses_reference_target(self, cfg_homogeneous, tmp_path):
        """data-only 路径会加载 reference_envelope 并计算监督损失。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        reference_path = tmp_path / "reference_envelope.npy"
        _write_reference_envelope(reference_path, _grid_shape(cfg))
        cfg.training.lambda_pde = 0.0
        cfg.training.lambda_data = 1.0
        cfg.training.supervision = {
            "enabled": True,
            "reference_path": str(reference_path),
        }

        trainer = Trainer(cfg)
        losses = trainer.train(epochs=3)

        assert trainer.supervision_enabled is True
        assert trainer.reference_target.shape == (1, 2, *_grid_shape(cfg))
        assert len(losses) == 3
        assert trainer.loss_history_data[0] > 0.0
        assert np.all(np.isfinite(losses))
        assert losses[-1] == pytest.approx(trainer.loss_history_data[-1])

    def test_hybrid_training_combines_pde_and_data_losses(self, cfg_homogeneous, tmp_path):
        """hybrid 路径按权重组合 PDE 与 data 两项损失。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        reference_path = tmp_path / "reference_envelope.npy"
        _write_reference_envelope(reference_path, _grid_shape(cfg))
        cfg.training.lambda_pde = 1.0
        cfg.training.lambda_data = 0.25
        cfg.training.supervision = {
            "enabled": True,
            "reference_path": str(reference_path),
        }

        trainer = Trainer(cfg)
        losses = trainer.train(epochs=1)

        assert len(losses) == 1
        assert losses[0] == pytest.approx(
            trainer.loss_history_pde[0] + 0.25 * trainer.loss_history_data[0]
        )
        assert trainer.last_step_metrics["loss_total"] == pytest.approx(losses[0])

    def test_lambda_data_requires_enabled_supervision(self, cfg_homogeneous):
        """lambda_data>0 且 supervision 未启用时应直接报错。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        cfg.training.lambda_data = 1.0

        with pytest.raises(ValueError, match="supervision.enabled"):
            Trainer(cfg)

    def test_missing_reference_path_fails_fast(self, cfg_homogeneous, tmp_path):
        """缺少 reference 文件时在初始化阶段直接失败。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        cfg.training.lambda_data = 1.0
        cfg.training.supervision = {
            "enabled": True,
            "reference_path": str(tmp_path / "missing_reference.npy"),
        }

        with pytest.raises(FileNotFoundError, match="Reference envelope file"):
            Trainer(cfg)

    def test_reference_shape_mismatch_fails_fast(self, cfg_homogeneous, tmp_path):
        """reference shape 不匹配时在初始化阶段直接失败。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        reference_path = tmp_path / "reference_envelope.npy"
        np.save(reference_path, np.ones((8, 8), dtype=np.complex128))
        cfg.training.lambda_data = 1.0
        cfg.training.supervision = {
            "enabled": True,
            "reference_path": str(reference_path),
        }

        with pytest.raises(ValueError, match="shape mismatch"):
            Trainer(cfg)

    def test_reference_must_be_complex(self, cfg_homogeneous, tmp_path):
        """reference_envelope.npy 必须是复数数组。"""
        cfg = copy.deepcopy(cfg_homogeneous)
        reference_path = tmp_path / "reference_envelope.npy"
        np.save(reference_path, np.ones(_grid_shape(cfg), dtype=np.float64))
        cfg.training.lambda_data = 1.0
        cfg.training.supervision = {
            "enabled": True,
            "reference_path": str(reference_path),
        }

        with pytest.raises(ValueError, match="complex ndarray"):
            Trainer(cfg)
