"""
tests/test_train.py — 训练管线集成测试
"""

from pathlib import Path

import numpy as np
import torch
import pytest

from src.config import load_config
from src.train.trainer import Trainer, build_network_input
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
