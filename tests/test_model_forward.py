"""
tests/test_model_forward.py — 网络骨架前向测试
"""

from pathlib import Path

import torch
import pytest

from src.config import load_config
from src.models.nsno import NSNO2D


@pytest.fixture
def cfg():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    return load_config(base, debug)


@pytest.fixture
def model(cfg):
    return NSNO2D(cfg)


class TestNSNO2DForward:

    def test_output_shape(self, model, cfg):
        """输出 shape 应为 [B, 2, H, W]。"""
        B, H, W = 1, cfg.grid.ny + 2 * cfg.pml.width, cfg.grid.nx + 2 * cfg.pml.width
        x = torch.randn(B, 8, H, W)
        out = model(x)
        assert out.shape == (B, 2, H, W)

    def test_zero_init_output(self, model, cfg):
        """零初始化时输出应全零（无散射初态）。"""
        B, H, W = 1, cfg.grid.ny + 2 * cfg.pml.width, cfg.grid.nx + 2 * cfg.pml.width
        x = torch.randn(B, 8, H, W)
        out = model(x)
        # decoder 权重和偏置全零 → 输出全零
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    def test_batch_size(self, model, cfg):
        """支持多 batch。"""
        B, H, W = 4, cfg.grid.ny + 2 * cfg.pml.width, cfg.grid.nx + 2 * cfg.pml.width
        x = torch.randn(B, 8, H, W)
        out = model(x)
        assert out.shape == (B, 2, H, W)

    def test_no_nan_inf(self, model, cfg):
        """前向输出不含 NaN/Inf。"""
        B, H, W = 1, cfg.grid.ny + 2 * cfg.pml.width, cfg.grid.nx + 2 * cfg.pml.width
        x = torch.randn(B, 8, H, W)
        out = model(x)
        assert torch.all(torch.isfinite(out))

    def test_gradient_flows(self, model, cfg):
        """反向传播梯度可流通。"""
        B, H, W = 1, cfg.grid.ny + 2 * cfg.pml.width, cfg.grid.nx + 2 * cfg.pml.width
        x = torch.randn(B, 8, H, W, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
