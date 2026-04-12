"""
tests/test_background.py — BackgroundField 单元测试
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField


@pytest.fixture
def cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "base.yaml")


@pytest.fixture
def grid(cfg):
    return Grid2D(cfg)


@pytest.fixture
def medium(grid, cfg):
    return Medium2D(grid, cfg)


@pytest.fixture
def source(grid, cfg):
    return PointSource(grid, cfg)


@pytest.fixture
def bg(grid, medium, source, cfg):
    return BackgroundField(grid, medium, source, cfg.physics.omega)


class TestBackgroundField:

    def test_tau0_equals_s0_times_r(self, bg, source, medium):
        """tau0 应等于 s0 * r（距离场）。"""
        expected = medium.s0 * source.distance
        np.testing.assert_allclose(bg.tau0, expected)

    def test_grad_tau0_magnitude_equals_s0(self, bg, source, medium):
        """远离震源处，|∇tau0| 应等于 s0。"""
        # 排除震源附近（r < 5*eps 的点）
        far = source.distance > 5 * source.eps
        mag = np.sqrt(bg.grad_tau0_x**2 + bg.grad_tau0_y**2)
        np.testing.assert_allclose(mag[far], medium.s0, atol=1e-10)

    def test_no_nan_inf(self, bg):
        """所有场不含 NaN 或 Inf。"""
        for name in ["tau0", "grad_tau0_x", "grad_tau0_y", "lap_tau0", "u0"]:
            arr = getattr(bg, name)
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    def test_shapes(self, bg, grid):
        """所有 2D 场的 shape 应为 [ny_total, nx_total]。"""
        expected_shape = (grid.ny_total, grid.nx_total)
        for name in ["tau0", "grad_tau0_x", "grad_tau0_y", "lap_tau0", "u0"]:
            arr = getattr(bg, name)
            assert arr.shape == expected_shape, (
                f"{name}.shape = {arr.shape}, expected {expected_shape}"
            )

    def test_u0_oscillates(self, bg):
        """u0 的实部应有符号变化（振荡性质）。"""
        real_part = bg.u0.real
        has_positive = np.any(real_part > 0)
        has_negative = np.any(real_part < 0)
        assert has_positive and has_negative, "u0.real does not change sign"
