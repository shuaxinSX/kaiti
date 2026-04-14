"""
tests/test_medium.py — Medium2D 单元测试
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D, Medium2D


@pytest.fixture
def cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "base.yaml")


@pytest.fixture
def grid(cfg):
    return Grid2D(cfg)


@pytest.fixture
def medium(grid, cfg):
    return Medium2D(grid, cfg)


class TestMediumHomogeneous:
    """测试均匀介质：所有慢度 == 1.0 / c_background。"""

    def test_all_slowness_equals_s0(self, medium, cfg):
        expected = 1.0 / cfg.medium.c_background
        np.testing.assert_allclose(
            medium.slowness, expected, atol=1e-14,
            err_msg="Homogeneous medium: all slowness values should equal 1/c_background",
        )

    def test_s0_attribute(self, medium, cfg):
        assert medium.s0 == pytest.approx(1.0 / cfg.medium.c_background)


class TestMediumPmlDegradation:
    """测试 PML 退化 (D6d)：PML 层内慢度 == s0。"""

    def test_pml_slowness_equals_s0(self, grid, medium):
        pml_mask = grid.pml_mask()
        pml_slowness = medium.slowness[pml_mask]
        np.testing.assert_allclose(
            pml_slowness, medium.s0, atol=1e-14,
            err_msg="PML region slowness should degrade to s0 = 1/c_background",
        )

    def test_pml_velocity_equals_1_over_s0(self, grid, medium):
        pml_mask = grid.pml_mask()
        pml_velocity = medium.velocity[pml_mask]
        np.testing.assert_allclose(
            pml_velocity, 1.0 / medium.s0, atol=1e-14,
            err_msg="PML region velocity should degrade to 1/s0",
        )


class TestMediumS0Heterogeneous:
    """测试非均匀介质 s0 = s(x_s) 正确。"""

    def test_smooth_lens_s0(self, grid, cfg):
        """smooth_lens 震源在中心，速度 > c_background，所以 s0 < 1/c_background。"""
        cfg.medium.velocity_model = "smooth_lens"
        medium = Medium2D(grid, cfg)
        # 震源在 (0.5, 0.5) 即透镜中心，速度有正扰动
        assert medium.s0 < 1.0 / cfg.medium.c_background

    def test_pml_degrades_to_s0_lens(self, grid, cfg):
        """smooth_lens 下 PML 退化到 s0（非 1/c_background）。"""
        cfg.medium.velocity_model = "smooth_lens"
        medium = Medium2D(grid, cfg)
        pml_mask = grid.pml_mask()
        np.testing.assert_allclose(
            medium.slowness[pml_mask], medium.s0, atol=1e-14,
        )


class TestMediumShape:
    """测试介质场 shape 与 grid 一致。"""

    def test_slowness_shape(self, grid, medium):
        assert medium.slowness.shape == (grid.ny_total, grid.nx_total)

    def test_velocity_shape(self, grid, medium):
        assert medium.velocity.shape == (grid.ny_total, grid.nx_total)

    def test_slowness_tensor_shape(self, grid, medium):
        assert medium.slowness_t.shape == (grid.ny_total, grid.nx_total)
