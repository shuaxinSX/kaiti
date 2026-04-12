"""
tests/test_grid.py — Grid2D 单元测试
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D


@pytest.fixture
def cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "base.yaml")


@pytest.fixture
def grid(cfg):
    return Grid2D(cfg)


class TestGrid2DShape:
    """测试网格总尺寸 = 物理域尺寸 + 2 * PML 宽度。"""

    def test_nx_total(self, grid, cfg):
        assert grid.nx_total == cfg.grid.nx + 2 * cfg.pml.width

    def test_ny_total(self, grid, cfg):
        assert grid.ny_total == cfg.grid.ny + 2 * cfg.pml.width

    def test_coord_array_shapes(self, grid):
        assert grid.x_coords.shape == (grid.nx_total,)
        assert grid.y_coords.shape == (grid.ny_total,)

    def test_meshgrid_shape(self, grid):
        assert grid.xx.shape == (grid.ny_total, grid.nx_total)
        assert grid.yy.shape == (grid.ny_total, grid.nx_total)


class TestGrid2DSpacing:
    """测试均匀步长 h = (x_max - x_min) / nx。"""

    def test_h_value(self, grid, cfg):
        expected_h = (cfg.grid.domain[1] - cfg.grid.domain[0]) / cfg.grid.nx
        assert grid.h == pytest.approx(expected_h)

    def test_x_coords_spacing(self, grid):
        diffs = np.diff(grid.x_coords)
        np.testing.assert_allclose(diffs, grid.h, atol=1e-14)

    def test_y_coords_spacing(self, grid):
        diffs = np.diff(grid.y_coords)
        np.testing.assert_allclose(diffs, grid.h, atol=1e-14)


class TestGrid2DPmlMask:
    """测试 PML 掩码：PML 区域为 True，物理域为 False。"""

    def test_pml_mask_shape(self, grid):
        mask = grid.pml_mask()
        assert mask.shape == (grid.ny_total, grid.nx_total)

    def test_physical_domain_is_false(self, grid, cfg):
        mask = grid.pml_mask()
        pw = cfg.pml.width
        physical = mask[pw:pw + cfg.grid.ny, pw:pw + cfg.grid.nx]
        assert not physical.any(), "Physical domain should be all False"

    def test_pml_boundary_is_true(self, grid, cfg):
        mask = grid.pml_mask()
        pw = cfg.pml.width
        # Top PML rows
        assert mask[:pw, :].all(), "Top PML rows should be all True"
        # Bottom PML rows
        assert mask[pw + cfg.grid.ny:, :].all(), "Bottom PML rows should be all True"
        # Left PML columns
        assert mask[:, :pw].all(), "Left PML columns should be all True"
        # Right PML columns
        assert mask[:, pw + cfg.grid.nx:].all(), "Right PML columns should be all True"

    def test_is_in_pml_consistent_with_mask(self, grid):
        mask = grid.pml_mask()
        # Check a few sample points
        pw = grid.pml_width
        # PML corner
        assert grid.is_in_pml(0, 0) == mask[0, 0]
        # Physical domain center
        ci = pw + grid.ny // 2
        cj = pw + grid.nx // 2
        assert grid.is_in_pml(ci, cj) == mask[ci, cj]
