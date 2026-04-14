"""
tests/test_rhs.py — 等效体源与穿孔掩码单元测试
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver
from src.physics.rhs import compute_rhs, compute_loss_mask


@pytest.fixture
def cfg():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    return load_config(base, debug)


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


@pytest.fixture
def eik(grid, medium, source, bg, cfg):
    return EikonalSolver(grid, medium, source, bg, cfg)


class TestComputeRHS:

    def test_homogeneous_rhs_zero(self, grid, medium, source, bg, eik, cfg):
        """均匀介质: s=s0 → perturbation=0 → RHS 全零。"""
        rhs = compute_rhs(grid, medium, source, bg, eik, cfg.physics.omega)
        np.testing.assert_allclose(np.abs(rhs), 0.0, atol=1e-20)

    def test_source_defuse(self, grid, medium, source, bg, eik, cfg):
        """震源点 RHS 精确为 0。"""
        rhs = compute_rhs(grid, medium, source, bg, eik, cfg.physics.omega)
        assert rhs[source.i_s, source.j_s] == 0.0 + 0.0j

    def test_no_nan_inf(self, grid, medium, source, bg, eik, cfg):
        """RHS 不含 NaN 或 Inf。"""
        rhs = compute_rhs(grid, medium, source, bg, eik, cfg.physics.omega)
        assert np.all(np.isfinite(rhs))

    def test_shape(self, grid, medium, source, bg, eik, cfg):
        """RHS shape 正确。"""
        rhs = compute_rhs(grid, medium, source, bg, eik, cfg.physics.omega)
        assert rhs.shape == (grid.ny_total, grid.nx_total)


class TestLossMask:

    def test_mask_at_source_zero(self, grid, source, cfg):
        """震源处掩码为 0。"""
        mask = compute_loss_mask(grid, source, cfg)
        assert mask[source.i_s, source.j_s] == 0.0

    def test_mask_far_from_source_one(self, grid, source, cfg):
        """远离震源处掩码为 1。"""
        mask = compute_loss_mask(grid, source, cfg)
        assert mask[0, 0] == 1.0

    def test_mask_shape(self, grid, source, cfg):
        """掩码 shape 正确。"""
        mask = compute_loss_mask(grid, source, cfg)
        assert mask.shape == (grid.ny_total, grid.nx_total)
