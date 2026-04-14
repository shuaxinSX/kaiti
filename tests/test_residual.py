"""
tests/test_residual.py — PDE 残差前向检查
"""

from pathlib import Path

import torch
import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver
from src.physics.tau_ops import TauDerivatives
from src.physics.diff_ops import DiffOps
from src.physics.pml import PMLTensors
from src.physics.rhs import compute_rhs, compute_loss_mask
from src.physics.residual import ResidualComputer


@pytest.fixture
def cfg():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    return load_config(base, debug)


@pytest.fixture
def pipeline(cfg):
    """构建完整预处理管线。"""
    grid = Grid2D(cfg)
    medium = Medium2D(grid, cfg)
    source = PointSource(grid, cfg)
    omega = cfg.physics.omega
    bg = BackgroundField(grid, medium, source, omega)
    eik = EikonalSolver(grid, medium, source, bg, cfg)
    diff_ops = DiffOps(grid.h)
    tau_d = TauDerivatives(bg, eik, diff_ops)
    pml = PMLTensors(grid, cfg, omega)
    rhs = compute_rhs(grid, medium, source, bg, eik, omega)
    mask = compute_loss_mask(grid, source, cfg)
    rc = ResidualComputer(grid, pml, tau_d, rhs, mask, omega, diff_ops)
    return rc, grid


class TestResidualComputer:

    def test_forward_runs(self, pipeline, cfg):
        """前向可跑通，无报错。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        assert 'loss_pde' in result
        assert 'residual_real' in result
        assert 'residual_imag' in result

    def test_no_nan_inf(self, pipeline, cfg):
        """残差不含 NaN/Inf。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        assert torch.all(torch.isfinite(result['residual_real']))
        assert torch.all(torch.isfinite(result['residual_imag']))
        assert torch.isfinite(result['loss_pde'])

    def test_zero_network_homogeneous(self, pipeline, cfg):
        """零初始化网络 + 均匀介质 → 残差 = -RHS/ω²。
        均匀介质 RHS=0，所以残差应为 0。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        # 均匀介质 RHS 全零，零网络无散射 → 残差应接近零
        assert result['loss_pde'].item() < 1e-10

    def test_loss_pde_non_negative(self, pipeline, cfg):
        """L_pde 非负。"""
        rc, grid = pipeline
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        result = rc.compute(A_scat)
        assert result['loss_pde'].item() >= 0.0

    def test_residual_shape(self, pipeline, cfg):
        """残差 shape 正确。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        assert result['residual_real'].shape == (grid.ny_total, grid.nx_total)
        assert result['residual_imag'].shape == (grid.ny_total, grid.nx_total)
