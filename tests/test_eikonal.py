"""
tests/test_eikonal.py -- EikonalSolver unit tests
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver


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


@pytest.fixture
def solver(grid, medium, source, bg, cfg):
    return EikonalSolver(grid, medium, source, bg, cfg)


class TestEikonalSolver:

    def test_homogeneous_alpha_near_one(self, solver, grid):
        """For uniform medium, alpha should be approximately 1 everywhere."""
        # Only check finite points (boundary may remain inf)
        finite_mask = np.isfinite(solver.alpha)
        assert np.any(finite_mask), "No finite alpha values found"
        np.testing.assert_allclose(
            solver.alpha[finite_mask], 1.0, atol=1e-6,
            err_msg="alpha deviates from 1.0 in homogeneous medium"
        )

    def test_convergence(self, solver, cfg):
        """Solver should converge before max_iter."""
        assert solver.converged_iter < cfg.eikonal.fsm_max_iter, (
            f"FSM did not converge: converged_iter={solver.converged_iter}, "
            f"max_iter={cfg.eikonal.fsm_max_iter}"
        )

    def test_no_nan(self, solver):
        """No NaN in alpha for finite points."""
        finite_mask = np.isfinite(solver.alpha)
        assert not np.any(np.isnan(solver.alpha[finite_mask])), (
            "alpha contains NaN values"
        )

    def test_tau_positive(self, solver, source):
        """tau should be positive away from source."""
        # Exclude source point and its immediate neighbors
        si, sj = source.i_s, source.j_s
        mask = np.ones_like(solver.tau, dtype=bool)
        mask[max(0, si - 2):si + 3, max(0, sj - 2):sj + 3] = False
        finite_mask = np.isfinite(solver.tau) & mask
        assert np.all(solver.tau[finite_mask] > 0), (
            "tau contains non-positive values away from source"
        )

    def test_shapes(self, solver, grid):
        """All fields should have correct shapes."""
        expected_shape = (grid.ny_total, grid.nx_total)
        assert solver.alpha.shape == expected_shape, (
            f"alpha.shape = {solver.alpha.shape}, expected {expected_shape}"
        )
        assert solver.tau.shape == expected_shape, (
            f"tau.shape = {solver.tau.shape}, expected {expected_shape}"
        )
        assert solver.alpha_t.shape == (grid.ny_total, grid.nx_total), (
            f"alpha_t.shape = {tuple(solver.alpha_t.shape)}, expected {expected_shape}"
        )
        assert solver.tau_t.shape == (grid.ny_total, grid.nx_total), (
            f"tau_t.shape = {tuple(solver.tau_t.shape)}, expected {expected_shape}"
        )


class TestGodunovManufactured:
    """Manufactured solution: 均匀介质 τ = s₀·r 解析对比。"""

    def test_tau_matches_analytical(self, solver, grid, source, medium):
        """FSM τ 与解析解 s₀·r 在远场一致 (L∞ < 0.01)。"""
        s0 = medium.s0
        tau_analytical = s0 * source.distance

        # 排除震源近场（1/r 奇点）和边界 inf
        si, sj = source.i_s, source.j_s
        margin = 3
        ny, nx = grid.ny_total, grid.nx_total
        mask = np.ones((ny, nx), dtype=bool)
        mask[max(0, si - 4):si + 5, max(0, sj - 4):sj + 5] = False
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
        mask &= np.isfinite(solver.tau)

        error = np.abs(solver.tau[mask] - tau_analytical[mask])
        assert np.max(error) < 0.01, (
            f"FSM tau deviates from analytical by {np.max(error):.6f}"
        )

    def test_eikonal_equation_residual(self, solver, grid, source, medium):
        """验证 |∇τ|² ≈ s² = s₀² (eikonal 方程残差)。"""
        from src.physics.diff_ops import DiffOps
        import torch

        h = grid.h
        diff_ops = DiffOps(h)
        s0 = medium.s0

        # 数值求 tau 梯度（仅用于验证，与 tau_ops 不同路径）
        tau_safe = np.where(np.isfinite(solver.tau), solver.tau, 0.0)
        tau_t = torch.from_numpy(tau_safe)
        dtau_dx = diff_ops.diff_x(tau_t).squeeze().numpy()
        dtau_dy = diff_ops.diff_y(tau_t).squeeze().numpy()
        grad_sq = dtau_dx ** 2 + dtau_dy ** 2

        # 排除震源和边界
        si, sj = source.i_s, source.j_s
        ny, nx = grid.ny_total, grid.nx_total
        margin = 4
        mask = np.ones((ny, nx), dtype=bool)
        mask[max(0, si - 6):si + 7, max(0, sj - 6):sj + 7] = False
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
        mask &= np.isfinite(solver.tau)

        np.testing.assert_allclose(
            grad_sq[mask], s0 ** 2, atol=0.01,
            err_msg="Eikonal equation |∇τ|²=s² violated"
        )
