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
