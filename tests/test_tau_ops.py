"""
tests/test_tau_ops.py -- TauDerivatives unit tests (M4)
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver
from src.physics.diff_ops import DiffOps
from src.physics.tau_ops import TauDerivatives

BASE_PATH = Path(__file__).parent.parent / "configs" / "base.yaml"
DEBUG_PATH = Path(__file__).parent.parent / "configs" / "debug.yaml"


@pytest.fixture
def cfg():
    return load_config(BASE_PATH, DEBUG_PATH)


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


@pytest.fixture
def diff_ops(grid):
    return DiffOps(grid.h)


@pytest.fixture
def tau_d(bg, eik, diff_ops):
    return TauDerivatives(bg, eik, diff_ops)


class TestTauDerivatives:

    def test_no_nan_inf(self, tau_d):
        """All outputs must be finite (no NaN or Inf)."""
        for name in ["grad_tau_x", "grad_tau_y", "lap_tau",
                      "grad_alpha_x", "grad_alpha_y", "lap_alpha"]:
            arr = getattr(tau_d, name)
            assert torch.all(torch.isfinite(arr)), (
                f"{name} contains NaN or Inf"
            )

    def test_shapes(self, tau_d, grid):
        """All fields should have shape [ny_total, nx_total]."""
        expected_shape = (grid.ny_total, grid.nx_total)
        for name in ["grad_tau_x", "grad_tau_y", "lap_tau",
                      "grad_alpha_x", "grad_alpha_y", "lap_alpha"]:
            arr = getattr(tau_d, name)
            assert tuple(arr.shape) == expected_shape, (
                f"{name}.shape = {tuple(arr.shape)}, expected {expected_shape}"
            )

    def test_grad_tau_magnitude_homogeneous(self, tau_d, grid, medium, source):
        """For uniform medium, |grad tau|^2 should approximate s0^2 away from source."""
        s0 = medium.s0
        grad_mag_sq = tau_d.grad_tau_x ** 2 + tau_d.grad_tau_y ** 2

        # Build mask: exclude source neighborhood and boundaries
        si, sj = source.i_s, source.j_s
        ny, nx = grid.ny_total, grid.nx_total
        margin = 3  # exclude boundary cells affected by reflect padding
        mask = torch.ones(ny, nx, dtype=torch.bool)
        # Exclude source neighborhood (5x5 patch)
        r = 5
        mask[max(0, si - r):si + r + 1, max(0, sj - r):sj + r + 1] = False
        # Exclude boundary margin
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

        values = grad_mag_sq[mask].numpy()
        expected = s0 ** 2
        np.testing.assert_allclose(
            values, expected, atol=1e-4,
            err_msg=f"|grad tau|^2 deviates from s0^2={expected}"
        )

    def test_no_direct_tau_diff(self):
        """Source code must NOT contain direct diff_xx(tau) or diff_yy(tau)."""
        source_file = Path(__file__).parent.parent / "src" / "physics" / "tau_ops.py"
        content = source_file.read_text(encoding="utf-8")
        assert "diff_xx(tau)" not in content, (
            "FORBIDDEN: source contains diff_xx(tau)"
        )
        assert "diff_yy(tau)" not in content, (
            "FORBIDDEN: source contains diff_yy(tau)"
        )
