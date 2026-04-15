"""tests/test_pml.py -- PMLTensors unit tests"""

import math
import numpy as np
import pytest
from pathlib import Path

from src.config import load_config
from src.core import Grid2D, Medium2D
from src.physics.pml import PMLTensors


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
def pml(grid, cfg, medium):
    omega = cfg.physics.omega
    return PMLTensors(grid, cfg, omega, s0=medium.s0)


class TestPhysicalDomain:
    def test_physical_domain_gamma_one(self, pml, grid):
        """gamma = 1 in the physical (non-PML) domain."""
        pw = grid.pml_width
        phys_gamma_x = pml.gamma_x[pw:pw + grid.ny, pw:pw + grid.nx]
        phys_gamma_y = pml.gamma_y[pw:pw + grid.ny, pw:pw + grid.nx]
        np.testing.assert_allclose(phys_gamma_x, 1.0 + 0j, atol=1e-14)
        np.testing.assert_allclose(phys_gamma_y, 1.0 + 0j, atol=1e-14)

    def test_physical_domain_A_one_B_zero(self, pml, grid):
        """A = 1 and B = 0 in the physical domain."""
        pw = grid.pml_width
        phys_Ax = pml.A_x[pw:pw + grid.ny, pw:pw + grid.nx]
        phys_Ay = pml.A_y[pw:pw + grid.ny, pw:pw + grid.nx]
        phys_Bx = pml.B_x[pw:pw + grid.ny, pw:pw + grid.nx]
        phys_By = pml.B_y[pw:pw + grid.ny, pw:pw + grid.nx]
        np.testing.assert_allclose(phys_Ax, 1.0 + 0j, atol=1e-14)
        np.testing.assert_allclose(phys_Ay, 1.0 + 0j, atol=1e-14)
        np.testing.assert_allclose(phys_Bx, 0.0 + 0j, atol=1e-14)
        np.testing.assert_allclose(phys_By, 0.0 + 0j, atol=1e-14)


class TestPMLRegion:
    def test_pml_sigma_positive(self, pml, grid, cfg):
        """sigma > 0 in the PML region (gamma has positive imaginary part)."""
        pw = grid.pml_width
        omega = cfg.physics.omega
        # Left PML x-direction: columns 0..pw-1
        left_gamma_x = pml.gamma_x[pw, :pw]  # pick a physical-domain row
        left_sigma = left_gamma_x.imag * omega
        assert np.all(left_sigma > 0), "sigma must be positive in left PML"

        # Right PML x-direction
        right_gamma_x = pml.gamma_x[pw, pw + grid.nx:]
        right_sigma = right_gamma_x.imag * omega
        assert np.all(right_sigma > 0), "sigma must be positive in right PML"

        # Top PML y-direction
        top_gamma_y = pml.gamma_y[:pw, pw]
        top_sigma = top_gamma_y.imag * omega
        assert np.all(top_sigma > 0), "sigma must be positive in top PML"

        # Bottom PML y-direction
        bottom_gamma_y = pml.gamma_y[pw + grid.ny:, pw]
        bottom_sigma = bottom_gamma_y.imag * omega
        assert np.all(bottom_sigma > 0), "sigma must be positive in bottom PML"


class TestShapes:
    def test_shapes(self, pml, grid):
        """All PML tensors should have shape [ny_total, nx_total]."""
        expected = (grid.ny_total, grid.nx_total)
        assert pml.A_x.shape == expected
        assert pml.A_y.shape == expected
        assert pml.B_x.shape == expected
        assert pml.B_y.shape == expected
        assert pml.gamma_x.shape == expected
        assert pml.gamma_y.shape == expected

    def test_torch_tensor_shapes(self, pml, grid):
        """Torch tensor versions should match numpy shapes."""
        expected = (grid.ny_total, grid.nx_total)
        assert pml.A_x_t.shape == expected
        assert pml.A_y_t.shape == expected
        assert pml.B_x_t.shape == expected
        assert pml.B_y_t.shape == expected


class TestPMLAttenuation:
    def test_pml_theoretical_reflection(self, grid, cfg, medium):
        omega = cfg.physics.omega
        pml = PMLTensors(grid, cfg, omega, s0=medium.s0)
        L_pml = grid.pml_width * grid.h
        p = cfg.pml.power
        c_pml = 1.0 / medium.s0

        log_reflection = -2.0 * pml.sigma_max * L_pml / ((p + 1) * c_pml)
        R_theoretical = math.exp(
            np.nextafter(log_reflection, -np.inf)
        )

        assert R_theoretical <= cfg.pml.R0
