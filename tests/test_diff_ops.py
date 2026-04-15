"""tests/test_diff_ops.py -- DiffOps unit tests"""

import torch
import pytest

from src.physics.diff_ops import DiffOps


@pytest.fixture
def ops():
    h = 0.01
    return DiffOps(h)


class TestDiffX:
    def test_diff_x_linear(self, ops):
        """d/dx of f(x)=x should be 1 at interior points."""
        H, W = 32, 32
        h = ops.h
        # Build f(x) = x: x varies along W dimension
        x = torch.arange(W, dtype=torch.float64) * h
        u = x.unsqueeze(0).expand(H, -1)  # [H, W]
        u4d = u.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        result = ops.diff_x(u4d)
        # Check interior points (exclude boundary columns affected by reflect pad)
        interior = result[0, 0, :, 2:-2]
        torch.testing.assert_close(
            interior,
            torch.ones_like(interior),
            atol=1e-10,
            rtol=0,
        )

    def test_diff_x_constant(self, ops):
        """d/dx of constant should be 0."""
        u = torch.ones(1, 1, 16, 16, dtype=torch.float64) * 5.0
        result = ops.diff_x(u)
        torch.testing.assert_close(
            result,
            torch.zeros_like(result),
            atol=1e-10,
            rtol=0,
        )


class TestDiffXX:
    def test_diff_xx_quadratic(self, ops):
        """d^2/dx^2 of f(x)=x^2 should be 2."""
        H, W = 32, 32
        h = ops.h
        x = torch.arange(W, dtype=torch.float64) * h
        u = (x ** 2).unsqueeze(0).expand(H, -1)
        u4d = u.unsqueeze(0).unsqueeze(0)
        result = ops.diff_xx(u4d)
        interior = result[0, 0, :, 2:-2]
        torch.testing.assert_close(
            interior,
            torch.full_like(interior, 2.0),
            atol=1e-10,
            rtol=0,
        )


class TestLaplacian:
    def test_laplacian_quadratic(self, ops):
        """Laplacian of f(x,y) = x^2 + y^2 should be 4."""
        H, W = 32, 32
        h = ops.h
        x = torch.arange(W, dtype=torch.float64) * h
        y = torch.arange(H, dtype=torch.float64) * h
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        u = xx ** 2 + yy ** 2
        u4d = u.unsqueeze(0).unsqueeze(0)
        result = ops.laplacian(u4d)
        interior = result[0, 0, 2:-2, 2:-2]
        torch.testing.assert_close(
            interior,
            torch.full_like(interior, 4.0),
            atol=1e-10,
            rtol=0,
        )


class TestOutputShape:
    def test_output_shape(self, ops):
        """Output shape should match input shape."""
        u = torch.randn(1, 1, 20, 30, dtype=torch.float64)
        assert ops.diff_x(u).shape == u.shape
        assert ops.diff_y(u).shape == u.shape
        assert ops.diff_xx(u).shape == u.shape
        assert ops.diff_yy(u).shape == u.shape
        assert ops.laplacian(u).shape == u.shape

    def test_2d_input_output_shape(self, ops):
        """2D input should produce 4D output after _ensure_4d."""
        u = torch.randn(20, 30, dtype=torch.float64)
        result = ops.diff_x(u)
        assert result.shape == (1, 1, 20, 30)
