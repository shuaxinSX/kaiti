"""tests/test_complex_utils.py — 复数工具函数单元测试"""

import torch
import pytest

from src.core.complex_utils import (
    complex_to_dual, dual_to_complex, complex_mul_dual, to_network_input,
)


class TestComplexDualRoundtrip:

    def test_roundtrip_2d(self):
        """complex → dual → complex 往返无损。"""
        z = torch.randn(4, 4) + 1j * torch.randn(4, 4)
        d = complex_to_dual(z)
        z2 = dual_to_complex(d)
        torch.testing.assert_close(z.real, z2.real)
        torch.testing.assert_close(z.imag, z2.imag)

    def test_roundtrip_batched(self):
        """带 batch 维度的往返。"""
        z = torch.randn(2, 4, 4) + 1j * torch.randn(2, 4, 4)
        d = complex_to_dual(z)
        assert d.shape == (2, 2, 4, 4)
        z2 = dual_to_complex(d)
        torch.testing.assert_close(z.real, z2.real)
        torch.testing.assert_close(z.imag, z2.imag)


class TestComplexMul:

    def test_mul_matches_native(self):
        """双通道乘法结果与原生复数乘法一致。"""
        a = torch.randn(1, 2, 4, 4)
        b = torch.randn(1, 2, 4, 4)
        result = complex_mul_dual(a, b)
        a_c = torch.complex(a[0, 0], a[0, 1])
        b_c = torch.complex(b[0, 0], b[0, 1])
        ref = a_c * b_c
        torch.testing.assert_close(result[0, 0], ref.real, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result[0, 1], ref.imag, atol=1e-6, rtol=1e-5)


class TestToNetworkInput:

    def test_shape_and_dtype(self):
        """输出 shape 为 [1, 2, H, W]，dtype 为 float32。"""
        z = torch.randn(8, 8, dtype=torch.cfloat)
        out = to_network_input(z)
        assert out.shape == (1, 2, 8, 8)
        assert out.dtype == torch.float32


class TestDualToComplexValidation:

    def test_wrong_channels_raises(self):
        """通道数 ≠ 2 时应抛出 ValueError。"""
        u = torch.randn(1, 3, 4, 4)
        with pytest.raises(ValueError):
            dual_to_complex(u)
