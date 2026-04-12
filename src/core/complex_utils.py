"""
复数工具函数
=============

冻结决策 D1:
    采用实部/虚部双通道表示 [B, 2, H, W]。
    不直接依赖 torch.complex 做核心网络输入输出。
    内部预处理允许使用 complex128，送入网络前转双通道 float32。
"""

import torch


def complex_to_dual(z):
    """复数张量 → 双通道实虚部张量。

    Args:
        z: 复数张量 [B, H, W] 或 [H, W]。

    Returns:
        双通道张量 [B, 2, H, W] 或 [2, H, W]。
    """
    return torch.stack([z.real, z.imag], dim=-3)


def dual_to_complex(u):
    """双通道实虚部张量 → 复数张量。

    Args:
        u: 双通道张量 [B, 2, H, W] 或 [2, H, W]。

    Returns:
        复数张量 [B, H, W] 或 [H, W]。
    """
    if u.shape[-3] != 2:
        raise ValueError(f"期望通道维度为 2，实际为 {u.shape[-3]}")
    real = u[..., 0, :, :]
    imag = u[..., 1, :, :]
    return torch.complex(real.float(), imag.float())


def complex_mul_dual(a, b):
    """双通道复数乘法: (a_r + i*a_i)(b_r + i*b_i)。

    Args:
        a, b: 双通道张量 [..., 2, H, W]。

    Returns:
        双通道张量 [..., 2, H, W]。
    """
    a_r, a_i = a[..., 0:1, :, :], a[..., 1:2, :, :]
    b_r, b_i = b[..., 0:1, :, :], b[..., 1:2, :, :]
    out_r = a_r * b_r - a_i * b_i
    out_i = a_r * b_i + a_i * b_r
    return torch.cat([out_r, out_i], dim=-3)


def to_network_input(z):
    """复数预处理结果 → 网络输入格式。

    将 complex128/complex64 张量转为 float32 双通道。

    Args:
        z: 复数张量 [H, W]。

    Returns:
        float32 双通道张量 [1, 2, H, W]（含 batch 维度）。
    """
    real = z.real.float()
    imag = z.imag.float()
    return torch.stack([real, imag], dim=0).unsqueeze(0)
