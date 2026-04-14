"""
SpectralConv2d — FNO 频域截断卷积层
=====================================

冻结决策 D11 (零填充拓扑修复):
    - FFT 前在空间维度后方填充至少 N 个零（总长 ≥ 2N-1）
    - FFT 后裁剪回原始物理尺寸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """FNO 核心层：频域权重乘法 + 高频截断 + 零填充拓扑修复。"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            modes1: y 方向保留的傅里叶模式数。
            modes2: x 方向保留的傅里叶模式数。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W] float32 张量。

        Returns:
            [B, C_out, H, W] float32 张量。
        """
        B, C, H, W = x.shape

        # 零填充拓扑修复 (D11)：总长 >= 2N-1
        pad_h = H
        pad_w = W
        x_padded = F.pad(x, [0, pad_w, 0, pad_h])

        # FFT
        x_ft = torch.fft.rfft2(x_padded)

        # 频域权重乘（只保留低频模式）
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.shape[2], x_ft.shape[3],
            dtype=torch.cfloat, device=x.device,
        )

        # 正频率区
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )
        # 负频率区（y 方向对称）
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        # IFFT
        x_out = torch.fft.irfft2(out_ft, s=x_padded.shape[-2:])

        # 裁剪回原尺寸
        return x_out[:, :, :H, :W]
