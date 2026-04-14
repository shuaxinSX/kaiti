"""
NSNO2D — Neumann Series Neural Operator (2D)
==============================================

冻结决策 D10:
    基于 Neumann 级数展开同构映射。
    每层 Block 对应一次多重散射迭代:
        h^(k+1) = σ(W_skip·u_inc + K_θ(M_φ(V, h^(k))))

输入通道: 8 (s实虚, u₀实虚, ∇τ_x, ∇τ_y, Δτ, ω)
输出通道: 2 (Â_scat 实虚部)
"""

import torch
import torch.nn as nn

from .spectral_conv import SpectralConv2d


class NeumannSpectralBlock(nn.Module):
    """单层 Neumann 散射迭代 Block。

    h^(k+1) = σ(W_skip · u_inc + K_θ(M_φ(V, h^(k))))
    """

    def __init__(self, channels, modes1, modes2):
        super().__init__()
        # M_φ: 逐点局部散射（1x1 conv）
        self.local_scatter = nn.Conv2d(channels, channels, 1)
        # K_θ: 全局积分核（FNO 谱卷积）
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        # W_skip: 源项残差注入
        self.skip = nn.Conv2d(channels, channels, 1)
        # 激活函数：GELU（C∞ 平滑）
        self.activation = nn.GELU()

    def forward(self, h, u_inc):
        """
        Args:
            h: 隐层状态 [B, C, H, W]。
            u_inc: 入射场编码 [B, C, H, W]。

        Returns:
            更新后的隐层状态 [B, C, H, W]。
        """
        # M_φ: 逐点散射
        scattered = self.local_scatter(h)
        # K_θ: 全局传播
        propagated = self.spectral_conv(scattered)
        # W_skip: 残差注入 + 激活
        return self.activation(self.skip(u_inc) + propagated)


class NSNO2D(nn.Module):
    """结构化神经算子完整网络。

    输入: [B, 8, H, W] — 8 通道物理特征
    输出: [B, 2, H, W] — Â_scat 实虚部
    """

    INPUT_CHANNELS = 8  # s实虚, u₀实虚, ∇τ_x, ∇τ_y, Δτ, ω

    def __init__(self, cfg):
        super().__init__()
        channels = cfg.model.nsno_channels
        modes = cfg.model.fno_modes
        n_blocks = cfg.model.nsno_blocks

        # 输入编码: 8 → hidden
        self.encoder = nn.Conv2d(self.INPUT_CHANNELS, channels, 1)

        # N 个 Neumann 散射 Block
        self.blocks = nn.ModuleList([
            NeumannSpectralBlock(channels, modes, modes)
            for _ in range(n_blocks)
        ])

        # 输出解码: hidden → 2 (实虚部)
        self.decoder = nn.Conv2d(channels, 2, 1)

        # Zero-State Ignition (D10): 输出层零初始化
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 8, H, W] 输入特征。

        Returns:
            [B, 2, H, W] 散射包络 Â_scat 的实虚部。
        """
        # 编码
        u_inc = self.encoder(x)
        h = u_inc

        # 多重散射迭代
        for block in self.blocks:
            h = block(h, u_inc)

        # 解码
        return self.decoder(h)
