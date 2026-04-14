"""
训练循环
=========

单样本 overfit 验证训练管线可行性。
后续扩展：多波数课程学习。
"""

import logging
import math

import torch
import numpy as np

from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver
from src.physics.tau_ops import TauDerivatives
from src.physics.diff_ops import DiffOps
from src.physics.pml import PMLTensors
from src.physics.rhs import compute_rhs, compute_loss_mask
from src.physics.residual import ResidualComputer
from src.models.nsno import NSNO2D
from src.train.losses import loss_total

logger = logging.getLogger(__name__)


def build_network_input(grid, medium, source, bg, tau_d, omega):
    """构建网络输入张量 [1, 8, H, W]。

    通道定义 (D10):
        0: s(x) 实部 (慢度)
        1: s(x) 虚部 (=0, 慢度为实数)
        2: u₀ 实部
        3: u₀ 虚部
        4: ∇τ_x
        5: ∇τ_y
        6: Δτ
        7: ω (标量广播)
    """
    H, W = grid.ny_total, grid.nx_total

    channels = []
    # 0-1: 慢度实虚部
    channels.append(torch.from_numpy(medium.slowness).float())
    channels.append(torch.zeros(H, W, dtype=torch.float32))
    # 2-3: u₀ 实虚部
    channels.append(torch.from_numpy(bg.u0.real).float())
    channels.append(torch.from_numpy(bg.u0.imag).float())
    # 4-5: ∇τ
    channels.append(tau_d.grad_tau_x.float())
    channels.append(tau_d.grad_tau_y.float())
    # 6: Δτ
    channels.append(tau_d.lap_tau.float())
    # 7: ω 广播
    channels.append(torch.full((H, W), omega, dtype=torch.float32))

    # 处理可能的 NaN/Inf（来自边界）
    x = torch.stack(channels, dim=0).unsqueeze(0)  # [1, 8, H, W]
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


class Trainer:
    """最小训练循环。"""

    def __init__(self, cfg):
        self.cfg = cfg
        omega = cfg.physics.omega

        # 构建预处理管线
        logger.info("构建预处理管线...")
        self.grid = Grid2D(cfg)
        self.medium = Medium2D(self.grid, cfg)
        self.source = PointSource(self.grid, cfg)
        self.bg = BackgroundField(self.grid, self.medium, self.source, omega)
        self.eik = EikonalSolver(self.grid, self.medium, self.source, self.bg, cfg)
        self.diff_ops = DiffOps(self.grid.h)
        self.tau_d = TauDerivatives(self.bg, self.eik, self.diff_ops)
        self.pml = PMLTensors(self.grid, cfg, omega)
        self.rhs = compute_rhs(
            self.grid, self.medium, self.source, self.bg, self.eik, omega, cfg
        )
        self.loss_mask = compute_loss_mask(self.grid, self.source, cfg)
        self.omega = omega

        # 残差计算器
        self.residual_computer = ResidualComputer(
            self.grid, self.pml, self.tau_d, self.rhs,
            self.loss_mask, omega, self.diff_ops,
        )

        # 网络输入
        self.net_input = build_network_input(
            self.grid, self.medium, self.source, self.bg, self.tau_d, omega
        )

        # 模型
        self.model = NSNO2D(cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.lr)

        logger.info("预处理完成。网格: %dx%d, ω=%.1f",
                     self.grid.nx_total, self.grid.ny_total, omega)

    def train(self, epochs=None):
        """执行训练循环。

        Args:
            epochs: 训练轮次，默认使用配置值。

        Returns:
            loss 历史列表。
        """
        if epochs is None:
            epochs = self.cfg.training.epochs

        lambda_pde = self.cfg.training.lambda_pde
        lambda_data = self.cfg.training.lambda_data
        losses = []

        logger.info("开始训练: %d epochs", epochs)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 前向
            A_scat = self.model(self.net_input)

            # 残差 + 损失
            result = self.residual_computer.compute(A_scat)
            total = loss_total(result['loss_pde'], lambda_pde=lambda_pde,
                               lambda_data=lambda_data)

            # 反向
            total.backward()
            self.optimizer.step()

            loss_val = total.item()
            losses.append(loss_val)

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                logger.info("Epoch %d/%d, loss=%.6e", epoch + 1, epochs, loss_val)

        logger.info("训练完成。最终 loss=%.6e", losses[-1])
        return losses

    def reconstruct_wavefield(self):
        """重构全波场 u_total = u₀ + Â·exp[i(ωτ+π/4)]。

        Returns:
            complex128 numpy [ny_total, nx_total]。
        """
        with torch.no_grad():
            A_scat = self.model(self.net_input)

        A_r = A_scat[0, 0].numpy().astype(np.float64)
        A_i = A_scat[0, 1].numpy().astype(np.float64)
        A_complex = A_r + 1j * A_i

        # 安全 tau
        tau_safe = np.where(np.isfinite(self.eik.tau), self.eik.tau, 0.0)
        phase_restorer = np.exp(1j * (self.omega * tau_safe + math.pi / 4.0))

        u_total = self.bg.u0 + A_complex * phase_restorer
        return u_total
