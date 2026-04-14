"""
PDE 残差计算器
===============

冻结决策 D7 (阻尼池解耦):
    R_damping = ω²·Σ_ξ [(∂ξτ)²·(1 − 1/γξ²)]·Â
    物理区 γ=1 → 截断误差自动湮灭

冻结决策 D12 (残差公式):
    R_pde = [Δ̃Â + iω(2∇̃Â·∇̃τ + Â·Δ̃τ) + R_damping − RHS_eq] / ω²

流水线:
    网络输出 Â → ∇̃Â → Δ̃Â → 组装 LHS → R_pde → L_pde
"""

import torch
import numpy as np


class ResidualComputer:
    """PDE 残差完整计算流水线。

    将预处理量（PML 张量、走时导数、RHS、掩码）与网络输出拼接，
    计算 PDE 残差和 L_pde 损失。
    """

    def __init__(self, grid, pml, tau_d, rhs, loss_mask, omega, diff_ops):
        """
        Args:
            grid: Grid2D 实例。
            pml: PMLTensors 实例。
            tau_d: TauDerivatives 实例。
            rhs: complex128 numpy [H, W]，等效体源。
            loss_mask: float64 numpy [H, W]，穿孔掩码。
            omega: 圆频率。
            diff_ops: DiffOps 实例。
        """
        self.omega = omega
        self.diff_ops = diff_ops

        # 预处理量转为 float32 torch 张量
        self.grad_tau_x = tau_d.grad_tau_x.float()
        self.grad_tau_y = tau_d.grad_tau_y.float()
        self.lap_tau = tau_d.lap_tau.float()

        # PML 张量（取实部和虚部分开处理，因为网络输出是实数双通道）
        self.A_x = pml.A_x_t  # complex128
        self.A_y = pml.A_y_t
        self.B_x = pml.B_x_t
        self.B_y = pml.B_y_t

        # 阻尼池解耦系数 (D7): (∂ξτ)²·(1 − 1/γξ²)
        gamma_x_t = torch.from_numpy(pml.gamma_x)
        gamma_y_t = torch.from_numpy(pml.gamma_y)
        self.damp_x = tau_d.grad_tau_x.double() ** 2 * (1.0 - 1.0 / gamma_x_t ** 2)
        self.damp_y = tau_d.grad_tau_y.double() ** 2 * (1.0 - 1.0 / gamma_y_t ** 2)

        # RHS 和掩码
        self.rhs_real = torch.from_numpy(rhs.real).float()
        self.rhs_imag = torch.from_numpy(rhs.imag).float()
        self.loss_mask = torch.from_numpy(loss_mask).float()

    def compute(self, A_scat_dual):
        """计算 PDE 残差和 L_pde。

        Args:
            A_scat_dual: [B, 2, H, W] float32 网络输出（Â_scat 实虚部）。

        Returns:
            dict: {
                'residual_real': [H, W], 'residual_imag': [H, W],
                'loss_pde': scalar tensor
            }
        """
        omega = self.omega

        # 拆分实虚部
        A_r = A_scat_dual[:, 0:1, :, :]  # [B, 1, H, W]
        A_i = A_scat_dual[:, 1:2, :, :]

        # PML 拉普拉斯 Δ̃Â（对实虚部分别计算）
        lap_A_r = self.diff_ops.diff_xx(A_r).squeeze() * self.A_x.real.float() \
                - self.diff_ops.diff_x(A_r).squeeze() * self.B_x.real.float() \
                + self.diff_ops.diff_yy(A_r).squeeze() * self.A_y.real.float() \
                - self.diff_ops.diff_y(A_r).squeeze() * self.B_y.real.float()

        lap_A_i = self.diff_ops.diff_xx(A_i).squeeze() * self.A_x.real.float() \
                - self.diff_ops.diff_x(A_i).squeeze() * self.B_x.real.float() \
                + self.diff_ops.diff_yy(A_i).squeeze() * self.A_y.real.float() \
                - self.diff_ops.diff_y(A_i).squeeze() * self.B_y.real.float()

        # ∇̃Â·∇̃τ（实部通道）
        grad_A_r_x = self.diff_ops.diff_x(A_r).squeeze()
        grad_A_r_y = self.diff_ops.diff_y(A_r).squeeze()
        grad_A_i_x = self.diff_ops.diff_x(A_i).squeeze()
        grad_A_i_y = self.diff_ops.diff_y(A_i).squeeze()

        dot_r = grad_A_r_x * self.grad_tau_x + grad_A_r_y * self.grad_tau_y
        dot_i = grad_A_i_x * self.grad_tau_x + grad_A_i_y * self.grad_tau_y

        # Â·Δ̃τ
        A_lap_tau_r = A_r.squeeze() * self.lap_tau
        A_lap_tau_i = A_i.squeeze() * self.lap_tau

        # iω(2∇̃Â·∇̃τ + Â·Δ̃τ) 展开: iω·(X_r + i·X_i) = ω·(-X_i + i·X_r)
        transport_r = 2.0 * dot_r + A_lap_tau_r
        transport_i = 2.0 * dot_i + A_lap_tau_i
        i_omega_transport_r = -omega * transport_i
        i_omega_transport_i = omega * transport_r

        # 阻尼池 R_damping (D7): ω²·(damp_x + damp_y)·Â
        damp_coeff = (self.damp_x + self.damp_y).real.float()
        R_damp_r = omega ** 2 * damp_coeff * A_r.squeeze()
        R_damp_i = omega ** 2 * damp_coeff * A_i.squeeze()

        # LHS = Δ̃Â + iω(transport) + R_damping
        lhs_r = lap_A_r + i_omega_transport_r + R_damp_r
        lhs_i = lap_A_i + i_omega_transport_i + R_damp_i

        # R_pde = (LHS - RHS) / ω²  (D12 归一化)
        res_r = (lhs_r - self.rhs_real) / omega ** 2
        res_i = (lhs_i - self.rhs_imag) / omega ** 2

        # L_pde = mean(mask · |R_pde|²)
        res_sq = res_r ** 2 + res_i ** 2
        loss_pde = torch.mean(self.loss_mask * res_sq)

        return {
            'residual_real': res_r.detach(),
            'residual_imag': res_i.detach(),
            'loss_pde': loss_pde,
        }
