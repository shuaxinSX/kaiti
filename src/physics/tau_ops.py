"""
走时导数安全重组
=================

冻结决策 D4 / D4a:
    绝对不直接对 τ 做二阶差分!

安全重组公式:
    ∇τ = α·∇τ₀ + τ₀·∇α
    Δτ = α·Δτ₀ + 2·(∇τ₀·∇α) + τ₀·Δα

其中:
    - τ₀, ∇τ₀, Δτ₀ 全部解析计算
    - 仅对平滑的 α 做数值差分求 ∇α 和 Δα
"""

import torch
import numpy as np


class TauDerivatives:
    """安全导数重组 (D4, D4a): 链式法则构造 nabla tau 和 Delta tau."""

    def __init__(self, bg, eikonal, diff_ops):
        """
        Chain rule reconstruction:
        nabla tau = alpha * nabla tau0 + tau0 * nabla alpha          (D4a)
        Delta tau = alpha * Delta tau0 + 2 * nabla tau0 . nabla alpha + tau0 * Delta alpha  (D4)

        Only alpha (smooth) gets numerical differentiation. tau0 derivatives are analytical.

        Args:
            bg: BackgroundField instance with tau0, grad_tau0_x, grad_tau0_y, lap_tau0.
            eikonal: EikonalSolver instance with alpha.
            diff_ops: DiffOps instance for numerical differentiation.
        """
        # Convert alpha to torch for numerical diff.
        # FSM leaves alpha=inf on the outermost boundary rows/columns
        # (sweep range is 1..ny-2, 1..nx-2). Replace boundary infs with
        # nearest interior values so numerical differentiation stays finite.
        alpha_np = eikonal.alpha.copy()
        inf_mask = ~np.isfinite(alpha_np)
        if np.any(inf_mask):
            # Extrapolate boundary: copy row 1 -> row 0, row -2 -> row -1, etc.
            alpha_np[0, :] = alpha_np[1, :]
            alpha_np[-1, :] = alpha_np[-2, :]
            alpha_np[:, 0] = alpha_np[:, 1]
            alpha_np[:, -1] = alpha_np[:, -2]
        alpha_t = torch.from_numpy(alpha_np)  # [H, W] float64

        # Numerical derivatives of smooth alpha
        grad_alpha_x = diff_ops.diff_x(alpha_t).squeeze()  # [H, W]
        grad_alpha_y = diff_ops.diff_y(alpha_t).squeeze()
        lap_alpha = (diff_ops.diff_xx(alpha_t) + diff_ops.diff_yy(alpha_t)).squeeze()

        # Analytical tau0 derivatives as torch
        alpha = alpha_t
        tau0 = torch.from_numpy(bg.tau0)
        grad_tau0_x = torch.from_numpy(bg.grad_tau0_x)
        grad_tau0_y = torch.from_numpy(bg.grad_tau0_y)
        lap_tau0 = torch.from_numpy(bg.lap_tau0)

        # Safe reconstruction (D4a): nabla tau = alpha * nabla tau0 + tau0 * nabla alpha
        self.grad_tau_x = alpha * grad_tau0_x + tau0 * grad_alpha_x
        self.grad_tau_y = alpha * grad_tau0_y + tau0 * grad_alpha_y

        # Safe reconstruction (D4): Delta tau = alpha * Delta tau0 + 2 * (nabla tau0 . nabla alpha) + tau0 * Delta alpha
        cross_term = grad_tau0_x * grad_alpha_x + grad_tau0_y * grad_alpha_y
        self.lap_tau = alpha * lap_tau0 + 2.0 * cross_term + tau0 * lap_alpha

        # Store intermediates
        self.grad_alpha_x = grad_alpha_x
        self.grad_alpha_y = grad_alpha_y
        self.lap_alpha = lap_alpha
