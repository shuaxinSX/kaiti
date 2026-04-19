"""
PDE 残差计算器
===============

冻结决策 D7 (阻尼池解耦):
    R_damping = ω²·Σ_ξ [(∂ξτ)²·(1 − 1/γξ²)]·Â
    物理区 γ=1 → 截断误差自动湮灭

冻结决策 D12 (残差公式):
    R_pde = [Δ̃Â + iω(2∇̃Â·∇̃τ + Â·Δ̃τ) + R_damping − RHS_eq] / ω²

所有 PML 相关运算在复数域完成，不丢弃虚部。

流水线:
    网络输出 Â → ∇̃Â → Δ̃Â → 组装 LHS → R_pde → L_pde
"""

import torch


class ResidualComputer:
    """PDE 残差完整计算流水线。

    使用复数张量正确处理 PML 区域的复坐标拉伸。
    """

    def __init__(
        self,
        grid,
        pml,
        tau_d,
        rhs,
        loss_mask,
        omega,
        diff_ops,
        lap_tau_mode="mixed_legacy",
    ):
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
        self.lap_tau_mode = self._validate_lap_tau_mode(lap_tau_mode)

        # PML 系数保持 complex（不取 real）
        self.A_x = pml.A_x_t.cfloat()  # [H, W] complex64
        self.A_y = pml.A_y_t.cfloat()
        self.B_x = pml.B_x_t.cfloat()
        self.B_y = pml.B_y_t.cfloat()

        # 走时导数（实数，转为 float32）
        self.grad_tau_x = tau_d.grad_tau_x.float()
        self.grad_tau_y = tau_d.grad_tau_y.float()

        # 阻尼池解耦系数 (D7): (∂ξτ)²·(1 − 1/γξ²) — 保持复数
        gamma_x_t = torch.from_numpy(pml.gamma_x).cfloat()
        gamma_y_t = torch.from_numpy(pml.gamma_y).cfloat()
        self.damp_coeff = (
            tau_d.grad_tau_x.cfloat() ** 2 * (1.0 - 1.0 / gamma_x_t ** 2)
            + tau_d.grad_tau_y.cfloat() ** 2 * (1.0 - 1.0 / gamma_y_t ** 2)
        )

        # 预计算 legacy / candidate 两套 Δ̃τ 缓存，并激活当前 mode。
        self._precompute_pml_lap_tau(pml, tau_d, diff_ops)

        # RHS 转为复数张量
        self.rhs_c = torch.from_numpy(rhs).cfloat()  # [H, W]

        # 掩码
        self.loss_mask = torch.from_numpy(loss_mask).float()
        self.device = torch.device("cpu")

    def _validate_lap_tau_mode(self, lap_tau_mode):
        valid_modes = ("mixed_legacy", "stretched_divergence")
        if lap_tau_mode not in valid_modes:
            raise ValueError(
                f"Unsupported lap_tau_mode={lap_tau_mode!r}. "
                f"Expected one of {valid_modes}."
            )
        return lap_tau_mode

    def _activate_lap_tau_mode(self):
        if self.lap_tau_mode == "mixed_legacy":
            self.lap_tau_c = self.lap_tau_legacy_c
        else:
            self.lap_tau_c = self.lap_tau_candidate_c

    def to(self, device):
        """Move cached tensors to the requested device."""
        self.device = torch.device(device)
        tensor_names = (
            "A_x",
            "A_y",
            "B_x",
            "B_y",
            "grad_tau_x",
            "grad_tau_y",
            "damp_coeff",
            "grad_tau_x_stretched",
            "grad_tau_y_stretched",
            "lap_tau_legacy_c",
            "lap_tau_candidate_c",
            "lap_tau_c",
            "rhs_c",
            "loss_mask",
        )
        for name in tensor_names:
            setattr(self, name, getattr(self, name).to(self.device))
        self._activate_lap_tau_mode()
        return self

    def _precompute_pml_lap_tau(self, pml, tau_d, diff_ops):
        """预计算 PML 拉伸走时拉普拉斯 Δ̃τ 和梯度系数。

        ∇̃τ·∇̃Â 中需要 (∂xτ/γx²)·∂xÂ + (∂yτ/γy²)·∂yÂ
        所以预计算: grad_tau_x_stretched = ∂xτ · A_x, grad_tau_y_stretched = ∂yτ · A_y

        Δ̃τ 用链式法则:
        Δ̃τ = α·Δ̃τ₀ + 2·(∇̃τ₀·∇̃α) + τ₀·Δ̃α
        其中 Δ̃f = A_x·f_xx - B_x·f_x + A_y·f_yy - B_y·f_y
        """
        # ∇τ 拉伸系数（用于输运项 ∇̃Â·∇̃τ = A_x·∂xÂ·∂xτ + A_y·∂yÂ·∂yτ）
        self.grad_tau_x_stretched = self.A_x * tau_d.grad_tau_x.cfloat()
        self.grad_tau_y_stretched = self.A_y * tau_d.grad_tau_y.cfloat()

        # legacy baseline：未拉伸 lap_tau 与已拉伸输运项混用。
        self.lap_tau_legacy_c = tau_d.lap_tau.cfloat()

        # A5 审计候选：对安全重组后的 grad_tau 再施加 stretched divergence。
        grad_tau_xx = diff_ops.diff_x(tau_d.grad_tau_x).squeeze().cfloat()
        grad_tau_yy = diff_ops.diff_y(tau_d.grad_tau_y).squeeze().cfloat()
        grad_tau_x_c = tau_d.grad_tau_x.cfloat()
        grad_tau_y_c = tau_d.grad_tau_y.cfloat()
        self.lap_tau_candidate_c = (
            self.A_x * grad_tau_xx
            - self.B_x * grad_tau_x_c
            + self.A_y * grad_tau_yy
            - self.B_y * grad_tau_y_c
        )
        self._activate_lap_tau_mode()

    def compute(self, A_scat_dual):
        """计算 PDE 残差和 L_pde。

        全部在复数域计算，正确处理 PML 系数虚部。

        Args:
            A_scat_dual: [B, 2, H, W] float32 网络输出（Â_scat 实虚部）。

        Returns:
            dict: {
                'residual_real': [H, W], 'residual_imag': [H, W],
                'loss_pde': scalar tensor
            }
        """
        omega = self.omega
        if A_scat_dual.device != self.device:
            raise RuntimeError(
                f"ResidualComputer device mismatch: expected {self.device}, "
                f"got {A_scat_dual.device}"
            )

        # 组装复数 Â = A_r + i·A_i
        A_r = A_scat_dual[:, 0:1, :, :]  # [B, 1, H, W]
        A_i = A_scat_dual[:, 1:2, :, :]
        A_c = torch.complex(A_r, A_i).squeeze()  # [H, W] complex

        # --- 1. PML 拉普拉斯 Δ̃Â (D6c) ---
        # Δ̃Â = A_x·∂xxÂ − B_x·∂xÂ + A_y·∂yyÂ − B_y·∂yÂ
        # 对实虚部分别做差分，再组合为复数
        Axx_r = self.diff_ops.diff_xx(A_r).squeeze()
        Ax_r = self.diff_ops.diff_x(A_r).squeeze()
        Ayy_r = self.diff_ops.diff_yy(A_r).squeeze()
        Ay_r = self.diff_ops.diff_y(A_r).squeeze()

        Axx_i = self.diff_ops.diff_xx(A_i).squeeze()
        Ax_i = self.diff_ops.diff_x(A_i).squeeze()
        Ayy_i = self.diff_ops.diff_yy(A_i).squeeze()
        Ay_i = self.diff_ops.diff_y(A_i).squeeze()

        Axx_c = torch.complex(Axx_r, Axx_i)
        Ax_c = torch.complex(Ax_r, Ax_i)
        Ayy_c = torch.complex(Ayy_r, Ayy_i)
        Ay_c = torch.complex(Ay_r, Ay_i)

        lap_A = self.A_x * Axx_c - self.B_x * Ax_c + self.A_y * Ayy_c - self.B_y * Ay_c

        # --- 2. 输运项 iω(2·∇̃Â·∇̃τ + Â·Δ̃τ) ---
        # ∇̃Â·∇̃τ = A_x·(∂xÂ·∂xτ) + A_y·(∂yÂ·∂yτ)  （PML 拉伸）
        dot_grad = self.grad_tau_x_stretched * Ax_c + self.grad_tau_y_stretched * Ay_c

        # Â·Δ̃τ
        A_lap_tau = A_c * self.lap_tau_c

        transport = 1j * omega * (2.0 * dot_grad + A_lap_tau)

        # --- 3. 阻尼池 R_damping (D7) ---
        R_damp = omega ** 2 * self.damp_coeff * A_c

        # --- 4. 组装 LHS 和残差 ---
        lhs = lap_A + transport + R_damp
        R_pde = (lhs - self.rhs_c) / omega ** 2

        # --- 5. L_pde = mean(mask · |R_pde|²) ---
        res_sq = R_pde.real ** 2 + R_pde.imag ** 2
        loss_pde = torch.mean(self.loss_mask * res_sq)

        return {
            'residual_real': R_pde.real.detach(),
            'residual_imag': R_pde.imag.detach(),
            'loss_pde': loss_pde,
        }
