"""
因子化 Eikonal 求解器
======================

乘法因子化: τ = τ₀ · α

求解流程:
    1. 初始化: α 全部设为 INF
    2. 震源解析烫印 (D3a): α(xₛ)=1, 邻居一阶泰勒
    3. 四方向交替扫描 (Godunov 迎风, D3b)
    4. 因果律检验 + 1D 退化
    5. 收敛判定

冻结决策 D3:
    - FSM 全程 Float64
    - 震源冻结点绝对禁止更新
    - 不直接离散展开后的因子化方程
"""

import numpy as np
import torch


class EikonalSolver:
    def __init__(self, grid, medium, source, bg, cfg):
        """Solve factored eikonal for alpha using FSM.

        Args:
            grid: Grid2D instance.
            medium: Medium2D instance.
            source: PointSource instance.
            bg: BackgroundField instance.
            cfg: Config object with eikonal parameters.
        """
        h = grid.h
        s0 = medium.s0
        slowness = medium.slowness  # [ny_total, nx_total]
        tau0 = bg.tau0
        # Analytical grad_tau0 for Godunov weights
        grad_tau0_x = bg.grad_tau0_x  # p_x
        grad_tau0_y = bg.grad_tau0_y  # p_y

        ny, nx = grid.ny_total, grid.nx_total
        max_iter = cfg.eikonal.fsm_max_iter
        tol = cfg.eikonal.fsm_tol
        freeze_radius = cfg.eikonal.source_freeze_radius

        # Step 1: Initialize alpha = INF everywhere
        alpha = np.full((ny, nx), np.inf, dtype=np.float64)
        frozen = np.zeros((ny, nx), dtype=bool)

        # Step 2: Analytical seeding (D3a)
        si, sj = source.i_s, source.j_s
        alpha[si, sj] = 1.0
        frozen[si, sj] = True

        # Gradient of slowness at source (central difference)
        grad_s_x = (slowness[si, sj + 1] - slowness[si, sj - 1]) / (2 * h)
        grad_s_y = (slowness[si + 1, sj] - slowness[si - 1, sj]) / (2 * h)
        grad_alpha_src_x = grad_s_x / (2 * s0)
        grad_alpha_src_y = grad_s_y / (2 * s0)

        # Seed neighbors (8-connected)
        for di in [-freeze_radius, 0, freeze_radius]:
            for dj in [-freeze_radius, 0, freeze_radius]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = si + di, sj + dj
                if 0 <= ni < ny and 0 <= nj < nx:
                    dx_phys = dj * h
                    dy_phys = di * h
                    alpha[ni, nj] = 1.0 + grad_alpha_src_x * dx_phys + grad_alpha_src_y * dy_phys
                    frozen[ni, nj] = True

        # Step 3: FSM - four-direction alternating sweeps
        sweep_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for iteration in range(max_iter):
            alpha_old = alpha.copy()

            for sx, sy in sweep_dirs:
                i_range = range(1, ny - 1) if sy == 1 else range(ny - 2, 0, -1)
                j_range = range(1, nx - 1) if sx == 1 else range(nx - 2, 0, -1)

                for i in i_range:
                    for j in j_range:
                        if frozen[i, j]:
                            continue

                        s_val = slowness[i, j]
                        tau0_ij = tau0[i, j]
                        if tau0_ij < 1e-30:
                            continue

                        # Neighbors in sweep direction
                        ax = alpha[i, j - sx]  # x-neighbor
                        ay = alpha[i - sy, j]  # y-neighbor

                        # Skip if both neighbors are inf (no info yet)
                        ax_fin = np.isfinite(ax)
                        ay_fin = np.isfinite(ay)
                        if not ax_fin and not ay_fin:
                            continue

                        px = grad_tau0_x[i, j]  # analytical
                        py = grad_tau0_y[i, j]

                        # Dynamic weights (D3b)
                        Wx = sx * px + tau0_ij / h
                        Wy = sy * py + tau0_ij / h

                        candidates = []

                        # Try 2D update only if both neighbors are finite
                        if ax_fin and ay_fin:
                            Vx = (tau0_ij / h) * ax
                            Vy = (tau0_ij / h) * ay

                            # 2D quadratic (D3b step 3)
                            A = Wx**2 + Wy**2
                            B = Wx * Vx + Wy * Vy
                            C = Vx**2 + Vy**2 - s_val**2

                            disc = B**2 - A * C

                            if disc >= 0 and A > 0:
                                alpha_2d = (B + np.sqrt(disc)) / A
                                # Causality check (D3b step 4)
                                if (Wx * alpha_2d - Vx > 0) and (Wy * alpha_2d - Vy > 0):
                                    candidates.append(alpha_2d)

                        # 1D fallbacks (D3b step 5)
                        if ax_fin and Wx > 0:
                            Vx = (tau0_ij / h) * ax
                            alpha_1x = (Vx + s_val) / Wx
                            candidates.append(alpha_1x)
                        if ay_fin and Wy > 0:
                            Vy = (tau0_ij / h) * ay
                            alpha_1y = (Vy + s_val) / Wy
                            candidates.append(alpha_1y)

                        if candidates:
                            alpha_new = min(candidates)
                            if alpha_new < alpha[i, j]:
                                alpha[i, j] = alpha_new

            # Convergence check (only compare points finite in both old and new)
            both_finite = np.isfinite(alpha) & np.isfinite(alpha_old)
            if not np.any(both_finite):
                continue
            diff = np.max(np.abs(alpha[both_finite] - alpha_old[both_finite]))
            if diff < tol:
                self.converged_iter = iteration + 1
                break
        else:
            self.converged_iter = max_iter

        self.alpha = alpha
        self.tau = tau0 * alpha
        self.alpha_t = torch.from_numpy(alpha)
        self.tau_t = torch.from_numpy(self.tau)
