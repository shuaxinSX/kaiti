"""
Medium2D — 2D 介质慢度场
==========================

s(x) = 1/c(x) 慢度场

冻结决策 D6d:
    PML 层内介质退化为背景介质 s(x) ≡ s₀

支持的速度模型:
    - homogeneous : 均匀介质
    - smooth_lens : 平滑透镜异常体（高斯）
    - layered     : 水平层状介质
"""

import numpy as np
import torch


class Medium2D:
    """2D 介质慢度场。

    Attributes:
        c_background: 背景波速。
        s0: 震源处慢度 s(x_s)。
        velocity: 波速场 [ny_total, nx_total]。
        slowness: 慢度场 [ny_total, nx_total]。
        slowness_t: 慢度场 torch 张量。
    """

    def __init__(self, grid, cfg):
        """
        Args:
            grid: Grid2D 实例。
            cfg: Config 对象，需包含 cfg.medium.{c_background, velocity_model}
                 和 cfg.physics.source_pos。
        """
        self.c_background = cfg.medium.c_background
        self.grid = grid

        model = cfg.medium.velocity_model
        if model == "homogeneous":
            velocity = self._homogeneous(grid)
        elif model == "smooth_lens":
            velocity = self._smooth_lens(grid)
        elif model == "layered":
            velocity = self._layered(grid)
        else:
            raise ValueError(f"未知速度模型: {model}")

        # 先从原始速度场提取震源处慢度 s0 = s(x_s)
        # （PML 退化之前，确保取到真实介质值）
        src_pos = cfg.physics.source_pos
        j_s = round((src_pos[0] - grid.x_min) / grid.h) + grid.pml_width
        i_s = round((src_pos[1] - grid.y_min) / grid.h) + grid.pml_width
        self.s0 = 1.0 / velocity[i_s, j_s]

        # PML 退化 (D6d)：PML 层内强制 c = 1/s0（震源处速度）
        pml_mask = grid.pml_mask()
        velocity[pml_mask] = 1.0 / self.s0

        self.velocity = velocity
        self.slowness = 1.0 / velocity
        self.slowness_t = torch.from_numpy(self.slowness)

    def _homogeneous(self, grid):
        """均匀介质: c(x) = c_background。"""
        return np.full(
            (grid.ny_total, grid.nx_total), self.c_background, dtype=np.float64
        )

    def _smooth_lens(self, grid):
        """平滑透镜异常体: 物理域中心高斯扰动。"""
        velocity = np.full(
            (grid.ny_total, grid.nx_total), self.c_background, dtype=np.float64
        )
        # 异常体中心：物理域中心
        cx = (grid.x_min + grid.x_max) / 2.0
        cy = (grid.y_min + grid.y_max) / 2.0
        # 异常体半径：物理域尺寸的 1/4
        Lx = grid.x_max - grid.x_min
        sigma = Lx / 4.0
        # 高斯扰动: 速度增加 30%
        perturbation = 0.3 * self.c_background * np.exp(
            -((grid.xx - cx) ** 2 + (grid.yy - cy) ** 2) / (2 * sigma ** 2)
        )
        velocity += perturbation
        return velocity

    def _layered(self, grid):
        """水平层状介质: 三层速度结构。"""
        velocity = np.full(
            (grid.ny_total, grid.nx_total), self.c_background, dtype=np.float64
        )
        # 三层分界线（物理域 y 方向三等分）
        y_third = grid.y_min + (grid.y_max - grid.y_min) / 3.0
        y_two_third = grid.y_min + 2.0 * (grid.y_max - grid.y_min) / 3.0

        for i in range(grid.ny_total):
            y = grid.y_coords[i]
            if y < y_third:
                velocity[i, :] = self.c_background * 0.8
            elif y < y_two_third:
                velocity[i, :] = self.c_background * 1.0
            else:
                velocity[i, :] = self.c_background * 1.2

        return velocity
