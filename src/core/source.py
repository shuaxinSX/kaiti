"""
PointSource — 点源数据结构
============================

管理震源的物理坐标、网格索引和距离场。

关键属性:
    - pos         : 物理坐标 (x_s, y_s)
    - idx         : 网格索引 (i_s, j_s)，已包含 PML 偏移
    - distance    : 距离场 r(x) = |x - x_s|
    - safe_dist   : 安全距离场 r_safe = max(r, eps)，eps = 0.1 * h

冻结决策 D5:
    穿孔域掩码半径: 1.5h（配置可调）
"""

import numpy as np
import torch


class PointSource:
    """点源数据结构。

    Attributes:
        x_s, y_s: 震源物理坐标。
        i_s, j_s: 震源网格索引（含 PML 偏移）。
        eps: 安全距离 = 0.1 * h。
        distance: 距离场 [ny_total, nx_total]。
        safe_distance: 安全距离场 [ny_total, nx_total]。
        distance_t: 距离场 torch 张量。
        safe_distance_t: 安全距离场 torch 张量。
    """

    def __init__(self, grid, cfg):
        """
        Args:
            grid: Grid2D 实例。
            cfg: Config 对象，需包含 cfg.physics.source_pos = [x_s, y_s]。
        """
        self.x_s = cfg.physics.source_pos[0]
        self.y_s = cfg.physics.source_pos[1]
        self.grid = grid

        # 物理坐标 → 网格索引（含 PML 偏移）
        self.j_s = round((self.x_s - grid.x_min) / grid.h) + grid.pml_width
        self.i_s = round((self.y_s - grid.y_min) / grid.h) + grid.pml_width

        # 安全距离参数
        self.eps = 0.1 * grid.h

        # 距离场
        self.distance = np.sqrt(
            (grid.xx - self.x_s) ** 2 + (grid.yy - self.y_s) ** 2
        )

        # 安全距离场：防止除零
        self.safe_distance = np.maximum(self.distance, self.eps)

        # torch 张量版本
        self.distance_t = torch.from_numpy(self.distance)
        self.safe_distance_t = torch.from_numpy(self.safe_distance)
