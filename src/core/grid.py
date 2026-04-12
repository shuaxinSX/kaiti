"""
Grid2D — 2D 规则笛卡尔网格
============================

坐标约定:
    grid[i, j] 对应物理坐标 (x_min + j*h, y_min + i*h)
    即 i 是行索引对应 y, j 是列索引对应 x

张量 shape:
    [H, W] 其中 H = ny_total, W = nx_total
    nx_total = nx + 2 * pml_width
    ny_total = ny + 2 * pml_width
"""

import numpy as np
import torch


class Grid2D:
    """2D 规则笛卡尔网格（含 PML 扩展）。

    Attributes:
        nx, ny: 物理域格点数（不含 PML）。
        pml_width: PML 层格点数。
        nx_total, ny_total: 总格点数（含 PML）。
        h: 均匀步长。
        x_min, x_max, y_min, y_max: 物理域边界。
        x_coords: x 方向坐标向量 [nx_total]。
        y_coords: y 方向坐标向量 [ny_total]。
        xx, yy: 网格坐标矩阵 [ny_total, nx_total]。
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Config 对象，需包含 cfg.grid.{nx, ny, domain} 和 cfg.pml.width。
        """
        self.nx = cfg.grid.nx
        self.ny = cfg.grid.ny
        self.pml_width = cfg.pml.width

        domain = cfg.grid.domain  # [x_min, x_max, y_min, y_max]
        self.x_min, self.x_max = domain[0], domain[1]
        self.y_min, self.y_max = domain[2], domain[3]

        # 均匀步长（x 和 y 方向一致）
        self.h = (self.x_max - self.x_min) / self.nx

        # 总格点数（含 PML 两侧）
        self.nx_total = self.nx + 2 * self.pml_width
        self.ny_total = self.ny + 2 * self.pml_width

        # 坐标向量（PML 向外延伸）
        # x 方向: 从 x_min - pml_width*h 到 x_max + (pml_width-1)*h
        self.x_coords = np.array(
            [self.x_min + (j - self.pml_width) * self.h for j in range(self.nx_total)],
            dtype=np.float64,
        )
        # y 方向: 从 y_min - pml_width*h 到 y_max + (pml_width-1)*h
        self.y_coords = np.array(
            [self.y_min + (i - self.pml_width) * self.h for i in range(self.ny_total)],
            dtype=np.float64,
        )

        # 网格坐标矩阵 [ny_total, nx_total]
        # xx[i, j] = x坐标, yy[i, j] = y坐标
        self.xx, self.yy = np.meshgrid(self.x_coords, self.y_coords)

        # torch 张量版本（float64 预处理用）
        self.xx_t = torch.from_numpy(self.xx)
        self.yy_t = torch.from_numpy(self.yy)

    def physical_slice(self):
        """返回物理域（不含 PML）的切片索引 (slice_y, slice_x)。"""
        pw = self.pml_width
        return slice(pw, pw + self.ny), slice(pw, pw + self.nx)

    def is_in_pml(self, i, j):
        """判断网格点 (i, j) 是否在 PML 层内。"""
        pw = self.pml_width
        return (i < pw or i >= pw + self.ny or
                j < pw or j >= pw + self.nx)

    def pml_mask(self):
        """返回 PML 区域的布尔掩码 [ny_total, nx_total]，PML 内为 True。"""
        mask = np.ones((self.ny_total, self.nx_total), dtype=bool)
        pw = self.pml_width
        mask[pw:pw + self.ny, pw:pw + self.nx] = False
        return mask
