"""
等效体源与穿孔掩码 (D5, D8, D9)
=================================

RHS_eq = -ω²·(s²−s₀²)·u₀·exp[-i(ωτ+π/4)]

- D8: 2D 相位补偿 (π/4 汉克尔相移)
- D9: 震源处防爆 (硬编码清零)
- D5: 穿孔域掩码 (Loss 中震源近场权重为 0)
"""

import math
import numpy as np


def compute_rhs(grid, medium, source, bg, eikonal, omega, cfg=None):
    """计算等效体源 RHS_eq。

    Args:
        cfg: Config 对象，用于读取 source_mask_radius。若为 None 则默认 1.5h。

    Returns:
        complex128 numpy 数组 [ny_total, nx_total]。
    """
    # 介质扰动：PML 内为 0（由介质退化保证）
    perturbation = medium.slowness ** 2 - medium.s0 ** 2

    # 安全 tau：FSM 最外层边界可能残留 inf，替换为 0（PML 外无物理意义）
    tau_safe = np.where(np.isfinite(eikonal.tau), eikonal.tau, 0.0)

    # 相位剥离（含 2D π/4 补偿, D8）
    phase_strip = np.exp(-1j * (omega * tau_safe + math.pi / 4.0))

    # 等效体源
    rhs = -omega ** 2 * perturbation * bg.u0 * phase_strip

    # 震源防爆 (D9)：近场清零，半径与穿孔掩码一致
    defuse_radius = cfg.loss.source_mask_radius if cfg is not None else 1.5
    source_near = source.distance <= defuse_radius * grid.h
    rhs[source_near] = 0.0 + 0.0j

    return rhs


def compute_loss_mask(grid, source, cfg):
    """计算穿孔域掩码 (D5)。

    Returns:
        float64 numpy 数组 [ny_total, nx_total]，震源近场为 0，远场为 1。
    """
    mask_radius = cfg.loss.source_mask_radius
    return (source.distance > mask_radius * grid.h).astype(np.float64)
