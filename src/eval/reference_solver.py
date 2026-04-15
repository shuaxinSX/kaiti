"""
参考解求解器
============

基于当前 ResidualComputer 使用的同一离散算子，组装复数稀疏线性系统：

    L(Â_ref) = RHS_eq

其中 L 与训练时的 PDE 残差离散保持一致，用于生成可比较的数值参考解。
"""

import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


def _flat_index(i, j, width):
    return i * width + j


def _add_entry(rows, cols, data, row, col, value):
    rows.append(row)
    cols.append(col)
    data.append(np.complex128(value))


def assemble_reference_operator(trainer):
    """Assemble the sparse linear operator matching ResidualComputer."""
    residual = trainer.residual_computer
    height, width = trainer.grid.ny_total, trainer.grid.nx_total
    omega = trainer.omega
    h = trainer.grid.h

    a_x = residual.A_x.detach().cpu().numpy().astype(np.complex128)
    a_y = residual.A_y.detach().cpu().numpy().astype(np.complex128)
    b_x = residual.B_x.detach().cpu().numpy().astype(np.complex128)
    b_y = residual.B_y.detach().cpu().numpy().astype(np.complex128)
    g_x = residual.grad_tau_x_stretched.detach().cpu().numpy().astype(np.complex128)
    g_y = residual.grad_tau_y_stretched.detach().cpu().numpy().astype(np.complex128)
    lap_tau = residual.lap_tau_c.detach().cpu().numpy().astype(np.complex128)
    damp = residual.damp_coeff.detach().cpu().numpy().astype(np.complex128)

    rows = []
    cols = []
    data = []

    for i in range(height):
        for j in range(width):
            row = _flat_index(i, j, width)

            cx2 = a_x[i, j] / (h ** 2)
            cy2 = a_y[i, j] / (h ** 2)
            dx1 = (2.0j * omega * g_x[i, j] - b_x[i, j]) / (2.0 * h)
            dy1 = (2.0j * omega * g_y[i, j] - b_y[i, j]) / (2.0 * h)

            self_coeff = 1.0j * omega * lap_tau[i, j] + (omega ** 2) * damp[i, j]

            if j == 0:
                self_coeff += -2.0 * cx2
                _add_entry(rows, cols, data, row, _flat_index(i, 1, width), 2.0 * cx2)
            elif j == width - 1:
                self_coeff += -2.0 * cx2
                _add_entry(rows, cols, data, row, _flat_index(i, width - 2, width), 2.0 * cx2)
            else:
                self_coeff += -2.0 * cx2
                _add_entry(rows, cols, data, row, _flat_index(i, j - 1, width), cx2 - dx1)
                _add_entry(rows, cols, data, row, _flat_index(i, j + 1, width), cx2 + dx1)

            if i == 0:
                self_coeff += -2.0 * cy2
                _add_entry(rows, cols, data, row, _flat_index(1, j, width), 2.0 * cy2)
            elif i == height - 1:
                self_coeff += -2.0 * cy2
                _add_entry(rows, cols, data, row, _flat_index(height - 2, j, width), 2.0 * cy2)
            else:
                self_coeff += -2.0 * cy2
                _add_entry(rows, cols, data, row, _flat_index(i - 1, j, width), cy2 - dy1)
                _add_entry(rows, cols, data, row, _flat_index(i + 1, j, width), cy2 + dy1)

            _add_entry(rows, cols, data, row, row, self_coeff)

    size = height * width
    operator = coo_matrix((data, (rows, cols)), shape=(size, size), dtype=np.complex128)
    return operator.tocsr()


def solve_reference_scattering(trainer):
    """Solve the discrete factored Helmholtz system for a reference envelope."""
    logger.info("组装参考解稀疏线性系统...")
    operator = assemble_reference_operator(trainer)
    rhs = trainer.rhs.reshape(-1).astype(np.complex128)

    logger.info("求解参考解线性系统，未知量数=%d", rhs.size)
    solution = spsolve(operator, rhs)
    return solution.reshape(trainer.grid.ny_total, trainer.grid.nx_total)
