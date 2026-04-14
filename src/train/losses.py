"""
损失函数
=========

冻结决策 D12:
    L_total = λ_pde · L_pde + λ_data · L_data

    L_pde 由 ResidualComputer.compute() 返回。
    L_data = mean(|Â_pred − Â_true|²)  (可选，有监督标签时启用)
"""

import torch


def loss_data(A_pred, A_true):
    """数据损失: mean(|Â_pred − Â_true|²)。

    Args:
        A_pred: [B, 2, H, W] 预测散射包络（实虚双通道）。
        A_true: [B, 2, H, W] 真值散射包络。

    Returns:
        标量 tensor。
    """
    return torch.mean((A_pred - A_true) ** 2)


def loss_total(loss_pde, loss_data_val=None, lambda_pde=1.0, lambda_data=0.0):
    """总损失: λ_pde · L_pde + λ_data · L_data。

    Args:
        loss_pde: PDE 残差损失（标量 tensor）。
        loss_data_val: 数据损失（标量 tensor，可选）。
        lambda_pde: PDE 损失权重。
        lambda_data: 数据损失权重。

    Returns:
        标量 tensor。
    """
    total = lambda_pde * loss_pde
    if loss_data_val is not None and lambda_data > 0:
        total = total + lambda_data * loss_data_val
    return total
