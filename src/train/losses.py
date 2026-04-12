"""
损失函数
=========

冻结决策 D12:
    L_total = λ_pde · L_pde + λ_data · L_data

    L_pde = mean(loss_mask · |R_pde|²)
    L_data = mean(|Â_pred − Â_true|²)  (可选)

    残差两端统一除以 ω²，防止阻尼项淹没衍射项梯度。
"""

# TODO: M8/M9 实现
