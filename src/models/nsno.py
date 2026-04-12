"""
NSNO2D — Neumann Series Neural Operator (2D)
==============================================

冻结决策 D10:
    基于 Neumann 级数展开同构映射。
    每层 Block 对应一次多重散射迭代:
        h^(k+1) = σ(W_skip·u_inc + K_θ(M_φ(V, h^(k))))

组件:
    - M_φ : 逐点介质-波场局部非线性散射 (1×1 conv)
    - K_θ : 数据驱动全局积分核 (FNO 频域截断层)
    - W_skip : 源项残差注入 (skip connection)
    - 激活函数: GELU (C∞ 平滑性)

初始化:
    输出层零权重零偏置 (Zero-State Ignition)

输入通道: 8 (s实虚, u₀实虚, ∇τ_x, ∇τ_y, Δτ, ω)
输出通道: 2 (Â_scat 实虚部)
"""

# TODO: M7 实现
