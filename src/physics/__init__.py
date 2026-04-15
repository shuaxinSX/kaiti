"""
physics 子包 — 物理场计算
==========================

- background.py : u₀ — 2D 汉克尔格林函数 + 安全距离
- eikonal.py    : τ₀, α, FSM 求解器, 解析烫印
- tau_ops.py    : ∇τ, Δτ 的链式法则安全重组
- pml.py        : σ, γ, γ', A, B 张量预计算
- diff_ops.py   : 固定卷积核微分引擎
- rhs.py        : RHS_eq 等效体源组装 + 震源防爆
- residual.py   : PDE 残差计算器（含阻尼池解耦）
"""
