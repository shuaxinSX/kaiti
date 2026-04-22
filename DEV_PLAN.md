# DEV_PLAN.md — 高波数 Helmholtz 神经算子 2D 原型开发规范

> **本文档是开发推进规范，不是理论说明。**
> 所有实现决策在此冻结，代码必须严格遵循本文档。
> 如需修改决策，先改文档，再开分支重做。

---

## 1. 项目概述

### 1.1 项目目标

本项目实现一个面向高波数 Helmholtz 方程的分阶段原型系统：先完成 **2D 单源、双精度预处理、物理残差验证**，再逐步扩展到神经算子训练与多波数泛化。

核心管线：

```
介质 s(x) ──> [Eikonal预处理] ──> τ, ∇τ, Δτ, u₀, RHS_eq ──> [NSNO网络] ──> Â_scat ──> u_total
```

### 1.2 当前阶段目标（Phase 1 — 2D Prototype）

- 2D、单点源、标量声学、规则笛卡尔网格
- 双精度预处理（Float64 求解 → Float32 送入网络）
- 单机 PyTorch 原型
- 完成前向推断 + PDE 残差验证 + 最小 overfit 实验

**暂不处理**：3D、多源批处理、分布式训练、大规模 benchmark、论文级实验。

### 1.3 开发原则

1. 一次只推进一个里程碑
2. 上一步未通过检查，不进入下一步
3. 所有公式先冻结（见第 3 节），再编码
4. 每个里程碑必须有最小测试
5. 每次合并到 `dev` 前，必须更新文档和实验记录
6. 遇到理论与实现冲突，**优先修改文档**，不直接改代码绕过去

---

## 2. 范围冻结（Scope Freeze）

### 本版本包含

| 类别 | 内容 |
|------|------|
| PDE | 2D 标量 Helmholtz：Δu + ω²s²u = −δ(x−xₛ) |
| 源项 | 单个点源（Dirac δ） |
| 网格 | 规则笛卡尔网格，均匀步长 h |
| 复数表示 | 实部/虚部双通道 `[B, 2, H, W]` |
| 相位先验 | 因子化 Eikonal：τ = τ₀·α |
| 背景场 | 2D 汉克尔格林函数 u₀ = (i/4)H₀⁽¹⁾(ωs₀r) |
| 走时导数 | 链式法则半解析重组（禁止直接对 τ 做二阶差分） |
| 边界 | PML 复坐标拉伸 |
| 微分引擎 | 固定卷积核中心差分（`F.conv2d`） |
| 网络 | 结构化神经算子（NSNO）前向骨架 |
| 验证 | PDE residual 前向检查 + 最小 overfit |
| 2D 相位补偿 | 汉克尔 π/4 相移补偿（维度自适应拟设） |

### 本版本不包含

- 3D
- 多源联合训练
- `torch.autograd` 求二阶空间导数
- 端到端大规模训练
- 论文级 benchmark
- 工程部署优化
- 复数射线追踪（阴影区用椭圆代偿处理）

---

## 3. 冻结决策（Frozen Decisions）

### D1. 复数表示

- 采用实部/虚部双通道，不直接依赖 `torch.complex` 做核心网络输入输出
- 所有物理张量接口统一为 `[B, C, H, W]`，其中 `C` 的前两通道为实虚部
- 内部预处理计算允许使用 `complex128`，送入网络前转为双通道 `float32`

### D2. 空间导数实现

- **绝对不使用** `torch.autograd.grad` 计算二阶空间导数
- 一阶、二阶导数统一使用固定卷积核（不可训练参数）通过 `F.conv2d` 实现
- 二阶导数模板：`[1, -2, 1] / h²`
- 一阶导数模板：`[-1, 0, 1] / (2h)`（中心差分）
- 边界处理：至少 1 层 reflect padding（PML 外侧无物理意义，可用零填充）

### D3. 走时计算

- **不直接离散**展开后的因子化方程三项式
- 走时求解采用 Godunov 迎风 + 快速扫描法（FSM）路线
- 乘法因子化：`τ = τ₀ · α`
- FSM 在 Float64 精度下运行
- 震源处 α 解析初始化（见 D3a）

### D3a. 震源区解析烫印（Analytical Seeding）

- 震源点：`α(xₛ) = 1.0`
- 震源紧邻一圈网格：利用介质微观斜率定理解析赋值

  ```
  ∇α(xₛ) = ∇s(xₛ) / (2·s₀)
  α(x_nb) = 1.0 + ∇α(xₛ) · (x_nb − xₛ)
  ```

- 上述点标记为 `FROZEN`，FSM 迭代中**绝对禁止更新**
- `∇s(xₛ)` 使用二阶中心差分从慢度矩阵提取

### D3b. Godunov 迎风离散

对于每个扫描方向 `(sₓ, sᵧ) ∈ {±1}²`：

1. 计算动态权重：`Wₓ = sₓ·pₓ + τ₀/h`，`Wᵧ = sᵧ·pᵧ + τ₀/h`
   - 其中 `pₓ = ∂τ₀/∂x`，`pᵧ = ∂τ₀/∂y` 为**解析值**（不用数值差分）
2. 计算邻居常数：`Vₓ = (τ₀/h)·αₓ`，`Vᵧ = (τ₀/h)·αᵧ`
3. 求解一元二次方程 `A·α² − 2B·α + C = 0`
   - `A = Wₓ² + Wᵧ²`，`B = Wₓ·Vₓ + Wᵧ·Vᵧ`，`C = Vₓ² + Vᵧ² − s²`
4. **因果律检验**：候选根必须同时满足 `Wₓ·α − Vₓ > 0` 且 `Wᵧ·α − Vᵧ > 0`
5. 不满足则退化为 1D 更新：`α₁ₓ = (Vₓ + s) / Wₓ` 或 `α₁ᵧ = (Vᵧ + s) / Wᵧ`
6. 取所有合法候选根的**最小值**更新

### D4. `Δτ` 的构造

- **绝对不直接对 τ 做二阶差分**
- 使用链式法则重组：

  ```
  Δτ = α·Δτ₀ + 2·∇τ₀·∇α + τ₀·Δα
  ```

- 其中 `τ₀`、`∇τ₀`、`Δτ₀` 全部用解析公式计算
  - 2D：`τ₀ = s₀·r`，`∇τ₀ = s₀·r̂`（r̂ = (x−xₛ)/r），`Δτ₀ = s₀/r`
  - 震源处 `Δτ₀` 的 `1/r` 奇点：由穿孔掩码处理（见 D5）
- 仅对**平滑的 α** 做数值差分求 `∇α` 和 `Δα`

### D4a. `∇τ` 的构造

- 同样使用链式法则：

  ```
  ∇τ = α·∇τ₀ + τ₀·∇α
  ```

- `∇τ₀` 为解析值，`∇α` 为数值差分值

### D5. 震源点处理

- PDE residual 在震源邻域采用**穿孔域掩码（Punctured Domain Mask）**
- 掩码半径：当前版本固定为 `1.5h`
- 掩码逻辑：`loss_mask = (distance_grid > 1.5 * h).float()`
- **不修改**网络输出值（允许网络在震源处自由预测 A₀ ≠ 0）
- 仅在 Loss 计算中将震源近场残差权重置为 0

**物理依据**：

> 真实散射包络在震源处为非零常数（反射回声）。包络场因 WKBJ 剥离
> 产生的 `r` 型锥形尖峰（C⁰ 连续但 C¹ 不连续），与 Δτ 的 1/r 奇点
> 在连续强形式下完美对冲。但 C∞ 平滑的神经网络无法输出锥形拓扑，
> 造成"拓扑死锁"。穿孔掩码是泛函弱解意义上唯一正确的工业级策略。

### D6. PML 处理

#### D6a. 阻尼剖面

- 二次方衰减：`σ(x) = σ_max · ((x − x_start) / L_pml)²`
- `σ'(x)` **必须用解析公式**：`σ'(x) = 2·σ_max / L_pml² · (x − x_start)`
- **绝对禁止**对 σ 做数值差分

#### D6b. 复数拉伸因子

```
γ(x) = 1 + i·σ(x)/ω
γ'(x) = i·σ'(x)/ω
```

#### D6c. PML 拉普拉斯的张量解耦

将空间异性的复数微分解耦为**固定卷积 + 逐元素乘法**：

```
Δ̃U = Aₓ ⊙ (Kₓₓ * U) − Bₓ ⊙ (Kₓ * U) + Aᵧ ⊙ (Kᵧᵧ * U) − Bᵧ ⊙ (Kᵧ * U)
```

其中：
- `Aξ = 1/γξ²`，`Bξ = γξ'/γξ³`（在每个网格点预计算的复数张量）
- `Kₓₓ`、`Kᵧᵧ`：标准二阶中心差分卷积核 `[1,-2,1]/h²`
- `Kₓ`、`Kᵧ`：标准一阶中心差分卷积核 `[-1,0,1]/(2h)`

#### D6d. PML 内介质处理

- **强制假定**：PML 层内介质退化为背景介质 `s(x) ≡ s₀`
- 因而等效体源 RHS_eq 在 PML 内**严格等于 0**
- u₀ 和 τ 只在实数空间计算，无需复坐标格林函数

#### D6e. PML 衰减充分性

- 必须保证 PML 外层波场衰减至 FP32 机器精度以下
- 验收条件：`max(|u(x_boundary)|) < 1.19e-7`
- 典型参数起点：`L_pml = 20 格点`，`σ_max = 按经验公式计算`

  ```
  σ_max = −(p+1)·c_max·ln(R₀) / (2·L_pml·h)
  ```

  其中 p=2（二次方），R₀=1e-6（目标反射系数）

### D7. 阻尼池代数解耦（Anti-Parasitic Source）

控制方程中 `ω²(s² − ∇̃τ·∇̃τ)·Â` 在物理区内应精确为 0，但因迎风/中心差分格式不匹配会产生 O(h) 截断误差，被 ω² 放大后成为虚假寄生源。

**强制使用安全形式**：

```
R_damping = ω² · [ (∂ₓτ)²·(1 − 1/γₓ²) + (∂ᵧτ)²·(1 − 1/γᵧ²) ] · Â
```

> 物理区 γ=1 → (1−1)≡0，截断误差自动湮灭。
> PML 区 γ≠1 → 阻尼正常生效。

### D8. 2D 维度自适应相位补偿

2D 汉克尔函数渐近展开自带 +π/4 相移。若不补偿，网络实虚通道被强制锁在 45° 对角线上，产生严重特征串扰。

**修正拟设**：

```
u_total = u₀ + Â_scat · exp[i(ωτ + π/4)]
```

**RHS 组装时同步补偿**：

```
RHS_eq = −ω²·(s² − s₀²)·u₀·exp[−i(ωτ + π/4)]
```

**PDE 方程左端（LHS）不需要任何修改**，因为 π/4 是空间常数，在 ∇ 和 Δ 作用下完全湮灭。

### D9. 等效体源（RHS_eq）震源处防爆

- 2D 情形下，通过洛必达法则严格证明：`lim(r→0) RHS_eq = 0`
  - 因为 O(r) 的介质扰动 × O(ln r) 的汉克尔奇点 → r·ln(r) → 0
- 代码中**不能**直接让框架算 `0 * inf`
- 策略：先用安全距离 `r_safe = max(r, eps)` 计算 u₀，再用 loss_mask 将震源处 RHS 硬编码为 `0.0 + 0.0j`

### D10. 网络架构（NSNO）

- 基于 Neumann 级数展开同构映射
- 每层 Block 对应一次多重散射迭代：

  ```python
  h^(k+1) = σ(W_skip·u_inc + K_θ(M_φ(V, h^(k))))
  ```

- `M_φ`：逐点介质-波场局部非线性散射（1×1 conv 或 MLP）
- `K_θ`：数据驱动全局积分核（FNO 频域截断层）
- `W_skip`：源项残差注入（skip connection = 物理强迫项）
- 激活函数：GELU（保持 C∞ 平滑性）
- 输出层初始化为 **零权重零偏置**（Zero-State Ignition：第 0 步输出"无散射"物理初态）

### D11. FNO 层零填充拓扑修复

- FFT 的循环卷积 ≠ 物理线性卷积（环面拓扑 T² vs 开放空间 R²）
- 每次 FFT 前在空间维度后方填充至少 N 个零（总长 ≥ 2N−1）
- FFT 后裁剪回原始物理尺寸
- **前提**：PML 已将边界波场衰减至机器精度以下（见 D6e），消除 Gibbs 振铃

### D12. Loss 函数

```
L_total = λ_pde · L_pde + λ_data · L_data（如有标签）
```

- `L_pde = mean(loss_mask · |R_pde|²)`
  - R_pde = Δ̃Â + iω(2∇̃Â·∇̃τ + Â·Δ̃τ) + R_damping − RHS_eq
- 等式两端统一除以 ω²，防止阻尼项淹没衍射项梯度
- `L_data`（可选）= `mean(|Â_pred − Â_true|²)`，其中 Â_true 需从全波真值反向剥离相位

### D13. 训练策略

- 当前阶段**不启动正式训练**，先完成前向、残差、shape、数值稳定性检查
- 后续启动时采用多波数课程学习：
  1. 低频预热（ω 小）→ 宏观包络
  2. 中频过渡 → 衍射细节
  3. 高频极限 → PML 全阻尼

---

## 4. 仓库结构

```
开题/
├── DEV_PLAN.md              # 本文档（开发规范）
├── CHANGELOG.md             # 变更记录
├── configs/
│   ├── base.yaml            # 默认参数
│   └── debug.yaml           # 调试用小网格参数
├── docs/
│   ├── README.md            # 文档索引
│   ├── theory/
│   │   ├── document.tex     # 原始开发思路
│   │   └── main.tex         # 完整理论资料
│   └── archive/             # 历史协作文档归档
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── grid.py          # Grid2D: 规则网格、坐标系、距离场
│   │   ├── medium.py        # Medium2D: 介质慢度场、PML内退化
│   │   ├── source.py        # PointSource: 源坐标、索引、邻域
│   │   └── complex_utils.py # 实虚拆分、合并、复数乘法辅助
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── background.py    # u₀: 2D汉克尔格林函数 + 安全距离
│   │   ├── eikonal.py       # τ₀, α, FSM求解器, 解析烫印
│   │   ├── tau_ops.py       # ∇τ, Δτ 的链式法则安全重组
│   │   ├── pml.py           # σ, γ, γ', A, B 张量预计算
│   │   ├── diff_ops.py      # 固定卷积核微分引擎
│   │   ├── rhs.py           # RHS_eq 等效体源组装 + 震源防爆
│   │   └── residual.py      # PDE残差计算器（含阻尼池解耦）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spectral_conv.py # SpectralConv2d (FNO频域截断层)
│   │   └── nsno.py          # NSNO2D 完整网络
│   └── train/
│       ├── __init__.py
│       ├── losses.py        # L_pde, L_data, L_total
│       └── trainer.py       # 训练循环（含课程学习调度）
├── tests/
│   ├── test_grid.py
│   ├── test_medium.py
│   ├── test_background.py
│   ├── test_eikonal.py
│   ├── test_tau_ops.py
│   ├── test_pml.py
│   ├── test_diff_ops.py
│   ├── test_rhs.py
│   ├── test_residual.py
│   └── test_model_forward.py
├── notebooks/               # 可视化与调试notebook
├── scripts/                 # 独立运行脚本
├── requirements.txt
└── README.md
```

---

## 5. 里程碑推进表（Milestones）

---

### M0. 建仓与最小骨架

**目标**：完成仓库初始化、目录结构、配置系统、基础依赖。

**输出**：
- 可运行的 Python 环境
- 目录结构固定
- `configs/base.yaml` 定义所有默认参数
- `configs/debug.yaml`（小网格快速调试）
- `requirements.txt`

**检查项**：
- [ ] 能安装依赖 (`pip install -r requirements.txt`)
- [ ] `pytest` 能跑（至少一个空测试通过）
- [ ] 配置能读 (`yaml.safe_load`)
- [ ] 日志能输出 (`logging` 配置)
- [ ] `import src` 不报错

**通过标准**：新机器克隆后 10 分钟内可启动，无手工改路径。

**base.yaml 最小参数集**：

```yaml
# 物理参数
omega: 30.0                # 圆频率 (先用低频调试)
source_pos: [0.5, 0.5]    # 震源物理坐标 (归一化域中心)

# 网格参数
domain: [0.0, 1.0, 0.0, 1.0]  # [x_min, x_max, y_min, y_max]
nx: 128                    # x方向格点数（不含PML）
ny: 128                    # y方向格点数（不含PML）

# 介质参数
c_background: 1.0          # 背景波速
# velocity_model: "homogeneous" | "smooth_lens" | "layered"

# PML参数
pml_width: 20              # PML层格点数
pml_power: 2               # 阻尼剖面幂次
pml_R0: 1.0e-6             # 目标反射系数

# Eikonal参数
fsm_max_iter: 100          # FSM最大扫描轮次
fsm_tol: 1.0e-10           # FSM收敛阈值
source_freeze_radius: 1    # 震源冻结圈数（网格数）
eikonal_precision: "float64"

# 网络参数 (M7之后启用)
nsno_blocks: 4             # NSNO Block数（多重散射迭代数）
nsno_channels: 32          # 隐层通道数
fno_modes: 16              # FNO保留的傅里叶模式数
activation: "gelu"

# Loss参数
source_mask_radius: 1.5    # 震源穿孔掩码半径 (单位: h)
pml_rhs_zero: true         # PML内RHS强制为0
omega2_normalize: true     # 残差两端除以ω²

# 训练参数 (M9之后启用)
lr: 1.0e-3
epochs: 1000
batch_size: 1              # 当前单样本
lambda_pde: 1.0
lambda_data: 0.0           # 无监督标签时为0
```

**Git**：
- 分支：`feat/m0-bootstrap`
- 合并条件：目录、环境、README、基础测试齐全

---

### M1. 网格、介质、源项

**目标**：实现规则网格、介质场、点源数据结构和距离场。

**输出**：
- `Grid2D`：坐标系统、网格步长、含 PML 的总网格尺寸
- `Medium2D`：慢度场 s(x)、PML 内退化逻辑
- `PointSource`：物理坐标、网格索引、到源距离场 r(x)

**关键实现细节**：

```python
class Grid2D:
    # 坐标约定: grid[i, j] 对应物理坐标 (x_min + j*h, y_min + i*h)
    # 即 i 是行索引对应 y, j 是列索引对应 x
    # 张量 shape: [H, W] 其中 H=ny_total, W=nx_total
    # nx_total = nx + 2*pml_width, ny_total = ny + 2*pml_width
```

```python
class Medium2D:
    # s(x) = 1/c(x) 慢度场
    # PML内强制: s_pml = s0 (背景慢度)
    # s0 = s(x_source) 震源处慢度
```

```python
class PointSource:
    # 物理坐标 -> 网格索引（考虑PML偏移）
    # distance_field: r(x) = |x - x_s|
    # safe_distance: r_safe = max(r, eps) 防止除零
    # eps 取值: 0.1 * h (远小于网格但远大于浮点零)
```

**检查项**：
- [ ] 网格 shape 正确（含 PML 扩展）
- [ ] 坐标系定义统一（i↔y, j↔x 不混淆）
- [ ] 源点索引和物理坐标双向转换一致
- [ ] 距离场在源点处为 eps（不为 0）
- [ ] PML 内介质退化为 s₀
- [ ] 3 个手写 case 对上（均匀介质、线性梯度、含PML边界）
- [ ] 可视化正常（`matplotlib` 画网格、介质、距离场）

**通过标准**：可视化检查 + 单元测试全通过。

**Git**：`feat/m1-grid-source`

---

### M2. 背景场与解析先验

**目标**：实现 2D 背景场 u₀、背景走时 τ₀ 及其解析导数。

**输出**：
- `s₀ = s(xₛ)` 背景慢度
- `τ₀ = s₀·r` 背景走时
- `∇τ₀ = s₀·r̂`（解析梯度向量）
- `Δτ₀ = s₀/r`（解析拉普拉斯）— 震源处用 safe 值
- `u₀ = (i/4)·H₀⁽¹⁾(ω·s₀·r)` 2D 汉克尔格林函数

**关键实现细节**：

```python
# 背景场（含安全距离处理）
r_safe = torch.clamp(r, min=eps)
k0 = omega * s0
u0 = (1j / 4) * scipy.special.hankel1(0, k0 * r_safe)  # CPU预处理
# 震源格点: u0[src_i, src_j] 使用 r=eps 的值（后续被mask清零）

# 解析导数
grad_tau0_x = s0 * (x - xs) / r_safe   # 震源处 → 0/eps 有限值
grad_tau0_y = s0 * (y - ys) / r_safe
lap_tau0 = s0 / r_safe                  # 震源处 → s0/eps 大值（被mask处理）
```

**震源安全处理策略**：
- `τ₀(xₛ) = 0`（精确值）
- `∇τ₀(xₛ)` 方向多值 → 用 `r_safe` 近似，最终被 `τ₀=0` 乘法抑制
- `Δτ₀(xₛ) = s₀/eps` → 极大值，但在穿孔掩码中不参与 Loss

**检查项**：
- [ ] 远离震源时 τ₀ 趋势合理（线性增长）
- [ ] ∇τ₀ 模长 ≡ s₀（常数，允许浮点误差 < 1e-12）
- [ ] u₀ 远场渐近行为：`|u₀| ∝ 1/√r`（2D柱面扩散）
- [ ] 震源点无 NaN/Inf 泄漏
- [ ] u₀ 可视化：实部呈同心环振荡

**通过标准**：可视化检查 + 与 scipy 参考值对齐 + 无 NaN。

**Git**：`feat/m2-background`

---

### M3. 因子化 Eikonal 求解器

**目标**：实现 FSM 求解因子化程函方程，得到平滑因子 α。

**输出**：
- `alpha` 场：平滑修正因子
- `tau = tau0 * alpha` 总走时
- FSM 求解器（含解析烫印、Godunov 迎风、因果律检验）

**关键实现细节**：

```python
def fsm_solve(grid, medium, source, config):
    """
    1. 初始化: alpha 全部设为 INF
    2. 震源解析烫印: alpha(xs)=1, 邻居用一阶泰勒
    3. 四方向交替扫描 (sx,sy) ∈ {(-1,-1),(-1,+1),(+1,-1),(+1,+1)}
    4. 每个非冻结点: Godunov 更新
    5. 收敛判定: max|alpha_new - alpha_old| < tol
    """
    # 全程 float64!
    alpha = np.full((ny, nx), np.inf, dtype=np.float64)
    
    # 解析烫印
    alpha[src_i, src_j] = 1.0
    grad_alpha_src = grad_s_at_src / (2 * s0)
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        ni, nj = src_i + di, src_j + dj
        dx_phys = dj * h
        dy_phys = di * h
        alpha[ni, nj] = 1.0 + grad_alpha_src[0]*dx_phys + grad_alpha_src[1]*dy_phys
        frozen[ni, nj] = True
```

**检查项**：
- [ ] 震源初始化正确（α=1，邻居解析）
- [ ] 扫描顺序完整（4 方向）
- [ ] 更新单调（α 只能减小或保持不变）
- [ ] 因果律检验不遗漏
- [ ] 无明显假波纹
- [ ] 收敛轮次合理（通常 < 20 轮）

**通过标准**：
- [ ] 常速介质中 `max|α − 1| < 1e-8`
- [ ] 平滑变速介质中 τ 连续、无突变
- [ ] 无 NaN
- [ ] 可视化 α 场平滑
- [ ] 与参考 FMM（如 scikit-fmm）结果对比误差 < 1e-4

**Git**：`feat/m3-eikonal`

---

### M4. 安全导数重组

**目标**：构造 ∇τ 和 Δτ，禁止危险直接差分。

**输出**：
- `grad_alpha`（数值差分）
- `lap_alpha`（数值差分）
- `grad_tau = alpha * grad_tau0 + tau0 * grad_alpha`（安全重组）
- `lap_tau = alpha * lap_tau0 + 2 * dot(grad_tau0, grad_alpha) + tau0 * lap_alpha`（安全重组）

**关键实现细节**：

```python
# 对极其平滑的 alpha 做中心差分（安全无噪声）
grad_alpha_x = diff_x(alpha)   # 使用 [-1, 0, 1]/(2h)
grad_alpha_y = diff_y(alpha)
lap_alpha = diff_xx(alpha) + diff_yy(alpha)  # 使用 [1, -2, 1]/h²

# 安全重组
grad_tau_x = alpha * grad_tau0_x + tau0 * grad_alpha_x
grad_tau_y = alpha * grad_tau0_y + tau0 * grad_alpha_y
lap_tau = alpha * lap_tau0 + 2*(grad_tau0_x*grad_alpha_x + grad_tau0_y*grad_alpha_y) + tau0 * lap_alpha
```

**反模式（绝对禁止）**：
```python
# 错! 直接对 tau 做二阶差分 → 震源附近星芒噪声
lap_tau = diff_xx(tau) + diff_yy(tau)  # FORBIDDEN!
```

**检查项**：
- [ ] 不直接对 τ 做二阶差分（代码审查）
- [ ] 链式法则重组正确（手写 3 个点验证）
- [ ] 震源附近 Δτ 数值可控（不出现异常尖峰）
- [ ] 常速介质中 `|∇τ|² ≈ s₀²`（验证程函一致性）

**通过标准**：
- [ ] 可视化 Δτ 无星芒噪声
- [ ] 无异常尖峰扩散
- [ ] 常速介质程函残差 < 1e-6

**Git**：`feat/m4-tau-derivatives`

---

### M5. PML 与差分引擎

**目标**：实现 PML 复数拉伸系数和固定卷积核微分算子。

**输出**：
- `sigma_x, sigma_y`：阻尼剖面
- `gamma_x, gamma_y`：复数拉伸因子
- `gamma_prime_x, gamma_prime_y`：**解析**导数
- `A_x, A_y, B_x, B_y`：PML 乘子张量
- `diff_ops.py`：`diff_x`, `diff_y`, `diff_xx`, `diff_yy`（固定卷积核）
- PML 拉普拉斯算子 `pml_laplacian(U, A_x, B_x, A_y, B_y)`

**关键实现细节**：

```python
# diff_ops.py — 固定卷积核（不可训练）
class DiffOps:
    def __init__(self, h):
        # 二阶: [1, -2, 1] / h²
        self.kernel_xx = torch.tensor([[[1, -2, 1]]], dtype=torch.float32) / h**2
        self.kernel_yy = torch.tensor([[[[1], [-2], [1]]]], dtype=torch.float32) / h**2
        # 一阶: [-1, 0, 1] / (2h)
        self.kernel_x = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32) / (2*h)
        self.kernel_y = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32) / (2*h)
        # 所有 kernel 注册为 buffer（不参与训练）

    def diff_x(self, u):
        return F.conv2d(u, self.kernel_x, padding=(0,1))  # reflect pad
    
    # ... 类似
```

```python
# pml.py — PML张量预计算
def compute_pml_tensors(grid, config, omega):
    """
    sigma_x[j]: 仅在 j < pml_width 或 j >= nx + pml_width 时非零
    gamma_x = 1 + 1j * sigma_x / omega
    gamma_prime_x = 1j * sigma_prime_x / omega   # 解析!
    A_x = 1 / gamma_x**2
    B_x = gamma_prime_x / gamma_x**3
    """
```

**检查项**：
- [ ] 差分核不使用 autograd
- [ ] γ' 用解析式（不做数值差分）
- [ ] PML 层只在边界起作用，内部 γ≡1
- [ ] 内部区域 A=1, B=0，退化为标准中心差分
- [ ] 卷积核 shape 和 padding 正确（不错位）

**通过标准**：
- [ ] 平面波 `e^{ikx}` 进入 PML 后衰减（可视化）
- [ ] 衰减后振幅低于机器精度
- [ ] 无明显边界反射

**Git**：`feat/m5-pml-diffops`

---

### M6. 等效体源与穿孔掩码

**目标**：实现 RHS_eq 组装与 Loss 掩码。

**输出**：
- `RHS_eq`：等效体源（含 2D π/4 相位补偿）
- `loss_mask`：穿孔域掩码
- 震源邻域防爆规则

**关键实现细节**：

```python
def compute_rhs(grid, medium, source, background, tau, omega):
    """
    perturbation = s**2 - s0**2
    phase_strip = exp(-1j * (omega * tau + pi/4))   # 2D 相位补偿!
    rhs = -omega**2 * perturbation * u0 * phase_strip
    
    # 震源防爆: 硬编码清零
    rhs[source_mask] = 0.0 + 0.0j
    
    # PML内: perturbation ≡ 0 → rhs ≡ 0（已由介质退化保证）
    """

def compute_loss_mask(grid, source, config):
    """
    mask = (distance_field > mask_radius * h).float()
    """
```

**检查项**：
- [ ] 震源点不出现 `0 * inf`（数值稳定）
- [ ] PML 内 RHS 为 0
- [ ] mask 半径写进配置文件
- [ ] π/4 相位补偿正确应用
- [ ] RHS 在全空间有界

**通过标准**：
- [ ] 数值稳定（无 NaN/Inf）
- [ ] RHS 可视化连续、震源处归零
- [ ] 常速介质中 RHS ≡ 0（因 s²−s₀²=0）

**Git**：`feat/m6-rhs-mask`

---

### M7. 网络骨架

**目标**：只实现前向网络结构，不训练。

**输出**：
- `SpectralConv2d`：FNO 频域截断卷积（含零填充拓扑修复）
- `NeumannSpectralBlock`：单层散射迭代 Block
- `NSNO2D`：完整网络（输入编码 → N个Block → 输出解码）

**网络输入通道设计**：

| 通道 | 内容 | 维度 |
|------|------|------|
| 0-1 | s(x) 实虚部（慢度是实数，虚部=0） | 2 |
| 2-3 | u₀ 实虚部 | 2 |
| 4-5 | ∇τ_x, ∇τ_y | 2 |
| 6 | Δτ | 1 |
| 7 | ω（标量广播） | 1 |
| **总计** | | **8** |

**网络输出**：`[B, 2, H, W]` — Â_scat 的实部和虚部

**关键实现细节**：

```python
class SpectralConv2d(nn.Module):
    """FNO核心层: 频域权重乘法 + 高频截断"""
    def forward(self, x):
        # 1. 零填充 (拓扑修复)
        x_padded = F.pad(x, [0, self.nx, 0, self.ny])
        # 2. FFT
        x_ft = torch.fft.rfft2(x_padded)
        # 3. 截断 + 频域权重乘
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", 
                         x_ft[:, :, :self.modes1, :self.modes2],
                         self.weights)
        # 4. IFFT
        x_out = torch.fft.irfft2(out_ft, s=x_padded.shape[-2:])
        # 5. 裁剪回原尺寸
        return x_out[:, :, :self.orig_h, :self.orig_w]

class NSNO2D(nn.Module):
    def __init__(self, config):
        # 输入编码: 8通道 → hidden_channels
        self.encoder = nn.Conv2d(8, config.nsno_channels, 1)
        # N个散射Block
        self.blocks = nn.ModuleList([
            NeumannSpectralBlock(config) for _ in range(config.nsno_blocks)
        ])
        # 输出解码: hidden_channels → 2通道 (实虚)
        self.decoder = nn.Conv2d(config.nsno_channels, 2, 1)
        # 零初始化输出层
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
```

**检查项**：
- [ ] 通道维度统一（8 → hidden → 2）
- [ ] 实虚双通道一致
- [ ] 前向不爆显存（debug 网格上测试）
- [ ] 零填充逻辑正确
- [ ] 输出层零初始化

**通过标准**：
- [ ] 前向可跑通（无报错）
- [ ] 输出 shape 正确 `[B, 2, H, W]`
- [ ] 零初始化时输出全零（无散射初态）
- [ ] 显存占用合理

**Git**：`feat/m7-model`

---

### M8. PDE Residual 前向检查

**目标**：把预处理、网络、残差拼起来，只做 forward residual 验证。

**输出**：
- `ResidualComputer`：完整残差计算流水线
- `loss_pde` 值
- debug 可视化图

**残差计算流水线**：

```
网络输出 Â_scat
    ↓
[diff_ops] → ∇̃Â (PML一阶导) → Δ̃Â (PML二阶导)
    ↓
组装 LHS = Δ̃Â + iω(2∇̃Â·∇̃τ + Â·Δ̃τ) + R_damping
    ↓
R_pde = (LHS − RHS_eq) / ω²
    ↓
L_pde = mean(loss_mask · |R_pde|²)
```

**检查项**：
- [ ] 物理域内残差有限（无 NaN/Inf）
- [ ] PML 行为合理（残差在边界衰减）
- [ ] 震源附近 mask 生效（掩码区残差为 0）
- [ ] 阻尼池解耦正确（物理区内该项 ≡ 0）
- [ ] ω² 归一化生效

**通过标准**：
- [ ] 全图无 NaN/Inf
- [ ] 小样例 forward 成功
- [ ] 常速介质 + 零初始化网络 → 残差 = −RHS_eq/ω²（纯源项）
- [ ] 可视化残差分布合理

**Git**：`feat/m8-residual`

---

### M9. 最小训练实验

**目标**：单样本 overfit，验证训练管线可行。

**输出**：
- 1 个常速 / 1 个平滑变速介质的 overfit 实验
- loss 曲线
- 包络场 Â_scat 重构图
- 全波场 u_total 重构图

**关键实现细节**：

```python
# 全波场重构（含2D相位补偿）
phase_restorer = torch.exp(1j * (omega * tau + math.pi / 4))
u_total = u0 + (A_scat_real + 1j * A_scat_imag) * phase_restorer
```

**检查项**：
- [ ] loss 能下降
- [ ] 不出现数值爆炸
- [ ] 输出图像合理（包络平滑、全波场有干涉条纹）
- [ ] 常速介质：Â_scat ≈ 0（无散射）
- [ ] 变速介质：Â_scat 非零且平滑

**通过标准**：
- [ ] 小样本可拟合（loss 下降 2 个数量级以上）
- [ ] 训练稳定（无 NaN 爆炸）
- [ ] 全波场可视化物理合理

**Git**：`feat/m9-min-train`

---

## 6. 验收模板

每个里程碑合并前必须满足：

- [ ] 功能完成
- [ ] 单元测试通过
- [ ] 可视化检查通过
- [ ] 无 NaN/Inf
- [ ] DEV_PLAN.md 已更新（勾选对应检查项）
- [ ] 配置文件已更新（如有新参数）
- [ ] 至少 1 个 debug case 已保存（可复现）
- [ ] commit message 清晰
- [ ] 可以被他人在新环境复现

---

## 7. Git 工作流

### 分支命名

| 前缀 | 用途 | 示例 |
|------|------|------|
| `main` | 稳定主线 | — |
| `dev` | 开发集成 | — |
| `feat/` | 里程碑功能 | `feat/m3-eikonal` |
| `fix/` | Bug 修复 | `fix/pml-gamma-sign` |
| `exp/` | 实验性探索 | `exp/debug-source-mask` |

### 提交规范

```
init: bootstrap repository
feat: add 2d grid and point source
feat: implement background field u0
feat: add factored eikonal FSM solver
fix: stabilize source mask near singularity
refactor: split residual operator and diff ops
docs: update DEV_PLAN for M4
test: add eikonal convergence test
```

### 合并规则

1. 不直接往 `main` 推
2. 每个 milestone 单独分支
3. milestone 通过验收后 merge 到 `dev`
4. `dev` 稳定后 merge 到 `main` 并打 tag
5. merge 前必须补文档
6. 未通过测试不得合并

### Tag 规范

```
v0.1-m0-bootstrap
v0.1-m1-grid
v0.1-m2-background
v0.1-m3-eikonal
...
```

---

## 8. 回退规则

1. 如果某里程碑连续两次修改后仍不稳定，**停止继续叠加功能**
2. 回到上一个稳定 tag
3. 在 `CHANGELOG.md` 记录失败原因
4. 若问题来自理论未冻结，先改 `DEV_PLAN.md`，再开新分支重做
5. 永远不要在不稳定的基础上继续开发

---

## 9. 当前未决问题（Open Questions）

> 以下问题在实现过程中需要通过实验确定，确定后更新本文档并冻结。

| 编号 | 问题 | 当前倾向 | 需在哪个 M 前确定 |
|------|------|---------|-------------------|
| Q1 | α 求解器最终选 FSM 还是 FMM | FSM（实现简单，2D够用） | M3 |
| Q2 | source mask 半径取 1.0h / 1.5h / 2.0h | 1.5h（需实验） | M6 |
| Q3 | PML 厚度默认值 | 20 格点（需测衰减充分性） | M5 |
| Q4 | σ_max 取值 | 按反射系数公式计算 | M5 |
| Q5 | FNO 保留模式数 | 16（需实验平衡精度/速度） | M7 |
| Q6 | NSNO Block 数 | 4（对应4阶多重散射） | M7 |
| Q7 | 是否需要焦散面 Δτ 高斯软化 | 2D简单介质暂不需要 | M4 |
| Q8 | 复数激活函数选型 | 先用 split GELU（实虚分开激活） | M7 |
| Q9 | 课程学习频率调度策略 | 线性递增 ω | M9 |
| Q10 | 输入特征是否需要 Z-score 归一化 | 网络输入可以，PDE Loss 系数禁止 | M8 |

---

## 10. 核心公式速查表

> 所有代码实现必须与此表一一对应，不得自行推导变体。

### 10.1 基本定义

| 符号 | 定义 | 代码对应 |
|------|------|---------|
| s(x) = 1/c(x) | 慢度场 | `medium.slowness` |
| s₀ = s(xₛ) | 背景慢度 | `medium.s0` |
| k₀ = ω·s₀ | 背景波数 | `omega * s0` |
| r = \|x−xₛ\| | 到源距离 | `source.distance_field` |
| τ₀ = s₀·r | 背景走时 | `tau0` |
| α | 平滑修正因子 | `alpha` |
| τ = τ₀·α | 总走时 | `tau` |

### 10.2 核心方程

**因子化程函方程**（FSM 求解目标）：
```
s₀²·α² + τ₀²·|∇α|² + 2·α·τ₀·(∇τ₀·∇α) = s²(x)
```

**安全走时重组**：
```
∇τ = α·∇τ₀ + τ₀·∇α
Δτ = α·Δτ₀ + 2·(∇τ₀·∇α) + τ₀·Δα
```

**2D 背景场**：
```
u₀ = (i/4)·H₀⁽¹⁾(ω·s₀·r)
```

**等效体源**（2D 含相位补偿）：
```
RHS_eq = −ω²·(s²−s₀²)·u₀·exp[−i(ωτ + π/4)]
```

**PML 拉普拉斯**：
```
Δ̃U = Σ_{ξ∈{x,y}} [ (1/γξ²)·∂²U/∂ξ² − (γξ'/γξ³)·∂U/∂ξ ]
```

**阻尼池安全形式**：
```
R_damping = ω²·Σ_{ξ} [(∂ξτ)²·(1 − 1/γξ²)]·Â
```

**PDE 残差**（ω² 归一化后）：
```
R = [Δ̃Â + iω(2∇̃Â·∇̃τ + Â·Δ̃τ) + R_damping − RHS_eq] / ω²
```

**全波场重构**（2D）：
```
u_total = u₀ + Â_scat · exp[i(ωτ + π/4)]
```

---

## 11. 补充：原文档未覆盖的关键实现细节

> 以下是基于理论资料推导但原文档未明确提及的实现要点。

### 11.1 FSM 震源处的极限自洽验证

当网格点落在震源中心时，τ₀=0 导致 Godunov 方程自动退化为纯代数式：
```
α² · (pₓ² + pᵧ²) = s² → α² · s₀² = s₀² → α = 1
```
这说明 FSM 在震源处天然自洽，但**邻居精度**依赖解析烫印。

### 11.2 Hankel 函数计算的数值稳定性

`scipy.special.hankel1(0, z)` 在 `z→0` 时有对数奇点。对于 `z < 1e-15`，应使用小参数渐近展开：
```
H₀⁽¹⁾(z) ≈ 1 + (2i/π)·[ln(z/2) + γ_euler]
```
其中 γ_euler ≈ 0.5772。但更安全的做法是使用 `r_safe = max(r, eps)` 避免触发。

### 11.3 PML 四角重叠区

2D PML 在网格四角同时有 x 和 y 方向的阻尼叠加：
```
γₓ ≠ 1 且 γᵧ ≠ 1
```
此处 PML 拉普拉斯的两个方向项独立相加，无需特殊处理。

### 11.4 边界 padding 策略

| 区域 | padding 方式 |
|------|-------------|
| 物理域内部 | reflect padding |
| PML 外侧边界 | zero padding |

差分卷积核在 PML 最外一层需要 padding。由于 PML 已将波衰减至零，零填充在此是物理正确的。

### 11.5 梯度冲突防护

当同时使用 `L_pde` 和 `L_data` 时，两者梯度可能方向冲突。建议：
- 初期纯 PDE Loss（无标签）
- 有标签时逐步退火 λ_pde，避免梯度互搏
- 监控两个 Loss 分量的独立走势

### 11.6 数值精度隔离

| 模块 | 精度 | 原因 |
|------|------|------|
| FSM 求解器 | Float64 | 走时相位累积误差敏感 |
| 解析预处理（u₀, τ₀, ∇τ₀） | Float64 | 汉克尔函数、除法精度 |
| PML 张量 | Complex64 (= 2×Float32) | GPU 友好 |
| 网络输入特征 | Float32 | GPU 训练 |
| PDE Loss 计算 | Float32 | 网络梯度回传 |
| Loss 中的物理系数（ω, s, γ） | Float32 | 与网络精度匹配 |

预处理完成后统一 `tensor.float()` 截断送入 GPU。

### 11.7 背向反射波的处理策略

单向 WKBJ 拟设下，背向反射波在包络中表现为 `exp(−2iωτ)` 的高频寄生振荡（波数倍增悖论）。在当前 2D 原型中：

1. **FNO 频域截断**天然切除 `2ω` 以上频率（架构级防护）
2. **黎曼-勒贝格引理**保证高频残差在全局积分 Loss 中自平均趋零
3. 网络学习的是"宏观有效散射截面"而非微观高频相位
4. 这是**设计特性**而非缺陷——粗网格上本就无法分辨 `2ω` 结构

### 11.8 阴影区（Shadow Zone）处理

当介质存在强速度异常（如低速透镜）时：

1. FMM/FSM 返回纯实数走时（丢失隐波虚部 τ_I）
2. 控制方程中产生 `−ω²|∇τ_I|²` 的巨大误差
3. **但** `ΔÂ` 的指数崩塌项产生等量反向代偿 `+ω²|∇τ_I|²`
4. 完美对冲——这依赖于完整保留拉普拉斯项 ΔÂ
5. 方程从双曲型（波动）自动退化为椭圆型（扩散），网络像求解热方程一样拟合隐波

**工程要求**：绝对不能在包络方程中丢弃 ΔÂ 项。

---

## 12. 防爆清单（Anti-NaN Checklist）

> 每次提交前过一遍此清单。

- [ ] FSM 震源冻结区已解析烫印
- [ ] 距离场 r 使用 `r_safe = max(r, eps)`
- [ ] u₀ 在震源处无 Inf（使用 r_safe 计算）
- [ ] RHS_eq 在震源处硬编码为 0
- [ ] Loss 掩码覆盖震源 1.5h 邻域
- [ ] Δτ 使用链式法则重组（不直接差分 τ）
- [ ] γ' 使用解析公式（不数值差分 γ）
- [ ] 阻尼池使用安全解耦形式（物理区内 ≡ 0）
- [ ] FNO 层有零填充（防止 FFT 环形穿越）
- [ ] PML 衰减充分（边界波场 < 1e-7）
- [ ] 网络输出层零初始化
- [ ] 残差两端除以 ω²
- [ ] Float64 预处理 → Float32 送入网络
