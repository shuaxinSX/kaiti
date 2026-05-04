# 高波数 2D Helmholtz 神经算子实验矩阵设计

> 目标：把当前 `2D + 单源 + WKBJ envelope + NSNO + strict PML residual` 原型扩展为面向高波数 Helmholtz 方程的系统性 benchmark campaign，并与 2022-2026 年高波数神经算子文献中的核心叙事衔接。

## 1. 总体研究问题

本实验矩阵不只是调参 sweep，而是服务于以下研究判断：

1. 当前方法能否在高波数 regime 下稳定求解 2D Helmholtz 方程。
2. 误差瓶颈来自数值离散、Eikonal 走时、PML、模型容量、优化，还是监督标签。
3. WKBJ 相位剥离是否显著缓解 FNO/神经算子的谱偏置。
4. `stretched_divergence` 严格 PML residual 是否优于 `mixed_legacy`。
5. NSNO 的主要容量瓶颈是否来自 Neumann 散射阶数深度，而非简单 Fourier mode 数。
6. 高频下 `rel_l2≈1e-3` 是否可由 `omega*h^2` 或 `omega*h` 相位误差预算解释。
7. curriculum / warm-start / denser sampling 是否能把模型推向更高可解波数。
8. 当前方法在文献 benchmark 坐标系中应如何定位：相位剥离、Neumann unrolling、PML 一致性、hybrid supervision、curriculum training。

## 2. 当前工程基线

当前代码已支持：

- 2D 规则网格与 PML 扩展。
- `homogeneous` / `smooth_lens` / `layered` 三类介质。
- 单点源 `physics.source_pos`。
- WKBJ / Eikonal 预处理：`tau = tau0 * alpha`。
- `ResidualComputer.lap_tau_mode`：
  - `stretched_divergence`：当前默认，严格 PML 复拉伸。
  - `mixed_legacy`：历史审计模式。
- supervision 只允许离线相位剥离后的 `reference_envelope.npy`：
  - `training.supervision.target_kind = scattering_envelope`
- 评估指标已包含：
  - residual metrics
  - reference comparison
  - phase/amplitude error
  - phase reconstruction budgets
  - Neumann capacity diagnostics

当前仍不应作为有效矩阵轴的配置：

- `training.seed`：当前模型初始化固定，seed sweep 没有意义。
- `eikonal.precision`：配置存在，但 runtime 未消费。
- `loss.omega2_normalize`：配置存在，但 residual path 未消费。
- `loss.pml_rhs_zero`：配置存在，但 residual path 未消费。

## 2.1 相关配置总表

所有实验均以 `configs/base.yaml` 为底座，通过 `configs/experiments/*.yaml` 的 `fixed` 与单个 run 的 `overrides` 覆盖。矩阵设计时只应把当前 runtime 已消费的字段作为有效实验轴。

### 2.1.1 物理与网格配置

| YAML key | 默认值 | 当前是否可作为矩阵轴 | 作用 |
|---|---:|---|---|
| `physics.omega` | `30.0` | 是 | 圆频率，高波数实验的主轴。 |
| `physics.source_pos` | `[0.5, 0.5]` | 是 | 点源位置，用于源位置鲁棒性与近边界退化测试。 |
| `grid.domain` | `[0.0, 1.0, 0.0, 1.0]` | 暂不建议 | 当前理论量纲和 ppw 估算都以单位正方形为默认。 |
| `grid.nx` | `128` | 是 | 物理域 x 方向格点数，不含 PML。 |
| `grid.ny` | `128` | 是 | 物理域 y 方向格点数，不含 PML；目前建议与 `nx` 同步扫描。 |
| `medium.c_background` | `1.0` | 暂不建议 | 背景波速；改动会影响所有 `omega*s0*h` 预算，建议后期单独做物理量纲测试。 |
| `medium.velocity_model` | `homogeneous` | 是 | 介质族：`homogeneous` / `smooth_lens` / `layered`。 |

高波数实验中 `grid.nx = grid.ny = N` 不应随便取值，而应从 `ppw` 反推：

```text
h = 1 / N
ppw = 2*pi / (omega * s0 * h)
N ~= ppw * omega * s0 / (2*pi)
```

在当前 `s0≈1` 下，推荐按以下规则选网格：

| 实验目的 | 推荐 ppw | 适用 omega |
|---|---:|---|
| 快速扫描 / 失败边界 | 8 | 60-180 |
| 主结果 baseline | 12 | 60-240 |
| 可信高波数结果 | 16 | 90-300 |
| reference-only 或 finalist | 24 | 120-300 |

### 2.1.2 PML 与 residual 配置

| YAML key | 默认值 | 当前是否可作为矩阵轴 | 作用 |
|---|---:|---|---|
| `pml.width` | `20` | 是 | PML 厚度，单位为格点数。 |
| `pml.power` | `2` | 是 | PML 阻尼剖面幂次。 |
| `pml.R0` | `1.0e-6` | 是 | PML 目标反射系数，用于决定最大阻尼。 |
| `residual.lap_tau_mode` | `stretched_divergence` | 是 | `tau` 在 PML 内的拉普拉斯装配策略。 |

`residual.lap_tau_mode` 只能取：

```yaml
residual:
  lap_tau_mode: stretched_divergence  # 主结果默认，严格复拉伸
```

或：

```yaml
residual:
  lap_tau_mode: mixed_legacy          # 仅用于消融审计，不作为最终方法
```

高频主结果建议固定为：

```yaml
pml:
  width: 24        # N<=192 可先用 16；N>=256 建议 24 或 32
  power: 2
  R0: 1.0e-6
residual:
  lap_tau_mode: stretched_divergence
```

PML 消融时再扫描：

```yaml
pml:
  width: [8, 16, 24, 32, 48]
  power: [1, 2, 3]
  R0: [1.0e-4, 1.0e-6, 1.0e-8]
```

### 2.1.3 Eikonal 配置

| YAML key | 默认值 | 当前是否可作为矩阵轴 | 作用 |
|---|---:|---|---|
| `eikonal.fsm_max_iter` | `100` | 是，但不建议大扫 | FSM 最大扫描轮次。 |
| `eikonal.fsm_tol` | `1.0e-10` | 是，但不建议大扫 | FSM 收敛阈值。 |
| `eikonal.source_freeze_radius` | `1` | 谨慎 | 震源冻结半径，影响近场和穿孔区。 |
| `eikonal.precision` | `float64` | 否 | 配置存在，但当前 runtime 未实际消费。 |

高波数主实验建议固定：

```yaml
eikonal:
  fsm_max_iter: 100
  fsm_tol: 1.0e-10
  source_freeze_radius: 1
  precision: float64  # 文档保留；当前不作为有效轴解释结果
```

不要把 `eikonal.precision` 的结果写成论文结论，除非后续代码把它真正接入 FSM。

### 2.1.4 模型容量配置

| YAML key | 默认值 | 当前是否可作为矩阵轴 | 作用 |
|---|---:|---|---|
| `model.nsno_blocks` | `4` | 是 | NSNO 深度，对应 Neumann/Born 多重散射阶数。 |
| `model.nsno_channels` | `32` | 是 | 隐层通道数，控制 pointwise mixing 容量。 |
| `model.fno_modes` | `16` | 是 | SpectralConv 保留的 Fourier modes。 |
| `model.activation` | `gelu` | 谨慎 | 激活函数，当前不作为主论证轴。 |

建议把模型容量拆成三个独立实验，不要一次性混扫：

```yaml
# Depth sweep: 验证 Neumann 散射阶数瓶颈
model:
  nsno_blocks: [2, 4, 6, 8, 12, 16]
  nsno_channels: 64
  fno_modes: 16
  activation: gelu
```

```yaml
# Mode sweep: 验证相位剥离后 M 是否已经解绑高频 Nyquist
model:
  nsno_blocks: 8
  nsno_channels: 64
  fno_modes: [4, 8, 12, 16, 24, 32]
  activation: gelu
```

```yaml
# Channel sweep: 验证点乘/非线性混合容量是否不足
model:
  nsno_blocks: 8
  nsno_channels: [16, 32, 64, 96, 128]
  fno_modes: 16
  activation: gelu
```

### 2.1.5 Loss 与监督配置

| YAML key | 默认值 | 当前是否可作为矩阵轴 | 作用 |
|---|---:|---|---|
| `loss.source_mask_radius` | `1.5` | 是 | 震源穿孔半径，单位为 `h`。 |
| `loss.pml_rhs_zero` | `true` | 否 | 配置存在，但当前 residual path 未消费。 |
| `loss.omega2_normalize` | `true` | 否 | 配置存在，但当前 residual path 未消费。 |
| `training.lambda_pde` | `1.0` | 是 | PDE residual loss 权重。 |
| `training.lambda_data` | `0.0` | 是 | envelope data loss 权重。 |
| `training.supervision.enabled` | `false` | 是 | 是否启用离线参考标签监督。 |
| `training.supervision.reference_path` | `null` | 是 | `reference_envelope.npy` 路径；矩阵脚本可自动生成。 |
| `training.supervision.target_kind` | `scattering_envelope` | 固定 | 只允许相位剥离后的散射包络标签。 |

PDE-only 配置：

```yaml
training:
  lambda_pde: 1.0
  lambda_data: 0.0
  supervision:
    enabled: false
    reference_path: null
    target_kind: scattering_envelope
```

Data-only 配置：

```yaml
training:
  lambda_pde: 0.0
  lambda_data: 1.0
  supervision:
    enabled: true
    reference_path: /path/to/reference_envelope.npy
    target_kind: scattering_envelope
```

Hybrid 配置：

```yaml
training:
  lambda_pde: 1.0
  lambda_data: 1.0
  supervision:
    enabled: true
    reference_path: /path/to/reference_envelope.npy
    target_kind: scattering_envelope
```

在 `scripts/run_matrix.py` 的矩阵文件中，推荐使用 `reference_prep` 自动生成标签，而不是手写 `reference_path`：

```yaml
matrix:
  - run_id: smooth_lens__omega90__hybrid
    overrides:
      medium: {velocity_model: smooth_lens}
      physics: {omega: 90.0}
      training:
        lambda_pde: 1.0
        lambda_data: 1.0
    reference_prep:
      enabled: true
      label_name: matched
```

启动器会先运行 `scripts/solve_reference.py` 生成：

```text
<output-root>/_reference_cache/<batch>/<run>/<label>__<hash>/reference_envelope.npy
```

然后把该路径写入训练 run 的有效配置。

### 2.1.6 训练与日志配置

| YAML key | 默认值 | 当前是否可作为矩阵轴 | 作用 |
|---|---:|---|---|
| `training.lr` | `1.0e-3` | 是 | 学习率。 |
| `training.epochs` | `1000` | 是 | 训练轮次。 |
| `training.batch_size` | `1` | 暂不建议 | 当前单样本路径，先不要作为主轴。 |
| `logging.level` | `INFO` | 否 | 日志级别。 |
| `logging.save_dir` | `outputs` | 否 | 单次训练默认输出目录；矩阵脚本用 `--output-root` 接管。 |
| `logging.save_interval` | `100` | 可固定 | checkpoint / 日志保存间隔。 |

高波数长训建议：

```yaml
training:
  lr: 3.0e-4
  epochs: 5000
  batch_size: 1
```

最终 finalist 可使用命令行统一覆盖：

```bash
python scripts/run_matrix.py \
  --output-root outputs/highk_round3_finalists \
  --device cuda:0 \
  --epochs-override 20000 \
  --continue-on-error
```

## 2.2 矩阵 YAML 文件写法

当前矩阵启动器读取：

```text
configs/experiments/B*.yaml
```

每个实验批次文件的结构应保持如下：

```yaml
batch_id: C1
name: highk_scaling_ppw16
status: active
entrypoint: train              # train 或 reference_only
purpose: Fixed-ppw frequency scaling for high-wavenumber regime.
expected_runs: 5

fixed:
  medium:
    velocity_model: smooth_lens
  pml:
    power: 2
    R0: 1.0e-6
  residual:
    lap_tau_mode: stretched_divergence
  model:
    nsno_blocks: 8
    nsno_channels: 64
    fno_modes: 16
  training:
    epochs: 5000
    lr: 3.0e-4
    lambda_pde: 1.0
    lambda_data: 0.0
    supervision:
      enabled: false
      reference_path: null
      target_kind: scattering_envelope

matrix:
  - run_id: omega060__N153__ppw16
    overrides:
      physics: {omega: 60.0}
      grid: {nx: 153, ny: 153}
      pml: {width: 24}
  - run_id: omega090__N229__ppw16
    overrides:
      physics: {omega: 90.0}
      grid: {nx: 229, ny: 229}
      pml: {width: 24}
```

字段合并顺序是：

```text
configs/base.yaml
-> batch.fixed
-> run.overrides
-> --epochs-override
-> reference_prep 自动写入 supervision.reference_path
```

因此实验设计中必须避免在 `fixed` 与 `overrides` 中写相互矛盾的 key，尤其是：

- `training.lambda_data`
- `training.supervision.enabled`
- `training.supervision.reference_path`
- `residual.lap_tau_mode`

## 2.3 现有 `B*.yaml` 与本文 `C*` 矩阵的对应关系

当前仓库已有一组较早的 `B*.yaml`，可以复用，但它们不是完整高波数 campaign。建议关系如下：

| 本文矩阵 | 可复用配置文件 | 当前状态 | 需要补充 |
|---|---|---|---|
| C0 Reference Gate | `B10_reference_only.yaml` | 可直接跑 | 增加 `omega=120/180/240` 与 ppw 网格。 |
| C1 High-Wavenumber Scaling | `B2_frequency_sweep.yaml` | 可直接跑 | 现有 B2 固定 `N=128`，需新增 fixed-ppw 版本。 |
| C2 PPW Resolution Law | `B3_grid_pml.yaml` | 部分可跑 | B3 是 `omega=30`，需扩展到 `omega>=90`。 |
| C3 Phase Budget | `B3_grid_pml.yaml` + summary metrics | 部分可跑 | 需按 `omega*h^2` 与 `omega*h` 分层设计。 |
| C4 PML Stability | `B3_grid_pml.yaml`, `B12_pml_profile.yaml` | 可直接跑 | 高频版本应加入 `omega=90/120/180`。 |
| C5 LapTau Strictness | `B9_lap_tau_audit.yaml` | 配置文件状态偏旧 | runtime 已支持；建议下一步把 B9 从 blocked 改为 active 并写入真实 overrides。 |
| C6 Medium Complexity | `B1_media_sweep.yaml` | 可直接跑 | 仅三类介质，后续需要更多强散射参数化介质。 |
| C7 Source Robustness | `B7_source_pos.yaml` | 可直接跑 | 高频版需配合 `omega=90/120`。 |
| C8 NSNO Depth | `B4_capacity.yaml` | 部分可跑 | B4 同时改 depth/modes/channels，建议拆出纯 depth sweep。 |
| C9 FNO Mode Bandwidth | `B4_capacity.yaml` | 部分可跑 | 建议固定 depth/channels，只扫 `fno_modes`。 |
| C10 Channel Capacity | `B4_capacity.yaml` | 部分可跑 | 建议固定 depth/modes，只扫 `nsno_channels`。 |
| C11 Supervision / Loss | `B1_media_sweep.yaml`, `B6_hybrid_weights.yaml`, `B14_label_mismatch.yaml` | 可直接跑 | 需要显式加入 data-only。 |
| C12 Optimization Budget | `B5_optim.yaml` | 可直接跑 | 高频版建议固定最佳容量后重扫。 |
| C13 Curriculum / Warm Start | 无 | 需扩展 | 需要训练脚本支持跨频率 checkpoint warm-start。 |
| C14 Phase Pathology | `B14_label_mismatch.yaml` + phase metrics | 部分可跑 | 需补 phase-loss / random-phase 初始化记录。 |
| C15 SOTA Narrative Cards | 无单独配置 | 需手动汇总 | 以最终 winner configs 生成 benchmark card。 |

注意：`B8_seed.yaml`、`B13_loss_switches.yaml`、`B15_eikonal_precision.yaml` 当前仍应保持 blocked，除非后续代码把对应字段真正接入 runtime。

## 2.4 第一轮建议启动配置

第一轮目标不是追求最终最优，而是快速决定高波数路线是否成立。建议先写 5 个新批次配置，或用 `--batch-filter` 从现有 B 批次里筛选同类实验。

### 2.4.1 R1A: reference gate

建议配置：

```yaml
entrypoint: reference_only
fixed:
  residual:
    lap_tau_mode: stretched_divergence
  training:
    epochs: 1
matrix:
  medium.velocity_model: [homogeneous, smooth_lens, layered]
  physics.omega: [30, 60, 90, 120, 180]
  ppw: [12, 16]
  pml.width: [16, 24, 32]
```

实际 YAML 不能直接写笛卡尔积，需要展开成多条 `matrix` run。网格按下式生成：

```text
N = ceil(ppw * omega / (2*pi))
```

建议先排除 `N>512` 的训练 run；reference-only 可以保留到 `N≈700`。

### 2.4.2 R1B: fixed-ppw high-k training

主配置：

```yaml
fixed:
  medium: {velocity_model: smooth_lens}
  residual: {lap_tau_mode: stretched_divergence}
  model:
    nsno_blocks: 8
    nsno_channels: 64
    fno_modes: 16
  training:
    epochs: 5000
    lr: 3.0e-4
    lambda_pde: 1.0
    lambda_data: 0.0
    supervision:
      enabled: false
      reference_path: null
      target_kind: scattering_envelope
matrix:
  physics.omega: [60, 90, 120, 180]
  ppw: [12, 16]
```

展开后典型 run：

```yaml
- run_id: smooth_lens__omega120__ppw16__N306__pde
  overrides:
    physics: {omega: 120.0}
    grid: {nx: 306, ny: 306}
    pml: {width: 32}
```

### 2.4.3 R1C: depth vs modes 解耦

Depth sweep：

```yaml
fixed:
  physics: {omega: 90.0}
  grid: {nx: 229, ny: 229}   # ppw≈16
  pml: {width: 24}
  model:
    nsno_channels: 64
    fno_modes: 16
matrix:
  model.nsno_blocks: [2, 4, 6, 8, 12, 16]
```

Mode sweep：

```yaml
fixed:
  physics: {omega: 90.0}
  grid: {nx: 229, ny: 229}
  pml: {width: 24}
  model:
    nsno_blocks: 8
    nsno_channels: 64
matrix:
  model.fno_modes: [4, 8, 12, 16, 24, 32]
```

Channel sweep：

```yaml
fixed:
  physics: {omega: 90.0}
  grid: {nx: 229, ny: 229}
  pml: {width: 24}
  model:
    nsno_blocks: 8
    fno_modes: 16
matrix:
  model.nsno_channels: [16, 32, 64, 96, 128]
```

### 2.4.4 R1D: PDE-only / data-only / hybrid

同一个物理配置必须跑三种监督方式：

```yaml
# PDE-only
training:
  lambda_pde: 1.0
  lambda_data: 0.0
  supervision:
    enabled: false
    reference_path: null
    target_kind: scattering_envelope
```

```yaml
# Data-only
training:
  lambda_pde: 0.0
  lambda_data: 1.0
  supervision:
    enabled: true
    reference_path: null
    target_kind: scattering_envelope
reference_prep:
  enabled: true
  label_name: matched
```

```yaml
# Hybrid
training:
  lambda_pde: 1.0
  lambda_data: 1.0
  supervision:
    enabled: true
    reference_path: null
    target_kind: scattering_envelope
reference_prep:
  enabled: true
  label_name: matched
```

建议在以下场景上跑：

| medium | omega | ppw | N | pml.width |
|---|---:|---:|---:|---:|
| `homogeneous` | 90 | 16 | 229 | 24 |
| `smooth_lens` | 90 | 16 | 229 | 24 |
| `smooth_lens` | 120 | 16 | 306 | 32 |
| `layered` | 90 | 16 | 229 | 24 |
| `layered` | 120 | 16 | 306 | 32 |

### 2.4.5 R1E: strict LapTau paired audit

每个 run 必须只改变 `residual.lap_tau_mode`，其余配置完全一致：

```yaml
fixed:
  physics: {omega: 90.0}
  grid: {nx: 229, ny: 229}
  pml:
    width: 24
    power: 2
    R0: 1.0e-6
  model:
    nsno_blocks: 8
    nsno_channels: 64
    fno_modes: 16
  training:
    epochs: 5000
    lr: 3.0e-4
matrix:
  - run_id: smooth_lens__strict
    overrides:
      medium: {velocity_model: smooth_lens}
      residual: {lap_tau_mode: stretched_divergence}
  - run_id: smooth_lens__legacy
    overrides:
      medium: {velocity_model: smooth_lens}
      residual: {lap_tau_mode: mixed_legacy}
```

这组实验应作为 C5 的高优先级验证，因为它直接对应 PML 算子不可交换性的理论证明。

## 3. 核心无量纲控制量

高波数实验应围绕无量纲量组织，而不是只扫原始参数。

### 3.1 每波长点数

```text
lambda = 2*pi / (omega * s0)
ppw = lambda / h = 2*pi / (omega * s0 * h)
```

在 `s0≈1`、单位正方形区域下：

```text
N ~= ppw * omega / (2*pi)
```

示例：

| omega | ppw=8 | ppw=12 | ppw=16 | ppw=24 |
|---:|---:|---:|---:|---:|
| 60 | 76 | 115 | 153 | 229 |
| 90 | 115 | 172 | 229 | 344 |
| 120 | 153 | 229 | 306 | 458 |
| 180 | 229 | 344 | 458 | 688 |
| 240 | 306 | 458 | 611 | 917 |
| 300 | 382 | 573 | 764 | 1146 |

服务器 RTX A6000 可支撑较大网格，但 `N>700` 的训练 run 成本会明显增加，应优先做 reference-only 或 finalist long-run。

### 3.2 相位误差预算

用于判断误差是否由 FSM / Eikonal 走时主导：

```text
smooth medium floor proxy: omega * h^2
kink/order-loss proxy:    omega * h
```

对应当前指标：

- `phase_tau_error_budget_h2`
- `phase_tau_error_budget_h`
- `wavefield_phase_error_budget_mean_h2`
- `wavefield_phase_error_budget_p95_h2`
- `wavefield_phase_error_budget_mean_h`
- `wavefield_phase_error_budget_p95_h`

### 3.3 Neumann 容量代理

用于判断散射强度和网络深度是否匹配：

```text
scattering_strength_proxy ~ omega * diam(domain) * ||s^2 - s0^2|| / s0
```

对应当前指标：

- `full_wave_nyquist_mode_floor`
- `fno_mode_to_full_wave_nyquist`
- `scattering_strength_proxy`
- `neumann_proxy_convergent`
- `neumann_depth_tail_proxy`

## 4. 实验矩阵总览

建议把完整 campaign 分为 16 组。

| 组别 | 名称 | 核心问题 | 是否当前可直接跑 |
|---|---|---|---|
| C0 | Sanity / Reference Gate | reference floor 是否可信 | 是 |
| C1 | High-Wavenumber Scaling | 固定 ppw 下高频能否求解 | 是 |
| C2 | PPW Resolution Law | 多少 ppw 足够 | 是 |
| C3 | Grid Pollution / Phase Budget | 误差是否由 `omega*h^2` / `omega*h` 解释 | 是 |
| C4 | PML Stability | 高频 PML 是否稳定 | 是 |
| C5 | LapTau Strictness | strict PML residual 是否优于 legacy | 是 |
| C6 | Medium Complexity | 介质复杂度与强散射影响 | 部分可跑 |
| C7 | Source Robustness | 单源位置是否造成偏置 | 是 |
| C8 | NSNO Depth | Neumann 深度是否为主瓶颈 | 是 |
| C9 | FNO Mode Bandwidth | Fourier modes 是否为主瓶颈 | 是 |
| C10 | Channel Capacity | 宽度是否限制模型 | 是 |
| C11 | Supervision / Loss | PDE-only / data-only / hybrid 差异 | 是 |
| C12 | Optimization Budget | lr / epochs 是否限制收敛 | 是 |
| C13 | Curriculum / Warm Start | 频率 curriculum 是否改善高频训练 | 需扩展 |
| C14 | Phase Pathology | phase loss / pi/2 随机相位是否出现 | 是 |
| C15 | SOTA Narrative Cards | 与文献 benchmark 对齐 | 部分可跑 |

## 5. C0: Sanity / Reference Gate

### 目的

在正式训练前确认 reference solver、strict residual、PML 和基础配置没有数值问题。

### 矩阵

| 轴 | 值 |
|---|---|
| `medium.velocity_model` | `homogeneous`, `smooth_lens`, `layered` |
| `physics.omega` | 30, 60, 90, 120 |
| `grid.nx = grid.ny` | 64, 128, 192 |
| `pml.width` | 8, 16, 24 |
| `residual.lap_tau_mode` | `stretched_divergence` |
| entrypoint | `reference_only` |

### 规模

```text
3 media * 4 omega * 3 grids * 3 pml = 108 reference runs
```

### 必看指标

- `reference_residual_rmse`
- `reference_residual_max`
- `phase_tau_error_budget_h2`
- `phase_tau_error_budget_h`
- `pml_thickness_in_wavelengths`

### 通过标准

- homogeneous reference floor 应接近机器/离散底噪。
- smooth_lens / layered 不应出现 NaN/Inf。
- reference floor 若已高于训练目标，后续训练 run 不进入 finalist。

## 6. C1: High-Wavenumber Scaling

### 目的

验证在固定 ppw 条件下，频率升高时模型是否仍能逼近 reference。

### 矩阵

| 轴 | 值 |
|---|---|
| `omega` | 30, 60, 90, 120, 180, 240, 300 |
| `ppw` | 8, 12, 16 |
| `medium` | `smooth_lens`, `layered` |
| `model tier` | base, deep |
| `supervision` | PDE-only, hybrid 1:10 |
| `lap_tau_mode` | `stretched_divergence` |

模型定义：

| tier | `nsno_blocks` | `nsno_channels` | `fno_modes` |
|---|---:|---:|---:|
| base | 4 | 32 | 16 |
| deep | 8 | 32 | 16 |

### 规模

```text
7 omega * 3 ppw * 2 media * 2 model * 2 supervision = 168 runs
```

### 建议执行

先跑子集：

```text
omega = 60, 120, 180
ppw = 12, 16
medium = smooth_lens, layered
model = base, deep
supervision = PDE-only, hybrid 1:10
```

约 48 runs。

### 关键图

- `omega` vs `rel_l2_to_reference`
- `omega` vs `phase_mae_to_reference`
- `omega` vs `residual_rmse_evaluation`
- 按 ppw 分线。

## 7. C2: PPW Resolution Law

### 目的

确定高波数下最小可用分辨率。该组是最终 benchmark 的关键基础。

### 矩阵

| 轴 | 值 |
|---|---|
| `omega` | 90, 180, 300 |
| `ppw` | 6, 8, 10, 12, 16, 24 |
| `medium` | `smooth_lens`, `layered` |
| `model` | best-so-far |
| `supervision` | best-so-far |

### 规模

```text
3 omega * 6 ppw * 2 media = 36 runs
```

可加 reference-only companion，共 72 jobs。

### 判读

- `ppw < 8` 如果 reference floor 已经变差，说明是数值分辨率失败。
- `ppw >= 12` 如果 reference 好但模型差，说明模型/优化瓶颈。
- `ppw >= 16` 如果误差平台化，说明继续加网格收益有限。

## 8. C3: Grid Pollution / Phase Budget

### 目的

检验 `omega*h^2` 与 `omega*h` 是否解释标签/重构误差下界。

### 矩阵

| 轴 | 值 |
|---|---|
| `omega` | 30, 60, 90, 120, 180 |
| `N` | 64, 96, 128, 192, 256, 384 |
| `medium` | `homogeneous`, `smooth_lens`, `layered` |
| entrypoint | reference-only + trained subset |

### 规模

Reference-only:

```text
5 omega * 6 N * 3 media = 90 runs
```

训练子集：

```text
omega = 60, 120, 180
N = 96, 128, 192, 256
medium = smooth_lens, layered
=> 24 runs
```

### 核心图

- x: `phase_tau_error_budget_h2`, y: `rel_l2_to_reference`
- x: `phase_tau_error_budget_h`, y: `rel_l2_to_reference`
- 按 medium 分面。

### 预期

- `smooth_lens` 更接近 `omega*h^2`。
- `layered` 更可能接近 `omega*h`。
- 若训练误差低于预算 proxy，应谨慎检查 reference 和指标。

## 9. C4: High-Wavenumber PML Stability

### 目的

确认高波数下 PML 不是主要伪反射来源，并选择高频默认 PML 配置。

### 矩阵

| 轴 | 值 |
|---|---|
| `omega` | 90, 180, 300 |
| `pml_thickness_in_wavelengths` | 0.5, 0.75, 1.0, 1.5, 2.0 |
| PML profile | soft, default, strong, steep, aggressive |
| `medium` | `smooth_lens`, `layered` |

PML profile：

| profile | `pml.power` | `pml.R0` |
|---|---:|---:|
| soft | 1 | 1e-4 |
| default | 2 | 1e-6 |
| strong | 2 | 1e-8 |
| steep | 3 | 1e-6 |
| aggressive | 4 | 1e-8 |

### 规模

```text
3 omega * 5 thickness * 5 profiles * 2 media = 150 runs
```

### 指标

- `residual_rmse_evaluation`
- `reference_residual_rmse`
- physical boundary residual
- centerline edge oscillation
- heatmaps at PML interface

### 判读

低 PML 厚度若 residual 好但 `rel_l2` 差，可能是吞能偏置。最终默认应选择物理可解释而非单点最优的 PML profile。

## 10. C5: LapTau Strictness

### 目的

验证 `mixed_legacy` 在 PML 中引入常量级截断误差源的理论判断。

### 矩阵

| 轴 | 值 |
|---|---|
| `lap_tau_mode` | `mixed_legacy`, `stretched_divergence` |
| `omega` | 30, 90, 180 |
| `pml_width` | 8, 16, 24 |
| `medium` | `homogeneous`, `smooth_lens`, `layered` |
| `source_pos` | center, near_boundary |

### 规模

```text
2 modes * 3 omega * 3 pml * 3 media * 2 source = 108 runs
```

### 指标

- `reference_residual_rmse`
- `residual_rmse_evaluation`
- `phase_mae_to_reference`
- PML/physical boundary heatmaps

### 预期

高频、薄 PML、near-boundary source 下 strict mode 应更稳。

## 11. C6: Medium Complexity / Strong Scattering

### 当前可跑矩阵

| 轴 | 值 |
|---|---|
| `medium` | `homogeneous`, `smooth_lens`, `layered` |
| `omega` | 60, 120, 180, 300 |
| `ppw` | 12, 16 |
| `nsno_blocks` | 4, 8, 12 |

规模：

```text
3 media * 4 omega * 2 ppw * 3 depths = 72 runs
```

### 建议后续扩展介质族

当前三种介质不足以支撑“comprehensive benchmark”。建议后续新增：

| medium family | 参数 |
|---|---|
| smooth_lens_contrast | contrast = 0.1, 0.3, 0.5, 0.8 |
| multi_lens | inclusions = 2, 4, 8 |
| random_smooth | correlation length = 0.05, 0.1, 0.2 |
| sharp_interface_lens | contrast = 0.2, 0.5 |
| checkerboard_smooth | cell count = 4, 8, 16 |
| fault_layered | angle = 15, 30, 45 degrees |

这些介质能与文献中的 heterogeneous / high-contrast / strong scattering 叙事衔接。

## 12. C7: Source Robustness

### 目的

验证单源位置是否引入系统偏置。

### 矩阵

| 轴 | 值 |
|---|---|
| `source_pos` | center, x_quarter, y_quarter, off_center, near_left, near_corner |
| `omega` | 60, 120, 180 |
| `medium` | `smooth_lens`, `layered` |
| `ppw` | 16 |
| `model` | best-so-far |

建议 source 坐标：

| name | `source_pos` |
|---|---|
| center | [0.5, 0.5] |
| x_quarter | [0.25, 0.5] |
| y_quarter | [0.5, 0.25] |
| off_center | [0.35, 0.65] |
| near_left | [0.15, 0.5] |
| near_corner | [0.2, 0.2] |

规模：

```text
6 sources * 3 omega * 2 media = 36 runs
```

## 13. C8: NSNO Depth Sweep

### 目的

验证 Neumann series unrolling 是否需要更高散射阶数。

### 矩阵

| 轴 | 值 |
|---|---|
| `nsno_blocks` | 1, 2, 4, 6, 8, 12, 16, 24 |
| `omega` | 60, 120, 180 |
| `medium` | `smooth_lens`, `layered` |
| `ppw` | 16 |
| `fno_modes` | 16 |
| `nsno_channels` | 32, optionally 48 |

规模：

```text
8 depths * 3 omega * 2 media = 48 runs
```

若加 `channels=48`，共 96 runs。

### 预期

- `L=1/2` 明显欠表达。
- `L=8/12` 应显著改善。
- `L=16/24` 可能平台化。

## 14. C9: FNO Mode Bandwidth Sweep

### 目的

验证相位剥离后模型是否仍受 Fourier mode 截断限制。

### 矩阵

| 轴 | 值 |
|---|---|
| `fno_modes` | 4, 8, 12, 16, 24, 32, 48 |
| `omega` | 120, 180 |
| `medium` | `smooth_lens`, `layered` |
| `nsno_blocks` | best depth, e.g. 12 |
| `nsno_channels` | 48 |
| `ppw` | 16 |

规模：

```text
7 modes * 2 omega * 2 media = 28 runs
```

### 预期

若 envelope 真正低频化，则 `M>=12/16` 后收益应明显平台化。

## 15. C10: Channel Capacity Sweep

### 目的

判断网络宽度是否为瓶颈。

### 矩阵

| 轴 | 值 |
|---|---|
| `nsno_channels` | 16, 32, 48, 64, 96, 128 |
| `nsno_blocks` | 8, 12 |
| `fno_modes` | 16 |
| `omega` | 120 |
| `medium` | `smooth_lens`, `layered` |

规模：

```text
6 channels * 2 depths * 2 media = 24 runs
```

## 16. C11: Supervision / Loss Matrix

### 目的

对齐 PINO / HyPINO / hybrid training 叙事。

### Loss 组合

| mode | `lambda_pde` | `lambda_data` |
|---|---:|---:|
| PDE-only | 1 | 0 |
| data-only | 0 | 1 |
| balanced hybrid | 1 | 1 |
| data-dominant hybrid | 1 | 10 |
| PDE-dominant hybrid | 10 | 1 |
| weak PDE hybrid | 0.1 | 1 |
| weak data hybrid | 1 | 0.1 |

### 矩阵

| 轴 | 值 |
|---|---|
| `omega` | 60, 120, 180 |
| `medium` | `smooth_lens`, `layered` |
| `loss mode` | 7 modes |
| `model` | best-so-far |
| `ppw` | 16 |

规模：

```text
3 omega * 2 media * 7 loss = 42 runs
```

### 判读

- PDE-only residual 好但 phase 差，说明 phase lock 不足。
- data-only rel_l2 好但 PDE residual 差，说明 label fitting 不等于物理可解。
- hybrid 1:10 若最稳，支持 data supervision 对 phase lock 有帮助。

## 17. C12: Optimization Budget

### 目的

确认高频失败是否只是训练预算不足。

### 矩阵

| 轴 | 值 |
|---|---|
| `epochs` | 1000, 3000, 5000, 10000, 20000, 40000 |
| `lr` | 1e-4, 3e-4, 1e-3 |
| `omega` | 120, 180 |
| `medium` | `smooth_lens`, `layered` |
| `model` | best-so-far |

规模：

```text
6 epochs * 3 lr * 2 omega * 2 media = 72 runs
```

建议每个 run 保存：

- full loss history
- final component losses
- best epoch
- tail slope

当前 full per-epoch loss components 仍是未来增强项。

## 18. C13: Curriculum / Warm Start

### 目的

强衔接 docx 中的 frequency curriculum、warm start、denser sampling、architectural expansion 叙事。

### 当前状态

当前训练入口没有原生 warm-start / stage curriculum 管理，需要后续扩展脚本支持：

- 从上一个 stage 加载 `model_state.pt`
- 修改 `omega`
- 修改网格/ppw
- 可选扩展 `nsno_blocks`
- 可选重置 optimizer / lr

### Curriculum schedules

| schedule | frequency path | grid policy | model policy | lr policy |
|---|---|---|---|---|
| direct | target only | target ppw | final model | normal |
| octave-2 | 30 -> 60 | ppw fixed | same model | stage reset |
| octave-3 | 30 -> 60 -> 120 | ppw fixed | same model | stage reset |
| octave-4 | 30 -> 60 -> 120 -> 180 | ppw fixed | same model | stage reset |
| dense | 30 -> 60 -> 120 -> 180 | denser sampling / fixed ppw | same model | stage reset |
| grow-depth | 30:L4 -> 60:L8 -> 120:L12 -> 180:L16 | fixed ppw | expand depth | warm restart |
| no-restart | same as octave-4 | fixed ppw | same model | no lr reset |

### 矩阵

| 轴 | 值 |
|---|---|
| target omega | 120, 180 |
| medium | `smooth_lens`, `layered` |
| schedule | 7 schedules |

规模：

```text
2 omega * 2 media * 7 schedules = 28 multi-stage jobs
```

### 核心对照

- direct high-frequency random init
- low-to-high warm-start
- low-to-high + denser sampling
- low-to-high + depth expansion
- with vs without warm restart

## 19. C14: Phase Pathology Matrix

### 目的

对应 docx 中 phase loss、sign-flip saddle、随机相位 `pi/2` 的讨论。

### 矩阵

| 轴 | 值 |
|---|---|
| `omega` | 60, 120, 180, 300 |
| `medium` | `smooth_lens`, `layered` |
| `supervision` | PDE-only, data-only, hybrid 1:10 |
| `model` | base, deep |
| `ppw` | 16 |

规模：

```text
4 omega * 2 media * 3 supervision * 2 model = 48 runs
```

### 判据

- `phase_mae_to_reference ≈ pi/2`：相位随机猜测。
- amp error 低但 phase error 高：典型 phase loss。
- deep/curriculum 降低 phase error：结构或训练策略有效。

## 20. C15: SOTA Narrative Benchmark Cards

### 目的

形成最终论文或报告中的固定 benchmark card，便于和文献 SOTA 叙事对齐。

### Cards

| Card | 设置 | 叙事目标 |
|---|---|---|
| A | `omega=30, N=256, smooth_lens` | 当前 proprietary 1e-3 benchmark anchor |
| B | `omega=120, ppw=16, smooth_lens` | resolved high-wavenumber smooth medium |
| C | `omega=180, ppw=16, layered` | high-frequency strong scattering |
| D | `omega=300, ppw=12, smooth_lens` | extreme high-frequency stress |
| E | `omega=180, ppw=8, layered` | under-resolved failure mode |
| F | `omega=120, ppw=16, near-boundary source` | source/PML stress |
| G | `omega=180, ppw=16, best curriculum` | curriculum-aligned final result |

### 每张 card 跑的 variants

| variant | 说明 |
|---|---|
| reference-only | reference floor |
| base NSNO | `L=4, C=32, M=16` |
| deep NSNO | best depth from C8 |
| best hybrid | best loss from C11 |
| best curriculum | best schedule from C13 |
| mixed legacy audit | 少量保留，用于 PML 叙事 |

规模：

```text
7 cards * 5-6 variants = 35-42 runs
```

## 21. 三轮执行计划

### Round 1: Diagnostic Wide Sweep

目标：快速确定可行区域，筛掉数值不可解释点。

包含：

- C0
- C1 子集
- C2
- C5
- C8/C9 小规模

预计：

```text
250-350 runs
```

### Round 2: High-Wavenumber Main Sweep

目标：形成高波数主结果。

包含：

- C1 完整
- C3
- C4
- C6
- C11
- C12

预计：

```text
400-600 runs
```

### Round 3: Paper Benchmark / Long Runs

目标：生成最终图表、长训结果、文献对齐 benchmark。

包含：

- C13
- C14
- C15
- long-run finalists

预计：

```text
80-150 runs
```

## 22. 服务器运行建议

RTX A6000 可长期跑，但应分层调度：

1. reference-only 优先，筛选可信配置。
2. 小 epochs 训练广扫，筛掉明显失败区域。
3. 中等 epochs 验证容量与 loss。
4. finalist 才跑 20000-40000 epochs。

建议输出目录：

```text
outputs/highk_round1_reference_gate
outputs/highk_round1_wide
outputs/highk_round2_main
outputs/highk_round3_finalists
```

每轮都应运行：

```bash
python scripts/run_matrix.py --output-root <output-root> --dry-run --include-blocked
python scripts/run_matrix.py --output-root <output-root> --device cuda:0 --continue-on-error
python scripts/summarize_matrix.py --output-root <output-root>
```

如果需要统一 epochs：

```bash
python scripts/run_matrix.py --output-root <output-root> --device cuda:0 --epochs-override 5000 --continue-on-error
```

## 23. 最终报告图表清单

必须产出的核心图：

1. `omega` vs `rel_l2_to_reference`
2. `omega` vs `residual_rmse_evaluation`
3. `ppw` vs `rel_l2_to_reference`
4. `phase_tau_error_budget_h2/h` vs `rel_l2_to_reference`
5. `nsno_blocks` vs `rel_l2_to_reference`
6. `fno_modes` vs `rel_l2_to_reference`
7. `scattering_strength_proxy` vs error
8. `phase_mae_to_reference` heatmap
9. PML thickness vs residual
10. `lap_tau_mode` strict vs legacy paired plot
11. PDE-only / data-only / hybrid comparison
12. epoch budget vs final error
13. curriculum vs direct training loss curve
14. runtime vs accuracy Pareto
15. SOTA benchmark card summary table

## 24. 最终判断标准

### 成功标准

高波数主张成立至少需要：

1. 在 `omega >= 120`、`ppw >= 12/16` 下，reference floor 可信。
2. `stretched_divergence` 相比 `mixed_legacy` 在 PML/high-frequency 场景下更稳。
3. 深度增加比单纯增加 `fno_modes` 更有效。
4. `rel_l2_to_reference` 与相位误差预算存在可解释关系。
5. high-frequency finalist 能在至少一个 `omega>=180` 场景达到稳定、可复现误差。
6. phase MAE 不应长期停在 `pi/2`。

### 失败也有价值的判据

以下失败结果也能形成有效结论：

- reference floor 本身随 ppw 崩坏：数值离散不足。
- layered 在高频贴近 `omega*h` 预算：Eikonal 降阶主导。
- 增加 `fno_modes` 无效但增加 depth 有效：Neumann 散射阶数主导。
- hybrid 比 PDE-only 差：label phase noise 或 reference mismatch 主导。
- curriculum 明显优于 direct：与文献 spectral bias/curriculum 叙事一致。

## 25. 推荐优先启动矩阵

如果现在立刻上服务器，建议第一批不是全量，而是：

1. C0: Reference Gate
2. C1: High-Wavenumber Scaling 子集
3. C2: PPW Resolution Law
4. C8: NSNO Depth Sweep
5. C11: Supervision / Loss Matrix

这五组最能快速回答：

- 高频是否可解。
- 多少 ppw 足够。
- 深度是否是瓶颈。
- hybrid 是否帮助 phase lock。
- 目前 1e-3 误差是否接近理论底噪。

建议第一批规模：

```text
C0: 108
C1 subset: 48
C2: 36-72
C8: 48
C11: 42
Total: about 282-318 runs
```

这对 A6000 连续运行是合理的，并且结果足够决定第二轮大矩阵怎么收缩。
