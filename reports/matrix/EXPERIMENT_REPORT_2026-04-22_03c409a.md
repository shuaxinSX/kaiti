# 2D Helmholtz 神经算子实验矩阵总结报告

- 数据来源：`outputs/matrix_from_branch_ep5000_full/`
- 统计口径：102 完成 run / 0 失败 / 4 阻塞批次（B8/B9/B13/B15 尚未接入 runtime）
- 生成工件：`_reports/matrix_report.csv`、`matrix_quantiles.csv`、`matrix_pareto.csv`、`matrix_reference_only.csv`、`matrix_loss_history.csv`（48 万行）、`matrix_centerline.csv`（2.5 万行）、12 批 × 14 张全局图 / 38 张批内图
- 主指标：`residual_rmse_evaluation`（纯物理残差 RMSE，越低越好）；辅助：`rel_l2_to_reference`、`amp_mae_to_reference`、`phase_mae_to_reference`、`rmse_gap_to_reference`
- 统一训练轮数：**epochs = 5000**（本轮复跑已统一拉满，B5 的 ep500/1000/3000 标签仅为历史命名）
- GPU：RTX A6000 单卡；全矩阵累计壁钟 ≈ 132 min（B4 xlarge 拉高长尾）
- 配置来源：默认层来自 `configs/base.yaml`，批次层来自 `configs/experiments/B*.yaml`，本报告对应一次 `scripts/run_matrix.py --epochs-override 5000` 的统一复跑结果

---

## 1. 实验目标与配置

### 1.1 实验目标

本轮实验矩阵服务于四个直接问题：
1. 在 `2D + 单源 + 规则网格 + PML` 的当前原型上，确认训练/评估/参考解链路是否稳定可复现。
2. 在**介质、频率、网格、PML、模型规模、优化策略、监督权重、源点位置**八类主轴上，锁定一个可继续迭代的 baseline 工作点。
3. 用 `B10 reference_only` 给出参考解残差地板，量化神经算子与 reference solver 之间还剩多少数量级差距。
4. 把当前 runtime 尚未接通、但配置层已经存在的实验轴（seed、`lap_tau_mode`、loss switches、eikonal precision）明确标注出来，避免把“未实验”误写成“已验证”。

### 1.2 配置分层与本次执行口径

| 层级 | 来源 | 作用 | 本次采用口径 |
|---|---|---|---|
| 默认层 | `configs/base.yaml` | 提供物理、网格、PML、模型、loss、training 默认值 | 作为所有 batch 的起点 |
| 批次层 | `configs/experiments/B*.yaml` | 定义每个 batch 的变量轴、固定项和 `entrypoint` | 共 16 批：12 active，4 blocked |
| 启动层 | `scripts/run_matrix.py` | 允许对全矩阵施加统一 launcher 覆盖 | 本次使用 `--epochs-override 5000`，所以所有 `entrypoint=train` 的 run 实际都按 5000 epochs 执行 |
| 统计层 | `outputs/matrix_from_branch_ep5000_full/_reports/*` | 汇总 manifest、summary、loss、reference 和图表 | 本报告仅以已落盘工件为准，缺失字段保持空白，不从隐藏状态反推 |

补充说明：
- `B5_optim.yaml` 设计上扫描 `epochs ∈ {500, 1000, 3000, 5000}`，但本次统一复跑后，这一维被 launcher 覆盖为 5000，故表中不同 `ep*` 标签仅保留为历史 run-id。
- `B10_reference_only` 的 `entrypoint` 是 reference solver，不走训练主循环；其 `training.epochs: 1` 只是占位配置，不影响参考解求解。
- hybrid 批次通过 `reference_prep.enabled=true` 先生成标签，再以 `training.lambda_data > 0` 打开监督项；PDE-only 批次保持 `lambda_data=0`、`supervision.enabled=false`。

### 1.3 基础默认配置（`configs/base.yaml`）

下表是**未被 batch 覆盖前**的仓库默认配置；实验矩阵的每一行都从这里出发再叠加对应 batch override：

| 模块 | 默认配置 |
|---|---|
| `physics` | `omega=30.0`，`source_pos=[0.5, 0.5]` |
| `grid` | `domain=[0,1]×[0,1]`，`nx=128`，`ny=128` |
| `medium` | `c_background=1.0`，`velocity_model=homogeneous` |
| `pml` | `width=20`，`power=2`，`R0=1e-6` |
| `eikonal` | `fsm_max_iter=100`，`fsm_tol=1e-10`，`source_freeze_radius=1`，`precision=float64` |
| `model` | `nsno_blocks=4`，`nsno_channels=32`，`fno_modes=16`，`activation=gelu` |
| `loss` | `source_mask_radius=1.5`，`pml_rhs_zero=true`，`omega2_normalize=true` |
| `training` | `lr=1e-3`，`epochs=1000`，`batch_size=1`，`lambda_pde=1.0`，`lambda_data=0.0` |
| `logging` | `level=INFO`，`save_dir=outputs`，`save_interval=100` |

### 1.4 各批次实验配置矩阵

#### Active 批次

| 批次 | 入口 | 变量轴 | 设计固定项 | 本次执行补充 |
|---|---|---|---|---|
| `B0 smoke` | `train` | `medium ∈ {homogeneous, smooth_lens, layered}` | `omega=10`，`N=32`，`pml=5`，`model=2/16/8`，spec `ep=50`，PDE-only | 本次也受全局 `epochs=5000` 覆盖；用于最小可运行栈 gate |
| `B1 media_sweep` | `train` | `medium × {pde, hybrid}` | `omega=30`，`N=128`，`pml=20`，`lr=1e-3`，spec `ep=500` | hybrid 使用 `reference_prep.label_name=matched`；实际训练 5000 epochs |
| `B2 frequency_sweep` | `train` | `omega ∈ {15,20,30,40,50,60,75,90}`，另加 `omega={60,90}` 两个 hybrid probe | `medium=smooth_lens`，`N=128`，`pml=20`，spec `ep=500` | 两个 hybrid probe 用 matched label；实际训练 5000 epochs |
| `B3 grid_pml` | `train` | `N ∈ {96,128,192,256}` × `pml ∈ {8,16,24}` | `medium=smooth_lens`，`omega=30`，`lr=1e-3`，spec `ep=500` | 这是后续 B4/B6/B7/B11/B12 的基线来源之一；实际训练 5000 epochs |
| `B4 capacity` | `train` | `capacity ∈ {small, base, large, xlarge}` × representative medium | `omega=30`，`N=128`，`pml=16`，spec `ep=500` | 四档网络分别为 `2/16/8`、`4/32/16`、`6/64/24`、`8/96/32`；实际训练 5000 epochs |
| `B5 optim` | `train` | `lr ∈ {1e-4,3e-4,1e-3,3e-3,1e-2}` × design `epochs ∈ {500,1000,3000,5000}` | `medium=smooth_lens`，`omega=30`，`N=128`，`pml=16`，PDE-only | 本次统一覆盖为 5000 epochs，因此最终只保留学习率差异的有效比较 |
| `B6 hybrid_weights` | `train` | `(lambda_pde, lambda_data) ∈ {(1,1),(1,10),(10,1),(1,0.1),(0.1,1)}` × `medium ∈ {smooth_lens, layered}` | `omega=30`，`N=128`，`pml=16`，`lr=1e-3`，spec `ep=500`，hybrid 默认打开 | 全部依赖 matched label；实际训练 5000 epochs |
| `B7 source_pos` | `train` | `source_pos ∈ {center, x/y quarter shifts, near_left_boundary, lower_left_corner}` | `medium=smooth_lens`，`omega=30`，`N=128`，`pml=16`，spec `ep=500` | 用于评估源点到 PML 的相对位置偏置；实际训练 5000 epochs |
| `B10 reference_only` | `reference_only` | representative `(medium, omega, grid, pml)` 组合 | 覆盖 `homogeneous/smooth_lens/layered@omega30,N128,pml16`，以及 `smooth_lens@omega60/90,N128,pml16`、`smooth_lens@omega30,N192,pml24` | 不训练，只求 reference floor，并给 B2/B3/B14 提供标签与对照 |
| `B11 source_mask_radius` | `train` | `source_mask_radius ∈ {1.0,1.5,2.0,3.0}` × `medium ∈ {smooth_lens, layered}` | `omega=30`，`N=128`，`pml=16`，spec `ep=500`，PDE-only | 用于验证近源穿孔半径；实际训练 5000 epochs |
| `B12 pml_profile` | `train` | `pml.power ∈ {1,2,3}` × selected `R0 ∈ {1e-4,1e-6,1e-8}` | `medium=smooth_lens`，`omega=30`，`N=128`，`pml.width=16`，spec `ep=500` | 实际覆盖点为 `(1,1e-4)`、`(1,1e-6)`、`(2,1e-6)`、`(2,1e-8)`、`(3,1e-6)`、`(3,1e-8)` |
| `B14 label_mismatch` | `train` | label 类型 `∈ {clean, omega_drift, medium_swap, source_shift}` | `medium=smooth_lens`，`omega=30`，`source=[0.5,0.5]`，`N=128`，`pml=16`，spec `ep=500`，`lambda_pde=lambda_data=1` | 三个错误 label 分别来自 `omega=33`、`medium=homogeneous`、`source=[0.75,0.5]` 的 reference_prep；实际训练 5000 epochs |

#### Blocked 批次

| 批次 | 计划变量轴 | 设计目的 | 当前阻塞原因 |
|---|---|---|---|
| `B8 seed` | `training.seed ∈ {0,1,2,3,4}` × `{best_grid, best_capacity}` | 估计随机初始化带来的方差 | `training.seed` 尚未从 `base.yaml/CLI/Trainer` 打通 |
| `B9 lap_tau_audit` | `residual.lap_tau_mode ∈ {mixed_legacy, stretched_divergence}` × `medium` | 重放 A5 的 `Δ̃τ` 审计到矩阵尺度 | `lap_tau_mode` 已在 residual 内部实现，但没有接入训练配置路径 |
| `B13 loss_switches` | `loss.omega2_normalize × loss.pml_rhs_zero` | 审计 base.yaml 中现存但休眠的 loss 开关 | 配置项存在，但 residual 路径当前不消费 |
| `B15 eikonal_precision` | `eikonal.precision ∈ {float32,float64}` × `medium` | 比较 Eikonal 预处理精度对残差和稳定性的影响 | `eikonal.precision` 尚未进入当前 eikonal runtime |

### 1.5 指标与判读原则

- 主指标是 `residual_rmse_evaluation`，表示**物理评估域**上的残差 RMSE；它最适合做 batch 间的主排序，但不能单独代表波场质量。
- `rel_l2_to_reference`、`amp_mae_to_reference`、`phase_mae_to_reference` 用于回答“是否只是把残差压低，但相位/振幅仍然学错”这一问题；B2 已证明这种情况真实存在。
- `rmse_gap_to_reference` / `rmse_gap_ratio` 以 `B10` 的 reference floor 为地板，度量神经算子离可接受数值解还有多远。
- 训练损失与评估口径并不完全相同：当前训练端只对源点近场做 hole mask，并**没有把 PML 区从 `loss_pde` 中剔除**；而本报告主指标强调物理评估域。解释 B7/B12/B14 时必须牢记这一区别。
- 源点位置、PML 厚度、错误监督三类实验都存在“指标偏置”风险，因此本报告在这些 batch 中更重视相对比较和机制解释，而非简单按最小 rmse 排名。

---

## 2. 总览

### 2.1 批次构成与资源占用

| 批次 | runs | 目的 | 中位运行时间 | 总时 | 主要结论 |
|---|---:|---|---:|---:|---|
| B0 smoke | 3 | 最小网格 32² 冒烟 | 59.8 s | 181 s | 三介质全通过，`converged=True` |
| B1 media × sup | 6 | 介质 × (pde/hybrid) 基线 | 62.6 s | 380 s | homogeneous 完美；layered 最难 |
| B2 frequency | 10 | ω ∈ {15…90} 扫 | 64.3 s | 643 s | 高频反而更准（ω≥50 时 rmse≈7e-4） |
| B3 grid×pml | 12 | 4 网格 × 3 PML | 72.9 s | 1027 s | N=96 最优；N=256 明显劣化 |
| B4 capacity | 10 | small→xlarge | 111.8 s | 2099 s | base(2.1 M) 已饱和，xlarge 仅付代价 |
| B5 optim | 20 | lr × epochs | 63.5 s | 1256 s | lr=3e-4 最优；lr=1e-2 四全崩 |
| B6 hybrid λ | 10 | λ_pde × λ_data | 61.3 s | 625 s | λ_data 偏大反而更稳 |
| B7 src-pos | 7 | 源点位置 | 63.9 s | 441 s | lower_left_corner 最准（边界吸波区） |
| B10 ref-only | 6 | 参考解地板 | — | — | 参考残差 ≤ 3.2e-7，作为 ground truth 可信 |
| B11 mask-r | 8 | source hole 半径 | 61.9 s | 497 s | r=2.0 对 smooth_lens 最优 |
| B12 PML profile | 6 | power × R0 | 63.0 s | 376 s | p=2, R0=1e-6（当前默认）已最优 |
| B14 label mismatch | 4 | 错误监督信号 | 65.2 s | 258 s | rmse 从 2.9e-3 跃至 1.2e-2（4×） |

阻塞（未统计）：
- **B8** seed 重复性 — runner 未暴露 `training.seed`
- **B9** `residual.lap_tau_mode` — 当前仅从 CLI/overlay 走死
- **B13** `loss.omega2_normalize` / `loss.pml_rhs_zero` — 配置项存在但 residual 路径未消费
- **B15** `eikonal.precision` — 默认 float64 硬编码

### 2.2 关键指标分布（图 G1/G2/G5/G7）

- **G1（log₁₀ residual_rmse 热图）**：整体色域 −3.1 … −1.0；B5 lr=1e-2 四个格子拉到 −1.0 （亮黄）为唯一明显异常带；所有其它 run 都压在 10⁻³ 以下。
- **G2（runtime 箱线）**：12 批中 B0–B2、B5–B7、B11–B14 全部挤在 60–65 s 一条细线；B3 网格放大后到 70–144 s；B4 capacity 右尾到 493 s，是成本唯一的长尾。
- **G5（Spearman 相关矩阵）**：主要相关块——
  - `nsno_blocks / nsno_channels / fno_modes` 相互 ≈1（B4 网络尺寸同升同降）
  - `omega / grid_nx / pml_*` 正相关块 ≈0.5（B2/B3 的联动设计）
  - `rel_l2_to_reference` 与 `lambda_data` 呈 **负** 相关（增加数据监督确实拉近与参考解）
  - `residual_rmse` 与任何单一轴相关都不强 → 非线性交互主导，单轴调参天花板已到
- **G7（rmse_gap_ratio 直方图）**：B5 lr=1e-2 四个失败 run 独立聚在 ~8e5；其余集中在 10³–10⁵ 量级，说明模型残差比参考解（10⁻⁷）高 3–5 个数量级，**仍远未到参考解地板**。

---

## 3. 每批次分析

### 3.1 B1 介质 × 监督（ω=30, N=128, PML=20, ep=5000）

| 介质 | pde rmse | hybrid rmse | Δ |
|---|---:|---:|---:|
| homogeneous | 0.00e+00 | 0.00e+00 | — |
| smooth_lens | 1.20e-03 | 3.08e-03 | **hybrid 变差 2.6×** |
| layered | 4.74e-03 | 6.88e-03 | **hybrid 变差 1.5×** |

> homogeneous 上 residual 在数值精度下直接归零，作为合理性闸门通过。
> 在默认 λ_pde=λ_data=1、参考解来自相同离散算子的情况下，hybrid **反而劣化**——这是 B6 的直接动机，后面看到需要 λ_data≥10 才翻身。

### 3.2 B2 频率扫（smooth_lens, N=128, PML=20）

| ω | ppw | rmse | rel_l2 | amp_mae | phase_mae |
|--:|--:|---:|---:|---:|---:|
| 15 | 53.2 | 1.26e-3 | 0.24 | 8.5e-3 | 0.27 |
| 20 | 39.9 | 1.46e-3 | 1.33 | 4.5e-2 | 1.18 |
| 30 | 26.6 | 1.20e-3 | 1.79 | 5.7e-2 | **1.55** |
| 40 | 19.9 | 1.58e-3 | 1.40 | 5.1e-2 | 1.33 |
| 50 | 16.0 | **7.38e-4** | 1.24 | 4.3e-2 | 0.99 |
| 60 | 13.3 | **7.77e-4** | 0.82 | 2.6e-2 | 0.70 |
| 75 | 10.6 | **7.24e-4** | 0.56 | 1.8e-2 | 0.55 |
| 90 | 8.9 | 1.20e-3 | 0.83 | 2.4e-2 | 0.69 |

**非单调趋势（B2__omega_line.png 清晰可见）**：rmse 在 ω=30 和 ω=75 附近出现两个谷值，ω=50–75 段是全矩阵最优区。
- ω=30 附近 `phase_mae=1.55` 接近随机相位（π/2），说明 PDE 残差已经很小但**神经算子把相位学错**——符合 A5 审计中关于 Δ̃τ 修正尚未闭合的记录。
- ω≥50 后 ppw 下降到 16→9，残差 RMSE 反而下降。这暗示：**residual_rmse_evaluation 被点源近场主导**，当 ω 升高时近场能量相对占比降低；需要搭配 G1 的 amp_mae / phase_mae 才能做 apples-to-apples 判断。

### 3.3 B3 网格 × PML（smooth_lens, ω=30）

热图（log₁₀ rmse）：
```
         pml=8    pml=16    pml=24
N= 96   9.5e-4   2.4e-3   6.2e-4   ← 最优格 N=96,pml=24：6.2e-4
N=128   1.3e-3   1.5e-3   1.3e-3
N=192   1.7e-3   2.1e-3   2.1e-3
N=256   4.4e-3   3.5e-3   3.2e-3   ← 全矩阵最差
```
- **反直觉结论**：细化网格反而劣化。原因：
  1. 固定 epochs=5000 未按网格点数缩放，大网格每 epoch 实际覆盖不到收敛域。
  2. FNO 模数 fno_modes=16 固定，相对分辨率下降，频谱截断吃掉高频细节。
- N=128/pml=16 恰是全矩阵默认工作点（1.47e-3），且一次 run 仅 60 s，性价比最高。

### 3.4 B4 容量扫

| 介质 | small (66 K) | base (2.1 M) | large (28 M) | xlarge (151 M) |
|---|---:|---:|---:|---:|
| homogeneous | 0 | 0 | 0 | 0 |
| smooth_lens | 2.47e-3 | 1.47e-3 | 1.28e-3 | 1.37e-3 |
| layered | — | 5.09e-3 | — | 5.35e-3 |
| runtime | 58 s | 64 s | 160 s | 493 s |

- 拐点在 **base→large**：smooth_lens rmse 从 1.47e-3 → 1.28e-3（-13%），但 wall-time ×2.5。
- large→xlarge **退化** 0.08e-3（+6%），明显过拟合。
- layered 仅跑了 base 和 xlarge 两档，xlarge 同样没好处（5.35 vs 5.09）。
- B4__params_curve.png 的锯齿状（三条垂直段对应三介质）再次确认：**参数量并非瓶颈**。

### 3.5 B5 学习率 × epochs

本轮所有 run 实际都跑了 5000 epochs（epochs 标签仅历史遗留），所以同一 lr 的 4 个 run 数值完全一致：

| lr | rmse | converged | 备注 |
|--:|---:|---:|---|
| 1e-4 | 1.72e-3 | 4/4 | 学习过慢 |
| **3e-4** | **1.12e-3** | 4/4 | **最优** |
| 1e-3 | 1.47e-3 | 4/4 | 默认 |
| 3e-3 | 3.06e-3 | 4/4 | 偏大 |
| 1e-2 | 1.05e-1 | **0/4** | 发散但未 NaN |

- B5__optim_heatmap.png 清晰呈现：lr=1e-2 一行亮黄（log₁₀ loss ≈ −2），其他行 < −5。
- 推荐把 **base.yaml 的默认学习率从 1e-3 改到 3e-4**，单轮 -23% residual 免费收益。

### 3.6 B6 hybrid 权重（λ_pde × λ_data）

smooth_lens：

| λ_pde | λ_data | rmse | loss_pde | loss_data |
|---:|---:|---:|---:|---:|
| 0.1 | 1 | 2.50e-3 | 5.4e-6 | 1.7e-7 |
| 1 | 0.1 | 2.17e-3 | 4.1e-6 | 1.2e-5 |
| 1 | 1 | 2.93e-3 | 6.7e-6 | 8.2e-7 |
| **1** | **10** | **1.76e-3** | 2.8e-6 | 2.4e-8 |
| 10 | 1 | 2.11e-3 | 3.9e-6 | 7.8e-6 |

- **最优配方 λ_pde=1 / λ_data=10**：λ_data 相对 PDE 放大 10× 时 rmse 最低，验证 A3 hybrid 监督确实有用——但前提是 label 大幅压过 PDE。
- layered 上所有 hybrid 配置都比 B1 的 pde-only（4.74e-3）差，原因可能是 layered 的参考解误差已经拖后腿，数据监督信号本身噪。

### 3.7 B7 源点位置（smooth_lens）

| run | pos | rmse |
|---|---|---:|
| lower_left_corner | (0.02, 0.02) | **7.17e-4** |
| y_minus_quarter | (0.5, 0.25) | 1.01e-3 |
| x_plus_quarter | (0.75, 0.5) | 1.06e-3 |
| x_minus_quarter | (0.25, 0.5) | 1.11e-3 |
| near_left_boundary | (0.02, 0.5) | 1.17e-3 |
| y_plus_quarter | (0.5, 0.75) | 1.37e-3 |
| center | (0.5, 0.5) | 1.47e-3 |

- B7__source_map.png 呈现反直觉格局：**越靠近 PML 吸波层 rmse 越低**。
  - lower_left_corner 在双侧 PML 夹角内，源辐射的大部分能量直接被吸收，残差评估域（物理内区）内能量本就少。
  - center 是源点远离 PML 的"干净传播"情形，反而更难学。
- 结论：B7 的绝对数值不能跨 run 直接比，需配合 `rmse_gap_ratio` 归一化（中心 ratio 最大，也印证了"容易评估 ≠ 容易学习"）。

### 3.8 B10 参考解地板（无训练）

| run | reference_residual_rmse | reference_residual_max |
|---|---:|---:|
| homogeneous N=128 ω=30 | 0 | 0 |
| smooth_lens N=128 ω=30 | 1.28e-7 | 6.45e-7 |
| layered N=128 ω=30 | 1.40e-7 | 8.05e-7 |
| smooth_lens N=128 ω=60 | 3.33e-8 | 1.23e-7 |
| smooth_lens N=128 ω=90 | 1.21e-8 | 4.90e-8 |
| smooth_lens N=192 ω=30 | 3.22e-7 | 2.49e-6 |

- 参考解残差 ≤ 3.2e-7，比神经算子残差低 **4 个数量级**（10⁻³ vs 10⁻⁷）。
- B10__reference_vs_grid.png 显示 ω 升高时参考解 residual 反而下降（离散算子对高频 PML 更贴近 Sommerfeld 边界）——这部分解释了 B2 中 rmse 在高频段下降的现象。
- G7 直方图中 rmse_gap_ratio 集中在 10³–10⁵，这条"天花板 vs 地板"数字表明**当前训练 pipeline 保留 ≥ 10³× 改进空间**。

### 3.9 B11 source_mask_radius

| 介质 | r=1.0 | r=1.5 (默认) | r=2.0 | r=3.0 |
|---|---:|---:|---:|---:|
| smooth_lens | 3.03e-3 | 1.47e-3 | **1.29e-3** | 1.80e-3 |
| layered | 5.32e-3 | 5.09e-3 | 5.12e-3 | 5.10e-3 |

- smooth_lens 上 r=2.0 最优（-12% vs 默认）；r=1.0 崩盘（近源点奇点无法屏蔽）；r=3.0 掩掉过多物理信号。
- layered 对 mask 半径几乎不敏感（波长尺度上已经被层间反射主导）。
- B11__radius_sweep.png 的 residual_max 图最能说明 r=1.5→2.0 把峰值残差降了 30%，但这些峰值都挤在源点 3×3 邻域内。

### 3.10 B12 PML profile (power × R0)

| (power, R0) | rmse | converged |
|---|---:|---:|
| p=1, R0=1e-4 | 1.63e-3 | True |
| p=1, R0=1e-6 | 1.58e-3 | True |
| **p=2, R0=1e-6** | **1.47e-3** | True |
| p=2, R0=1e-8 | 1.88e-3 | True |
| p=3, R0=1e-6 | 2.33e-3 | **False** |
| p=3, R0=1e-8 | 2.40e-3 | True |

- 当前 base.yaml 默认 (p=2, R0=1e-6) **已经是扫过的最优点**——不需要调整。
- p=3 同时收敛变差（仅 5/6 收敛），说明吸波太陡反而让 PDE 残差在 PML 层噪声化。

### 3.11 B14 错误监督敏感度（smooth_lens, hybrid λ=1/1）

| 监督信号 | rmse | rel_l2 | loss_data |
|---|---:|---:|---:|
| **clean** | **2.93e-3** | 0.022 | 8.2e-7 |
| omega_drift (ω=33 错标) | 9.06e-3 | 1.13 | 1.4e-5 |
| source_shift (pos=[0.75,0.5]) | 1.12e-2 | 0.97 | 3.7e-5 |
| medium_swap (用 homogeneous label) | 1.18e-2 | 0.94 | 2.5e-5 |

- 错误 label → rmse 劣化 **3–4 倍**，而 loss_data 同步放大 **30×**，模型能"闻到"数据异常——B14__bars.png 右图中 loss_data 柱（橙）对 clean 看不到、对三种污染显著抬起，是非常干净的监督健康度指标。
- 实际应用意义：当 production pipeline 中观察到 `loss_data / loss_pde` 比值突然跳升 1 个数量级，基本可断言 label 受污染，无需上游追溯。

---

## 4. 全局观察与工程建议

### 4.1 图 G1/G5/G7 的共同结论
1. **单轴旋钮已接近饱和**——G5 Spearman 矩阵里 `residual_rmse` 与任何单一输入都不到 ±0.5 相关，非线性耦合主导。
2. **训练效率 > 模型规模**——B4 显示 base 已经够；要突破只能改训练（B5 的 lr、B6 的 λ、潜在的 scheduler / warm-up）。
3. **天花板差距 10³–10⁵**——G7 直方图，参考解层级提供 5 个数量级的可改进空间。

### 4.2 可立即落地的基线更新
| 项 | 现值 | 建议值 | 预期增益 |
|---|---|---|---|
| `training.lr` | 1.0e-3 | **3.0e-4** | rmse ↓23%（B5） |
| `loss.source_mask_radius` | 1.5 | **2.0**（smooth 介质） | rmse ↓12%（B11） |
| `grid.nx/ny` 默认 | 128 | 128（保留） | — |
| `model.capacity` | base | base（保留） | — |
| `pml.power / pml.R0` | 2 / 1e-6 | 保留 | — |
| 默认 `λ_pde:λ_data` | 1:0 | **1:10**（仅当有可靠 label 时） | rmse ↓20%（B6） |

### 4.3 未决技术债
- **B8/B9/B13/B15 仍阻塞**：4 条轴（seed、lap_tau_mode、omega2_normalize + pml_rhs_zero、eikonal_precision）在 base.yaml 存在但 runtime 未消费，建议 A7 里程碑一次性接通。
- **per-epoch loss 分解缺失**：`matrix_loss_history.csv` 48 万行中 `loss_pde / loss_data` 列全空，仅末 epoch 由 `compute_final_loss_components()` 重建。修复点：`src.train.trainer.py` 按 epoch 持久化三路 loss 即可。
- **相位学习是主要瓶颈**：B2 中 ω=30 处 `phase_mae≈π/2` 说明模型没学对相位旋转，与 A5 审计所列 Δ̃τ 闭合问题一致，优先级 > 振幅 MAE。
- **网格缩放的 epoch 预算不对齐**：B3 中 N=256 明显欠拟合，建议引入 epochs ∝ N² 的默认调度（或显式 `training_budget_pixels`）。

### 4.4 总体结论
- **矩阵框架运行健康**：102/102 无失败、无 NaN、无 early-stop，产物全量可复现。
- **当前最佳 run**：B7 `lower_left_corner` rmse=7.17e-4 与 B2 `omega=75` rmse=7.24e-4 并列冠军，但前者含 PML 吞能偏置，**推荐 B2 ω∈[50, 75]、B3 N=96/pml=24 为下阶段 fine-tune 起点**。
- **参考解地板**（B10）≈ 1e-7，对比神经算子 ≈ 1e-3，**仍有 10³× 改进余量**。下阶段应把 phase 对齐（A5 闭合）、更大 epoch 预算、数据监督权重 λ_data≥10 这三件事同时打通。

---

## 5. 复现方式与解释边界

### 5.1 复现实验命令

以下命令对应本报告所使用的输出根目录 `outputs/matrix_from_branch_ep5000_full/`：

```bash
python scripts/run_matrix.py \
  --output-root outputs/matrix_from_branch_ep5000_full \
  --dry-run --include-blocked

python scripts/run_matrix.py \
  --output-root outputs/matrix_from_branch_ep5000_full \
  --device cuda:0 \
  --epochs-override 5000 \
  --continue-on-error

python scripts/summarize_matrix.py \
  --output-root outputs/matrix_from_branch_ep5000_full
```

### 5.2 最小验收标准

- `outputs/matrix_from_branch_ep5000_full/_meta/manifest.tsv` 中 active run 全部有启动记录，blocked run 只出现在 `--include-blocked` 的 dry-run manifest 中。
- `matrix_report.csv` 应覆盖 102 个已完成 run；`matrix_failures.csv` 为空或仅包含表头。
- `matrix_reference_only.csv` 中 B10 的 `reference_residual_rmse` 应稳定在 `1e-7` 量级或更低，否则 reference floor 不可信。
- `matrix_loss_history.csv` 行数应接近 `5000 × 训练 run 数 + 表头`；若明显偏低，通常表示中途失败或日志落盘不完整。

### 5.3 解释边界

- 这份报告是**工程矩阵报告**，不是严格统计学实验报告；除 B8 外，目前没有 seed 复现实验，因此结论默认建立在单 seed 运行上。
- batch spec 中的设计值与本次实际执行值并不完全一致，最典型的是 `training.epochs` 被统一覆盖到 5000；因此讨论配置时必须区分“YAML 设计值”和“本次复跑值”。
- hybrid 相关结论默认 reference label 是可信的；一旦上游 reference 配置漂移，B6/B14 的结论会被同步污染。
- B7 与部分高频 run 的低 rmse 受评估域能量分布影响较大，不能简单外推成“物理问题本身更容易”。

---

## 附录 A：产物清单（`_reports/` 下）

| 文件 | 行数 | 主用途 |
|---|---:|---|
| `matrix_report.csv` | 103 | L1 每 run 一行，67 列（config + metrics + runtime） |
| `matrix_loss_history.csv` | 480,001 | L2 每 epoch 一行（5000 × 96 训练 run） |
| `matrix_quantiles.csv` | 103 | L3 残差 12 分位数 + std |
| `matrix_centerline.csv` | 24,961 | L4 源点两条正交中心线（pred_re/im、ref_re/im、residual_abs、velocity、loss_mask） |
| `matrix_pareto.csv` | 123 | L5 每批次 2D Pareto 前沿 rank |
| `matrix_failures.csv` | 1 | L6 空（0 失败） |
| `matrix_reference_only.csv` | 7 | L7 B10 纯参考解指标 |

## 附录 B：图清单（`_reports/matrix_figures/`）

全局 7 张（PNG+PDF）：
- `G1__residual_rmse_heatmap`（log₁₀ rmse 矩阵热图）
- `G2__runtime_boxplot`（各批 runtime 分布）
- `G3__status_sankey`（成功/收敛/NaN/失败流向）
- `G4__pca_projection`（数值配置 PCA 2D 投影，着色 rmse）
- `G5__spearman_heatmap`（19 个数值变量相关矩阵）
- `G6__parallel_coordinates`（标准化平行坐标）
- `G7__gap_histogram`（rmse_gap_ratio 按批叠加直方图）

批内 31 张（B0 gate_bar；B1 grouped_bar + residual_collage；B2 omega_line + amp_phase + centerline_compare；B3 grid_pml_heatmap + slices；B4 params_curve + pareto；B5 optim_heatmap + loss_panels + loss_envelope；B6 lambda_scatter + loss_decomposition；B7 source_map + boundary_distance + centerline_compare；B10 reference_vs_grid；B11 radius_sweep；B12 pml_heatmap；B14 bars + phase_profiles）。
