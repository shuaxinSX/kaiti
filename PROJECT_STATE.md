# PROJECT_STATE.md — 当前集成基线快照

> 本文档反映 `int/multi-agent` 分支在本次快照时的实际代码状态，是 README / CHANGELOG / DEV_PLAN 的事实底稿。
> 快照口径：`int/multi-agent @ b80dc88`（A1 + A2 + A3 合并完成，A4 正在同步文档）。

---

## 1. 快照基本信息

| 项 | 值 |
|----|----|
| 集成分支 | `int/multi-agent` |
| 快照 commit | `b80dc88` |
| 快照日期 | 2026-04-19 |
| 活跃文档分支 | `chore/project-state`（本次 A4 工作分支） |
| Python 版本要求 | `>=3.9,<3.10` |
| 测试入口 | `pytest -q`（共 16 个测试文件） |
| 训练入口 | `helmholtz-train` 或 `python scripts/run_train.py` |

---

## 2. 一句话定位

一个 2D 标量 Helmholtz 高波数散射的神经算子原型：物理预处理（Float64）→ NSNO 网络（Float32）→ PDE 残差 / 可选监督损失 → 训练产物 + 参考解评估。

---

## 3. 已集成能力清单

### 3.1 物理预处理管线（M0 – M6）

- `src/core/`：`Grid2D`、`Medium2D`、`PointSource`、复数工具
- `src/physics/`：
  - `background.py`：2D 汉克尔格林函数 u₀（含 `r_safe` 防奇点）
  - `eikonal.py`：因子化 Eikonal 的 FSM 求解器 + 解析烫印
  - `tau_ops.py`：链式法则安全重组的 ∇τ、Δτ
  - `pml.py`：PML 复坐标拉伸张量 A/B（γ' 用解析式）
  - `diff_ops.py`：固定卷积核的一阶/二阶差分引擎
  - `rhs.py`：等效体源 RHS_eq（含 2D π/4 相位补偿、穿孔掩码）
  - `residual.py`：PDE 残差计算器（含阻尼池解耦、ω² 归一化）

### 3.2 网络骨架（M7）

- `src/models/spectral_conv.py`：FNO 频域截断卷积（含零填充拓扑修复）
- `src/models/nsno.py`：NSNO2D 完整网络，输出层零初始化

### 3.3 训练与最小 overfit（M8 – M9）

- `src/train/trainer.py`：
  - `Trainer.__init__` 会在构建时快照 `lambda_pde` / `lambda_data`，此后在 `train()` 中不再重新读取
  - 模型初始化前后 `torch.manual_seed(0)` 打桩，保证网络初态可复现
  - `train()` 每个 epoch 同时记录 `loss_history_total` / `loss_history_pde` / `loss_history_data` 与 `last_step_metrics`，并保持旧的 `list[float]` 返回契约（返回的是 total 历史）
- `src/train/losses.py`：`loss_data`、`loss_total`（按 D12 冻结决策）
- `src/train/supervision.py`（A3 新增）：
  - `resolve_supervision_config`：解析 `training.supervision.{enabled, reference_path}`，并对 `lambda_data > 0` 的非法组合抛错
  - `load_reference_target`：读盘 + 校验 shape/dtype，加载为 `[1, 2, H, W]` float32 张量
  - `complex_array_to_dual_channel_tensor`：复数 `[H, W]` → float32 `[1, 2, H, W]` 的统一通道转换
- `src/train/runner.py`：`run_training`、`evaluate_saved_run`、`run_train.py` 的 CLI 实现；CLI 接受 `--config` / `--overlay` / `--device` / `--epochs` / `--output-dir` / `--velocity-model`，**当前不提供 `--reference-path`**

### 3.4 参考解与评估（A2）

- `src/eval/reference_solver.py`：`assemble_reference_operator` + `solve_reference_scattering`（离散算子稀疏直解，作为一致性基线，**不是外部解析真值**）
- `src/eval/reference_eval.py`：`compute_prediction_envelope`、`export_reference_artifacts`、`solve_reference_from_config`、`solve_reference_from_run_dir`、常量 `REFERENCE_SUMMARY_KEYS`、`_COMPARISON_PHASE_EPS=1e-8`、`_COMPARISON_NORM_EPS=1e-12`
- 产物在 `evaluate_saved_run` 中自动补齐：`run_training` 的默认路径**不会**自动求参考解
- 评估区域一律是 `evaluation_mask = physical_mask & loss_mask.astype(bool)`

### 3.5 打包 / CI（A1）

- `pyproject.toml` 是唯一的打包事实来源；`requirements.txt` 退化为兼容壳
- `pip install -e ".[dev]"` 同时装运行依赖和 `pytest` / `pytest-cov`
- 入口脚本 `helmholtz-train` 注册在 `[project.scripts]` 下，映射到 `src.train.runner:main`
- `src/config.load_config` 支持多 overlay：`load_config(base, overlay1, overlay2, ...)`，`None` 会被跳过，从左到右合并
- `.github/workflows/ci.yml`：最小 install + pytest + 1-epoch 训练 smoke 流水线

---

## 4. 统一命令手册

> 所有命令前置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 以抑制 OpenMP/MKL 线程爆增（debug 网格下推荐）。

### 4.1 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4.2 测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
```

### 4.3 训练（PDE-only 默认路径）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
helmholtz-train \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir outputs/smoke_pde_only
```

### 4.4 评估（为已有 run 目录补算 CSV / 热力图 / 参考解比较）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/evaluate_run.py \
  --run-dir outputs/smoke_pde_only \
  --device cpu
```

### 4.5 参考解（config 模式和 run-dir 模式二选一）

```bash
# config 模式：独立生成 envelope/wavefield/metrics，不比较预测
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/solve_reference.py \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --device cpu \
  --velocity-model smooth_lens \
  --output-dir outputs/reference_smooth_lens

# run-dir 模式：使用 run 内的 config_merged.yaml + model_state.pt 生成比较产物
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/solve_reference.py \
  --run-dir outputs/smoke_pde_only \
  --device cpu
```

### 4.6 Hybrid / data-only 监督训练（A3）

- CLI 没有 `--reference-path`；目前必须通过 YAML overlay 或程序化 cfg 修改注入
- 样例 overlay `configs/experiments/hybrid.yaml` 默认仍为 PDE-only（`lambda_data=0.0`、`supervision.enabled=false`），是作为字段模板，不是可直接跑的 hybrid 配置

```yaml
training:
  lambda_pde: 1.0
  lambda_data: 0.5
  supervision:
    enabled: true
    reference_path: outputs/reference_smooth_lens/reference_envelope.npy
```

---

## 5. 产物契约（以 int/multi-agent 为准）

### 5.1 `run_training` 默认写出（由 `tests/test_runner.py` 锁定，**任何重命名都会挂测试**）

```
config_merged.yaml
summary.json
losses.npy
losses.csv
loss_tail.csv
loss_curve.png
loss_curve_log.png
wavefield.npy
wavefield.png
model_state.pt
metrics_summary.csv
quantiles.csv
centerline_profiles.csv
x_coords_physical.csv
y_coords_physical.csv
velocity_physical.csv
wavefield_magnitude_physical.csv
residual_magnitude_physical.csv
field_heatmaps.png
residual_heatmaps.png
```

### 5.2 已导出但未被测试断言的文件（仍需维持存在）

- `loss_mask_physical.csv`
- `scattering_magnitude_physical.csv`

### 5.3 `evaluate_saved_run` / `solve_reference.py` 追加写入

```
reference_envelope.npy         # complex [ny_total, nx_total]，含 PML
reference_wavefield.npy        # complex [ny_total, nx_total]
reference_metrics.json
reference_comparison.csv       # 仅在 run-dir 模式（存在模型预测）时生成
reference_error_heatmaps.png   # 同上
```

### 5.4 `summary.json` 键（增量由 A2 贡献）

新增字段（`REFERENCE_SUMMARY_KEYS`）：

- `reference_available`
- `rel_l2_to_reference`
- `amp_mae_to_reference`
- `phase_mae_to_reference`
- `reference_residual_rmse`
- `reference_residual_max`

实测参考值（debug 网格 + smooth_lens）：`reference_residual_rmse ≈ 7.21e-08`，`reference_residual_max ≈ 2.62e-07`。

---

## 6. 配置与依赖事实

### 6.1 `configs/base.yaml` 顶层键

`physics`、`grid`、`medium`、`pml`、`eikonal`、`model`、`loss`、`training`、`logging`。

### 6.2 `configs/debug.yaml` 覆盖项

- `physics.omega: 10.0`
- `grid.nx=ny=32`
- `pml.width=5`
- `eikonal.fsm_max_iter=20`、`fsm_tol=1e-8`
- `model.nsno_blocks=2`、`nsno_channels=16`、`fno_modes=8`
- `training.epochs=50`
- `logging.level="DEBUG"`

### 6.3 `configs/experiments/hybrid.yaml` 字段模板

- 新增 `training.supervision.enabled` + `training.supervision.reference_path`
- 默认仍为 PDE-only，只把字段摆在显眼位置

### 6.4 依赖范围（由 `pyproject.toml` 决定）

| 依赖 | 约束 |
|------|------|
| numpy | `>=1.24,<2.0` |
| scipy | `>=1.10,<2.0` |
| PyYAML | `>=6.0,<7.0` |
| matplotlib | `>=3.7,<4.0` |
| torch | `>=2.7,<2.8` |
| tqdm | `>=4.65,<5.0` |
| pytest | `>=7.0,<9.0` (dev) |
| pytest-cov | `>=4.0,<6.0` (dev) |

---

## 7. 测试覆盖现状

`tests/` 下共 16 个文件：

| 测试 | 覆盖面 |
|------|--------|
| test_bootstrap.py | 仓库引导 |
| test_grid.py | Grid2D |
| test_medium.py | Medium2D / PML 退化 |
| test_complex_utils.py | 复数工具 |
| test_background.py | 汉克尔背景场 |
| test_eikonal.py | FSM + 烫印 |
| test_tau_ops.py | ∇τ / Δτ 重组 |
| test_pml.py | σ / γ / A / B |
| test_diff_ops.py | 一/二阶差分核 |
| test_rhs.py | 等效体源 + 掩码 |
| test_residual.py | 残差 + 阻尼池 |
| test_model_forward.py | NSNO 前向 |
| test_config.py | 多 overlay 合并（A1） |
| test_runner.py | run_training + evaluate_saved_run 产物契约 |
| test_train.py | Trainer 训练 + 监督 fail-fast（A3 新增 6 条） |
| test_reference_solver.py | 参考解 + 评估产物（A2） |

---

## 8. 已知技术债

| 项 | 位置 | 影响 | 建议 |
|----|------|------|------|
| `resolve_output_dir` 硬编码 `exist_ok=False` | `src/train/runner.py:55` | 同名目录二次运行会抛错；自动化脚本必须先 `rm -rf` | 后续引入 `--overwrite` 开关或容忍标志 |
| CLI 无 `--reference-path` | `src/train/runner.py` / `scripts/run_train.py` | hybrid / data-only 只能走 YAML overlay 或程序化 cfg 改写 | 评估期后补加 flag |
| `lambda_pde` / `lambda_data` 在 `Trainer.__init__` 时快照 | `src/train/trainer.py:133` | 构造 Trainer 后修改 `cfg.training` 不会影响运行时权重 | 与 A0 约定；若要动态调度需重构 |
| `epochs=0` 边缘情况 | `trainer.train()` 尾部日志 | 不跑任何 step 时收尾日志会索引空 history | 在 runner / trainer 任选一处加守卫 |
| Δ̃τ 张量解耦未闭合 | `src/physics/pml.py` + `residual.py` | A5 待审计重点；当前 PML 内残差依赖全波衰减假设 | 留待 A5 物理审计结论落地 |
| `handoffs/` 位于仓库根 | 项目级目录污染 | 不影响打包，但会出现在 `ls` 和搜索结果中 | 等多 agent 流程完结后考虑归档到 `docs/handoffs/` |

---

## 9. 并行分支状态（与 `int/multi-agent` 的相对关系）

| Agent | 分支 | 状态 | 关键产物 |
|-------|------|------|---------|
| A0（集成者） | `int/multi-agent` | 活动中 | `--no-ff` 合入 A1/A2/A3，board §16 Gate 管控 |
| A1 | `feat/packaging-ci` | ✅ MERGED | `pyproject.toml`、多 overlay、CI |
| A2 | `feat/reference-eval` | ✅ MERGED | `src/eval/`、`solve_reference.py`、参考摘要键 |
| A3 | `feat/hybrid-loss` | ✅ MERGED | `src/train/supervision.py`、三路 loss 历史、fail-fast 规则 |
| A4（文档） | `chore/project-state`（本分支） | 🟡 进行中 | 本 PROJECT_STATE.md + README / CHANGELOG / DEV_PLAN 同步 |
| A5（物理审计） | `audit/physics-review`（规划） | ⏸ 未开始 | 基于 A2 基线做 Δ̃τ、PML、π/4 相位闭合审计 |
| A6（实验矩阵） | `exp/experiment-matrix`（规划） | ✅ READY | 消费 A2 的参考解、A3 的 hybrid/data-only 管线 |

### 依赖边

- A5 依赖 A2 的参考产物和 A3 的监督接口；A2 + A3 已合并，A5 可启动
- A6 依赖 A2 + A3 + A4 文档；A4 本次完成后 A6 进入"可启动"状态
- A4（本分支）**禁止**改动 `WORKTREE_BOARD.md`、`pyproject.toml`、`src/**`、`tests/**`、`configs/**`、`scripts/**`；只改文档

---

## 10. 推荐推进顺序

1. 先把 A4 的四份文档同步合入，锁定"当前快照"的定义
2. 启动 A5 物理审计：Δ̃τ 张量解耦闭合性 + PML 衰减充分性 + π/4 相位补偿在残差中的落点
3. A6 基于 A5 结论设计实验矩阵；消费 `solve_reference.py` 的产物作为 ground truth；消费 `hybrid.yaml` 模板展开 PDE-only / data-only / hybrid 对照实验
4. A6 结果回流后，再由 A0 汇总、tag `v0.1-m9-baseline`，进入 Phase 2（多波数 / 多源 / 3D）的独立规划
