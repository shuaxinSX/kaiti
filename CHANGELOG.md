# CHANGELOG

## [0.1.0] - 2026-04-22

Phase 1（2D 原型）核心管线完成，并叠加四轮并行开发与一轮实验矩阵。

### M0-M9 — 核心管线

- **M0** 建仓与最小骨架，`DEV_PLAN.md` 冻结决策
- **M1** `Grid2D` / `Medium2D` / `PointSource` / 复数工具
- **M2** 汉克尔背景场 `u₀ = (i/4)H₀⁽¹⁾(ωs₀r)` 及解析先验
- **M3** 因子化 Eikonal FSM 求解器（τ = τ₀·α）
- **M4** 安全导数重组（∇τ, Δτ 半解析链式法则）
- **M5** PML 复坐标拉伸 + 固定卷积核中心差分
- **M6** 等效体源与穿孔掩码（`source_mask_radius`）
- **M7** `SpectralConv2d` + `NSNO2D` 网络骨架
- **M8** PDE residual 前向检查
- **M9** 最小训练实验

### 关键物理修复（M9 之后）

- `s0` 改为取震源位置慢度 `s(x_s)`
- `residual.py` PML 系数保持复数域，输运项正确拉伸
- RHS 震源防爆半径接入 `cfg.loss.source_mask_radius`
- PML `sigma_max` 采用 `1/s0` 作 PML 波速
- Godunov FSM 与链式法则制造解测试

### A1 — Packaging / CI

- `pip install -e ".[dev]"` 作为唯一安装路径
- 移除 `scripts/*.py` 的 `sys.path` hack，统一通过安装包导入
- `pyproject.toml` 为依赖唯一事实源
- `configs/` 加载支持多 overlay
- 最小 GitHub Actions CI（install + pytest + smoke）

### A2 — 参考解与评估

- `src/eval/reference_solver.py` 组装稀疏 Helmholtz 算子 + `spsolve`
- `src/eval/reference_eval.py` 生成参考对照 CSV/JSON/热力图
- `scripts/solve_reference.py` CLI（config 模式 + run-dir 模式）
- `evaluate_run.py` 集成参考指标（`reference_residual_rmse` 等）

### A3 — 混合损失

- `src/train/supervision.py` 载入 reference 作为 data 目标
- `Trainer.train()` 保持返回契约，同时支持 `pde-only` / `data-only` / `hybrid`
- `configs/hybrid.yaml` 训练 overlay 示例
- `config.py` fail-fast 校验 supervision 配置

### A5 — Residual / PML 审计

- `src/physics/residual.py` 新增 `lap_tau` 审计模式
- 默认 `mixed_legacy` 对 A2 参考解逐元素等价
- `stretched_divergence` 作为审计候选保留（仅 programmatic 入口）
- 覆盖测试：legacy vs candidate 交叉损失对照

### A6 — B0-B15 实验矩阵

- `configs/experiments/B0_smoke.yaml` 至 `B15_eikonal_precision.yaml`（16 个 sweep overlay）
- `scripts/run_matrix.py` / `run_matrix_from_branch.py` 矩阵扫查
- `scripts/run_benchmark.py` 单格 benchmark 入口
- `scripts/summarize_matrix.py` 汇总 CSV / MD / 图
- 首份实验报告 `reports/matrix/EXPERIMENT_REPORT_2026-04-22_03c409a.md`

### 工程约定

- `.gitignore` 追加 `.vendor_sx/`（服务器 vendored 依赖）与 `*.cmd`（Windows 启动器）
- 报告采用快照命名：`EXPERIMENT_REPORT_<日期>_<commit>.md`
- 多 agent 历史文档归档至 `docs/archive/multi-agent-2026-04/`

## [Unreleased]

- `residual.lap_tau_mode` 现已通过 `configs/base.yaml` / `Trainer` 接入运行路径。
- 默认残差装配切换为严格 PML 拉伸的 `stretched_divergence`；`mixed_legacy` 保留为显式审计/历史回放模式。
