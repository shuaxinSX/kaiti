# CHANGELOG

> 与 `int/multi-agent` 分支对齐。每个条目的事实依据见 `PROJECT_STATE.md` 与 `handoffs/`。

## [Unreleased] — int/multi-agent @ b80dc88

### Added

- **A3 hybrid 监督支持**（`feat/hybrid-loss`）
  - `src/train/supervision.py`：新增 `resolve_supervision_config`、`load_reference_target`、`complex_array_to_dual_channel_tensor`
  - `configs/experiments/hybrid.yaml`：hybrid/data-only 字段模板（默认仍为 PDE-only）
  - 新配置字段：`training.supervision.enabled`、`training.supervision.reference_path`
  - 新 Trainer 字段：`supervision_enabled`、`reference_path`、`reference_target`、`loss_history_total`、`loss_history_pde`、`loss_history_data`、`last_step_metrics`
  - 新测试：`tests/test_train.py` 覆盖 PDE-only/data-only/hybrid 三种模式与 5 条 fail-fast 规则
- **A2 参考解评估链**（`feat/reference-eval`）
  - `src/eval/reference_eval.py`：`compute_prediction_envelope`、`export_reference_artifacts`、`solve_reference_from_config`、`solve_reference_from_run_dir`、常量 `REFERENCE_SUMMARY_KEYS`
  - `src/eval/reference_solver.py` 收归到 `src/eval/__init__.py` 的对外出口
  - `scripts/solve_reference.py`：支持 `--config` 与 `--run-dir` 互斥模式
  - `evaluate_saved_run` 自动补写 `reference_envelope.npy` / `reference_wavefield.npy` / `reference_metrics.json` / `reference_comparison.csv` / `reference_error_heatmaps.png`
  - `summary.json` 新增 `reference_available` / `rel_l2_to_reference` / `amp_mae_to_reference` / `phase_mae_to_reference` / `reference_residual_rmse` / `reference_residual_max`
  - 评估区域固定为 `evaluation_mask = physical_mask & loss_mask.astype(bool)`
  - 新测试：`tests/test_reference_solver.py`
- **A1 打包与 CI**（`feat/packaging-ci`）
  - `pyproject.toml` 成为打包唯一事实源，`helmholtz-train` 入口注册到 `[project.scripts]`
  - `src/config.load_config` 支持 `load_config(base, overlay1, overlay2, ...)` 多 overlay（左到右合并，`None` 跳过）
  - `.github/workflows/ci.yml`：最小 install + `pytest -q` + 1-epoch 训练 smoke
  - 新测试：`tests/test_config.py`

### Changed

- `src/train/trainer.py`：
  - `Trainer.__init__` 在构造期一次性快照 `lambda_pde` / `lambda_data`，`train()` 不再动态回读
  - 模型构造前后 `torch.manual_seed(0)` 存储/恢复 RNG，避免全局随机状态干扰
  - `train()` 仍返回 `list[float]`（total loss 历史），保持与既有 runner 调用契约一致
  - 日志同步输出 `loss_total` / `loss_pde` / `loss_data` 三列
- `src/train/runner.py`：`evaluate_saved_run` 在一次预测中同时产出评估产物与参考解比较，消除重复 forward
- `requirements.txt` 降级为兼容壳，真正的依赖清单来自 `pyproject.toml`
- `scripts/run_train.py` / `scripts/evaluate_run.py` 移除 `sys.path` hack，依赖 editable install 解析包

### Tests

- 当前 `tests/` 共 16 个测试文件：物理层（9 个）、网络（1 个）、config（1 个）、runner（1 个）、训练循环与监督（1 个）、参考解（1 个）、基础引导（2 个：`test_bootstrap.py` / `test_complex_utils.py`）
- `tests/test_runner.py` 锁定 20 个产物文件名，重命名会挂测试（详见 `PROJECT_STATE.md §5.1`）

### Notes

- `run_training` 的默认路径**不会**自动求参考解；参考解由 `evaluate_saved_run` 或 `solve_reference.py` 显式调用
- CLI 尚未暴露 `--reference-path`；hybrid/data-only 必须通过 YAML overlay 或程序化 cfg 注入
- `resolve_output_dir` 以 `exist_ok=False` 保护产物，重复运行同一目录前需要手动 `rm -rf`
- Δ̃τ / PML 闭合性待 A5 物理审计复核
- 已完成里程碑：M0 – M9（详见 `DEV_PLAN.md §5`、`PROJECT_STATE.md §3`）

---

## [0.0.0] — 初始开发规范（历史入口）

### Added

- `DEV_PLAN.md`：完整开发规范，含 10 个里程碑、13 条冻结决策、防爆清单
- `CHANGELOG.md`：本文件
- `WORKTREE_BOARD.md`：多 agent 并行工作面板（A0 维护）

### Notes

- 2D 原型范围已冻结，见 `DEV_PLAN.md` 第 2 节
