# 高波数 2D Helmholtz 神经算子原型

本项目面向高波数二维 Helmholtz 方程求解，研究路线是：

```text
medium s(x), source
  -> Eikonal / WKBJ phase preprocessing
  -> tau, grad tau, Lap tau, incident field, equivalent RHS
  -> NSNO scattering-envelope network
  -> reconstructed total wavefield
  -> PDE residual, reference comparison, PML/interface diagnostics
```

核心问题不是普通调参，而是判断当前
`WKBJ phase stripping + scattering envelope + NSNO + strict PML residual`
是否能在高波数 regime 下同时控制：

- reference solver floor
- PDE residual
- phase / amplitude / relative error
- PML 和 physical-PML interface 一致性
- supervision、curriculum、网络容量对高频稳定性的影响

## 当前状态

项目已经完成三层实验资产。

| 阶段 | 范围 | 状态 |
|---|---|---|
| M0-M9 | 2D 网格、介质、点源、Eikonal、PML、RHS、NSNO、PDE residual、最小训练 | 已完成 |
| A1-A6 | packaging、reference solver、hybrid loss、strict PML residual、B0-B15 基础矩阵 | 已完成 |
| C0-C21 | 高波数主实验矩阵，覆盖 reference floor、ppw、PML、phase、capacity、curriculum、architecture audit | 已完成并归档 |

最新完整高波数矩阵来自 C0-C21：

- 1413 planned active rows
- 1412 completed summaries
- 1207 completed train runs
- 205 completed reference-only runs
- final missing run: 1
- known missing artifact: `C5_laptau_strictness/layered__omega180__N459__pml24__center__stretched_divergence`
- full matrix was run with `epochs_override=10000`

结论简述：当前 pipeline 能稳定跑到高波数设置，并且 reference floor 可信、strict PML residual 有配对证据支撑；但 PDE residual 与 phase/reference error 之间仍存在明显 tradeoff，curriculum、PML loss weighting 和简单 architecture antidote 尚未形成稳定收益。因此不能把最低 residual 直接解释为“高波数问题已经解决”。

## 方法组件

主要实现位于 `src/`：

| 模块 | 作用 |
|---|---|
| `src/core/` | `Grid2D`、`Medium2D`、`PointSource`、复数双通道工具、region masks |
| `src/physics/` | background field、Eikonal/FSM、tau derivatives、PML、finite differences、RHS、residual |
| `src/models/` | spectral convolution 和 NSNO block |
| `src/train/` | trainer、losses、reference supervision、curriculum/warm-start runner |
| `src/eval/` | finite-difference reference solver 和 reference comparison metrics |

当前默认 residual 路线是：

```yaml
residual:
  lap_tau_mode: stretched_divergence
```

历史审计模式 `mixed_legacy` 仍保留，用于 PML 消融和严格 PML 对照，不作为主结果默认。

## 关键实验结论

完整解释见 [高波数实验报告](reports/highk/EXPERIMENT_REPORT_2026-05-14_highk_ep10000.md)。README 只保留首页摘要。

| 主题 | 结论 |
|---|---|
| Reference floor | C0 reference gate 和 C3/C15 reference-only 结果表明 reference floor 足够低，神经解 `1e-3` 到 `1e-2` 量级 residual 不是 reference solver 噪声造成的 |
| High-k scaling | omega 300 已有训练结果，pipeline 能跑到高波数，但 phase/residual tradeoff 明显 |
| Strict PML residual | C5/C17 paired comparisons 支持 `stretched_divergence` 优于 legacy，尤其在 interface/PML residual 上更稳 |
| Supervision | PDE-only residual 低但 phase/reference error 大；data/hybrid 明显改善 phase/rel_l2，但 residual 变差 |
| ppw / grid | ppw 增大不是单调收益，固定容量和固定 epoch 下大网格会引入优化压力 |
| Capacity | depth、FNO modes、channels 都不是简单越大越好，best residual 和 best reference error 常不在同一模型 |
| Curriculum | 可运行性已修复，但相对 direct training 还没有稳定收益 |
| PML weighting / architecture antidote | 简单 PML loss weighting 和 quadratic curriculum 在部分高频组合中不稳定，C20 出现 NaN 和爆炸 run |

重要解释口径：

- C12 的 `ep01000/ep40000` 等 run_id 标签不能当作真实 epoch-budget 轴，因为全矩阵使用了 `epochs_override=10000`。
- `_meta/failures.jsonl` 是恢复和重试 ledger，最终完整性应以各 run 的 `summary.json` 为准。
- C 系列结果用于高波数 Helmholtz 求解论证，不是泛化的超参数排行榜。

## 实验资产索引

| 路径 | 内容 |
|---|---|
| [reports/highk/README.md](reports/highk/README.md) | C0-C21 高波数结果目录说明 |
| [reports/highk/EXPERIMENT_REPORT_2026-05-14_highk_ep10000.md](reports/highk/EXPERIMENT_REPORT_2026-05-14_highk_ep10000.md) | 最新高波数完整实验矩阵总结 |
| [reports/highk/MATRIX_PROVENANCE.md](reports/highk/MATRIX_PROVENANCE.md) | provenance、完整性、恢复说明 |
| [reports/highk/CODE_BACKUP_NOTES.md](reports/highk/CODE_BACKUP_NOTES.md) | 从服务器运行快照导入的代码/配置范围 |
| [reports/highk/matrix_report.csv](reports/highk/matrix_report.csv) | C0-C21 row-level 汇总表 |
| [reports/highk/matrix_batch_summary.csv](reports/highk/matrix_batch_summary.csv) | C0-C21 batch-level 汇总表 |
| [reports/matrix/README.md](reports/matrix/README.md) | 旧 A6/B0-B15 矩阵说明 |
| [reports/matrix/EXPERIMENT_REPORT_2026-04-22_03c409a.md](reports/matrix/EXPERIMENT_REPORT_2026-04-22_03c409a.md) | A6 基础矩阵报告 |
| [reports/next_phase/HIGH_WAVENUMBER_EXPERIMENT_MATRIX.md](reports/next_phase/HIGH_WAVENUMBER_EXPERIMENT_MATRIX.md) | 高波数实验矩阵设计文档 |
| [reports/next_phase/HIGH_WAVENUMBER_RESUME_INCIDENT_2026-05-14.md](reports/next_phase/HIGH_WAVENUMBER_RESUME_INCIDENT_2026-05-14.md) | 服务器恢复事故记录 |

## 实验矩阵

基础矩阵：

- `configs/experiments/B0_*.yaml` 到 `B15_*.yaml`
- 作用：介质、频率、网格/PML、容量、优化器、hybrid weights、source position、seed、strict PML audit 等基础扫查

高波数主矩阵：

- `configs/experiments/C0_*.yaml` 到 `C21_*.yaml`
- 作用：系统评估高波数 Helmholtz 求解能力

C 系列按功能可分为：

| 组别 | 批次 |
|---|---|
| reference / scaling / discretization | C0, C1, C2, C3, C15, C21 |
| PML strictness / interface / window / weighting | C4, C5, C16, C17, C18, C19 |
| medium / source / phase / supervision | C6, C7, C11, C14 |
| capacity / modes / channels / architecture | C8, C9, C10, C20 |
| optimization / curriculum | C12, C13 |

矩阵生成脚本：

```bash
python scripts/generate_highk_matrix.py --output-dir configs/experiments
```

注意：该命令会生成或覆盖 C 系列 YAML。复现实验前应先确认是否需要保留当前已归档配置。

## 项目结构

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── debug.yaml
│   ├── hybrid.yaml
│   └── experiments/
│       ├── B0-B15 basic matrix specs
│       └── C0-C21 high-wavenumber matrix specs
├── src/
│   ├── config.py
│   ├── core/
│   ├── physics/
│   ├── models/
│   ├── train/
│   └── eval/
├── scripts/
│   ├── run_train.py
│   ├── solve_reference.py
│   ├── evaluate_run.py
│   ├── run_benchmark.py
│   ├── run_matrix.py
│   ├── summarize_matrix.py
│   └── generate_highk_matrix.py
├── tests/
├── docs/
├── reports/
│   ├── matrix/
│   ├── highk/
│   └── next_phase/
└── outputs/
```

`outputs/` 是运行时产物目录，应保持 gitignored。完整实验结果体不进入仓库，仓库只保存代码、配置、报告和轻量汇总表。

## 安装

项目要求 Python 3.9：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

依赖事实源是 [pyproject.toml](pyproject.toml)。`requirements.txt` 只作为 pip 兼容入口保留。

## 基本用法

小网格 sanity run：

```bash
helmholtz-train \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --velocity-model smooth_lens \
  --device auto
```

等价脚本入口：

```bash
python scripts/run_train.py \
  --overlay configs/debug.yaml \
  --velocity-model smooth_lens \
  --device auto
```

为已有 run 补 reference solve 和评估：

```bash
python scripts/solve_reference.py --run-dir outputs/run_YYYYMMDD_HHMMSS
python scripts/evaluate_run.py --run-dir outputs/run_YYYYMMDD_HHMMSS
```

混合监督训练：

```bash
helmholtz-train \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --overlay configs/hybrid.yaml \
  --device auto
```

矩阵 dry run：

```bash
python scripts/run_matrix.py --output-root outputs/matrix_local --dry-run
```

矩阵运行与汇总：

```bash
python scripts/run_matrix.py --output-root outputs/matrix_local --device cuda:0 --continue-on-error
python scripts/summarize_matrix.py --output-root outputs/matrix_local
```

完整 C0-C21 矩阵计算量很大，应在服务器/GPU 环境运行。普通本地机器只建议做小网格 sanity run、静态检查、报告整理和代码维护。

## 测试

单元测试入口：

```bash
pytest -q
```

这些测试覆盖基础数值组件、配置加载、PML、residual、reference solver、model forward 和最小训练链路。它们不是 C0-C21 完整实验复现的替代。

## 复现和归档原则

- 已完成的 C0-C21 结果优先从 `reports/highk/` 读取，不应因为忘记结果位置而重跑。
- 大规模输出保存在运行环境的 `outputs/` 或外部结果根目录，不提交到 Git。
- 新矩阵应保留 manifest、failure ledger、summary、report 和 provenance。
- 解释结果时同时报告 residual、reference error、phase error 和 PML/interface 指标。
- 对高波数 Helmholtz，最低 residual 不等同于最佳物理解。

## 开发文档

- [DEV_PLAN.md](DEV_PLAN.md)：开发规范与冻结决策
- [CHANGELOG.md](CHANGELOG.md)：变更记录
- [docs/README.md](docs/README.md)：文档索引
- [reports/next_phase/HIGH_WAVENUMBER_EXPERIMENT_MATRIX.md](reports/next_phase/HIGH_WAVENUMBER_EXPERIMENT_MATRIX.md)：高波数矩阵设计

## 许可

本项目仅供学术研究使用。
