# 高波数 Helmholtz 神经算子 2D 原型

> 面向高波数 Helmholtz 方程的分阶段原型系统：
> 2D 单源 → 双精度预处理 → 物理残差验证 → 神经算子训练 → 参考解评估。

当前快照：`int/multi-agent @ b80dc88`（A1 打包/CI、A2 参考解评估、A3 hybrid 监督已合并）。
更完整的事实底稿见 [PROJECT_STATE.md](PROJECT_STATE.md)。

## 核心管线

```
介质 s(x) ──> [Eikonal预处理] ──> τ, ∇τ, Δτ, u₀, RHS_eq ──> [NSNO网络] ──> Â_scat ──> u_total
                                                                            │
                                                          参考解 A_ref ─────┘ (A2 评估链)
```

## 当前阶段

**Phase 1 — 2D Prototype**

- 2D、单点源、标量声学、规则笛卡尔网格
- 双精度预处理（Float64 求解 → Float32 送入网络）
- 单机 PyTorch 原型
- 前向推断 + PDE 残差验证 + 最小 overfit 实验
- 参考解（离散算子稀疏直解）作为一致性基线
- PDE-only / data-only / hybrid 三种损失模式（A3）

## 项目结构

```
开题/
├── DEV_PLAN.md              # 开发规范（冻结决策 + 里程碑）
├── PROJECT_STATE.md         # 当前集成基线快照
├── CHANGELOG.md             # 变更记录
├── README.md                # 本文件
├── WORKTREE_BOARD.md        # 多 agent 工作面板（A0 维护）
├── pyproject.toml           # 打包 / 依赖单一事实来源
├── requirements.txt         # 兼容壳（保留给旧脚本）
├── pytest.ini
├── .github/workflows/
│   └── ci.yml               # install + pytest + 1-epoch smoke
├── configs/
│   ├── base.yaml            # 默认参数
│   ├── debug.yaml           # 小网格调试覆盖
│   └── experiments/
│       └── hybrid.yaml      # hybrid 监督字段模板
├── handoffs/                # 每个 agent 之间的事实交接单
├── scripts/
│   ├── run_train.py         # 训练 CLI 包装
│   ├── evaluate_run.py      # 评估已有 run 的 CLI 包装
│   └── solve_reference.py   # 参考解 CLI（config/run-dir 两种模式）
├── src/
│   ├── __init__.py
│   ├── config.py            # load_config 多 overlay
│   ├── core/                # Grid2D / Medium2D / PointSource / 复数工具
│   ├── physics/             # background / eikonal / tau_ops / pml / diff_ops / rhs / residual
│   ├── models/              # spectral_conv / nsno
│   ├── train/
│   │   ├── losses.py        # L_pde, L_data, L_total
│   │   ├── supervision.py   # A3 监督解析与标签加载
│   │   ├── trainer.py       # 训练循环（三路 loss 历史 + last_step_metrics）
│   │   └── runner.py        # run_training / evaluate_saved_run + helmholtz-train 入口
│   └── eval/
│       ├── reference_solver.py  # 离散算子稀疏直解
│       └── reference_eval.py    # 参考解指标、产物导出、config/run-dir 工作流
├── tests/                   # 16 个测试文件，覆盖物理、网络、runner、监督、参考解
├── notebooks/               # 可视化 notebook
└── outputs/                 # 实验输出（不入库）
```

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd 开题

# 2. 创建虚拟环境（项目要求 Python 3.9.x）
python -m venv .venv
source .venv/bin/activate

# 3. 安装（pyproject.toml 为唯一事实源）
pip install -e ".[dev]"

# 4. 跑全测试
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q

# 5. 验证配置加载
python -c "from src.config import load_config; cfg = load_config('configs/base.yaml'); print(cfg)"

# 6. 跑一次完整训练（debug 网格 + smooth_lens 介质）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
helmholtz-train \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir outputs/smoke_pde_only

# 7. 为已有 run 补算评估 CSV / 热力图 / 参考解比较
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/evaluate_run.py --run-dir outputs/smoke_pde_only --device cpu

# 8. 独立生成参考解（config 模式）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/solve_reference.py \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --device cpu \
  --velocity-model smooth_lens \
  --output-dir outputs/reference_smooth_lens
```

> 备注：`outputs/` 目录不存在时会自动创建，但 `--output-dir` 指定的目标路径必须不存在（当前 `resolve_output_dir` 以 `exist_ok=False` 保护既有产物）。

## 开发规范

详见 [DEV_PLAN.md](DEV_PLAN.md)。

### 关键原则

1. 一次只推进一个里程碑
2. 上一步未通过检查，不进入下一步
3. 所有公式先冻结（见 DEV_PLAN.md 第 3 节），再编码
4. 每个里程碑必须有最小测试
5. 遇到理论与实现冲突，**优先修改文档**

### 里程碑进度

| 里程碑 | 状态 | 描述 |
|--------|------|------|
| M0 | ✅ | 建仓与最小骨架 |
| M1 | ✅ | 网格、介质、源项 |
| M2 | ✅ | 背景场与解析先验 |
| M3 | ✅ | 因子化 Eikonal 求解器 |
| M4 | ✅ | 安全导数重组 |
| M5 | ✅ | PML 与差分引擎 |
| M6 | ✅ | 等效体源与穿孔掩码 |
| M7 | ✅ | 网络骨架 |
| M8 | ✅ | PDE Residual 前向检查 |
| M9 | ✅ | 最小训练实验 |

### 集成增量

| 增量 | 状态 | 主要交付 |
|------|------|---------|
| A1 打包 / CI | ✅ 合并 | `pyproject.toml` 单一事实源、多 overlay、`helmholtz-train` CLI |
| A2 参考解评估 | ✅ 合并 | `src/eval/`、`scripts/solve_reference.py`、`reference_*` 产物与 summary 键 |
| A3 hybrid 监督 | ✅ 合并 | `src/train/supervision.py`、三路 loss 历史、`configs/experiments/hybrid.yaml` 模板 |
| A4 文档同步 | 🟡 进行中 | 本 README、PROJECT_STATE、CHANGELOG、DEV_PLAN 勾选 |
| A5 物理审计 | ⏸ 未开始 | Δ̃τ / PML / π/4 相位闭合审计 |
| A6 实验矩阵 | ✅ READY | 基于 A2 参考解、A3 三种 loss 模式展开对照实验 |

## 许可

本项目仅供学术研究使用。
