# 高波数 Helmholtz 神经算子 2D 原型

> 面向高波数 Helmholtz 方程的分阶段原型系统：
> 2D 单源 → 双精度预处理 → 物理残差验证 → 神经算子训练 → 参考解对照 → 实验矩阵扫查

## 核心管线

```
介质 s(x) ──> [Eikonal预处理] ──> τ, ∇τ, Δτ, u₀, RHS_eq ──> [NSNO网络] ──> Â_scat ──> u_total
                                                                           │
                                                                           ▼
                                                     [参考解对照 + PDE 残差 + 混合损失]
```

## 当前阶段

**Phase 1 — 2D Prototype**，M0-M9 核心管线完成，并在其上叠加了四轮并行开发（A1-A5）和一轮实验矩阵（A6）。

- 2D、单点源、标量声学、规则笛卡尔网格
- 双精度预处理（Float64 求解 → Float32 送入网络）
- 单机 PyTorch 原型
- 前向推断 + PDE 残差验证 + 参考解对照 + 混合损失训练
- B0-B15 实验矩阵（介质/频率/网格/PML/容量/优化器/源位/seed 等 16 个 sweep）

## 项目结构

```
开题/
├── DEV_PLAN.md              # 开发规范与冻结决策
├── CHANGELOG.md              # 变更记录
├── README.md                 # 本文件
├── pyproject.toml            # 依赖与安装（唯一事实源）
├── requirements.txt          # pip 向后兼容入口
├── pytest.ini
├── configs/
│   ├── base.yaml             # 默认参数
│   ├── debug.yaml            # 调试用小网格参数
│   ├── hybrid.yaml           # 混合损失训练 overlay 示例
│   └── experiments/          # B0-B15 实验矩阵 overlay
├── src/
│   ├── config.py             # 配置加载（支持多 overlay）
│   ├── core/                 # Grid2D / Medium2D / PointSource / 复数工具
│   ├── physics/              # background / eikonal / tau_ops / pml / diff_ops / rhs / residual
│   ├── models/               # spectral_conv / nsno
│   ├── train/                # losses / supervision / trainer / runner
│   └── eval/                 # reference_solver / reference_eval
├── scripts/
│   ├── run_train.py          # 单次训练
│   ├── evaluate_run.py       # 为已有实验目录补评估
│   ├── solve_reference.py    # 求参考解
│   ├── run_benchmark.py      # 单格 benchmark
│   ├── run_matrix.py         # 矩阵扫（本地）
│   ├── run_matrix_from_branch.py  # 矩阵扫（从分支启动）
│   ├── summarize_matrix.py   # 汇总 matrix 结果
│   └── _bootstrap.py / sitecustomize.py  # 服务器侧 import path
├── tests/                    # pytest 套件
├── notebooks/                # 可视化 notebook
├── docs/
│   ├── README.md             # 文档索引
│   ├── theory/               # TeX 理论资料与研究草稿
│   └── archive/              # 历史规范文档归档
├── reports/
│   └── matrix/               # 实验矩阵报告（README + 快照命名的 EXPERIMENT_REPORT）
└── outputs/                  # 实验输出（gitignored，本地专用）
```

## 快速开始

```bash
# 1. 克隆
git clone <repo-url>
cd 开题

# 2. 创建虚拟环境（要求 Python 3.9.x）
python -m venv .venv
source .venv/bin/activate

# 3. 以可编辑方式安装（含开发依赖）
pip install -e ".[dev]"

# 4. 跑测试
pytest -q

# 5. 单次训练（debug 小网格）
helmholtz-train --config configs/base.yaml --overlay configs/debug.yaml --velocity-model smooth_lens
# 等价写法：python scripts/run_train.py --overlay configs/debug.yaml --device auto

# 6. 为已有运行补参考解对照
python scripts/solve_reference.py --run-dir outputs/run_YYYYMMDD_HHMMSS
python scripts/evaluate_run.py --run-dir outputs/run_YYYYMMDD_HHMMSS

# 7. 混合损失训练
helmholtz-train --config configs/base.yaml --overlay configs/debug.yaml --overlay configs/hybrid.yaml

# 8. 实验矩阵（单格）
python scripts/run_benchmark.py --experiment configs/experiments/B0_smoke.yaml --output-root outputs/bench_B0

# 9. 实验矩阵（全扫 + 汇总）
python scripts/run_matrix.py --output-root outputs/matrix_local
python scripts/summarize_matrix.py --output-root outputs/matrix_local
```

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
| M3 | ✅ | 因子化 Eikonal 求解器（FSM） |
| M4 | ✅ | 安全导数重组（链式法则） |
| M5 | ✅ | PML 与固定核差分引擎 |
| M6 | ✅ | 等效体源与穿孔掩码 |
| M7 | ✅ | NSNO 网络骨架 |
| M8 | ✅ | PDE Residual 前向检查 |
| M9 | ✅ | 最小训练实验 |
| A1 | ✅ | Packaging / 最小 CI |
| A2 | ✅ | 参考解求解器 + 评估集成 |
| A3 | ✅ | 混合损失（PDE + data + reference） |
| A5 | ✅ | Residual / PML 审计（`lap_tau` mode） |
| A6 | ✅ | B0-B15 实验矩阵与汇总 |

### 报告产出

矩阵实验报告采用快照命名：`reports/matrix/EXPERIMENT_REPORT_<YYYY-MM-DD>_<commit_short>.md`。每次完整复跑新增一份，历史累积。

## 许可

本项目仅供学术研究使用。
