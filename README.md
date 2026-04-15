# 高波数 Helmholtz 神经算子 2D 原型

> 面向高波数 Helmholtz 方程的分阶段原型系统：  
> 2D 单源 → 双精度预处理 → 物理残差验证 → 神经算子训练 → 多波数泛化

## 核心管线

```
介质 s(x) ──> [Eikonal预处理] ──> τ, ∇τ, Δτ, u₀, RHS_eq ──> [NSNO网络] ──> Â_scat ──> u_total
```

## 当前阶段

**Phase 1 — 2D Prototype**

- 2D、单点源、标量声学、规则笛卡尔网格
- 双精度预处理（Float64 求解 → Float32 送入网络）
- 单机 PyTorch 原型
- 前向推断 + PDE 残差验证 + 最小 overfit 实验

## 项目结构

```
开题/
├── DEV_PLAN.md              # 开发规范（冻结决策 + 里程碑）
├── CHANGELOG.md             # 变更记录
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
├── pytest.ini               # 测试配置
├── configs/
│   ├── base.yaml            # 默认参数
│   └── debug.yaml           # 调试用小网格参数
├── src/
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── core/                # 基础数据结构
│   │   ├── grid.py          # Grid2D: 规则网格
│   │   ├── medium.py        # Medium2D: 介质慢度场
│   │   ├── source.py        # PointSource: 点源
│   │   └── complex_utils.py # 复数工具
│   ├── physics/             # 物理场计算
│   │   ├── background.py    # u₀: 汉克尔格林函数
│   │   ├── eikonal.py       # FSM 因子化程函求解
│   │   ├── tau_ops.py       # ∇τ, Δτ 安全重组
│   │   ├── pml.py           # PML 复坐标拉伸
│   │   ├── diff_ops.py      # 固定卷积核微分
│   │   ├── rhs.py           # 等效体源
│   │   └── residual.py      # PDE 残差
│   ├── models/              # 神经算子
│   │   ├── spectral_conv.py # FNO 频域截断层
│   │   └── nsno.py          # NSNO2D 网络
│   └── train/               # 训练
│       ├── losses.py        # 损失函数
│       └── trainer.py       # 训练循环
├── tests/                   # 单元测试
├── notebooks/               # 可视化 notebook
├── scripts/                 # 独立运行脚本
└── outputs/                 # 实验输出（不入库）
```

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd 开题

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试
pytest -v

# 5. 验证配置加载
python -c "from src.config import load_config; cfg = load_config('configs/base.yaml'); print(cfg)"

# 6. 运行一次完整训练并保存结果
python scripts/run_train.py --overlay configs/debug.yaml --device auto --velocity-model smooth_lens

# 7. 为已有实验目录补充评估 CSV 和热力图
python scripts/evaluate_run.py --run-dir outputs/run_YYYYMMDD_HHMMSS --device auto
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
| M1 | ⬜ | 网格、介质、源项 |
| M2 | ⬜ | 背景场与解析先验 |
| M3 | ⬜ | 因子化 Eikonal 求解器 |
| M4 | ⬜ | 安全导数重组 |
| M5 | ⬜ | PML 与差分引擎 |
| M6 | ⬜ | 等效体源与穿孔掩码 |
| M7 | ⬜ | 网络骨架 |
| M8 | ⬜ | PDE Residual 前向检查 |
| M9 | ⬜ | 最小训练实验 |

## 许可

本项目仅供学术研究使用。
