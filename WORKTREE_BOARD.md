# WORKTREE_BOARD.md

> 适用于当前"高波数 Helmholtz 神经算子 2D 原型"仓库的多 `git worktree` / 多 agent 并行推进。
> 本文档是**执行规范**，不是讨论稿。所有 agent 开工前必须先读完。
> 本轮目标不是"同时改很多东西"，而是把当前仓库从"能跑的研究原型"推进到"有稳定基线、可并行演进、可集成验收"的状态。

---

## 1. 本轮目的

本轮并行开发只做四件事：

1. 把**当前真实可工作的代码状态**冻结成统一基线。
2. 用 `git worktree` 拉出多个隔离工作区，降低 agent 之间的物理文件冲突。
3. 优先推进**基础设施、参考解基线、训练闭环、文档同步**四条主线。
4. 把高风险的 **Residual / PML 物理审计**放到第二阶段，避免一开始把所有热点文件同时点燃。

---

## 2. 当前仓库事实基线

### 2.1 当前真实开发主线

* 当前真实实现主线：`feat/m0-bootstrap`（当前已包含 M0-M9 全部里程碑 + 后续修复）
* `main` / `dev` 仍停留在更早的文档阶段，**本轮不得作为 worktree 母分支**
* 当前 git 历史已经包含（按时间顺序）：

  * M0: 建仓与最小骨架
  * M1: 网格、介质、源项与复数工具
  * M2: 背景场与解析先验
  * M3: 因子化 Eikonal FSM 求解器
  * M4: 安全导数重组（链式法则）
  * M5: PML 复坐标拉伸与固定卷积核微分引擎
  * M6: 等效体源与穿孔掩码
  * M7: 网络骨架 SpectralConv2d + NSNO2D
  * M8: PDE Residual 前向检查
  * M9: 最小训练实验
  * `fix(critical)`: s0 改为 s(x\_s) 震源处慢度
  * `fix(critical)`: residual.py PML 系数保持复数域运算，修正输运项拉伸
  * `test`: PML 激活 + 非均匀介质残差测试
  * `fix`: RHS 震源防爆半径改用 cfg.loss.source\_mask\_radius
  * `test`: Godunov FSM 和链式法则制造解测试
  * `fix(pml)`: sigma\_max 改用 1/s0 作为 PML 波速
  * `test`: 非均匀介质 eikonal 测试
  * `fix`: Grid 步长验证 + complex\_utils/layered 测试补全
  * `feat`: explicit trainer device support
  * `feat`: runnable training entrypoint
  * `fix`: persist runtime config overrides
  * `feat`: run diagnostics exports
  * `feat`: reference solver evaluation module

### 2.2 当前工作区状态

* `src/eval/` 已进入 git 历史（commit `e28f185`）
* 当前 `git status` 干净（仅 `WORKTREE_BOARD.md` 未跟踪）
* 分支 `feat/m0-bootstrap` 与远程 `origin/feat/m0-bootstrap` 一致
* 冻结基线可直接从当前 HEAD 创建，无需额外 commit

### 2.3 当前热点文件

以下文件是本轮冲突高发区，必须单写者或受控共享：

* `src/train/trainer.py`
* `src/train/runner.py`
* `src/physics/residual.py`
* `src/config.py`
* `configs/base.yaml`
* `configs/debug.yaml`

### 2.4 当前入口与输出契约

现有脚本入口：

* `scripts/run_train.py`（含 `sys.path.insert` hack）
* `scripts/evaluate_run.py`（含 `sys.path.insert` hack）

现有输出文件名由 `tests/test_runner.py` 断言锁定，**任何分支不得随意改名**：

* `config_merged.yaml`
* `summary.json`
* `losses.npy`
* `losses.csv`
* `loss_tail.csv`
* `loss_curve.png`
* `loss_curve_log.png`
* `wavefield.npy`
* `wavefield.png`
* `model_state.pt`
* `metrics_summary.csv`
* `quantiles.csv`
* `centerline_profiles.csv`
* `x_coords_physical.csv`
* `y_coords_physical.csv`
* `velocity_physical.csv`
* `wavefield_magnitude_physical.csv`
* `residual_magnitude_physical.csv`
* `field_heatmaps.png`
* `residual_heatmaps.png`

以下文件由 `save_diagnostic_csvs()` 产出但**未被测试断言覆盖**（新分支若新增产物，应同时补测试断言）：

* `loss_mask_physical.csv`
* `scattering_magnitude_physical.csv`

### 2.5 当前 Python 环境约束

* Python 版本：**>=3.9,<3.10**（`pyproject.toml` 约束）
* 构建后端：`setuptools>=68`
* `pyproject.toml` 已存在，定义了：
  * 项目名：`helmholtz-nsno-2d`
  * 版本：`0.1.0`
  * 依赖带上下界（如 `torch>=2.7,<2.8`、`numpy>=1.24,<2.0`）
  * 可选开发依赖：`pytest`、`pytest-cov`（通过 `pip install -e ".[dev]"` 安装）
  * CLI 入口：`helmholtz-train = "src.train.runner:main"`
  * 包发现：`src*`
* `requirements.txt` 仍保留（使用 `>=` 下界无上界），与 `pyproject.toml` 存在冗余，A1 需统一

### 2.6 当前配置系统限制

`src/config.py` 中的 `load_config(base_path, overlay_path)` **仅支持单层 overlay**。如果 A3 需要在 `configs/experiments/hybrid.yaml` 中新增训练配置项，有两种策略：

1. **推荐**：将新参数直接写入 `configs/experiments/hybrid.yaml`，作为完整 overlay 使用
2. **备选**：扩展 `load_config()` 支持多 overlay（需在 A1 Packaging 分支中处理）

不论哪种策略，`configs/base.yaml` 本轮保持冻结。

### 2.7 当前已有测试文件清单

```text
tests/
├── test_bootstrap.py          # M0 基础环境
├── test_grid.py               # Grid2D
├── test_medium.py             # Medium2D
├── test_complex_utils.py      # 复数工具
├── test_background.py         # 背景场 u₀
├── test_eikonal.py            # FSM 求解器
├── test_tau_ops.py            # 链式法则走时导数
├── test_diff_ops.py           # 固定卷积核微分
├── test_pml.py                # PML 张量
├── test_rhs.py                # 等效体源
├── test_residual.py           # PDE 残差
├── test_model_forward.py      # NSNO2D 前向
├── test_train.py              # 训练循环
└── test_runner.py             # 运行器 + 产物验证
```

**尚缺测试**（本轮需补）：

* `tests/test_reference_solver.py` — 参考解求解器（A2 负责）
* `tests/test_config.py` — 配置加载（可选，A1 负责）
* `tests/test_losses.py` — 独立损失函数测试（可选，A3 负责）

### 2.8 当前 reference\_solver 状态

`src/eval/reference_solver.py` 已实现且已入库，功能包括：

* `assemble_reference_operator(trainer)` — 组装与 ResidualComputer 一致的稀疏线性算子
* `solve_reference_scattering(trainer)` — 用 `scipy.sparse.linalg.spsolve` 求解参考散射包络

**当前限制**：

* 入口依赖 `Trainer` 对象（无独立 CLI 脚本）
* 无标准指标输出
* 未与 `evaluate_saved_run()` 集成
* 无独立测试

### 2.9 本轮明确不做

本轮不做以下事情，除非另开独立后续计划：

* 3D
* 多源 batch
* 分布式训练
* 论文级大 benchmark
* 大规模架构替换
* 工程部署优化
* 全仓重构 / 全仓格式化

---

## 3. 总体协作原则

### 3.1 基线优先

所有 worktree 都必须从同一个冻结基线拉出，不允许直接从 `main` / `dev` / 各自 feature 分支继续野生生长。

### 3.2 单写者原则

高冲突文件必须有主责分支。未获得主责，不得"顺手改一下"。

### 3.3 追加优先

* 配置优先新增 overlay，不改 `configs/base.yaml` / `configs/debug.yaml`
* 输出优先追加新文件，不改已有文件名
* 文档优先补充现状，不重写历史叙述

### 3.4 物理变更先文档后代码

凡是涉及物理公式、离散策略、PML 处理、损失定义的改动：

1. 先在对应分支更新 `DEV_PLAN.md` / `PROJECT_STATE.md`
2. 再改代码
3. 再补测试

### 3.5 小步提交

禁止一个分支里堆"大杂烩提交"。每个分支至少做到：

* 功能提交
* 测试提交
* 文档提交

### 3.6 输出目录隔离

所有 agent 跑训练 / 评估时，必须使用独立输出目录，不得混用默认输出。

**注意**：`runner.py` 中 `resolve_output_dir()` 使用 `exist_ok=False`，相同路径重复运行会报错。smoke 测试前须删除旧目录或使用唯一路径：

```bash
rm -rf ../runs/smoke/train_1ep 2>/dev/null
python scripts/run_train.py ... --output-dir ../runs/smoke/train_1ep
```

### 3.7 不提交临时产物

以下路径不得入库：

* `outputs/`
* `.pytest_cache/`
* `.venv/`
* 临时 notebook
* 临时截图
* 本地 benchmark 原始结果

### 3.8 新增产物必须补测试

任何分支新增输出文件时，必须同步在 `tests/test_runner.py`（或对应测试文件）中添加 `assert (output_dir / "xxx").exists()` 断言，避免出现"代码产出但测试未覆盖"的空白。

---

## 4. 分支与 worktree 拓扑

### 4.1 推荐拓扑

```text
feat/m0-bootstrap                # 当前真实实现主线（已完成 M0-M9 + 修复）
└── int/base-current             # 本轮统一基线（从当前 HEAD 冻结）
    ├── int/multi-agent          # 集成分支（只做 merge / smoke / 冲突修复）
    ├── feat/packaging-ci        # Packaging / CI / entrypoint 清理
    ├── feat/reference-eval      # reference solver 接入与评估基线
    ├── feat/hybrid-loss         # loss_data 真正接入训练
    ├── chore/project-state      # README / CHANGELOG / PROJECT_STATE / BOARD 同步
    └── fix/residual-pml-audit   # 第二阶段：Residual / PML 审计
```

### 4.2 worktree 目录建议

```text
../wt/int
../wt/pkg
../wt/ref
../wt/train
../wt/docs
../wt/phys      # 第二阶段再创建
```

---

## 5. 启动命令

### 5.1 先冻结统一基线

> 当前 `feat/m0-bootstrap` 的 `git status` 已干净，`src/eval/` 已入库。
> 需要将 `WORKTREE_BOARD.md` 和 `pyproject.toml` 入库后创建基线。

```bash
git switch feat/m0-bootstrap
git add WORKTREE_BOARD.md pyproject.toml
git commit -m "chore: add WORKTREE_BOARD.md and pyproject.toml before multi-agent worktrees"

git branch int/base-current
```

必须保证：

* `git status` 干净
* 所有人从同一个 commit 起步

### 5.2 创建 worktree

```bash
mkdir -p ../wt
mkdir -p ../runs

git worktree add ../wt/int   -b int/multi-agent      int/base-current
git worktree add ../wt/pkg   -b feat/packaging-ci    int/base-current
git worktree add ../wt/ref   -b feat/reference-eval  int/base-current
git worktree add ../wt/train -b feat/hybrid-loss     int/base-current
git worktree add ../wt/docs  -b chore/project-state  int/base-current
```

第二阶段再创建：

```bash
git worktree add ../wt/phys  -b fix/residual-pml-audit int/multi-agent
```

### 5.3 初始化实验配置目录

每个 worktree 中首次使用前：

```bash
mkdir -p configs/experiments
```

### 5.4 查看 / 清理

```bash
git worktree list
git worktree prune
git worktree remove ../wt/<name>
```

---

## 6. 文件所有权地图

### 6.1 单写者 / 受控共享表

| 路径                               | 主责分支                                      | 是否共享 | 规则                                                    |
| ---------------------------------- | --------------------------------------------- | -------: | ------------------------------------------------------- |
| `pyproject.toml`                   | `feat/packaging-ci`                           |       否 | 已存在，本轮仅 Packaging agent 可调整                      |
| `pytest.ini`                       | `feat/packaging-ci`                           |       否 | 仅 Packaging agent 可改                                  |
| `requirements.txt`                 | `feat/packaging-ci`                           |     受控 | 仅为安装 / CI 兼容性修改                                   |
| `src/config.py`                    | `feat/packaging-ci`                           |     受控 | A1 可做导入清理；如需多 overlay 支持，在 A1 中扩展           |
| `scripts/run_train.py`             | `feat/packaging-ci`                           |     受控 | 只做入口与导入清理（去掉 sys.path hack），非功能扩展         |
| `scripts/evaluate_run.py`          | `feat/packaging-ci`                           |     受控 | 同上                                                    |
| `scripts/solve_reference.py`       | `feat/reference-eval`                         |       否 | 由 Reference agent 新建                                  |
| `src/eval/*`                       | `feat/reference-eval`                         |       否 | 由 Reference agent 全权负责                               |
| `src/train/losses.py`              | `feat/hybrid-loss`                            |       否 | 由 Training agent 主写                                   |
| `src/train/trainer.py`             | `feat/hybrid-loss`                            |       否 | 热点文件，Training agent 单写                              |
| `src/train/runner.py`              | `int/multi-agent` + `feat/reference-eval`     |       是 | Reference agent 只可追加参考评估；最终冲突由 Integrator 收口  |
| `src/physics/residual.py`          | `fix/residual-pml-audit`                      |       否 | 第二阶段前禁止其他分支改                                    |
| `src/physics/background.py`        | 冻结                                          |       否 | 本轮只读                                                 |
| `src/physics/eikonal.py`           | 冻结                                          |       否 | 本轮只读                                                 |
| `src/physics/tau_ops.py`           | 冻结                                          |       否 | 本轮只读                                                 |
| `src/physics/pml.py`               | 冻结（A5 可选修改）                             |     受控 | A5 如需修改 PML 行为，须先记录后改                          |
| `src/physics/diff_ops.py`          | 冻结                                          |       否 | 本轮只读                                                 |
| `src/physics/rhs.py`               | 冻结                                          |       否 | 本轮只读                                                 |
| `src/core/*`                       | 冻结                                          |       否 | 本轮只读，所有 agent 不得修改                               |
| `src/models/*`                     | 冻结                                          |       否 | 本轮只读                                                 |
| `tests/test_runner.py`             | `int/multi-agent`                             |     受控 | 如需扩展，必须保持现有产物断言不破                           |
| `tests/test_train.py`              | `feat/hybrid-loss`                            |     受控 | 训练 agent 可扩                                          |
| `tests/test_residual.py`           | `fix/residual-pml-audit`                      |       否 | 物理审计专属                                              |
| `tests/test_pml.py`                | `fix/residual-pml-audit`                      |       否 | 物理审计专属                                              |
| `tests/test_reference_solver.py`   | `feat/reference-eval`                         |       否 | Reference agent 新建                                     |
| `README.md`                        | `chore/project-state`                         |     受控 | Docs agent 主写；Packaging agent 仅可改安装/运行片段        |
| `CHANGELOG.md`                     | `chore/project-state`                         |       否 | Docs agent 主写                                          |
| `PROJECT_STATE.md`                 | `chore/project-state`                         |       否 | Docs agent 新建并维护                                     |
| `DEV_PLAN.md`                      | `chore/project-state` + `fix/residual-pml-audit` | 受控 | Docs agent 更新里程碑状态；A5 可更新物理决策部分             |
| `WORKTREE_BOARD.md`                | `chore/project-state`                         |     受控 | Integrator 可更新状态，Docs agent 负责结构                  |
| `configs/base.yaml`                | 冻结                                          |       否 | 本轮默认不改                                              |
| `configs/debug.yaml`               | 冻结                                          |       否 | 本轮默认不改                                              |
| `configs/experiments/*`            | 对应功能分支                                   |       是 | 所有实验新增都放这里                                       |

### 6.2 热点文件硬规则

#### `src/train/trainer.py`

* 仅 `feat/hybrid-loss` 主写
* 其他分支不得顺手改
* 如确需兼容性补丁，由 Integrator 最后统一做

#### `src/train/runner.py`

* 只允许两类改动：

  1. `feat/reference-eval` 追加参考评估产物
  2. `int/multi-agent` 处理合并后兼容问题
* `feat/hybrid-loss` 不得在此文件上扩展日志格式或重命名输出

#### `src/physics/residual.py`

* 第二阶段前只读
* 禁止"顺手优化"
* 禁止与 Training / Reference 同期并行修改

#### `src/config.py`

* A1 可做导入路径清理和 `pyproject.toml` 适配
* 如需支持多 overlay（A3 依赖），在 A1 中扩展
* 其他分支不得直接修改

#### `configs/base.yaml` / `configs/debug.yaml`

* 本轮视为"冻结配置"
* 谁都不要改，除非出现**所有分支都无法继续**的阻塞级问题
* 实验配置一律新建：

```text
configs/experiments/reference.yaml
configs/experiments/hybrid.yaml
configs/experiments/pml_audit.yaml
```

---

## 7. 统一运行规范

### 7.1 环境约束

在 Packaging 分支合并前，统一使用：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

在 Packaging 分支合并前，统一从仓库根目录运行，并临时带：

```bash
PYTHONPATH=.
```

Python 版本要求：**3.9+**

---

### 7.2 统一 smoke 命令

#### 基础测试

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
```

#### 最小训练冒烟

> 注意：`--output-dir` 路径不得与已有目录冲突，`resolve_output_dir()` 使用 `exist_ok=False`。
> 每次 smoke 前先清理旧输出，或使用带时间戳的路径。

```bash
rm -rf ../runs/smoke/train_1ep 2>/dev/null

PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_train.py \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir ../runs/smoke/train_1ep
```

#### 评估冒烟

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/evaluate_run.py \
  --run-dir ../runs/smoke/train_1ep \
  --device cpu
```

---

### 7.3 Packaging 合并后的统一命令

Packaging 合并后，统一改为：

```bash
pip install -e ".[dev]"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q

rm -rf ../runs/smoke/train_1ep 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_train.py \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir ../runs/smoke/train_1ep

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/evaluate_run.py \
  --run-dir ../runs/smoke/train_1ep \
  --device cpu
```

也可使用 `pyproject.toml` 定义的 CLI 入口（等价于 `python scripts/run_train.py`）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
helmholtz-train \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir ../runs/smoke/train_1ep
```

---

## 8. Agent 总看板

| ID | 分支                       | worktree      | 阶段      | 目标                                            | 依赖               | 合并顺序 | 状态 |
| -- | ------------------------ | ------------- | ------- | --------------------------------------------- | ---------------- | ---: | -- |
| A0 | `int/multi-agent`        | `../wt/int`   | 全程      | 集成、merge、冲突修复、统一 smoke                        | 基线快照             |    - | ⬜  |
| A1 | `feat/packaging-ci`      | `../wt/pkg`   | Phase 1 | `pip install -e .`、去掉脚本 path hack、加最小 CI      | 无                |    1 | ⬜  |
| A2 | `feat/reference-eval`    | `../wt/ref`   | Phase 1 | 接入 `reference_solver`，建立参考评估基线                | 建议 A1 先合         |    3 | ⬜  |
| A3 | `feat/hybrid-loss`       | `../wt/train` | Phase 2 | 打通 `loss_data` / `lambda_data` 监督混合训练         | A2               |    4 | ⬜  |
| A4 | `chore/project-state`    | `../wt/docs`  | Phase 1 | 更新 README / CHANGELOG / PROJECT_STATE / BOARD | 无                |    2 | ⬜  |
| A5 | `fix/residual-pml-audit` | `../wt/phys`  | Phase 2 | 审计并可选修正 PML 下 `Δ̃τ` 处理                        | A2 已稳定           |    5 | ⬜  |
| A6 | `exp/benchmark-matrix`   | `../wt/exp`   | Phase 3 | 运行小型实验矩阵并输出报告                                 | A1+A2+A3(+A5 可选) |    6 | ⬜  |

---

## 9. A0 — Integrator / 集成协调分支

### 9.1 分支信息

* 分支：`int/multi-agent`
* worktree：`../wt/int`

### 9.2 角色定义

A0 不负责"开发大功能"，只负责：

* 建立统一基线
* 接收其他分支合并
* 跑统一 smoke
* 处理冲突
* 更新本看板状态
* 决定某个分支是"继续修"还是"退回重做"

### 9.3 允许修改

* 合并冲突涉及的少量 glue code
* `WORKTREE_BOARD.md` 状态更新
* 必要的兼容性修补
* 极少量 test 修复（前提是不改变功能定义）

### 9.4 禁止修改

* 不得在 A0 分支里直接实现大型功能
* 不得把别人分支没完成的内容"临时补齐"成事实标准
* 不得越权改物理公式

### 9.5 A0 启动检查清单

* [ ] `WORKTREE_BOARD.md` 和 `pyproject.toml` 已入库
* [ ] `int/base-current` 已创建
* [ ] 所有 Phase 1 worktree 已创建
* [ ] 所有人拿到相同 smoke 命令
* [ ] `PROJECT_STATE.md` 已由 A4 创建（或预留）

### 9.6 合并流程

每次合并统一使用：

```bash
git switch int/multi-agent
git merge --no-ff <feature-branch>
```

合并后立即执行：

```bash
rm -rf ../runs/integration/smoke_after_merge 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_train.py --overlay configs/debug.yaml --device cpu --epochs 1 --velocity-model smooth_lens --output-dir ../runs/integration/smoke_after_merge
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/evaluate_run.py --run-dir ../runs/integration/smoke_after_merge --device cpu
```

若失败：

1. 记录失败原因和完整错误日志
2. 标记对应 agent 为 `BLOCKED`
3. 不继续合并后续分支
4. 如果是简单导入/路径冲突，A0 可直接修复后重试
5. 如果涉及逻辑/物理变更，退回给原分支 agent 处理

### 9.7 回退流程

若合并后 smoke 失败且无法快速修复：

```bash
git merge --abort   # 如果还在合并状态
# 或
git reset --hard HEAD~1   # 如果已经 commit 了合并
```

回退后在看板中记录原因，等原分支修复后再次合并。

### 9.8 A0 Definition of Done

* [ ] A1 合并并通过 smoke
* [ ] A4 合并并完成文档同步
* [ ] A2 合并并建立参考评估
* [ ] A3 合并并打通 hybrid loss
* [ ] A5（如开启）合并并无回归
* [ ] 最终 `int/multi-agent` 通过统一 smoke
* [ ] 形成下一轮基线快照

---

## 10. A1 — Packaging / CI / 入口清理

### 10.1 分支信息

* 分支：`feat/packaging-ci`
* worktree：`../wt/pkg`

### 10.2 目标

把仓库从"需要 path hack 才能跑"改成"可安装、可测试、可复现"的正常工程入口。

### 10.3 只解决这些问题

* 验证现有 `pyproject.toml` 并确保 `pip install -e .` 工作正常
* 统一 `pyproject.toml` 与 `requirements.txt` 的依赖声明（消除冗余）
* 去掉 `scripts/run_train.py` 和 `scripts/evaluate_run.py` 里的 `sys.path.insert(...)`
* 验证 CLI 入口 `helmholtz-train` 是否可用
* 清理根目录运行假设
* 加最小 CI
* 更新 README 的安装 / 测试 / 运行命令

### 10.4 当前 sys.path hack 位置

两处需要清理：

```python
# scripts/run_train.py:8-10
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# scripts/evaluate_run.py:8-10
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

### 10.5 允许修改

* `pyproject.toml`（已存在，可调整）
* `pytest.ini`
* `requirements.txt`（仅安装兼容性）
* `src/config.py`（仅导入路径适配）
* `scripts/run_train.py`
* `scripts/evaluate_run.py`
* `.github/workflows/*`（新建）
* `README.md` 中的快速开始片段
* 极少量导入路径修复
* 可选：`Makefile` / `justfile`

### 10.6 禁止修改

* 不改 `src/physics/*`
* 不改 `src/models/*`
* 不改 `src/core/*`
* 不改 `src/train/trainer.py` 训练逻辑
* 不改输出文件名
* 不顺手做 refactor

### 10.7 可选扩展：多 overlay 支持

如果 A3 需要多层 overlay（base + debug + hybrid），A1 可在 `src/config.py` 中扩展 `load_config()` 签名：

```python
def load_config(base_path, *overlay_paths):
```

这是 A1 的可选任务，但建议优先完成，以解除 A3 的配置依赖。

### 10.8 交付物

* [ ] `pyproject.toml` 已验证并与 `requirements.txt` 统一
* [ ] `pip install -e .` 可用（含 `pip install -e ".[dev]"` 开发模式）
* [ ] CLI 入口 `helmholtz-train` 可用
* [ ] `pytest -q` 可直接从仓库根目录执行
* [ ] `run_train.py` 无 `sys.path` hack
* [ ] `evaluate_run.py` 无 `sys.path` hack
* [ ] README 的 quickstart 改为新命令
* [ ] 最小 CI 工作流（至少安装 + 测试）
* [ ] 可选：`load_config()` 多 overlay 支持

### 10.9 验收命令

```bash
pip install -e .

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q

rm -rf ../runs/pkg/smoke 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_train.py \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir ../runs/pkg/smoke

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/evaluate_run.py \
  --run-dir ../runs/pkg/smoke \
  --device cpu
```

### 10.10 完成标准

* [ ] 上述命令全部通过
* [ ] 不需要 `PYTHONPATH=.`
* [ ] 不修改任何物理或训练语义

---

## 11. A2 — Reference / Evaluation 基线分支

### 11.1 分支信息

* 分支：`feat/reference-eval`
* worktree：`../wt/ref`

### 11.2 背景

仓库中已有 `src/eval/reference_solver.py`（commit `e28f185`），包含：

* `assemble_reference_operator(trainer)` — 基于 ResidualComputer 同一离散格式组装稀疏矩阵
* `solve_reference_scattering(trainer)` — 返回 `[H, W]` 复数参考散射包络

当前模块的限制：

* 只能通过 Python 代码调用（无 CLI 入口）
* 不输出标准指标
* 不与 `evaluate_saved_run()` 集成
* 无测试

这一分支的任务是把"参考解"从一个零散模块，升级为训练与评估的中心标尺。

### 11.3 目标

* 为 reference\_solver 新建独立 CLI 脚本
* 建立标准参考指标体系
* 把参考误差接入评估流程
* 让后续训练不再只看 PDE loss，而能看"距参考解有多远"

### 11.4 允许修改

* `src/eval/*`
* `scripts/solve_reference.py`（新建）
* `tests/test_reference_solver.py`（新建）
* 少量 `src/train/runner.py` 评估扩展
* 可选：`configs/experiments/reference.yaml`

### 11.5 禁止修改

* 不改 `src/physics/residual.py` 的离散定义
* 不改 `src/train/trainer.py` 的训练逻辑
* 不改模型结构
* 不改 `src/core/*`
* 不重命名已有产物

### 11.6 需要新增的标准产物

建议统一新增以下文件名：

* `reference_envelope.npy`
* `reference_wavefield.npy`
* `reference_metrics.json`
* `reference_comparison.csv`
* `reference_error_heatmaps.png`

如果需要保留更短文件名，可在此分支内统一一次后固定，后续不得再改。

### 11.7 需要新增的标准指标

建议至少输出：

* `rel_l2_to_reference`
* `amp_mae_to_reference`
* `phase_mae_to_reference`
* `reference_residual_rmse`
* `reference_residual_max`
* `reference_available`（bool）

### 11.8 任务清单

* [ ] 新建 `scripts/solve_reference.py` CLI 脚本
* [ ] 能从配置直接生成参考解
* [ ] 能从已有 `run_dir` 复用配置生成参考解
* [ ] 新建 `tests/test_reference_solver.py`
* [ ] `runner.py` 中的 `evaluate_saved_run()` 能识别并导出参考误差
* [ ] 对 `homogeneous` 做零散射 sanity check
* [ ] 对 `smooth_lens` 做 reference residual 检查
* [ ] 新增产物文件名加入测试断言

### 11.9 与 runner 的协作规则

由于 `src/train/runner.py` 是共享热点文件，A2 对 runner 的修改必须遵守：

* 只追加，不重命名
* 不打乱现有 summary 结构
* 不引入训练逻辑
* 不改默认 CLI 语义
* 新增指标追加到 `metrics` 字典和 `summary.json`，不替换已有键

### 11.10 验收标准

* [ ] `scripts/solve_reference.py` 能独立运行
* [ ] `tests/test_reference_solver.py` 通过
* [ ] 现有 `evaluate_saved_run()` 能识别并导出参考误差
* [ ] 对 `homogeneous`，参考散射包络在物理区接近 0
* [ ] 将参考解代回当前算子后 residual 显著小于未训练网络输出
* [ ] 全部现有测试不回归

---

## 12. A3 — Hybrid Loss / 训练闭环分支

### 12.1 分支信息

* 分支：`feat/hybrid-loss`
* worktree：`../wt/train`

### 12.2 背景

`loss_data()` 和 `lambda_data` 已经存在于 `src/train/losses.py`，但训练主循环（`trainer.py` 第 166 行）调用 `loss_total()` 时**未传入 `loss_data_val`**：

```python
total = loss_total(result['loss_pde'], lambda_pde=lambda_pde,
                   lambda_data=lambda_data)
```

这意味着即使 `lambda_data > 0`，监督损失也不会生效。
这一分支的目标不是"重写训练器"，而是**把现有接口打通**。

### 12.3 目标

* 支持三种训练模式：

  * PDE-only（当前默认，`lambda_data=0`）
  * data-only（`lambda_pde=0, lambda_data>0`）
  * PDE + data（两者都 > 0）
* 保持当前默认行为不变
* 接入 A2 产出的参考标签

### 12.4 允许修改

* `src/train/losses.py`
* `src/train/trainer.py`
* `tests/test_train.py`
* `configs/experiments/hybrid.yaml`（新建）
* 必要时新增少量训练数据加载辅助模块

### 12.5 禁止修改

* 不改 `src/physics/*`
* 不改 `src/models/*`
* 不改 `src/core/*`
* 不改现有 runner 产物文件名
* 不把 reference 生成逻辑复制一份到 trainer 里
* 不直接依赖未固定的临时输出路径

### 12.6 建议的接口约定

建议通过配置显式控制：

```yaml
training:
  lambda_pde: 1.0
  lambda_data: 0.0
  supervision:
    enabled: false
    reference_path: null
```

或等价结构，但必须满足：

* 默认无监督仍可正常训练
* 当有参考标签时，能显式启用监督项
* 不依赖硬编码文件路径

**重要**：新配置项写入 `configs/experiments/hybrid.yaml`，不改 `configs/base.yaml`。如需在 `base.yaml` 中设置默认值，与 A0 协商后由 A0 在集成时统一处理。

### 12.7 必做事项

* [ ] `loss_total()` 真正接收到 `loss_data_val`
* [ ] `trainer.py` 在有监督标签时计算 `loss_data`
* [ ] 默认 `lambda_data=0.0` 路径与当前行为完全一致
* [ ] 支持记录 `loss_pde` / `loss_data` / `loss_total`
* [ ] 增加至少两个测试：

  * 默认 PDE-only 不变
  * 有标签时 hybrid 生效

### 12.8 推荐最小实验

先只做最小 overfit 闭环：

* `configs/debug.yaml`
* `velocity_model=smooth_lens`
* 单样本
* 比较：

  * PDE-only
  * hybrid

### 12.9 验收标准

* [ ] 不带标签时训练流程与当前一致
* [ ] 带标签时能正确计算并使用 `loss_data`
* [ ] `lambda_data=0` 与 `lambda_data>0` 路径均有测试覆盖
* [ ] 不破坏现有 `tests/test_runner.py`
* [ ] 训练日志 / summary 至少能区分 `loss_pde` 与 `loss_data`
* [ ] 全部现有测试不回归

---

## 13. A4 — Docs / Project State 文档同步分支

### 13.1 分支信息

* 分支：`chore/project-state`
* worktree：`../wt/docs`

### 13.2 背景

顶层文档当前存在明显"文档阶段落后于代码阶段"的情况：

* `README.md` 里程碑表显示 M0 ✅ / M1-M9 全部 ⬜，但实际全部已完成
* `CHANGELOG.md` 仅记录 M0，遗漏全部后续开发
* `DEV_PLAN.md` 中的检查项未勾选
* 无 `PROJECT_STATE.md`

本分支的任务是让仓库叙事与真实实现同步，但**不夸大实验结果**。

### 13.3 目标

* 把 README、CHANGELOG、项目状态文件更新到与当前代码一致
* 让新加入的 agent 一眼看懂：

  * 现在代码到哪一步
  * 该怎么跑
  * 哪些是已完成
  * 哪些是下一步
  * 哪些分支在并行

### 13.4 允许修改

* `README.md`
* `CHANGELOG.md`
* `PROJECT_STATE.md`（新建）
* `WORKTREE_BOARD.md`
* `DEV_PLAN.md`（仅更新检查项状态和里程碑进度，不改冻结决策）
* 必要的 docstring 轻量修订

### 13.5 禁止修改

* 不改任何功能代码
* 不根据"想象中的结果"更新文档
* 不写论文式夸张表述
* 不补不存在的 benchmark 结论
* 不修改 `DEV_PLAN.md` 中的冻结决策（第 3 节）

### 13.6 当前文档与事实的具体偏差

A4 需要修正的已知偏差：

| 文件 | 当前状态 | 需要改为 |
| ---- | ------- | ------- |
| `README.md` 里程碑表 | M0 ✅ / M1-M9 ⬜ | M0-M9 全部 ✅ |
| `README.md` 项目结构 | 缺少 `src/eval/`、`src/train/runner.py` | 补全 |
| `README.md` 快速开始 | 未提及 `evaluate_run.py` | 补全 |
| `CHANGELOG.md` | 仅记录 M0 | 补全 M1-M9 + 后续修复 |
| `DEV_PLAN.md` 检查项 | 全部未勾选 | 根据实际完成情况勾选 |

### 13.7 `PROJECT_STATE.md` 必须包含

* 当前真实基线分支（`feat/m0-bootstrap`）
* 当前已实现能力（M0-M9 全部 + 修复清单）
* 当前已知问题（PML 区 Δ̃τ 混合策略、参考解未集成等）
* 当前 worktree / 分支拓扑
* 统一 smoke 命令
* 当前建议推进顺序
* 现有 artifact 契约（完整文件名列表）
* 参考解 / hybrid / residual 审计的依赖关系
* Python 版本和依赖约束

### 13.8 必做事项

* [ ] README 的阶段描述更新到真实状态
* [ ] README 里程碑表全部改为 ✅
* [ ] README 项目结构补全 `src/eval/`、`runner.py` 等
* [ ] CHANGELOG 补全 M1-M9 及后续修复记录
* [ ] 新建 `PROJECT_STATE.md`
* [ ] `WORKTREE_BOARD.md` 入库并完成初版状态
* [ ] README 快速开始和当前工程入口一致
* [ ] DEV_PLAN.md 检查项根据实际完成情况勾选

### 13.9 验收标准

* [ ] 新读者只看顶层文档即可知道仓库真实状态
* [ ] README / CHANGELOG / PROJECT_STATE 三者叙述一致
* [ ] 文档中明确说明当前版本不做 3D、多源、分布式、大 benchmark

---

## 14. A5 — Residual / PML Audit 第二阶段分支

### 14.1 分支信息

* 分支：`fix/residual-pml-audit`
* worktree：`../wt/phys`
* 仅在 A2 合并并建立参考评估后创建

### 14.2 背景

当前 `ResidualComputer` 中（`src/physics/residual.py:105-140`），`Δ̃τ` 的处理采用的是**混合策略**：

* 输运项 `∇̃Â·∇̃τ` 中的 `∇τ` 已正确使用 PML 拉伸系数（`grad_tau_x_stretched = A_x * grad_tau_x`）
* 但 `Δ̃τ` 仍使用未拉伸的 `lap_tau`（链式法则结果，但不含 PML 修正）
* 代码注释（119-137 行）解释了简化理由：PML 区 RHS=0 且波场应衰减，所以误差贡献小

这条线的任务是**带着参考解指标做物理审计**，而不是凭感觉重写。

### 14.3 目标

* 定量比较当前混合策略与更严格实现
* 在不破坏稳定性的前提下，降低参考误差或至少解释误差来源
* 补足相关测试

### 14.4 允许修改

* `src/physics/residual.py`
* `tests/test_residual.py`
* `tests/test_pml.py`
* 必要时更新 `DEV_PLAN.md`
* 可选：`configs/experiments/pml_audit.yaml`

### 14.5 禁止修改

* 不改 `src/train/trainer.py`
* 不改 `src/train/losses.py`
* 不改 `src/models/*`
* 不改 `src/core/*`
* 不与 Hybrid 分支同时争用训练接口

### 14.6 建议策略

优先做"可切换实验"而不是一次性替换：

* 保留当前实现为 baseline
* 新增严格实现开关（如 `pml_strict_lap_tau: bool` 配置项）
* 用 reference metrics 比较两者

### 14.7 必做事项

* [ ] 审计当前 `lap_tau_c` 的使用路径
* [ ] 实现一个更严格的 PML `Δ̃τ` 版本（可通过 flag 切换）
* [ ] 增加对比测试
* [ ] 在 `PROJECT_STATE.md` 记录对比结论

### 14.8 验收标准

* [ ] 所有相关测试通过
* [ ] 没有 NaN / Inf 回归
* [ ] 相比 baseline，reference 指标不退化
* [ ] 若未改善，也必须留下定量结论，而不是"感觉更正确"

---

## 15. A6 — Experiment Matrix 可选第三阶段分支

### 15.1 分支信息

* 分支：`exp/benchmark-matrix`
* worktree：`../wt/exp`
* 仅在 A1 + A2 + A3 合并后创建

### 15.2 目标

用小而清晰的实验矩阵验证当前系统的真实能力边界。

### 15.3 建议矩阵

* 介质：

  * `homogeneous`
  * `smooth_lens`
  * `layered`
* 频率：

  * `omega = 10`
  * `omega = 20`
  * `omega = 30`
* 网格：

  * `32`
  * `64`

### 15.4 允许修改

* `configs/experiments/*`
* `scripts/run_benchmark.py`（如需）
* `reports/*`
* `PROJECT_STATE.md` 中的实验摘要

### 15.5 禁止修改

* 原则上不改核心代码
* 如发现 bug，回流到对应功能分支处理

### 15.6 交付物

* [ ] 统一 CSV 汇总表
* [ ] 小型实验报告
* [ ] 指标对比：

  * final loss
  * rel L2 to reference
  * phase error
  * residual stats
  * runtime

---

## 16. 合并顺序与门禁

### 16.1 合并顺序

1. `feat/packaging-ci`
2. `chore/project-state`
3. `feat/reference-eval`
4. `feat/hybrid-loss`
5. `fix/residual-pml-audit`
6. `exp/benchmark-matrix`（可选）

---

### 16.2 Gate 0 — 基线冻结

* [ ] `WORKTREE_BOARD.md` 和 `pyproject.toml` 已入库
* [ ] `int/base-current` 已创建
* [ ] 所有 Phase 1 worktree 已拉起

### 16.3 Gate 1 — 工程入口可用

* [ ] A1 合并
* [ ] `pip install -e .` 可用
* [ ] `pytest -q` 可直跑
* [ ] CLI 入口无 path hack

### 16.4 Gate 2 — 项目状态同步

* [ ] A4 合并
* [ ] 顶层文档与代码一致
* [ ] `PROJECT_STATE.md` 已建立

### 16.5 Gate 3 — 参考解基线建立

* [ ] A2 合并
* [ ] reference solver 有脚本、有测试、有指标
* [ ] runner 可导出参考误差

### 16.6 Gate 4 — 训练闭环建立

* [ ] A3 合并
* [ ] `loss_data` 真正生效
* [ ] 默认 PDE-only 保持不变

### 16.7 Gate 5 — 物理审计

* [ ] A5 合并
* [ ] reference 指标无明显回归
* [ ] 相关结论写入 `PROJECT_STATE.md`

### 16.8 Gate 6 — 小型实验矩阵

* [ ] A6 完成
* [ ] 输出汇总报告
* [ ] 形成下一轮研发决策输入

---

## 17. 分支同步规则

### 17.1 Phase 1 分支同步

在 A1 合并后，其余未合并分支必须尽快同步 `int/multi-agent`：

```bash
git switch <feature-branch>
git merge int/multi-agent
```

### 17.2 禁止长期漂移

如果某个 feature 分支与 `int/multi-agent` 漂移超过两个已合并分支，必须先同步再继续开发。

### 17.3 冲突处理优先级

冲突统一按以下优先级处理：

1. 产物文件名契约
2. 测试稳定性
3. 工程入口可用性
4. 物理定义一致性
5. 文档表述

---

## 18. 提交规范

### 18.1 commit 前缀

建议统一使用：

* `feat:`
* `fix:`
* `test:`
* `docs:`
* `chore:`
* `refactor:`（谨慎使用）

### 18.2 每个分支最少提交结构

至少保证：

1. 功能提交
2. 测试提交
3. 文档 / 状态提交

### 18.3 禁止提交

* `misc`
* `update`
* `fix bugs`
* `改了一下`
* 大量无意义格式化

---

## 19. 每日状态更新模板

> 每个 agent 每次同步进度时，直接在对应小节下追加，不开新文件。

### 模板

```md
### YYYY-MM-DD / <Agent ID> / <Owner>

- 当前分支：
- 当前 commit：
- 今日完成：
- 正在处理：
- 阻塞项：
- 涉及文件：
- 已跑验证：
- 下一步：
```

---

## 20. 当前状态区

### 20.1 A0 Integrator

* 状态：`TODO`
* Owner：
* 最近更新：

### 20.2 A1 Packaging / CI

* 状态：`TODO`
* Owner：
* 最近更新：

### 20.3 A2 Reference / Eval

* 状态：`TODO`
* Owner：
* 最近更新：

### 20.4 A3 Hybrid Loss

* 状态：`TODO`
* Owner：
* 最近更新：

### 20.5 A4 Docs / Project State

* 状态：`TODO`
* Owner：
* 最近更新：

### 20.6 A5 Residual / PML Audit

* 状态：`WAITING_GATE_3`
* Owner：
* 最近更新：

### 20.7 A6 Experiment Matrix

* 状态：`WAITING_GATE_4`
* Owner：
* 最近更新：

---

## 21. 已知技术债与风险

### 21.1 ResidualComputer 的 PML Δ̃τ 混合策略

当前 `residual.py` 中输运项已拉伸但 `lap_tau` 未拉伸。代码注释解释了合理性（PML 区 RHS=0 且波场衰减），但严格正确性未经参考解定量验证。这是 A5 的核心审计目标。

### 21.2 `resolve_output_dir` 的 `exist_ok=False`

`runner.py:54` 使用 `final_dir.mkdir(parents=True, exist_ok=False)`，重复使用相同输出路径会报错。这是有意设计（防止覆盖实验结果），但 smoke 测试时需注意先清理旧目录。

### 21.3 依赖声明冗余

`pyproject.toml` 已有带上下界的依赖声明（如 `torch>=2.7,<2.8`），而 `requirements.txt` 仍保留宽松的 `>=` 下界。两套声明并存易导致不一致。A1 应统一为以 `pyproject.toml` 为准，`requirements.txt` 可保留为 `pip install -r` 的向后兼容入口或直接删除。

### 21.4 单 overlay 配置系统

`load_config()` 仅支持一层 overlay。A3 如需 `base + debug + hybrid` 三层叠加，需要 A1 先扩展配置系统或 A3 自行将 debug + hybrid 合并为单一 overlay。

### 21.5 trainer.py 中的固定随机种子

`trainer.py:126-129` 使用 `torch.manual_seed(0)` 固定模型初始化，这是为了测试可复现性。A3 修改训练逻辑时需注意保持这个行为（或提供配置项来控制）。

### 21.6 reference\_solver 的计算成本

`assemble_reference_operator()` 使用逐点循环构建稀疏矩阵，对于大网格（128x128 含 PML）约有 28000+ 未知量。`spsolve` 的内存和时间开销可能在 debug 网格上可接受，但在 base 网格上需要注意。A2 应在测试中使用 debug 配置。

---

## 22. 跨分支协调协议

### 22.1 阻塞通知

当某个 agent 发现自己被阻塞时（缺少前置分支的产出）：

1. 在本看板 §20 对应小节标记 `BLOCKED: <原因>`
2. 标注阻塞来源 agent ID
3. 切换到不依赖被阻塞内容的子任务

### 22.2 接口变更通知

如果 A1 扩展了 `load_config()`、A2 新增了 runner 指标键，需要：

1. 在本看板 §20 中简要描述接口变更
2. 列出受影响的下游分支
3. 下游分支在同步 `int/multi-agent` 后适配

### 22.3 紧急热修

如果某个 agent 发现基线中存在影响所有分支的 bug：

1. 不在自己的 feature 分支中修复
2. 通知 A0（Integrator）
3. A0 在 `int/multi-agent` 上做 hotfix
4. 所有活跃分支同步 `int/multi-agent`

---

## 23. 收尾规则

当本轮并行开发完成后：

1. 将 `int/multi-agent` 作为新的稳定基线
2. 从该基线创建新的集成快照分支，例如：

```bash
git switch int/multi-agent
git branch int/base-next
```

3. 清理已完成的 worktree：

```bash
git worktree remove ../wt/pkg
git worktree remove ../wt/ref
git worktree remove ../wt/train
git worktree remove ../wt/docs
git worktree remove ../wt/phys
git worktree prune
```

4. 更新：

   * `PROJECT_STATE.md`
   * `WORKTREE_BOARD.md`
   * `CHANGELOG.md`

5. 考虑是否将 `int/multi-agent` 合并回 `dev` / `main`：

```bash
git switch dev
git merge --no-ff int/multi-agent
git switch main
git merge --no-ff dev
git tag v0.2-multi-agent-complete
```

---

## 24. 一句话执行摘要

本轮多 worktree 协作的母分支不是 `main` / `dev`，而是**从当前真实实现分支 `feat/m0-bootstrap` 冻结出来的 `int/base-current`**；
先合 Packaging，再合 Docs，再合 Reference，再合 Hybrid，最后才做 Residual / PML 审计。
任何人都不要第一天就去同时改 `trainer.py`、`runner.py`、`residual.py` 和 `base.yaml`。
