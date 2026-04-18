# A1_PACKAGING_CI.md

> Agent A1 专用执行手册。  
> A1 只负责 packaging、安装、脚本入口、最小 CI，与之直接相关的配置加载兼容；不负责任何物理公式、训练逻辑、参考解逻辑、结果指标定义。  
> 本文件只定义 A1 的静态职责与动作顺序。动态状态、最新集成提交、Gate 开闭、合并结果，一律以 `WORKTREE_BOARD.md` 为准。

---

## 1. 身份与协作位置

**Agent ID**：A1  
**分支**：`feat/packaging-ci`  
**worktree**：`../wt/pkg`  
**合并顺序**：第 1 个进入 `int/multi-agent`  
**直接交付对象**：A0（集成）  
**文档交接对象**：A4（README / CHANGELOG / PROJECT_STATE）  
**配置兼容通知对象**：A3（是否已支持多 overlay）

A1 是本轮并行开发的**基础设施先手**。A1 不做大功能，只做三件事：

1. 把当前仓库变成一个**可安装、可测试、可运行**的 Python 项目。
2. 清理当前脚本入口里的 `sys.path` 临时 hack，让运行方式从“依赖当前目录技巧”切到“依赖安装后的包入口”。
3. 补一个**最小但真实**的 CI，让后续 A2 / A3 / A5 的改动都能落在同一套安装与测试路径上。

A1 的输出不是“看起来更整洁”，而是给后续所有 agent 提供一个统一事实：

- `pip install -e ".[dev]"` 是标准安装方式；
- `pytest -q` 是标准测试方式；
- `helmholtz-train ...` 和 `python scripts/evaluate_run.py ...` 是标准运行方式；
- 后续 agent 不再用 `PYTHONPATH=.` 或手工改 `sys.path` 规避导入问题。

---

## 2. 当前仓库中，A1 必须面对的真实现状

A1 不是从零开始设计 packaging，而是在**现有仓库事实**上做收口。当前与 A1 直接相关的现状如下：

1. 根目录已经存在 `pyproject.toml`，且已经声明：
   - 项目名 `helmholtz-nsno-2d`
   - Python 版本 `>=3.9,<3.10`
   - 运行依赖与开发依赖
   - CLI 入口 `helmholtz-train = "src.train.runner:main"`
2. 根目录还保留了 `requirements.txt`，但它和 `pyproject.toml` 维护的是**两套重复依赖信息**。
3. `scripts/run_train.py` 和 `scripts/evaluate_run.py` 目前都依赖：
   ```python
   ROOT = Path(__file__).resolve().parents[1]
   if str(ROOT) not in sys.path:
       sys.path.insert(0, str(ROOT))
   ```
   这种方式本质上是在用运行目录修补导入路径。
4. `src` 当前就是项目包名。这个命名不算理想，但**本轮禁止重命名包命名空间**，因为牵扯面太大，会把 A1 变成全仓重构。
5. `src/train/runner.py` 已经承担训练入口、评估入口、产物导出逻辑，A1 不能在这里扩 scope。

因此，A1 的正确策略不是“重新发明工程结构”，而是：

- 承认 `pyproject.toml` 已经是现成基础；
- 把它提升为**依赖与安装的唯一事实源**；
- 把脚本入口从“靠路径 hack”切到“靠安装后的包导入”；
- 用最小 CI 把这套路径钉死。

---

## 3. A1 的唯一目标

A1 完成后，仓库必须满足下面这组标准，而不是只满足其中一部分：

1. 在干净环境中执行 `pip install -e ".[dev]"` 可以成功安装项目与开发依赖。
2. 安装后，不设置 `PYTHONPATH=.`，也不依赖脚本内 `sys.path.insert(...)`，依然能运行：
   - `pytest -q`
   - `helmholtz-train --help`
   - `python scripts/run_train.py --help`
   - `python scripts/evaluate_run.py --help`
3. 安装后，可以完成一次最小训练 smoke 和一次评估 smoke。
4. 后续 agent 不需要再讨论“应该怎么装、怎么跑、导入为什么又坏了”。
5. 如果 A3 需要多 overlay，A1 要么已经在 `src/config.py` 里做了**向后兼容**支持，要么明确告诉 A3：当前仍只有单 overlay。

A1 的完成标准是**安装路径被固定**，不是“文件看起来更规范”。

---

## 4. 文件所有权与并改禁令

### 4.1 A1 独占文件

在 A1 未合并前，下面这些文件只允许 A1 修改；其他 agent 一律不得打开后直接提交：

- `pyproject.toml`
- `requirements.txt`
- `pytest.ini`
- `src/config.py`
- `scripts/run_train.py`
- `scripts/evaluate_run.py`
- `.github/workflows/*`
- `tests/test_config.py`（若新建）

### 4.2 A1 明确禁止修改的文件

A1 不得修改下列路径，即使“顺手改一下”也不允许：

- `src/train/trainer.py`
- `src/train/losses.py`
- `src/train/runner.py`
- `src/eval/*`
- `src/physics/*`
- `src/models/*`
- `src/core/*`
- `configs/base.yaml`
- `configs/debug.yaml`
- `tests/test_runner.py`
- `README.md`
- `CHANGELOG.md`
- `PROJECT_STATE.md`
- `DEV_PLAN.md`
- `WORKTREE_BOARD.md`

说明：

- 顶层文档由 A4 单写；A1 只提供文案 handoff，不直接提交文档。
- `src/train/runner.py` 是热点文件，A1 不碰，避免后续与 A2/A0 冲突。
- A1 不得改任何输出文件名、summary 键名、物理公式、训练语义。

---

## 5. 开工前同步协议

A1 每次**开始一个新的修改会话之前**，必须先完成下面动作。注意，是“每次开始新的修改会话”，不是只在第一次开工时做一次。

### 5.1 先认统一基线，不追别人的 feature 分支

A1 只认两种上游：

1. `int/base-current`：A1 正常起步基线
2. `int/multi-agent`：如果 A0 已经合并了比你更新的内容，则以它为最新集成状态

A1 **不得**：

- 去追 A2 / A3 / A4 / A5 的 feature branch
- 用别人的未合并提交当“最新状态”
- 在自己本地猜测哪个分支更新

### 5.2 同步检查步骤

进入 `../wt/pkg` 后，顺序执行：

```bash
git status --short
git fetch --all --prune
```

要求：

- `git status --short` 必须为空；有未提交改动时，先收尾或 stash，禁止脏树同步。
- 然后查看 `WORKTREE_BOARD.md`，确认：
  - A1 当前仍是允许开工状态
  - A0 没有记录“Packaging 冻结”或“等待回退”
  - 如看板已记录最新集成提交，则记下该 commit

之后按下面规则同步：

- 如果 A0 还没合并任何分支，A1 从 `int/base-current` 快进同步。
- 如果 `int/multi-agent` 已经领先于 `int/base-current`，A1 必须先同步 `int/multi-agent`，再开始新的文件修改。

推荐命令：

```bash
git merge --ff-only int/base-current
# 或（如果 int/multi-agent 已经领先）
git merge --ff-only int/multi-agent
```

### 5.3 硬规则

只要 `int/multi-agent` 的基线发生变化，A1 就**不能在旧基线上开启新的文件修改**。  
已经打开并接近完成的一个极小补丁可以先收尾，但收尾后必须立即同步，再继续下一个文件。

---

## 6. A1 的执行顺序

A1 必须按下面顺序做，不能乱序平推。原因很简单：A1 处理的是“安装事实”，顺序错了，会导致后面验证失真。

### 第 1 步：确定依赖与安装的唯一事实源

先改 `pyproject.toml` 与 `requirements.txt`，不要一上来先改脚本。

目标是把依赖真相收敛到一个地方：

- **唯一事实源**：`pyproject.toml`
- **兼容入口**：`requirements.txt`

A1 必须做出的决策：

1. `pyproject.toml` 继续作为唯一事实源保留。
2. `requirements.txt` 不再独立维护一整套依赖版本；它只作为“习惯用 `pip install -r requirements.txt` 的兼容入口”。

推荐做法：把 `requirements.txt` 缩成一个薄包装器，例如：

```text
# Compatibility entrypoint.
# Source of truth is pyproject.toml.
-e .[dev]
```

这样做的目的不是省字数，而是消灭“双份依赖声明迟早漂移”的风险。

A1 在这一阶段**不要做**：

- 不扩大 Python 版本到 3.10+
- 不把 `torch` 版本范围大改成完全不同区间
- 不重命名项目包
- 不新增第二套打包体系（如 `setup.py` / `setup.cfg`）

### 第 2 步：验证 editable install 真能工作

在改脚本前，先验证安装链路：

```bash
python -m pip install -U pip
pip install -e .
pip install -e ".[dev]"
```

然后至少做以下导入检查：

```bash
python -c "import src; from src.train.runner import main; print('ok')"
```

如果这里都没过，不准进入下一步；先把 `pyproject.toml` 修到通过。

### 第 3 步：清理脚本入口里的 `sys.path` hack

只允许修改：

- `scripts/run_train.py`
- `scripts/evaluate_run.py`

这一步只做两件事：

1. 删除 `ROOT = ...` 与 `sys.path.insert(...)`
2. 保持原有 CLI 参数、原有调用目标、原有行为不变

也就是说，清理后的脚本应该仍然只是一个很薄的入口壳，不要在这里顺手加业务逻辑。

**允许的变化**：

- 删除 `import sys`
- 删除 `from pathlib import Path`（如果不再需要）
- 保留 `from src.train.runner import main` 或 `evaluate_saved_run`

**禁止的变化**：

- 不改参数名
- 不改默认值
- 不把评估逻辑塞回 `src/train/runner.py`
- 不新增 `helmholtz-evaluate`

说明：本轮不新增 `helmholtz-evaluate`，不是因为它没价值，而是它会诱导 A1 去动 `src/train/runner.py` 或新建额外 CLI 模块，容易超 scope。

### 第 4 步：验证脚本入口与现有 CLI

在完成第 3 步后，必须验证下面 3 条都成立：

```bash
helmholtz-train --help
python scripts/run_train.py --help
python scripts/evaluate_run.py --help
```

注意这里的判断标准是：**安装后可用**。  
A1 不需要保证“完全没安装也能直接跑脚本”；A1 要保证的是“安装后不需要 path hack 也能跑”。

### 第 5 步：按需扩展 `src/config.py` 的多 overlay 支持

这是 A1 的**可选但推荐任务**。只有当 A3 预计需要 `base + debug + experiments/hybrid` 这种多层覆盖时，A1 才做。

建议目标接口：

```python
def load_config(base_path, *overlay_paths):
    ...
```

实现要求：

1. 向后兼容现有调用方式：
   - `load_config(base)` 仍然可用
   - `load_config(base, overlay)` 仍然可用
2. 多个 overlay 时按**从左到右**依次覆盖：
   - `base <- overlay1 <- overlay2 <- overlay3`
3. `None` 必须被安全跳过，避免现有 `load_config(base_config, overlay_config)` 在 `overlay_config is None` 时出错。
4. 只做“加载与深合并”，**不引入 schema 校验系统**。
5. 不改变 `Config` 类的属性访问语义。

如果 A1 决定**不做**多 overlay，就必须在 handoff 里明确写：

- 当前仍只支持单 overlay
- A3 若要多 overlay，需要自己在实验 overlay 里写完整覆盖

### 第 6 步：为配置加载补最小测试

如果做了第 5 步，就应同步补 `tests/test_config.py`。  
即使没做第 5 步，只要 `src/config.py` 被改过，也建议补测试。

最少覆盖以下场景：

1. 只加载 base
2. base + 一个 overlay
3. base + 多个 overlay（若已实现）
4. 覆盖文件不存在时抛 `FileNotFoundError`
5. 深层嵌套键合并行为正确

A1 不需要在这里引入大规模 schema 验证测试。

### 第 7 步：补最小 CI

A1 只补**最小真实可用** CI，不做大矩阵。

推荐新建：

- `.github/workflows/ci.yml`

建议工作流内容：

- 触发：`push`、`pull_request`
- 平台：`ubuntu-latest`
- Python：只测 `3.9`
- 安装：`pip install -e ".[dev]"`
- 测试：`pytest -q`
- CLI smoke：
  - `helmholtz-train --overlay configs/debug.yaml --device cpu --epochs 1 --velocity-model smooth_lens --output-dir ci_runs/train_1ep`
  - `python scripts/evaluate_run.py --run-dir ci_runs/train_1ep --device cpu`

环境变量建议固定：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

A1 不需要做：

- CUDA CI
- 多 Python 版本矩阵
- benchmark job
- coverage 门禁
- lint / formatter 全套流程

### 第 8 步：把文档变化交给 A4，而不是自己改 README

A1 会改变用户层命令，但 A1 不直接改 README。  
A1 在完成代码后，要给 A4 一个**明确、可直接粘贴**的命令 handoff，至少包含：

- 安装命令
- 测试命令
- 训练命令
- 评估命令
- 是否支持多 overlay
- 是否要求先安装后再运行 `python scripts/*.py`

---

## 7. 分文件改动规则

### 7.1 `pyproject.toml`

A1 对这个文件的目标只有两个：

1. 确认它真能支撑 editable install
2. 让它成为唯一依赖事实源

硬规则：

- 保留 `requires-python = ">=3.9,<3.10"`
- 保留 `helmholtz-train = "src.train.runner:main"`
- 不重命名项目，不改包发现策略为另一种大方案
- 不引入新的打包工具链

### 7.2 `requirements.txt`

A1 不再维护一整套和 `pyproject.toml` 平行的依赖版本。  
这个文件的目标是兼容，不是主配置。

优先方案：

```text
-e .[dev]
```

如果目标环境对这种写法有兼容问题，再退回“镜像 `pyproject.toml` 依赖”的方案；但那是次优方案。

### 7.3 `scripts/run_train.py`

只清理入口，不动业务逻辑。最终应满足：

- 无 `sys.path` hack
- 安装后 `python scripts/run_train.py --help` 可用
- 现有参数接口不变
- 调用仍落到 `src.train.runner:main`

### 7.4 `scripts/evaluate_run.py`

与 `run_train.py` 相同，只清理入口，不改变行为。

### 7.5 `src/config.py`

只有两类允许改动：

1. 与安装 / 导入相关的最小兼容修补
2. 多 overlay 支持（可选）

禁止：

- 引入 schema 框架
- 改变 `Config` 的核心访问风格
- 在这里塞 project-wide 业务逻辑

### 7.6 `pytest.ini`

尽量不动。只有在确实影响 `pytest -q` 统一执行时才改。  
禁止通过在 `pytest.ini` 里偷偷加 `pythonpath = .` 来掩盖 packaging 问题。

### 7.7 `.github/workflows/*`

只允许加入最小 CI。  
不要在 A1 顺手加 release、publish、wheel build、artifact upload 等非本轮必需流程。

---

## 8. 提交粒度要求

A1 不允许把所有改动揉成一个“大杂烩提交”。最少拆成下面 3 类提交；更细可以，但不能更粗：

1. **build 提交**  
   处理 `pyproject.toml` / `requirements.txt`
2. **entrypoint 提交**  
   处理两个 `scripts/*.py`
3. **ci/test 提交**  
   处理 `tests/test_config.py` 与 `.github/workflows/ci.yml`

推荐提交顺序：

```text
build: make pyproject.toml the packaging source of truth
refactor: remove sys.path hacks from train and eval entrypoints
test: add config loader coverage
ci: add minimal install-test-smoke workflow
```

---

## 9. 验证命令

A1 自测必须至少跑完下面这套命令，且记录结果交给 A0：

```bash
python -m pip install -U pip
pip install -e ".[dev]"

python -c "import src; from src.train.runner import main; print('import-ok')"
helmholtz-train --help
python scripts/run_train.py --help
python scripts/evaluate_run.py --help

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q

rm -rf ../runs/pkg/smoke 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
helmholtz-train \
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

如果 A1 做了多 overlay，还要额外加一条针对 `load_config()` 的测试或临时验证。

---

## 10. 与其他 agent 的协作规则

### 10.1 与 A0 的关系

A1 的所有交付都先给 A0，A1 自己不宣布“已成为新标准”。  
只有当 A0 合并进 `int/multi-agent` 并通过统一 smoke，A1 的输出才算进入全局最新状态。

### 10.2 与 A3 的关系

A3 依赖 A1 给出一个明确信号：

- **已支持多 overlay**，可直接在 `configs/experiments/hybrid.yaml` 上叠加；
- 或 **未支持多 overlay**，A3 必须把所需参数写进完整 overlay，不再假设多层合并。

A1 必须把这个结果写进 handoff，不能让 A3 自己猜。

### 10.3 与 A4 的关系

A1 不直接改顶层文档。A1 要把下面内容交给 A4：

- 安装命令从什么改成什么
- 测试命令从什么改成什么
- 训练入口是否推荐 `helmholtz-train`
- 评估入口是否仍为 `python scripts/evaluate_run.py`
- 是否需要先 `pip install -e ".[dev]"`
- 多 overlay 是否已支持

---

## 11. Handoff 模板

### 11.1 A1 -> A0

```text
[A1 -> A0 HANDOFF]
branch: feat/packaging-ci
base_commit: <A1 开工时使用的基线 commit>
final_commit: <A1 最后提交 commit>

files_changed:
- pyproject.toml
- requirements.txt
- scripts/run_train.py
- scripts/evaluate_run.py
- src/config.py          # 若改了就列
- tests/test_config.py   # 若新建就列
- .github/workflows/ci.yml

packaging_decision:
- pyproject.toml is the single source of truth: yes/no
- requirements.txt reduced to compatibility wrapper: yes/no
- multi-overlay support added in src/config.py: yes/no

commands_run:
- pip install -e ".[dev]"
- pytest -q
- helmholtz-train --help
- python scripts/run_train.py --help
- python scripts/evaluate_run.py --help
- 1-epoch train smoke
- evaluate smoke

result:
- all passed / failed

notes_for_integrator:
- any known limitation
- any behavior intentionally unchanged
```

### 11.2 A1 -> A4

```text
[A1 -> A4 DOC HANDOFF]
Install:
  pip install -e ".[dev]"

Test:
  pytest -q

Train:
  helmholtz-train --overlay configs/debug.yaml --device cpu --epochs 1 --velocity-model smooth_lens --output-dir <dir>

Evaluate:
  python scripts/evaluate_run.py --run-dir <dir> --device cpu

Important notes:
- running scripts now assumes the project has been installed
- no PYTHONPATH=. needed
- multi-overlay support: yes/no
```

### 11.3 A1 -> A3

```text
[A1 -> A3 CONFIG STATUS]
load_config multi-overlay support: yes/no
accepted calling pattern:
- load_config(base)
- load_config(base, overlay)
- load_config(base, overlay1, overlay2, ... )   # if supported

merge order:
- overlays are applied left to right / not supported
```

---

## 12. Blocked 条件与处理办法

出现下面任一情况，A1 应立即停在本阶段，不要私自扩 scope：

1. 为了补 packaging，不得不修改 `src/train/runner.py`
2. 为了补 CI，不得不改训练语义或物理逻辑
3. 为了新增 evaluate CLI，不得不改一大片模块结构
4. 为了让脚本直接运行，不得不重新设计包命名空间

处理规则：

- 能在 A1 独占文件里解决的，就继续。
- 不能在 A1 独占文件里解决的，记录为 `BLOCKED`，交给 A0 决定是否拆新任务。
- 不允许以“先把功能做出来”为理由越界修改训练、参考解、物理代码。

---

## 13. 完成定义（Definition of Done）

A1 只有在下面所有条件同时成立时，才算完成：

- [ ] `pyproject.toml` 被确认是依赖与安装的唯一事实源
- [ ] `requirements.txt` 不再维护独立依赖真相
- [ ] `pip install -e ".[dev]"` 可用
- [ ] `helmholtz-train --help` 可用
- [ ] `python scripts/run_train.py --help` 可用
- [ ] `python scripts/evaluate_run.py --help` 可用
- [ ] `pytest -q` 可在安装后直接通过
- [ ] 最小训练 smoke 通过
- [ ] 最小评估 smoke 通过
- [ ] 两个脚本里都已删除 `sys.path.insert(...)`
- [ ] 未修改任何物理语义、训练语义、输出文件名、summary 键名
- [ ] 已向 A0 提交 handoff
- [ ] 已向 A4 提交文档命令 handoff
- [ ] 已向 A3 明确多 overlay 状态

只做到“能安装”但没有 smoke，不算完成。  
只做到“脚本去掉 hack”但没有 CI，不算完成。  
只做到“本地能跑”但没有向 A0/A4/A3 明确交接，也不算完成。

---

## 14. 本轮明确不做

A1 本轮明确不做下面这些事：

- 不把包名从 `src` 改成其它名字
- 不改 `src/train/runner.py` 的业务逻辑
- 不新增 `helmholtz-evaluate`
- 不改 `README.md` / `CHANGELOG.md` / `PROJECT_STATE.md`
- 不引入 lint / formatter / pre-commit 全家桶
- 不做多 Python 版本 CI 矩阵
- 不做 CUDA CI
- 不做发布流程
- 不做全仓重构

A1 的任务是**把工程入口收紧**，不是借 packaging 的名义把全仓翻新。

---

## 附录 A. 源码核查与补充说明

本附录记录在将本手册落盘前对当前仓库代码所做的交叉核查结果，供 A1 在真正开工时参照，不改变正文中的职责定义与边界。

### A.1 现有 `pyproject.toml` 确认

核查文件：`pyproject.toml`（仓库根目录）。

当前事实（与本文件 §2 一致）：

- `build-system.requires` 使用 `setuptools >= 61`；`build-backend = "setuptools.build_meta"`。
- `project.name = "helmholtz-nsno-2d"`。
- `project.requires-python = ">=3.9,<3.10"`。
- `project.scripts` 中已存在 `helmholtz-train = "src.train.runner:main"`。
- `project.optional-dependencies` 已提供 `dev` 分组。

A1 无需从零建 packaging，只需把 `requirements.txt` 收敛与 `sys.path` hack 清理按 §6 执行。

### A.2 现有 `scripts/*.py` 中 `sys.path` hack 范围

核查文件：

- `scripts/run_train.py`
- `scripts/evaluate_run.py`

两份脚本顶部均存在下列结构，需在 §6 第 3 步中删除：

```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

仓库内除上述两处以外，目前未发现其他 `sys.path.insert(...)` 形式的入口修补。

### A.3 `src/config.py` 现状：单 overlay 且 `Config` 无 `.get()`

这是本手册正文中默认给 A3 / A5 的兼容信号，此处补强事实依据：

- `load_config(base_path, overlay_path=None)` 当前签名**只收一个 overlay**，无可变参数。
- `Config` 类目前只实现：
  - 属性访问（`cfg.training.lr`）；
  - `__setattr__` / `__getitem__` / `__contains__` / `to_dict` / `__repr__`。
- `Config` **没有实现** `.get()`、`keys()`、`values()`、`items()`、`update()`。

A1 在 §6 第 5 步做可选的多 overlay 扩展时，必须遵守：

1. 不要借“顺手完善接口”之名给 `Config` 添加 dict-like 方法；本轮只处理加载合并语义。
2. 新签名 `load_config(base, *overlays)` 必须保持旧调用 `load_config(base)` / `load_config(base, overlay)` 的结果与旧实现一致。
3. `None` overlay 必须被安全跳过（旧调用方可能把 `overlay_config` 设为 `None`）。

### A.4 A3 将依赖 A1 的兼容信号

A1 在 handoff 中需要明确给 A3 的事实（参见 §11.3）：

- 若 A1 本轮**未**做多 overlay，A3 必须把 `hybrid.yaml` 写成**单 overlay 完整配置**（即把 `debug.yaml` 中影响 smoke 成本的关键字段也直接写进去），不得依赖 `base + debug + hybrid` 三层叠加。
- 无论 A1 是否扩展 `load_config`，A3 的代码里都**不得**假设 `Config` 有 `.get()`。  
  这不是 A1 多 overlay 扩展的可选副产品，而是 `Config` 当前实现的硬事实。

### A.5 `pytest.ini` 现状提醒

核查文件：`pytest.ini`。

A1 在 §7.6 已明确禁止通过 `pythonpath = .` 类写法掩盖 packaging 问题；本附录在此重申：即便 A1 发现当前 `pytest.ini` 中已有某些与路径相关的设定，也**不得**在本轮扩大其作用面。若确实发现 `pytest.ini` 阻碍安装后的 `pytest -q` 执行，应改 packaging，不改 `pytest.ini`。

### A.6 本附录不改变 A1 职责

以上核查只是对 A1 正文中假设前提的事实对账，目的是：

- 让 A1 在开工前对仓库现状有明确确认，不必再做重复探查；
- 让 A1 的 `pyproject.toml` / `requirements.txt` / 脚本清理决策直接落在真实代码上；
- 让 A1 与 A3 的兼容信号不被遗漏。

附录**不追加**新任务，也**不放宽**本文件 §4、§7、§14 中的禁令。若仓库在 A1 真正开工时与本附录描述已经不一致，应以当时 `int/multi-agent` 中的代码为准。
