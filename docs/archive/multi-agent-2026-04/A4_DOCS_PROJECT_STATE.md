# A4_DOCS_PROJECT_STATE.md

> Agent A4 专用执行手册。  
> A4 只负责顶层静态文档同步：`README.md`、`CHANGELOG.md`、`PROJECT_STATE.md`、`DEV_PLAN.md`。A4 不负责任何功能代码、测试逻辑、物理公式、训练接口或集成看板状态写入。  
> 本文件只定义 A4 的静态职责、改单边界、动作顺序和交接格式。动态状态、最新集成提交、Gate 开闭、分支是否已被 A0 接受，一律以 `WORKTREE_BOARD.md` 为准。

---

## 1. 身份与协作位置

**Agent ID**：A4  
**分支**：`chore/project-state`  
**worktree**：`../wt/docs`  
**首次合并顺位**：第 2 个进入 `int/multi-agent`（A1 之后）  
**直接交付对象**：A0（集成）  
**上游信息来源**：A1 / A2 / A3 / A5 的已接受 handoff + `int/multi-agent` 中已合并代码  
**职责持续时间**：全流程；即使首次合并后，后续顶层文档仍由 A4 单写收口

A4 是本轮并行开发里的“**仓库叙事对齐者**”。A4 不是来写论文式说明，也不是来补业务代码；A4 只做四件事：

1. 让顶层文档与**已集成代码事实**一致；
2. 新建并维护 `PROJECT_STATE.md`，把“当前真实基线、输出契约、已知问题、下一步依赖关系”集中写清楚；
3. 把 `README.md` 保持在“新读者 5 分钟能上手”的粒度，而不是把所有细节都塞进去；
4. 把其他 agent 的文档输入统一收口，禁止任何人“顺手改 README / CHANGELOG / DEV_PLAN”。

A4 的产物不是“文档更好看了”，而是建立下面这条信息层级：

- `WORKTREE_BOARD.md`：**动态协作状态**（谁在做、谁已合并、Gate 开没开）——由 A0 单写；
- `PROJECT_STATE.md`：**静态项目快照**（当前基线已有什么、怎么跑、有哪些已知技术债）；
- `README.md`：**新人入口**（项目范围、快速开始、项目结构、里程碑概览）；
- `CHANGELOG.md`：**历史变更记录**（按基线 / 分支 / 集成结果记录）；
- `DEV_PLAN.md`：**冻结决策 + 里程碑检查表**（只更新进度与勾选，不改理论冻结内容）。

A4 不是一次性分支。A4 首次合并顺位虽然是第 2，但后续只要 A2 / A3 / A5 进入 `int/multi-agent` 并改变了用户可见行为、输出文件或配置字段，A4 仍要基于最新 `int/multi-agent` 继续做**doc-only follow-up**。

---

## 2. 当前仓库中，A4 必须面对的真实现状

A4 不是抽象地“更新一下文档”，而是要面对当前仓库已经存在、而顶层文档尚未准确反映的真实情况。

### 2.1 `README.md` 明显落后于当前代码事实

当前 `README.md` 至少有下面这些已确认偏差：

1. **里程碑进度失真**  
   README 里程碑表仍显示：
   - `M0 = ✅`
   - `M1-M9 = ⬜`

   但当前仓库实际上已经存在并入库：
   - `src/core/*`、`src/physics/*`、`src/models/*`、`src/train/*`
   - `tests/test_grid.py`、`tests/test_background.py`、`tests/test_eikonal.py`、`tests/test_tau_ops.py`、`tests/test_residual.py`、`tests/test_train.py`、`tests/test_runner.py`
   - 训练入口 `src/train/runner.py`
   - 评估模块 `src/eval/reference_solver.py`

   也就是说，README 的粗粒度里程碑表已经不能再停留在“只有 M0 完成”的状态。

2. **项目结构过时**  
   README 的树状结构当前缺少或未体现：
   - `pyproject.toml`
   - `src/eval/`
   - `src/train/runner.py`
   - 顶层脚本 `scripts/evaluate_run.py`
   - `PROJECT_STATE.md`（目前还不存在，但 A4 需要新建并接入）

3. **快速开始命令没有跟工程事实同步**  
   当前 README 仍使用：
   - `pip install -r requirements.txt`
   - `python scripts/run_train.py ...`
   - `python scripts/evaluate_run.py ...`

   但当前仓库已经存在：
   - `pyproject.toml`
   - `project.scripts.helmholtz-train = "src.train.runner:main"`

   而 A1 的职责就是进一步把安装入口和脚本导入路径收紧。因此 A4 必须以**A1 已合并后的真实状态**来更新安装 / 训练命令，而不是继续沿用旧 README 的表述。

4. **README 还没有明确区分“已集成事实”和“下一步计划”**  
   例如当前仓库已经有 `src/eval/reference_solver.py`，但还没有：
   - `scripts/solve_reference.py`
   - `tests/test_reference_solver.py`
   - `configs/experiments/hybrid.yaml`

   所以 A4 必须把“模块已在库中”与“完整功能链路已集成”明确分开写，不能把计划中的 A2 / A3 / A5 说成已经完成。

### 2.2 `CHANGELOG.md` 目前几乎只有骨架

当前 `CHANGELOG.md` 只记录了：

- `DEV_PLAN.md` 建立
- `CHANGELOG.md` 建立
- “当前处于 M0 阶段”

这与代码现实完全不符。至少以下已经在当前仓库真实存在，但 changelog 没反映：

- M1：网格 / 介质 / 源项
- M2：背景场与解析先验
- M3：因子化 Eikonal
- M4：安全导数重组
- M5：PML 与差分引擎
- M6：等效体源与穿孔掩码
- M7：NSNO 网络骨架
- M8：PDE residual 前向检查
- M9：最小训练实验
- 顶层运行器 `src/train/runner.py`
- 诊断导出产物与 `tests/test_runner.py`
- `src/eval/reference_solver.py`
- `pyproject.toml`

A4 不能再让 changelog 继续停在“只有 M0”。

### 2.3 `DEV_PLAN.md` 的检查项全部未勾选，但不能粗暴“一键全勾”

当前 `DEV_PLAN.md` 的里程碑检查项仍大面积保留为未勾选状态，这当然说明文档进度落后；但 A4 也**不能**因为看到代码文件存在，就把所有小项直接勾满。

A4 需要区分三类证据：

1. **代码 + 测试都支持**  
   例如：
   - `import src` 不报错
   - `tests/test_grid.py`、`tests/test_train.py`、`tests/test_runner.py` 等已有覆盖
   - runner 产物名被测试断言锁定

   这类检查项可以勾。

2. **代码存在，但缺少明确验证证据**  
   例如：
   - “可视化检查通过”
   - “与参考 FMM 误差 < 1e-4”
   - “远场渐近行为合理”

   如果当前仓库没有对应测试、记录或已接受实验摘要，A4 不能凭印象直接勾选。

3. **未来分支才会完成**  
   例如：
   - reference 评估闭环
   - hybrid loss 真正接通
   - residual / PML 审计定量结论

   这些必须保持未完成或在 `PROJECT_STATE.md` 中标注“待 A2 / A3 / A5 集成”。

因此，A4 的正确策略不是“全部勾上”，而是：

- README 的**粗粒度里程碑表**可以对齐到当前实现级别；
- `DEV_PLAN.md` 的**细粒度检查项**只勾有证据支撑的内容；
- 对于“代码有，但验证证据不足”的条目，要么保持未勾，要么补一句“待复核”，但不能伪造通过。

### 2.4 `PROJECT_STATE.md` 当前缺失，导致顶层信息没有合适承载位

当前仓库没有 `PROJECT_STATE.md`。这会导致两个问题：

1. README 被迫承担过多职责，既想做 quickstart，又想做状态同步；
2. 没有一个地方集中写：
   - 当前真实基线是什么
   - 已集成能力有哪些
   - 统一 smoke 命令是什么
   - artifact 契约是什么
   - Python 版本与依赖约束是什么
   - 现有已知技术债是什么
   - A2 / A3 / A5 的依赖顺序是什么

A4 必须把 `PROJECT_STATE.md` 建起来，否则顶层叙事仍然会分裂在 README / BOARD / 代码里。

### 2.5 当前代码里，已经有很多 A4 必须写进文档的“硬事实”

A4 更新文档时，必须至少吸收下面这些当前仓库里已经成立的事实：

1. **Python / packaging 事实**
   - `pyproject.toml` 已存在；
   - `requires-python = ">=3.9,<3.10"`；
   - build backend 是 `setuptools.build_meta`；
   - `helmholtz-train` 已在 `project.scripts` 中定义。

2. **配置系统事实**
   - `src/config.py` 当前只支持 `base + 单个 overlay`；
   - `Config` 支持属性访问和 `in` 检查，但没有 `.get()`。

3. **运行入口事实**
   - `scripts/run_train.py` 仍存在；
   - `scripts/evaluate_run.py` 仍存在；
   - `src/train/runner.py` 是训练与评估公共入口；
   - `scripts/evaluate_run.py` 支持 `--run-dir` 和 `--device`。

4. **输出契约事实**  
   由 `tests/test_runner.py` 锁定的已有文件包括：
   - `config_merged.yaml`
   - `summary.json`
   - `losses.npy`
   - `losses.csv`
   - `loss_tail.csv`
   - `loss_curve.png`
   - `loss_curve_log.png`
   - `wavefield.npy`
   - `wavefield.png`
   - `model_state.pt`
   - `metrics_summary.csv`
   - `quantiles.csv`
   - `centerline_profiles.csv`
   - `x_coords_physical.csv`
   - `y_coords_physical.csv`
   - `velocity_physical.csv`
   - `wavefield_magnitude_physical.csv`
   - `residual_magnitude_physical.csv`
   - `field_heatmaps.png`
   - `residual_heatmaps.png`

   另外还有：
   - `loss_mask_physical.csv`
   - `scattering_magnitude_physical.csv`

   这两个由 runner 导出，但当前测试尚未断言覆盖。A4 应在 `PROJECT_STATE.md` 中区分“已有且被测试锁定”与“已有但尚未被测试锁定”。

5. **范围边界事实**
   - 当前仍是 2D、单源、单机原型；
   - 当前不做 3D、多源联合训练、分布式、大 benchmark；
   - 当前 residual / PML 的严格审计还没做完；
   - reference solver 模块已存在，但 reference 评估基线仍待 A2 收口；
   - data supervision 公式已存在，但训练闭环仍待 A3 真正接通。

### 2.6 A4 最容易犯的错，就是把“希望”写成“事实”

A4 很容易被诱导去做这些错误动作：

- 看到 A2 / A3 / A5 的设计文件，就把这些功能写成“已支持”；
- 看到 A1 计划改 quickstart，就提前改 README，但代码还没合并；
- 为了让 README 与代码对齐，顺手去改脚本 / 配置 / 测试；
- 为了让 `DEV_PLAN.md` 看起来好看，把没有证据的可视化 / benchmark 检查项也勾掉；
- 为了更新看板状态，直接去改 `WORKTREE_BOARD.md`。

这些都属于越界。A4 的正确策略是：

- 只写**已集成**、可在 `int/multi-agent` 看到的事实；
- 对未合并分支，只能写“计划 / in-flight / 待集成”，不能写“已实现”；
- 如果事实与现有文档冲突，**改文档，不改代码**；
- `WORKTREE_BOARD.md` 由 A0 单写，A4 只能给 A0 提 board 文案建议。

---

## 3. A4 的唯一目标

A4 完成后，仓库必须具备下面这一整套文档能力，而不是只把 README 修几行：

1. 新读者只看顶层文档，就能知道：
   - 项目当前范围是什么；
   - 当前已集成到哪一步；
   - 怎么安装、怎么测试、怎么训练、怎么对已有 run 目录做评估；
   - 当前有哪些已知技术债和下一步依赖关系。

2. `README.md` 成为**短而准的入口文档**：
   - 只放项目简介、范围、核心管线、快速开始、项目结构、粗粒度里程碑；
   - 不承担完整 artifact 契约和所有技术债细节。

3. `PROJECT_STATE.md` 成为**静态项目快照**：
   - 精确说明当前真实基线；
   - 列出已实现能力、统一 smoke 命令、artifact 契约、依赖约束、已知问题、并行分支依赖；
   - 明确区分“已集成”和“计划中 / 待集成”。

4. `CHANGELOG.md` 不再停留在 M0，而是能回溯到当前 baseline 已具备的主要能力与后续 agent 合并结果。

5. `DEV_PLAN.md` 的进度状态与当前代码现实一致，但冻结决策不被 A4 改写。

6. 后续 A1 / A2 / A3 / A5 任何一个分支改变了用户可见命令、配置字段、输出文件或项目叙事，A4 都是唯一负责顶层文档最终落盘的人。

A4 的目标不是“把文字补齐”，而是让顶层文档成为**对当前项目真实状态的可信映射**。

---

## 4. 文件所有权与并改禁令

### 4.1 A4 独占文件

A4 在本轮中独占以下文件；其他 agent 一律不得直接改动：

- `README.md`
- `CHANGELOG.md`
- `PROJECT_STATE.md`（新建）
- `DEV_PLAN.md`（仅限进度 / 检查项 / 状态说明，见后文限制）

A4 是顶层静态文档的**唯一写手**。A1 / A2 / A3 / A5 只能给 A4 handoff，不得“顺手改文档”。

### 4.2 A4 明确禁止修改的文件

A4 不得修改以下任何路径，即使只是“补一行说明”也不允许：

- `WORKTREE_BOARD.md`
- `pyproject.toml`
- `requirements.txt`
- `pytest.ini`
- `configs/*`
- `scripts/*`
- `src/*`
- `tests/*`
- `.github/workflows/*`

说明：

- `WORKTREE_BOARD.md` 是动态调度板，归 A0 单写；
- A4 不能为了让文档与代码一致而去修改代码；
- A4 也不能通过改测试来“证明”自己的文档表述成立。

### 4.3 A4 对 `DEV_PLAN.md` 的特殊限制

A4 可以改 `DEV_PLAN.md`，但只限：

1. 里程碑检查项勾选状态；
2. 里程碑推进表中的进度标记；
3. 必要时，在**不触碰冻结决策语义**的前提下，加一小段状态说明，例如：
   - 当前 baseline 已实现到 M9；
   - 某些经验性 / 可视化类检查项仍待复核。

A4 **不得**：

- 改第 3 节 `冻结决策（Frozen Decisions）`；
- 改公式定义；
- 改范围冻结条款；
- 改 Git 工作流与回退规则；
- 把未来分支的设计提前写成已冻结事实。

---

## 5. 开工前同步协议

A4 每次**开始一个新的文档修改会话之前**，必须先完成下面动作。文档虽然不改代码，但一样必须建立在最新集成基线上。

### 5.1 只认 `int/multi-agent` 的最新集成状态

A4 的文档事实来源只有两个：

1. `WORKTREE_BOARD.md` 中由 A0 记录的最新状态；
2. `int/multi-agent` 中已经合并的代码。

A4 **不得**：

- 追 A1 / A2 / A3 / A5 的 feature branch 当事实源；
- 根据对话里“计划要做什么”提前更新文档；
- 把别人 handoff 里尚未被 A0 接受的内容写成“当前支持”。

### 5.2 每次开工前的同步步骤

进入 `../wt/docs` 后，顺序执行：

```bash
git status --short
git fetch --all --prune
```

要求：

- `git status --short` 必须为空；
- 如果文档工作区有未提交改动，先收尾或 stash；
- 然后查看 `WORKTREE_BOARD.md`，记下：
  - 最新已合并 agent；
  - `latest_integrated_commit`（若看板已维护）；
  - 哪些 Gate 已开；
  - 哪些分支只是计划中。

之后同步：

```bash
git merge --ff-only int/multi-agent
```

如果 `int/multi-agent` 还没领先于 `int/base-current`，也必须以看板记录为准，不得自行猜测哪个分支算最新。

### 5.3 A4 的“事实冻结”规则

只要 `int/multi-agent` 又合并了新内容，A4 就不能继续在旧基线上打开新一轮文档修改。  
A4 可以先把手头的一小块 doc patch 收尾，但收尾后必须立即：

1. 重新同步最新集成状态；
2. 重新检查是否有新的用户可见变更；
3. 再开始下一轮文档更新。

### 5.4 A4 的事实判断优先级

当多个来源冲突时，A4 必须按下面顺序认定事实：

1. `int/multi-agent` 中已合并代码；
2. `tests/*` 对输出契约和行为的断言；
3. `pyproject.toml` / `src/config.py` / `scripts/*` / `src/train/runner.py` 里的当前实现；
4. A1 / A2 / A3 / A5 已被 A0 接受的 handoff；
5. 旧 README / 旧 CHANGELOG / 旧 DEV_PLAN。

也就是说：**代码与测试优先级永远高于旧文档。**

---

## 6. A4 的执行顺序

A4 必须按下面顺序做，不能直接上手改 README。A4 的正确做法是先建立“事实清单”，再写用户入口文档。

### 第 1 步：先做事实盘点，不先写文字

A4 开始时，先从最新 `int/multi-agent` 提取一份“事实清单”。至少检查这些文件：

- `pyproject.toml`
- `requirements.txt`
- `src/config.py`
- `scripts/run_train.py`
- `scripts/evaluate_run.py`
- `src/train/runner.py`
- `tests/test_runner.py`
- `tests/test_train.py`
- `src/eval/reference_solver.py`
- `README.md`
- `CHANGELOG.md`
- `DEV_PLAN.md`
- `WORKTREE_BOARD.md`

A4 需要先回答下面问题，再开始写：

1. 当前安装命令到底是什么？
2. 训练推荐入口到底是什么？
3. 评估入口到底是什么？
4. 当前只支持单 overlay 还是多 overlay？
5. 当前已锁定的 artifact 文件名有哪些？
6. 当前已集成的功能到哪一步，哪些只是计划中的 agent？
7. 当前 Python 版本和依赖约束是什么？
8. 当前是否已经有 `PROJECT_STATE.md`，如果没有，应如何结构化新建？

A4 在没有完成这份事实盘点前，不得开始改 README 文案。

### 第 2 步：先写 `PROJECT_STATE.md`，把“真实状态”固定下来

A4 的第一落盘文件应是 `PROJECT_STATE.md`，而不是 README。

原因很简单：

- README 应该短；
- `PROJECT_STATE.md` 才适合承载完整状态快照；
- 先把状态写清楚，再去缩写成 README，能大幅减少反复改动。

`PROJECT_STATE.md` 必须先搭出固定框架，至少包含下面内容：

1. **快照头部**
   - 当前对齐的分支（通常是 `int/multi-agent`）；
   - 当前对齐的 commit；
   - 文档更新时间；
   - 当前仅说明“已集成事实”，不含未合并 feature branch 的承诺。

2. **当前项目一句话结论**
   - 当前是 2D、单源、单机 Helmholtz 原型；
   - 当前 baseline 已具备 M0-M9 主链；
   - reference / hybrid / residual audit 的完整闭环是否已集成，要按事实写。

3. **当前已实现能力**
   - 网格 / 介质 / 源项；
   - 背景场；
   - Eikonal；
   - 安全导数重组；
   - PML 与差分引擎；
   - 等效体源与掩码；
   - NSNO 网络前向；
   - PDE residual；
   - 最小训练与评估导出。

4. **统一命令**
   - 安装；
   - 测试；
   - 训练；
   - 对已有 `run_dir` 的评估；
   - 若 A2 / A3 / A5 已合并，再补对应 reference / hybrid / audit 命令。

5. **artifact 契约**
   - 测试锁定的已有输出文件名；
   - 由 runner 导出但暂未被测试锁定的文件；
   - 若 A2 / A3 / A5 已合并，再追加新 artifact，并明确来自哪个分支。

6. **配置与依赖约束**
   - Python 版本；
   - `pyproject.toml` 为准还是 `requirements.txt` 为准（按 A1 已集成状态写）；
   - 单 overlay / 多 overlay 支持状态；
   - 目前重要配置字段。

7. **当前已知问题 / 技术债**
   - residual / PML 的 `Δ̃τ` 混合策略；
   - reference solver 成本问题；
   - `resolve_output_dir(..., exist_ok=False)` 带来的使用注意事项；
   - 其他经 A0 接受并写进看板的已知问题。

8. **并行分支依赖关系**
   - 哪些已集成；
   - 哪些只是 planned / in-flight；
   - A2 / A3 / A5 的依赖顺序。

9. **建议推进顺序**
   - 当前下一步应该优先什么；
   - 哪些事情不应并行推进。

A4 写 `PROJECT_STATE.md` 时，要始终记住：这个文件是**静态项目快照**，不是第二份 WORKTREE_BOARD。

### 第 3 步：再回写 `README.md`

在 `PROJECT_STATE.md` 先成形后，再改 README。README 必须控制在“新人入口文档”粒度。

A4 对 README 的主要任务是：

1. **更新当前阶段描述**  
   不能继续写成只有 M0 完成的仓库。

2. **更新项目结构**  
   至少补齐：
   - `pyproject.toml`
   - `src/eval/`
   - `src/train/runner.py`
   - `scripts/evaluate_run.py`
   - `PROJECT_STATE.md`

3. **更新 quickstart**  
   快速开始必须跟当前已集成事实一致：
   - A1 若已合并，就使用 A1 收口后的安装 / 训练命令；
   - A1 若尚未合并，则 A4 不应最终化 quickstart，应先标记 blocked。

4. **更新里程碑概览表**  
   README 的里程碑表是**粗粒度功能概览**，不是 `DEV_PLAN.md` 的细粒度验收表。  
   因此，当前 baseline 若确实已经具备 M0-M9 对应主链能力，README 表可以更新到 M0-M9 ✅；  
   但对 A2 / A3 / A5 这类尚未集成的后续扩展，不能提前打 ✅。

5. **把长信息导向 `PROJECT_STATE.md`**  
   README 只做导流，不要把完整 artifact 契约、所有已知技术债、所有分支依赖全塞进去。

### 第 4 步：补全 `CHANGELOG.md`

A4 写 changelog 时，要把“当前 baseline 已有能力”与“后续 agent 合并结果”分层记录。

推荐写法：

1. **Baseline / bootstrap already present**  
   用一个大段记录当前 `feat/m0-bootstrap` 已经具备的主要能力，按 Added / Changed / Fixed 分组。

2. **Packaging / doc / eval / hybrid / audit follow-ups**  
   当 A1 / A2 / A3 / A5 合并后，再按功能块追加小节。

A4 写 CHANGELOG 时必须注意：

- 不伪造 release tag；
- 不伪造发布日期；
- 不写不存在的 benchmark 或结论；
- 不把“本轮计划做什么”写成 changelog 已发生条目。

如果仓库没有 tag / release 体系，A4 完全可以使用：

- `[Unreleased]`
- `Bootstrap baseline already present`
- `Integrated changes`

这样的组织方式，而不是硬凑语义化版本号。

### 第 5 步：最后再更新 `DEV_PLAN.md`

A4 最后处理 `DEV_PLAN.md`，因为这是最容易过度修改的文件。

A4 在 `DEV_PLAN.md` 中只做这三件事：

1. 勾掉**有明确证据**支持的检查项；
2. 让里程碑推进状态不再与当前代码事实冲突；
3. 必要时在第 5 节前后补一句“当前 baseline 已具备 M0-M9 主链，但部分经验性 / 可视化类检查项仍待复核”。

A4 不得把 `DEV_PLAN.md` 改成第二个 README，也不得在其中重写理论说明。

### 第 6 步：做一致性回扫

文档写完后，A4 还必须做一次**横向一致性检查**。至少核对下面这些点：

1. README 与 PROJECT_STATE 的安装 / 测试 / 训练 / 评估命令是否一致；
2. PROJECT_STATE 与 `tests/test_runner.py` 的 artifact 文件名是否一致；
3. CHANGELOG 与 PROJECT_STATE 是否对同一项能力给出相互冲突的叙述；
4. `DEV_PLAN.md` 的勾选状态是否与 A4 自己在 PROJECT_STATE 中写的“待复核项”冲突；
5. 未合并分支是否被任何顶层文档提前写成“已支持”。

### 第 7 步：后续滚动文档同步

A4 首次合并后，不代表任务结束。只要后续出现下面任一情况，A4 都要再来一轮 doc-only follow-up：

- A1 改了安装 / quickstart / config overlay 语义；
- A2 新增 reference 命令、artifact、summary 键；
- A3 新增 supervision 字段、训练模式表述；
- A5 给出 residual / PML 审计结论；
- A0 在 `WORKTREE_BOARD.md` 中记录了新的已知问题或新基线。

滚动同步时，A4 仍按本节同样的顺序做：

1. 同步 `int/multi-agent`
2. 更新 `PROJECT_STATE.md`
3. 更新 README / CHANGELOG
4. 必要时微调 `DEV_PLAN.md`

---

## 7. 分文件改动规则

### 7.1 `README.md` 只能写“入口级信息”

README 必须满足以下规则：

1. **首页定位**：一句话说明项目是 2D、单源、单机 Helmholtz 原型；
2. **范围边界**：明确当前不做 3D、多源、分布式、大 benchmark；
3. **核心管线**：保留已有的简图或精简文字；
4. **当前阶段**：对齐真实基线；
5. **快速开始**：命令必须能复制运行；
6. **项目结构**：列稳定目录与关键文件；
7. **里程碑概览**：粗粒度表述；
8. **导流链接**：把详细状态引导到 `PROJECT_STATE.md`。

README **不要**做的事：

- 不要列完整 artifact 文件名全集；
- 不要写所有技术债细节；
- 不要塞 A2 / A3 / A5 的完整使用说明；
- 不要把 `WORKTREE_BOARD.md` 的动态状态复制一遍；
- 不要写空泛口号式成果表述。

### 7.2 `CHANGELOG.md` 只记录“已发生变更”

CHANGELOG 必须满足以下规则：

1. 只记录已经在代码中成立的事情；
2. 对历史 baseline 可以按模块 / milestone 归并；
3. 对后续 agent 合并结果应单独成条；
4. 允许写“Known limitations / Notes”；
5. 若没有 release tag，可使用 `[Unreleased]` + 分组结构。

CHANGELOG **不要**做的事：

- 不写未来计划；
- 不写夸张效果结论；
- 不伪造版本号和日期；
- 不把 README 的 quickstart 复制过来。

### 7.3 `PROJECT_STATE.md` 是 A4 的核心文件

`PROJECT_STATE.md` 必须比 README 更具体，但仍要保持结构化。建议固定下面这些节：

1. 当前快照（branch / commit / last verified）
2. 一句话结论
3. 当前已集成能力
4. 当前统一命令
5. 当前 artifact 契约
6. 配置与依赖约束
7. 测试覆盖面概览
8. 当前已知技术债 / 风险
9. 并行分支依赖关系
10. 建议推进顺序

其中最关键的三条规则是：

- **必须明确区分** `already integrated` 与 `planned / in flight`；
- **artifact 名称必须精确**，不能凭记忆；
- **任何用户可见命令**都必须与最新集成代码一致。

### 7.4 `DEV_PLAN.md` 只做“进度对齐”，不做“内容重写”

A4 对 `DEV_PLAN.md` 的操作必须非常克制。具体规则如下：

1. 对**代码存在且测试覆盖明确**的检查项，可以勾；
2. 对**代码存在但缺少验证证据**的经验性、可视化、对比实验条目，不直接勾；
3. 对未来 agent 才会完成的条目，不勾；
4. 可以在第 5 节附近补一句状态说明，但必须明显标明是“当前进度备注”；
5. 禁止修改第 3 节冻结决策原文。

#### 7.4.1 A4 在 `DEV_PLAN.md` 中的证据判定准则

A4 可以使用下面的简单判定表：

| 证据类型 | 可否勾选 | 例子 |
| --- | --- | --- |
| 代码 + 自动化测试 | 可以 | `import src`、runner artifact 断言 |
| 代码 + 已接受 smoke 记录 | 可以 | `helmholtz-train --help`、1-epoch smoke |
| 代码存在但无验证记录 | 谨慎，通常不勾 | 可视化合理、误差小于某阈值 |
| 只是计划中的功能 | 不可以 | reference 评估闭环、hybrid loss 集成（未合并时） |

#### 7.4.2 A4 可以勾哪些类型的条目

常见可勾类型：

- 文件 / 模块已存在且测试覆盖；
- 基础 import / 配置读取 / runner 产物导出已被测试锁定；
- 命令行入口已通过 `--help` 或 smoke 验证；
- 已有明确的单元测试文件与断言支持。

#### 7.4.3 A4 不应擅自勾哪些类型的条目

通常不应擅自勾的类型：

- “可视化检查通过”；
- “与参考 FMM 对比误差 < 某阈值”；
- “远场渐近行为合理”；
- “物理上完全正确”；
- “benchmark 结果优秀”。

这些都需要额外证据，不应仅凭模块存在而通过。

---

## 8. 提交粒度要求

A4 不允许把 README、CHANGELOG、PROJECT_STATE、DEV_PLAN 全部揉成一个“超大文档提交”。最少拆成下面 4 类提交；后续滚动同步则按对应主题再追加小提交。

1. **state scaffold 提交**  
   新建 `PROJECT_STATE.md`，并写出当前项目快照骨架；
2. **readme sync 提交**  
   更新 `README.md` 的阶段、项目结构、quickstart、里程碑概览；
3. **changelog sync 提交**  
   补全 `CHANGELOG.md` 的 baseline 与已集成变更；
4. **dev-plan progress 提交**  
   只更新 `DEV_PLAN.md` 的进度与勾选状态。

推荐提交顺序：

```text
docs: add PROJECT_STATE snapshot for current integrated baseline
docs: sync README with current install-run-evaluate workflow
docs: reconstruct CHANGELOG for implemented baseline and integrated follow-ups
docs: update DEV_PLAN progress without touching frozen decisions
```

如果后续 A2 / A3 / A5 合并后再次做文档同步，推荐再拆成小提交，例如：

```text
docs: record integrated reference-eval workflow and artifacts
docs: document hybrid supervision config and training modes
docs: summarize residual-pml audit conclusions
```

A4 不允许把“首次文档同步”和“后续 agent 的 follow-up”混成一个无法审阅的大提交。

---

## 9. 验证命令

A4 是文档分支，但仍然必须跑最小验证，确保文档写的命令与当前集成状态一致。

### 9.1 事实提取与差异检查

```bash
git diff -- README.md CHANGELOG.md PROJECT_STATE.md DEV_PLAN.md
```

目的：确认 A4 只改了自己应改的文件，没有溢出到代码区。

### 9.2 安装 / 入口验证

在 A1 已合并的前提下，A4 至少应验证：

```bash
python -m pip install -U pip
pip install -e ".[dev]"
```

然后运行：

```bash
helmholtz-train --help
python scripts/evaluate_run.py --help
```

如果当前集成状态仍保留 `python scripts/run_train.py --help` 作为兼容入口，也应一并验证。

### 9.3 配置 / runner 最小守门

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_runner.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_bootstrap.py
```

目的：

- 验证 README / PROJECT_STATE 中写到的命令和产物契约没有与当前实现冲突；
- 验证 A4 没有基于陈旧事实写文档。

### 9.4 文档一致性人工检查

A4 在命令跑完后，至少人工核对下面 6 条：

1. README 与 PROJECT_STATE 的安装命令一致；
2. README 与 PROJECT_STATE 的训练 / 评估入口一致；
3. PROJECT_STATE 中的 artifact 文件名与 `tests/test_runner.py` 一致；
4. PROJECT_STATE 中的 Python 版本与 `pyproject.toml` 一致；
5. CHANGELOG 未把未合并分支写成已完成；
6. `DEV_PLAN.md` 中被勾掉的条目都有证据支持。

### 9.5 后续 agent 合并后的补充验证

若 A2 / A3 / A5 已合并，则 A4 还要分别增加相应验证：

- A2 合并后：验证 reference 命令 / artifact / summary 键文案；
- A3 合并后：验证 supervision 配置字段与模式表述；
- A5 合并后：验证审计结论确实有集成依据，不是会议纪要式猜测。

---

## 10. 与其他 agent 的协作规则

### 10.1 与 A0 的关系

A4 的所有文档输出都必须先交给 A0，只有当 A0 合并进 `int/multi-agent` 后，A4 的文案才算项目当前标准。

A4 不得自己宣布：

- README 已成为新标准；
- PROJECT_STATE 已代表当前真相；
- CHANGELOG 已锁定下一步计划；
- BOARD 状态已更新。

A0 是动态协调的唯一入口，A4 只负责静态文档落盘。

另外：

- 如果 A4 认为 `WORKTREE_BOARD.md` 里的某段文案也需要修订，A4 只能把建议交给 A0；
- A4 不直接改 `WORKTREE_BOARD.md`。

### 10.2 与 A1 的关系

A1 会影响：

- 安装命令；
- quickstart；
- `helmholtz-train` 是否成为推荐入口；
- `requirements.txt` 与 `pyproject.toml` 的叙述关系；
- 多 overlay 是否已支持。

A4 必须等到 A1 已被 A0 接受后，才能把这些内容写进 README / PROJECT_STATE。A4 不得根据 A1 的计划文档提前更新。

A1 需要给 A4 的最少 doc handoff 包括：

- 标准安装命令；
- 标准测试命令；
- 标准训练命令；
- 标准评估命令；
- 多 overlay 状态。

### 10.3 与 A2 的关系

A2 会影响：

- reference 生成命令；
- run-dir 模式补 reference 评估的命令；
- reference 新产物名；
- summary 中新增 reference 键；
- reference 指标的语义。

A4 在 A2 未合并前，最多只能写：

- 当前仓库已存在 `src/eval/reference_solver.py` 模块；
- 统一 reference 评估基线仍待 A2 集成。

A4 不能在 A2 未合并前写：

- `scripts/solve_reference.py` 已存在；
- `reference_envelope.npy` 是正式对外契约；
- README 已支持 reference workflow。

### 10.4 与 A3 的关系

A3 会影响：

- `training.supervision.enabled`
- `training.supervision.reference_path`
- PDE-only / data-only / hybrid 三种模式的描述
- fail-fast 规则

A4 只有在 A3 已集成后，才能把这些内容写进 PROJECT_STATE / README / CHANGELOG。  
在 A3 未合并前，只能写：

- `loss_data()` 公式已在仓库中存在；
- 监督数据通路尚未被训练主循环正式接通（若当前事实如此）。

### 10.5 与 A5 的关系

A5 会影响：

- residual / PML 的已知问题是否得到定量审计；
- `PROJECT_STATE.md` 中“当前已知技术债”的表述；
- 可能追加审计命令与结论摘要。

A4 在 A5 未合并前，不能写任何“问题已解决”或“误差已被证明可忽略”的表述。

### 10.6 A4 是唯一顶层文档写手

A4 必须坚持这一条边界：

- 其他 agent 可以提供 doc handoff；
- 其他 agent 不得直接改顶层文档；
- A4 也不能把自己的职责推回给其他 agent。

这条规则是为了彻底消除顶层文档的并改冲突。

---

## 11. Handoff 模板

### 11.1 A4 -> A0

```text
[A4 -> A0 HANDOFF]
branch: chore/project-state
base_commit: <A4 开工时基线 commit>
final_commit: <A4 最后提交 commit>

files_changed:
- README.md
- CHANGELOG.md
- PROJECT_STATE.md
- DEV_PLAN.md

integrated_state_documented_against:
- int/multi-agent @ <commit>

claims_verified_against:
- pyproject.toml
- src/config.py
- scripts/run_train.py
- scripts/evaluate_run.py
- src/train/runner.py
- tests/test_runner.py
- WORKTREE_BOARD.md

commands_checked:
- pip install -e ".[dev]"      # 若 A1 已合并
- helmholtz-train --help         # 若 A1 已合并
- python scripts/evaluate_run.py --help
- pytest -q tests/test_runner.py
- pytest -q tests/test_bootstrap.py

documented_as_integrated:
- baseline M0-M9: yes/no
- A1 packaging changes: yes/no
- A2 reference-eval changes: yes/no
- A3 hybrid-loss changes: yes/no
- A5 residual audit conclusions: yes/no

left_as_planned_or_in_flight:
- <列出仍未合并但在 PROJECT_STATE / README 中被标成 planned 的内容>

dev_plan_policy:
- only evidence-backed items checked: yes/no
- frozen decisions modified: yes/no (expected: no)

board_update_suggestions_for_A0:
- <若 A4 认为 BOARD 也需要改，由此处给建议，不直接改文件>
```

### 11.2 其他 agent -> A4（A4 要求的 doc intake 格式）

```text
[Ax -> A4 DOC INPUT]
agent: A1 / A2 / A3 / A5
merged_into_int_multi_agent: yes/no
accepted_by_A0: yes/no
final_commit: <commit>

user_visible_changes:
- <一句话列出这次真正影响用户看到的内容>

commands_changed_or_added:
- install:
- test:
- train:
- evaluate:
- other:

config_fields_added_or_changed:
- <字段名 + 简短语义>

artifacts_added_or_changed:
- <新增文件名>
- <summary 新键>

known_limitations_to_document:
- <真实限制，不要写目标态>

one_sentence_changelog_entry:
- <可直接写进 changelog 的一句话>
```

### 11.3 A4 -> 全体（顶层文档更新说明）

```text
[A4 DOC SYNC NOTICE]
Synced against: int/multi-agent @ <commit>
Updated files:
- README.md
- CHANGELOG.md
- PROJECT_STATE.md
- DEV_PLAN.md

What changed:
- <更新的安装/运行命令>
- <新增的项目状态信息>
- <新增的 artifact / config 说明>
- <哪些仍明确标记为 planned>

What did NOT change:
- WORKTREE_BOARD.md
- any code/config/test file
- frozen decisions in DEV_PLAN.md
```

---

## 12. Blocked 条件与处理办法

### 12.1 A1 尚未合并，但 README 需要写新 quickstart

这是 A4 的真实 blocker。A4 的处理方式应是：

- 可以先完成 `PROJECT_STATE.md` 骨架、README 结构更新、CHANGELOG baseline、DEV_PLAN 进度对齐；
- 但**不要最终化**安装 / 训练 quickstart 命令；
- 记录为 `BLOCKED: waiting for A1 integrated packaging commands`；
- 交给 A0 判断是否先合 A1 再收 A4。

### 12.2 代码事实与 handoff 说法冲突

处理规则：

- 以 `int/multi-agent` 中已合并代码为准；
- handoff 只作为辅助，不覆盖代码现实；
- 如冲突较大，标记 `BLOCKED` 并请 A0 裁决；
- 不允许 A4 自己“折中写法”把不确定内容包装成事实。

### 12.3 `DEV_PLAN.md` 某些检查项没有足够证据

处理规则：

- 不勾；
- 如有必要，在 `PROJECT_STATE.md` 或 `DEV_PLAN.md` 的进度说明中写“待复核”；
- 不要为了让文档整齐而勾掉。

### 12.4 后续 agent 已合并，但 A4 的旧分支已经先合过一次

这不是 blocker。A4 的处理方式是：

1. 同步最新 `int/multi-agent`；
2. 在 `chore/project-state` 上继续追加 doc-only commit；
3. 再次走 A4 -> A0 handoff；
4. 由 A0 决定是否再 merge 一次文档更新。

A4 不应把后续文档更新推给 A0 或其他 agent。

### 12.5 需要更新 `WORKTREE_BOARD.md`

A4 默认视为 blocker，不直接修改。  
处理方式：

- 把拟修改的 board 文案放进 handoff；
- 由 A0 统一改 board；
- A4 自己不落盘到 `WORKTREE_BOARD.md`。

### 12.6 为了让文档成立，似乎必须改代码

这也属于 blocker。正确做法：

- 记录文档与代码的偏差；
- 让文档如实描述当前行为；
- 如确实需要代码修复，交给对应 agent 或 A0；
- 不允许 A4 越界去补代码。

---

## 13. 完成定义（Definition of Done）

A4 只有在下面所有条件同时成立时，才算完成：

- [ ] `PROJECT_STATE.md` 已创建并能完整描述当前静态项目快照
- [ ] README 的当前阶段、项目结构、quickstart、里程碑概览已与最新集成事实一致
- [ ] CHANGELOG 不再停留在 M0，已能覆盖当前 baseline 的主要实现与后续已集成变更
- [ ] DEV_PLAN 的进度状态已对齐当前代码现实
- [ ] DEV_PLAN 第 3 节冻结决策未被修改
- [ ] 所有被勾掉的 DEV_PLAN 检查项都有证据支持
- [ ] README / CHANGELOG / PROJECT_STATE 三者表述一致
- [ ] PROJECT_STATE 中已明确区分“已集成”与“planned / in-flight”
- [ ] 安装 / 测试 / 训练 / 评估命令已验证并与文档一致
- [ ] artifact 文件名与 `tests/test_runner.py` 一致
- [ ] Python 版本与依赖约束表述与 `pyproject.toml` 一致
- [ ] 未修改任何代码、配置、测试、CI、BOARD 文件
- [ ] 已向 A0 提交 handoff

只做到“README 看起来更新了”但没有 `PROJECT_STATE.md`，不算完成。  
只做到“CHANGELOG 补了几条”但 README / PROJECT_STATE 互相冲突，也不算完成。  
只做到“DEV_PLAN 全部勾掉”但没有证据，更不算完成。

---

## 14. 本轮明确不做

A4 本轮明确不做下面这些事：

- 不修改 `WORKTREE_BOARD.md`
- 不修改任何代码文件、配置文件、测试文件、CI 文件
- 不根据未来计划写“已支持”
- 不为 README / CHANGELOG 伪造版本号、发布日期、benchmark 结论
- 不把 `DEV_PLAN.md` 的冻结决策、公式、范围定义改掉
- 不把 README 写成第二份 PROJECT_STATE
- 不把 PROJECT_STATE 写成第二份 WORKTREE_BOARD
- 不替 A1 / A2 / A3 / A5 代写功能说明而跳过已集成事实验证
- 不为了文档对齐而偷偷改变 artifact 名称、命令行参数或配置字段
- 不做论文式包装和夸大表述

A4 的任务是：**把顶层静态文档收紧成可信事实层，而不是借文档之名改代码、改看板、改理论。**

---

## 附录 A. 源码核查与补充说明

本附录记录将本手册落盘前对仓库代码的关键交叉核查结果，目的是让 A4 在写 `PROJECT_STATE.md` 与同步 README / CHANGELOG / DEV_PLAN 时有一份可信的事实清单作起点。附录不改变正文中的职责边界。

### A.1 `tests/test_runner.py` 已锁定的 artifact 清单（精确名单）

本文件 §2.5 已给出 A4 必须在 `PROJECT_STATE.md` 中区分的两类 artifact。此处再一次精确记录，帮助 A4 把“已测试锁定”与“已导出但未锁定”写清楚：

**已由 `tests/test_runner.py` 断言锁定的文件**：

- `config_merged.yaml`
- `summary.json`
- `losses.npy`
- `losses.csv`
- `loss_tail.csv`
- `loss_curve.png`
- `loss_curve_log.png`
- `wavefield.npy`
- `wavefield.png`
- `model_state.pt`
- `metrics_summary.csv`
- `quantiles.csv`
- `centerline_profiles.csv`
- `x_coords_physical.csv`
- `y_coords_physical.csv`
- `velocity_physical.csv`
- `wavefield_magnitude_physical.csv`
- `residual_magnitude_physical.csv`
- `field_heatmaps.png`
- `residual_heatmaps.png`

**由 runner 导出但当前测试**未**断言的 CSV**（在 `save_diagnostic_csvs(...)` 中产生）：

- `loss_mask_physical.csv`
- `scattering_magnitude_physical.csv`

A4 必须在 `PROJECT_STATE.md` 中把以上两份清单分开写，并明确第二份为**“已导出但未锁定”**；不要让读者误以为所有 CSV 都有测试守门。这也是 A4 可以建议 A0 未来拆单项（补断言或收敛文件集）的典型依据。

### A.2 `evaluate_saved_run()` 当前返回的 summary 最小锁定键

核查事实：`tests/test_runner.py` 对 `evaluate_saved_run()` 返回的 `summary` 仅断言：

- `summary["output_dir"]` 存在；
- `summary["residual_max_evaluation"] >= 0.0`。

A4 在 `PROJECT_STATE.md` 中描述 summary 键时，可以把“这两个是硬锁定键、其余字段仍可能随 A2/A3/A5 集成扩展”的区分写清楚，避免读者把当前完整 summary 列表误当作契约。

### A.3 `configs/base.yaml` 与 `configs/debug.yaml` 现状

核查事实：

- `configs/base.yaml`：`grid.nx/ny` 为本轮基础规模，`pml.width` 按本轮默认设计；`training` 块包含 `lr`、`epochs`、`batch_size`、`lambda_pde`、`lambda_data`。
- `configs/debug.yaml`：`grid.nx = 32, grid.ny = 32, pml.width = 5`；用于快速 smoke。
- 两份配置均**不包含** `training.supervision.*` 字段（这是 A3 将来要新增的 schema，尚未合并）。

A4 在 PROJECT_STATE “当前支持字段”节必须如实记录这一现状，不得提前写入 `training.supervision.*`。

### A.4 `src/config.py` 的访问语义

核查事实：`Config` 当前没有 `.get()`；`load_config(base, overlay=None)` 只支持单 overlay。A4 在 PROJECT_STATE 中的“配置与依赖约束”节至少应写出：

- Python 版本：`>=3.9,<3.10`（来自 `pyproject.toml`）；
- `pyproject.toml` 为 packaging 事实源（A1 合并后）；`requirements.txt` 为兼容入口；
- 配置加载：`load_config(base, overlay=None)`；
- 属性访问：`cfg.training.lr`；存在性检查：`"supervision" in cfg.training`；
- **`.get()` 不存在**。

这些是写顶层文档时最容易失真的技术事实，明确写进 PROJECT_STATE 有助于未来所有 agent 都按同一语义写配置代码。

### A.5 `resolve_output_dir(..., exist_ok=False)` 的用户可见影响

核查 `src/train/runner.py`：当前 `final_dir.mkdir(parents=True, exist_ok=False)` 的实现会在 `--output-dir` 已存在时抛错。这是用户文档中必须提到的使用注意事项：

- 在 README quickstart 与 PROJECT_STATE “已知技术债” 节都应明确：“重复跑同一 `--output-dir` 前需先删除旧目录”；
- 不建议 A4 把这条包装成“bug”——它是目前有意为之的防覆盖行为，应作为“使用注意”陈述，而非贬义描述。

### A.6 DEV_PLAN 证据判定清单（在当前 baseline 上）

本文件 §7.4 已给出证据判定表。此处结合当前仓库代码，直接列出**可勾**与**不应勾**的典型条目示例：

**当前可勾（代码 + 测试齐备）**：

- `src/core/*`、`src/physics/*`、`src/models/*`、`src/train/*` 模块存在且 `import` 不报错；
- `tests/test_grid.py`、`tests/test_background.py`、`tests/test_eikonal.py`、`tests/test_tau_ops.py`、`tests/test_residual.py`、`tests/test_train.py`、`tests/test_runner.py` 能跑通；
- runner 产物命名被 `tests/test_runner.py` 锁定（见 A.1）。

**当前不应擅自勾**：

- 任何“可视化检查通过”类条目（无自动化证据）；
- 任何“远场渐近行为合理”类经验判断条目；
- 任何“与参考 FMM 对比误差 < 阈值”类条目（A2 尚未合并）；
- 任何“hybrid loss 可用”“PML `Δ̃τ` 已审计”条目（A3 / A5 尚未合并）。

A4 勾选时请严格按此判据；若发现代码已具备但测试缺失，正确做法是**不勾**并在 PROJECT_STATE “待复核” 节记录一条，而不是替 A0 做范围决策。

### A.7 附录的职责

附录只是事实对账，目的是：

- 给 A4 一份可直接落入 `PROJECT_STATE.md` 的精确 artifact 清单；
- 让 A4 的“已集成”与“待集成”区分有代码证据支撑；
- 让 A4 的 DEV_PLAN 勾选判断有可重复的准绳。

附录**不新增** A4 职责，也**不放宽** §4、§7、§14 中的禁令。若仓库在 A4 真正开工时与本附录描述已经不一致，应以当时 `int/multi-agent` 内的实际代码为准；本附录不凌驾于 A4 “代码 + 测试为第一事实源” 的判断。
