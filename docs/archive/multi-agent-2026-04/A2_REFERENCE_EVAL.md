# A2_REFERENCE_EVAL.md

> Agent A2 专用执行手册。  
> A2 只负责把**现有 reference solver**升级成“可独立生成、可落盘、可进入评估流程、可成为后续训练与物理审计基线”的标准参考评估链路；不负责训练逻辑、不负责物理离散修正、不负责顶层文档。  
> 本文件只定义 A2 的静态职责、改单边界、动作顺序和交接格式。动态状态、最新集成提交、Gate 开闭、是否允许开始改共享文件，一律以 `WORKTREE_BOARD.md` 为准。

---

## 1. 身份与协作位置

**Agent ID**：A2  
**分支**：`feat/reference-eval`  
**worktree**：`../wt/ref`  
**启动 Gate**：`Gate 1` 已开（A1 已合并，工程入口稳定）  
**代码合并顺位**：A1 之后的第 2 个功能分支  
**直接交付对象**：A0（集成）  
**下游依赖对象**：A3（Hybrid Loss）、A5（Residual / PML Audit）  
**文档交接对象**：A4（README / PROJECT_STATE / CHANGELOG / DEV_PLAN）

A2 是本轮并行开发里最关键的“**标尺建立者**”。A2 的任务不是再造一个训练器，也不是去改 PDE；A2 只做三件事：

1. 把仓库里已经存在的 `src/eval/reference_solver.py`，从“能在 Python 里手调的模块”升级为“可标准运行的参考解通道”。
2. 把“模型输出离参考解有多远”这件事做成**统一指标、统一文件名、统一评估流程**。
3. 给 A3 和 A5 提供后续工作的统一基线：
   - A3 以后接监督标签时，不再自己发明 reference 文件格式；
   - A5 做 Residual / PML 审计时，不再只看 PDE residual，而能直接看“当前离散和 reference 的偏差”。

A2 的交付不是“又多了几个图”，而是给整个项目建立下面这条判断链：

- 训练输出不是只看 `loss_pde`；
- 评估结果不是只看 `residual_max_evaluation`；
- 而是还能回答：**模型散射包络 Â 和离散 reference Â_ref 差多少、reference 自己回代当前算子后 residual 多小、这些结果是否足够稳定，能被 A3 / A5 当作基线继续用。**

---

## 2. 当前仓库中，A2 必须面对的真实现状

A2 不是从空白页开始，而是要在**当前代码已经给出的结构**上收口。与 A2 直接相关的真实现状如下。

### 2.1 参考解求解器已经在库里，但只到“模块级”

当前仓库已经存在：

- `src/eval/reference_solver.py`
- `src/eval/__init__.py`

其中已实现并入库的公开能力只有两个：

- `assemble_reference_operator(trainer)`
- `solve_reference_scattering(trainer)`

它们的特点是：

1. **依赖 `Trainer` 对象**，不是独立于训练栈的纯配置接口；
2. 返回值只是一个 `[H, W]` 的复数散射包络 `Â_ref`；
3. 当前没有任何标准产物落盘；
4. 当前没有 CLI；
5. 当前没有测试。

这意味着：仓库已经有“参考解线性系统”，但还没有“参考评估链路”。A2 的工作不是重写求解器，而是把这条链路补齐。

### 2.2 当前 `runner.py` 已经有一套通用诊断管线，但完全不知道 reference

当前 `src/train/runner.py` 已经包含：

- `run_training()`：训练 + 导出常规产物
- `evaluate_saved_run()`：从已有 `run_dir` 读取 `config_merged.yaml`、`model_state.pt`、`losses.npy`，再重新导出诊断结果
- `compute_model_diagnostics()`：生成 `a_scat`、`u_total`、`residual_mag`、`physical_mask`、`evaluation_mask` 等通用诊断字段
- `export_evaluation_artifacts()`：写出 `metrics_summary.csv`、`quantiles.csv`、`centerline_profiles.csv`、热力图等

也就是说：

- **用于 reference 比较所需的数据，其实已经在 runner 里具备了**；
- A2 不需要去碰 `trainer.py` 才能拿到模型输出、掩码、wavefield、residual；
- A2 正确的做法是把 reference 相关逻辑尽量收在 `src/eval/*`，由 runner 以最小方式调用。

### 2.3 当前 `scripts/evaluate_run.py` 已经是现成入口壳

当前 `scripts/evaluate_run.py` 只做一件事：

- 调用 `src.train.runner.evaluate_saved_run(run_dir, device)`

并且它现在只有两个参数：

- `--run-dir`
- `--device`

因此，A2 **不需要重写评估脚本**。A2 只要让 `evaluate_saved_run()` 具备 reference 能力，现有脚本就会自动拥有“给已有 run 目录补 reference 评估”的能力。

### 2.4 当前 `tests/test_runner.py` 只锁定已有输出契约

当前 `tests/test_runner.py` 会检查：

- `config_merged.yaml`
- `summary.json`
- `losses.npy`
- `metrics_summary.csv`
- `quantiles.csv`
- `wavefield.png`
- `field_heatmaps.png`
- `residual_heatmaps.png`
- 等已有文件

并在 `evaluate_saved_run()` 后只确认：

- `summary["output_dir"]`
- `summary["residual_max_evaluation"] >= 0.0`

它**没有**任何 reference 产物或 reference 指标的断言。

这说明一件重要的事：

> A2 没必要把 `tests/test_runner.py` 也拖进自己的分支里，完全可以把 reference 产物与 reference 指标的验证放进 `tests/test_reference_solver.py`。

这样可以显著降低热点测试文件的冲突率。

### 2.5 当前 debug 配置就是 A2 的默认测试场

当前配置：

- `configs/base.yaml`
- `configs/debug.yaml`

在 `debug.yaml` 下：

- `grid.nx = 32`
- `grid.ny = 32`
- `pml.width = 5`

因此总网格大小为：

- `42 x 42`（含 PML）

这是 A2 的默认测试规模。A2 不得把默认测试建立在 base 网格上，因为 `assemble_reference_operator()` + `spsolve()` 的成本在大网格上会明显上升。

### 2.6 当前代码上，homogeneous + debug 已经给出一个非常好的 sanity check

基于当前仓库实际代码，`base + debug + homogeneous` 条件下，`solve_reference_scattering(trainer)` 返回的参考散射包络在当前 quick check 中为**全零**（`max_abs = 0`）。

这不是一句泛泛的“理论上应接近零”，而是当前代码路径已经表现出的行为。因此，A2 应把它固定成第一条 sanity check：

- homogeneous 场景：`Â_ref` 在物理区应接近 0；
- smooth_lens 场景：`Â_ref` 不必接近 0，但将 `Â_ref` 代回当前算子后，reference residual 应显著小于未训练网络输出的 residual。

### 2.7 A2 当前最大的诱惑，就是越界

A2 很容易被诱导去做这些错误动作：

- 为了更方便拿 reference，直接改 `src/train/trainer.py`
- 为了让 reference residual 更好看，直接改 `src/physics/residual.py`
- 为了让 CLI 更“优雅”，去改 `pyproject.toml` 新增 console script
- 为了让文档同步，顺手改 `README.md`
- 为了让测试一次全过，顺手改 `tests/test_runner.py`

这些都属于越界。A2 的正确策略是：

- reference 逻辑尽量放进 `src/eval/*`
- 共享文件 `src/train/runner.py` 只做最小接入
- 顶层文档全交给 A4
- 打包与 CLI 注册全交给 A1

---

## 3. A2 的唯一目标

A2 完成后，仓库必须具备下面这一整套能力，而不是只完成其中一半：

1. 可以从**配置**独立生成参考解：
   - `python scripts/solve_reference.py --config ... --overlay ...`
2. 可以从**已有 run 目录**补 reference 评估：
   - `python scripts/solve_reference.py --run-dir ...`
   - `python scripts/evaluate_run.py --run-dir ...`
3. 对已有 `run_dir` 执行评估时，会新增稳定的 reference 产物：
   - `reference_envelope.npy`
   - `reference_wavefield.npy`
   - `reference_metrics.json`
   - `reference_comparison.csv`
   - `reference_error_heatmaps.png`
4. `summary.json` 会追加统一 reference 指标键，而不是另起一套不一致命名。
5. `tests/test_reference_solver.py` 可以锁定：
   - homogeneous 零散射 sanity
   - smooth_lens 下 reference residual 的可信度
   - `evaluate_saved_run()` / `solve_reference.py` 生成 reference 产物的能力
6. A3 和 A5 可以把 A2 产出的 reference 文件与 reference 指标，直接作为下游接口使用。

A2 的目标不是“让评估更热闹”，而是把 reference 从一个零散模块，升级为**项目的统一标准标尺**。

---

## 4. 文件所有权与并改禁令

### 4.1 A2 独占文件

A2 未合并前，以下文件或目录只允许 A2 修改：

- `src/eval/*`
- `scripts/solve_reference.py`（新建）
- `tests/test_reference_solver.py`（新建）
- `configs/experiments/reference.yaml`（若新建）

说明：

- A2 可以在 `src/eval/` 下继续拆新文件，例如 `reference_eval.py`、`reference_artifacts.py`、`reference_metrics.py`；
- 只要留在 `src/eval/*`，都视作 A2 独占范围。

### 4.2 A2 允许有一个“受控共享窗口”的文件

A2 只允许**最小范围**修改以下共享热点文件：

- `src/train/runner.py`

但必须满足：

1. 只追加 reference 评估接入，不重构 runner；
2. 不改变 `run_training()` 默认行为；
3. 不改现有 CLI 参数语义；
4. 不重命名已有产物；
5. 只在完成 `src/eval/*` 逻辑之后，再进 runner 接口层。

### 4.3 A2 明确禁止修改的文件

A2 不得修改以下文件，即使“只是顺手补一下”：

- `src/train/trainer.py`
- `src/train/losses.py`
- `src/physics/*`
- `src/models/*`
- `src/core/*`
- `src/config.py`
- `pyproject.toml`
- `requirements.txt`
- `scripts/run_train.py`
- `scripts/evaluate_run.py`
- `tests/test_runner.py`
- `README.md`
- `CHANGELOG.md`
- `PROJECT_STATE.md`
- `DEV_PLAN.md`
- `WORKTREE_BOARD.md`

关键说明：

- `scripts/evaluate_run.py` 已经是现成入口壳，A2 不碰；A2 只让它的下游函数更强。
- `tests/test_runner.py` 不应成为 A2 的测试落点；reference 契约全部放进 `tests/test_reference_solver.py`。
- `src/physics/residual.py` 是 A5 的审计对象，不是 A2 的游乐场。

---

## 5. 开工前同步协议

A2 每次**开始一个新的修改会话之前**，都必须重新做一次同步检查。A2 不得把“我昨天同步过了”当成今天还能继续改共享文件的理由。

### 5.1 A2 只认 `int/multi-agent`，不追别人的 feature branch

A2 的上游只有两个：

1. `int/base-current`：理论冻结基线
2. `int/multi-agent`：实际最新集成状态

A2 不得：

- 追 A1 / A3 / A4 / A5 的 feature branch；
- 用别人的未合并提交当“最新状态”；
- 因为自己等不及就跳过 Gate 规则。

### 5.2 A2 的开工 Gate

A2 的正式开工前提是：

- `WORKTREE_BOARD.md` 中 `Gate 1 — 工程入口可用` 已开；
- A1 已合并；
- 当前没有 A0 记录的 packaging 回退或入口冻结。

在此之前，A2 最多只允许做**本地阅读和设计**，不允许开始修改共享文件，也不应开始写需要依赖 A1 安装路径的 CLI 行为。

### 5.3 每次开始新修改会话时的检查动作

进入 `../wt/ref` 后，顺序执行：

```bash
git status --short
git fetch --all --prune
```

要求：

- `git status --short` 必须为空；
- 若有未提交改动，先收尾或 stash，禁止脏树同步；
- 然后查看 `WORKTREE_BOARD.md`，确认：
  - `Gate 1` 已开；
  - `A2` 未被标记为 `BLOCKED`；
  - 若 `latest_integrated_commit` 已变，记下新 commit。

随后同步：

```bash
git merge --ff-only int/multi-agent
```

如果 `int/multi-agent` 还未领先于 `int/base-current`，也应优先认 `int/multi-agent`，不要自己判断“反正一样”。

### 5.4 对共享文件 `src/train/runner.py` 的额外规则

A2 只有在下面两个条件同时满足时，才允许打开 `src/train/runner.py`：

1. `src/eval/*` 的核心逻辑已经成型；
2. 本地已同步到当前 `int/multi-agent` 最新 commit。

只要 `int/multi-agent` 又前进了一步，A2 就不能继续在旧基线里开新一轮 runner 修改。

---

## 6. A2 的执行顺序

A2 必须按下面顺序推进，不能一开始就扎进 `runner.py`。

### 第 1 步：先冻结 reference 产物契约与指标契约

在写代码前，A2 先把下面两组名字和含义固定住，后面所有实现都按这套契约走。

#### 6.1 标准 reference 产物文件名

对于 **run-dir 评估模式**，A2 统一新增以下文件：

- `reference_envelope.npy`
- `reference_wavefield.npy`
- `reference_metrics.json`
- `reference_comparison.csv`
- `reference_error_heatmaps.png`

其中含义固定为：

- `reference_envelope.npy`：参考散射包络 `Â_ref`，复数数组，形状 `[H, W]`，保留全网格（含 PML）
- `reference_wavefield.npy`：参考总波场 `u_ref = u0 + Â_ref * exp(i(ωτ + π/4))`，复数数组，形状 `[H, W]`
- `reference_metrics.json`：reference 专属 JSON，包含 reference 是否可用、reference residual、comparison 指标、必要的辅助元数据
- `reference_comparison.csv`：reference 与当前模型预测的标量对比结果
- `reference_error_heatmaps.png`：物理区裁剪后的 reference / prediction / error 图

对于 **纯配置模式**（没有现成 `run_dir`、也没有已训练模型）至少必须产出：

- `reference_envelope.npy`
- `reference_wavefield.npy`
- `reference_metrics.json`

`reference_comparison.csv` 与 `reference_error_heatmaps.png` 在“没有模型可比”的配置模式下可以省略，但在 `run-dir` 模式下必须存在。

#### 6.2 标准 summary 指标键

A2 统一新增以下 summary 键：

- `reference_available`
- `rel_l2_to_reference`
- `amp_mae_to_reference`
- `phase_mae_to_reference`
- `reference_residual_rmse`
- `reference_residual_max`

含义必须固定，不得在后续分支中被重新解释。

#### 6.3 指标计算区域固定为 evaluation mask

reference 比较的默认区域应为：

- `evaluation_mask = physical_mask & loss_mask`

原因：

- `physical_mask` 去掉 PML；
- `loss_mask` 去掉震源穿孔区；
- 这样与当前 PDE loss 的关注区域保持一致。

A2 不得把 reference 比较偷偷切到另一套区域，而不在 handoff 里说清楚。

#### 6.4 指标公式固定

A2 推荐按下面公式落地：

**(1) relative L2**

在 `evaluation_mask` 上计算：

```text
rel_l2_to_reference = ||Â_pred - Â_ref||_2 / max(||Â_ref||_2, eps)
```

其中：

- `Â_pred`：模型当前散射包络预测（复数）
- `Â_ref`：reference solver 求得散射包络（复数）
- `eps`：推荐 `1e-12`

**(2) amplitude MAE**

```text
amp_mae_to_reference = mean( | |Â_pred| - |Â_ref| | )
```

同样在 `evaluation_mask` 上算。

**(3) phase MAE**

相位只在幅值不太小的位置上有意义。A2 必须使用相位掩码：

```text
phase_mask = evaluation_mask & (|Â_ref| > phase_eps)
```

推荐：

- `phase_eps = 1e-8`

相位误差应使用 wrap 到 `[-π, π]` 的差值：

```text
phase_err = angle( exp(i * (angle(Â_pred) - angle(Â_ref))) )
phase_mae_to_reference = mean( |phase_err| )
```

如果 `phase_mask` 为空：

- `phase_mae_to_reference` 在 `summary.json` 中写 `0.0`
- 在 `reference_metrics.json` 中补充 `phase_metric_points: 0`

这样下游 CSV / JSON 都不会失真到不可读。

**(4) reference residual**

A2 必须把 `Â_ref` 重新喂回当前 `trainer.residual_computer.compute()`，计算 reference 自己在当前离散下的 residual：

```text
reference_residual_rmse = sqrt(mean(|R_ref|^2)) on evaluation_mask
reference_residual_max  = max(|R_ref|) on evaluation_mask
```

这个指标非常重要，因为 A5 以后会用它判断 PML / residual 的离散是否还可信。

### 第 2 步：把 reference 专属逻辑尽量关进 `src/eval/*`

A2 不应把一堆 reference 细节直接堆进 `runner.py`。推荐结构是：

- `src/eval/reference_solver.py`：继续负责线性系统组装与求解
- `src/eval/reference_eval.py`（可新建）：负责 comparison 指标、reference residual、落盘、热力图
- `src/eval/__init__.py`：只做轻量导出，不塞业务实现

正确的依赖方向应是：

```text
runner.py -> src.eval.*
```

而不是：

```text
src.eval.* -> runner.py
```

A2 不得在 `src/eval/*` 里反向 import `run_training()` 或 `evaluate_saved_run()`。

### 第 3 步：做一个独立可运行的 `scripts/solve_reference.py`

A2 新建：

- `scripts/solve_reference.py`

这个脚本的目标不是取代 `evaluate_run.py`，而是提供**独立参考解入口**。建议支持两种模式：

#### 6.5 配置模式

从配置直接生成 reference：

```bash
python scripts/solve_reference.py \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --device cpu \
  --velocity-model smooth_lens \
  --output-dir ../runs/ref/from_config
```

用于：

- 纯 reference 生成
- sanity check
- 给 A3 预先生成参考标签

#### 6.6 run-dir 模式

从现有 run 目录生成 reference 并和当前模型比较：

```bash
python scripts/solve_reference.py \
  --run-dir ../runs/ref/train_1ep \
  --device cpu
```

用于：

- 给已有训练结果补 reference 产物
- 输出 `reference_comparison.csv`
- 输出 `reference_error_heatmaps.png`
- 写入 `reference_metrics.json`

脚本层规则：

1. `--run-dir` 和 `--config` 只能二选一；
2. 配置模式下可以带 `--overlay` 和 `--velocity-model`；
3. run-dir 模式下配置来源固定为 `run_dir/config_merged.yaml`；
4. run-dir 模式下若发现 `model_state.pt`，就比较模型与 reference；若没有模型文件，则只导出 reference 本身，不伪造 comparison；
5. 不新增 `helmholtz-reference` 之类的 `console_script`，本轮只保留 `scripts/solve_reference.py`。

### 第 4 步：先把 reference 逻辑做成“独立 helper + 独立 CLI”，再接入 runner

A2 的独立 helper 至少要覆盖：

1. 从 `Trainer` 求解 `Â_ref`
2. 从 `Â_ref` 重构 `u_ref`
3. 从 `Â_ref` 计算 reference residual
4. 从 `Â_pred` 与 `Â_ref` 计算 comparison metrics
5. 落盘 npy / json / csv / heatmap

只有当这些 helper 已独立工作后，A2 才能进入 runner 做最小接入。

### 第 5 步：最小接入 `evaluate_saved_run()`，不要动 `run_training()` 默认路径

A2 对 `runner.py` 的正确改法是：

- 在 `evaluate_saved_run()` 里，常规 diagnostics 完成后，再调用 reference helper
- 把得到的 reference metrics 合并进 `summary.json`
- 把 reference 产物落到当前 `run_dir`

A2 **不应该**在本轮做的事情：

- 不让 `run_training()` 默认每次训练都自动求 reference
- 不把 reference solve 塞进训练循环
- 不改变 `scripts/run_train.py` 的默认行为

原因很明确：

- reference solve 是离线评估动作，不是训练主路径；
- 一旦默认挂进训练，会把所有 smoke / CI 成本抬高；
- 这会让 A2 不必要地干扰 A1 与 A3。

### 第 6 步：把 reference 回归测试全部锁进 `tests/test_reference_solver.py`

A2 新建：

- `tests/test_reference_solver.py`

最少应覆盖下面三类行为。

#### 6.7 homogeneous 零散射 sanity

使用：

- `configs/base.yaml + configs/debug.yaml`
- `velocity_model = homogeneous`

断言：

- `Â_ref` 在物理区的幅值接近 0
- `reference_residual_rmse` 极小

不要求写成“必须绝对等于 0”，建议用小阈值锁定，例如 `1e-10` 或更宽松但稳定的数量级。

#### 6.8 smooth_lens 下 reference residual 可信

使用：

- `configs/base.yaml + configs/debug.yaml`
- `velocity_model = smooth_lens`

断言：

- `reference_residual_rmse` 小于未训练模型输出的 residual RMSE
- `reference_residual_max` 也显著小于未训练模型输出的 residual max

当前 quick check 显示，在 debug + smooth_lens 下：

- `reference_residual_rmse` 约为 `7e-08`
- 未训练模型 residual RMSE 约为 `2.7e-02`

A2 不需要把测试阈值锁得极端激进，但应至少保证“reference 比随机初始化网络显著更可信”。

#### 6.9 run-dir 评估会产出 reference 文件与 summary 键

流程：

1. 用 debug + `smooth_lens` 跑一个 1 epoch 的训练到临时目录
2. 调用 `evaluate_saved_run(tmp_run_dir, device="cpu")`
3. 断言 reference 文件存在
4. 断言 `summary.json` 或返回字典中包含标准 reference 键

这样 A2 就能在自己的测试文件里锁住整条 reference 评估链，而不必修改 `tests/test_runner.py`。

### 第 7 步：把下游接口定义清楚，再交给 A0 / A3 / A4 / A5

A2 交付不仅是代码，还包括接口说明：

- 给 A0：新增了哪些文件、哪些 summary 键、runner 改了哪几处
- 给 A3：reference 标签文件名、格式、网格维度语义
- 给 A4：如何运行 reference 评估，reference 指标代表什么
- 给 A5：哪些 reference residual 指标可以当物理审计基线

---

## 7. 分文件改动规则

### 7.1 `src/eval/reference_solver.py`

这个文件继续作为**离散算子组装 + 稀疏求解**核心。

A2 允许做：

- 补函数注释
- 补小型 helper
- 补更明确的日志
- 把与 comparison / 落盘无关的求解逻辑留在这里

A2 不要做：

- 在这里直接写 CLI
- 在这里直接处理 `summary.json`
- 在这里反向 import runner
- 为了让 solver 更“好用”去改 PDE 离散定义

### 7.2 `src/eval/reference_eval.py`（推荐新建）

A2 推荐把下面这些逻辑放在一个新文件里，例如：

- `solve_and_export_reference(...)`
- `compute_reference_metrics(...)`
- `compute_reference_residual(...)`
- `save_reference_artifacts(...)`
- `save_reference_heatmaps(...)`

原因：

- reference solver 本身保持纯粹；
- comparison / artifact / heatmap 不污染 runner；
- 下次 A5 如果需要复用 reference residual，也能直接 import 这个模块。

### 7.3 `src/eval/__init__.py`

只做轻量导出即可。不要把大量实现代码又塞进 `__init__.py`。

### 7.4 `scripts/solve_reference.py`

这个文件只做 CLI 参数解析和模式分发，不做具体数值实现。

应满足：

- 安装后的项目里，`python scripts/solve_reference.py --help` 可用
- 配置模式和 run-dir 模式互斥清晰
- 输出 JSON 到 stdout，便于 shell / CI 查看

不应满足：

- 不在脚本里复制一套 runner 逻辑
- 不在脚本里复制 reference metric 公式
- 不硬编码仓库根目录外的路径

### 7.5 `src/train/runner.py`

A2 对 `runner.py` 的修改必须遵守下面四条硬规则：

1. **只追加，不重写**
2. **只接入 reference helper，不内联大段 reference 逻辑**
3. **不改变 `run_training()` 默认成本**
4. **不改已有 summary 键含义**

推荐接入点：

- `evaluate_saved_run()` 完成常规 diagnostics 后

推荐合并方式：

```python
summary.update(reference_metrics)
```

同时保留原有：

- `summary["output_dir"]`
- `summary["device"]`
- 原先 generic metrics

### 7.6 `tests/test_reference_solver.py`

这是 A2 的主测试文件。它应成为：

- reference solver 正确性
- reference 产物存在性
- reference 指标回归

的唯一落点。

A2 不要把这些断言拆进多个原有测试文件里制造热点。

### 7.7 `configs/experiments/reference.yaml`（可选）

只有在 A2 确实需要固定某套 reference 专用参数时，才新建这个文件。否则直接使用：

- `configs/base.yaml`
- `configs/debug.yaml`

A2 不得因为“看起来更整齐”就无意义新增 overlay。

### 7.8 `tests/test_runner.py`

A2 明确禁止修改。

若需要验证 `evaluate_saved_run()` 的 reference 能力，请在 `tests/test_reference_solver.py` 中自行调用 `evaluate_saved_run()`，并断言 reference 产物与 summary 键。

---

## 8. 提交粒度要求

A2 不允许把 reference 求解、runner 接入、测试、CLI 全堆成一个大提交。最少拆成下面 4 类提交：

1. **eval core 提交**  
   处理 `src/eval/*` 中的 reference helper、metrics、artifact 导出
2. **cli 提交**  
   新建 `scripts/solve_reference.py`
3. **runner integration 提交**  
   在 `src/train/runner.py` 里做最小 reference 接入
4. **test 提交**  
   新建 `tests/test_reference_solver.py`

推荐提交顺序：

```text
feat(eval): add reference comparison and artifact helpers
feat(eval): add solve_reference CLI for config and run-dir modes
feat(eval): integrate reference metrics into evaluate_saved_run
test(eval): add reference solver regression coverage
```

如果有可选 `configs/experiments/reference.yaml`，可并入第 1 或第 2 个提交，但不要单独开一个“只加 yaml”的空洞提交。

---

## 9. 验证命令

A2 自测必须至少跑完下面这组命令，并把结果交给 A0。

### 9.1 reference 单测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q tests/test_reference_solver.py
```

### 9.2 homogeneous 配置模式 sanity

```bash
rm -rf ../runs/ref/hom_zero 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/solve_reference.py \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --device cpu \
  --velocity-model homogeneous \
  --output-dir ../runs/ref/hom_zero
```

检查：

- `reference_envelope.npy` 存在
- `reference_wavefield.npy` 存在
- `reference_metrics.json` 存在
- homogeneous 场景下 reference 包络接近 0

### 9.3 smooth_lens 配置模式 smoke

```bash
rm -rf ../runs/ref/smooth_cfg 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/solve_reference.py \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --device cpu \
  --velocity-model smooth_lens \
  --output-dir ../runs/ref/smooth_cfg
```

检查：

- reference 产物可生成
- `reference_metrics.json` 中 `reference_available = true`
- `reference_residual_rmse` 为小量

### 9.4 run-dir 模式：先训练再补 reference

```bash
rm -rf ../runs/ref/train_1ep 2>/dev/null

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
helmholtz-train \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir ../runs/ref/train_1ep

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/solve_reference.py \
  --run-dir ../runs/ref/train_1ep \
  --device cpu
```

检查：

- `reference_comparison.csv` 存在
- `reference_error_heatmaps.png` 存在
- `reference_metrics.json` 存在
- `summary.json` 若被脚本更新，新增 reference 键

### 9.5 现有评估入口 smoke

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/evaluate_run.py \
  --run-dir ../runs/ref/train_1ep \
  --device cpu
```

检查：

- 现有 generic 产物不回归
- reference 产物补齐
- `summary.json` 包含标准 reference 键

A2 不需要在本轮对 base 网格做 reference solve 验收；所有强制验收都应基于 debug 规模。

---

## 10. 与其他 agent 的协作规则

### 10.1 与 A0 的关系

A2 的输出只有在 A0 合并进 `int/multi-agent` 并通过统一 smoke 后，才算成为项目“当前最新状态”。

A2 不得自己宣布：

- reference 文件名从此固定了
- reference summary 键从此生效了
- A3 / A5 可以直接依赖了

这些都必须经过 A0 集成。

### 10.2 与 A3 的关系

A3 未来会需要 reference 标签。A2 必须明确告诉 A3：

- `reference_envelope.npy` 的格式是**复数 `[H, W]` 全网格数组**；
- 它表示散射包络 `Â_ref`，不是总波场；
- 若 A3 需要监督训练，应由 A3 在训练数据接入层把它转成双通道 `[1, 2, H, W]`，而不是要求 A2 改文件格式；
- `reference_wavefield.npy` 是总波场 `u_ref`，仅用于评估与可视化，不是 A3 的主监督标签。

A2 不要为了让 A3 更省事，就提前去改 `trainer.py` 接 reference。

### 10.3 与 A4 的关系

A2 不改顶层文档。A2 只把下面信息交给 A4：

- 如何从配置生成 reference
- 如何给已有 `run_dir` 补 reference 评估
- 新增了哪些 reference 文件
- 每个 reference 指标代表什么
- reference 只在 debug 规模上进入强制测试，不代表 base 网格的默认 CI 也会做同样动作

### 10.4 与 A5 的关系

A5 以后会用 A2 的结果做物理审计。A2 必须给 A5 说清楚：

- `reference_residual_rmse`
- `reference_residual_max`
- reference residual 的默认计算区域（`evaluation_mask`）
- 当前 reference solver 仍基于**现有** residual 离散组装，并不是“真连续解”

这条说明非常重要，因为 A5 审计的是**当前离散实现是否自洽**，不是拿 reference 当外部真理去压代码。

---

## 11. Handoff 模板

### 11.1 A2 -> A0

```text
[A2 -> A0 HANDOFF]
branch: feat/reference-eval
base_commit: <A2 开工时基线 commit>
final_commit: <A2 最后提交 commit>

files_changed:
- src/eval/reference_solver.py
- src/eval/reference_eval.py      # 若新建
- src/eval/__init__.py            # 若改了
- scripts/solve_reference.py
- src/train/runner.py             # 若改了
- tests/test_reference_solver.py
- configs/experiments/reference.yaml   # 若新建

standard_outputs_added:
- reference_envelope.npy
- reference_wavefield.npy
- reference_metrics.json
- reference_comparison.csv
- reference_error_heatmaps.png

summary_keys_added:
- reference_available
- rel_l2_to_reference
- amp_mae_to_reference
- phase_mae_to_reference
- reference_residual_rmse
- reference_residual_max

runner_change_scope:
- evaluate_saved_run only / evaluate_saved_run + other helper hook
- run_training default behavior changed: yes/no (expected: no)

commands_run:
- pytest -q tests/test_reference_solver.py
- solve_reference.py config-mode homogeneous
- solve_reference.py config-mode smooth_lens
- 1-epoch train smoke
- solve_reference.py run-dir mode
- evaluate_run.py run-dir mode

result:
- all passed / failed

notes_for_integrator:
- any known limitations
- any heavy-cost warnings
- whether comparison files are absent in config-only mode
```

### 11.2 A2 -> A3

```text
[A2 -> A3 LABEL HANDOFF]
reference label file:
- reference_envelope.npy

format:
- complex array
- shape: [H, W]
- full grid including PML
- semantic: scattering envelope A_ref

secondary artifact:
- reference_wavefield.npy
- semantic: total wavefield u_ref, for evaluation/visualization only

mask convention for metrics:
- evaluation_mask = physical_mask & loss_mask

do not assume:
- file is already dual-channel
- file is cropped to physical domain
```

### 11.3 A2 -> A4

```text
[A2 -> A4 DOC HANDOFF]
Generate reference from config:
  python scripts/solve_reference.py --config configs/base.yaml --overlay configs/debug.yaml --device cpu --velocity-model smooth_lens --output-dir <dir>

Add reference evaluation to existing run:
  python scripts/solve_reference.py --run-dir <run_dir> --device cpu
  python scripts/evaluate_run.py --run-dir <run_dir> --device cpu

Reference artifacts:
- reference_envelope.npy
- reference_wavefield.npy
- reference_metrics.json
- reference_comparison.csv
- reference_error_heatmaps.png

Reference metrics:
- reference_available
- rel_l2_to_reference
- amp_mae_to_reference
- phase_mae_to_reference
- reference_residual_rmse
- reference_residual_max

Important notes:
- reference default regression tests use debug-scale grids
- run_training default path does not automatically solve reference
- reference comparison is evaluated on evaluation_mask = physical_mask & loss_mask
```

### 11.4 A2 -> A5

```text
[A2 -> A5 AUDIT HANDOFF]
reference residual metrics:
- reference_residual_rmse
- reference_residual_max

calculation region:
- evaluation_mask = physical_mask & loss_mask

semantic note:
- reference is assembled from the current discrete operator and solved by sparse linear solve
- it is a discrete consistency baseline, not an external continuous closed-form truth
```

---

## 12. Blocked 条件与处理办法

出现下面任一情况，A2 应立即停在当前阶段，不要靠越界修改硬解：

1. 为了让 reference work，必须改 `src/train/trainer.py`
2. 为了让 reference residual 更小，必须改 `src/physics/residual.py`
3. 为了新增 CLI，必须改 `pyproject.toml` 或 packaging 结构
4. 为了让测试通过，必须修改 `tests/test_runner.py`
5. 因为多 overlay 需求，必须改 `src/config.py`
6. 为了在大网格上省时，想偷偷改 reference 方程或降阶近似

处理规则：

- 能留在 `src/eval/*` 内解决的，继续；
- 需要共享文件时，只在允许窗口里做最小接入；
- 一旦发现必须改 trainer / physics / packaging / config，立刻标记 `BLOCKED`，交给 A0 判断是否拆新任务；
- 不允许以“反正只是评估”为理由，越界改训练或物理逻辑。

---

## 13. 完成定义（Definition of Done）

A2 只有在下面所有条件同时成立时，才算完成：

- [ ] `src/eval/*` 中已形成独立的 reference 求解 + comparison + artifact helper
- [ ] `scripts/solve_reference.py` 可独立运行
- [ ] 支持配置模式生成 reference
- [ ] 支持 run-dir 模式给已有结果补 reference 评估
- [ ] `evaluate_saved_run()` 已能导出 reference 产物
- [ ] `summary.json` 已追加标准 reference 键
- [ ] `reference_envelope.npy` 文件格式固定为复数 `[H, W]` 全网格数组
- [ ] `reference_wavefield.npy` 已可导出
- [ ] `reference_metrics.json` 已可导出
- [ ] `reference_comparison.csv` 已可导出（run-dir 模式）
- [ ] `reference_error_heatmaps.png` 已可导出（run-dir 模式）
- [ ] `tests/test_reference_solver.py` 通过
- [ ] homogeneous 零散射 sanity 被测试锁定
- [ ] smooth_lens 下 reference residual 可信度被测试锁定
- [ ] 未修改 `trainer.py`、`residual.py`、`tests/test_runner.py`
- [ ] 未改变 `run_training()` 默认行为
- [ ] 未重命名任何已有产物或已有 summary 键
- [ ] 已向 A0 提交 handoff
- [ ] 已向 A3 提交 reference 标签 handoff
- [ ] 已向 A4 提交 reference 文档 handoff
- [ ] 已向 A5 提交物理审计 handoff

只做到“solver 能单独算出来”但没有 run-dir 集成，不算完成。  
只做到“runner 能多写几个文件”但没有独立 CLI，不算完成。  
只做到“本地看起来能跑”但没有稳定测试，也不算完成。

---

## 14. 本轮明确不做

A2 本轮明确不做下面这些事：

- 不修改 `src/train/trainer.py`
- 不修改 `src/physics/residual.py`
- 不修改 `src/train/losses.py`
- 不把 reference solve 默认挂进 `run_training()`
- 不新增 `console_script` 到 `pyproject.toml`
- 不改 `scripts/evaluate_run.py`
- 不改 `tests/test_runner.py`
- 不改 `README.md` / `CHANGELOG.md` / `PROJECT_STATE.md` / `DEV_PLAN.md`
- 不为大网格引入近似 reference 算法
- 不把 reference 标签直接接入训练（这是 A3 的任务）
- 不为了让 reference residual 更好看而改离散公式（这是 A5 的任务）
- 不做 base-grid reference 默认 CI

A2 的任务是**建立统一参考评估基线**，不是借 reference 的名义去改训练主链、物理主链或工程主链。

---

## 附录 A. 源码核查与补充说明

本附录记录将本手册落盘前对仓库代码做的关键交叉核查结果，目的是让 A2 在真正动手前对“已具备的 API 表面”有明确认知，从而避免重复发明。附录不改变正文中的职责边界。

### A.1 `src/eval/reference_solver.py` 现有 API

当前对外暴露的函数只有两个：

- `assemble_reference_operator(trainer)`
- `solve_reference_scattering(trainer)`

这两个函数**直接从 `trainer.residual_computer` 读取**以下字段：

- `A_x`、`A_y`、`B_x`、`B_y`（PML 复数系数）
- `grad_tau_x_stretched`、`grad_tau_y_stretched`（已拉伸的走时梯度）
- `lap_tau_c`（当前激活的 `Δ̃τ` 缓存）
- `damp_coeff`

A5 将来如果要在 `ResidualComputer` 内引入 `lap_tau_mode` 参数，**必须保留 `self.lap_tau_c` 作为激活态 alias**，否则 A2 的 reference solver 会在 A5 合并后直接失效。这条兼容链路 A5 已在自己的手册 §4.3 中写死，A2 在设计时可以信赖。

### A.2 `src/train/runner.py` 已经为 A2 提供现成信号

本文件 §2.2 已指出 runner 有通用诊断管线。实际核查下列事实对 A2 的实现选择影响较大，此处明确记录：

- `compute_model_diagnostics(trainer, ...)` 已经返回 `a_scat`、`u_total`、`residual_mag`、`physical_mask`、`loss_mask`、`evaluation_mask` 等字段。
- `evaluation_mask` 在 runner 中的定义就是：
  ```python
  evaluation_mask = physical_mask & trainer.loss_mask.astype(bool)
  ```
  **与本文件 §6.3 约定的 reference 比较区域完全一致**。因此 A2 不需要在 `src/eval/*` 内重新实现 `evaluation_mask`，而是可以直接从 `compute_model_diagnostics(...)` 的返回字典中取。
- `export_evaluation_artifacts(...)` 目前已写出 `metrics_summary.csv`、`quantiles.csv`、`centerline_profiles.csv`、`field_heatmaps.png`、`residual_heatmaps.png` 等产物；A2 只需**追加** reference 专属产物，不要重命名或重定位已有文件。
- `save_diagnostic_csvs(...)` 目前另外保存 `loss_mask_physical.csv`、`scattering_magnitude_physical.csv` 等辅助 CSV，它们不在 `tests/test_runner.py` 的断言列表里。A2 不需要替这些文件补断言（那是 A4/A0 范围），但应注意**不要把 reference 产物与这两个 CSV 命名冲突**。

### A.3 `evaluate_saved_run(run_dir, device)` 的正确接入点

核查 `src/train/runner.py` 中 `evaluate_saved_run()` 的当前流程，简化叙述如下：

1. 读取 `run_dir/config_merged.yaml` 构造 `Trainer`；
2. 加载 `run_dir/model_state.pt` 到 `trainer.model`；
3. 调用 `compute_model_diagnostics(trainer, ...)` 得到诊断字段；
4. 调用 `export_evaluation_artifacts(...)` 写出 generic 产物；
5. 组装并返回 `summary` 字典。

A2 在 §6 第 5 步所说的“最小接入”，推荐落点就在**第 4 步之后、第 5 步之前**：
```python
ref_metrics = solve_and_export_reference(trainer, diagnostics, run_dir=...)
summary.update(ref_metrics)
```
这样可以保证：

- 不影响 generic 产物生成；
- 不重复组装 `evaluation_mask`；
- summary 键追加语义清晰。

### A.4 `resolve_output_dir(..., exist_ok=False)` 的副作用

核查 `src/train/runner.py` 中的 `resolve_output_dir(...)` 当前行为：

- `final_dir.mkdir(parents=True, exist_ok=False)`

意味着如果 A2 在 CI / 本地反复跑同一个 `--output-dir`，第二次会因为目录已存在而失败。这个坑与 A2 本身没有直接关系，但由于 A2 的验证命令（§9）频繁构造 `../runs/ref/*` 目录，A2 在 handoff 中应提醒 A0：

- 在自动化脚本里必须先 `rm -rf` 目标目录（本文件 §9 已采用这一约定）；
- 此副作用是项目已知技术债，不应由 A2 本轮解决。

### A.5 `configs/debug.yaml` 实际规模确认

本文件 §2.5 给出的网格规模基于 debug overlay。再次核查：

- `configs/debug.yaml`：`grid.nx = 32, grid.ny = 32, pml.width = 5`。
- 因此总网格 `(nx + 2*pml, ny + 2*pml) = (42, 42)`。

A2 的默认验证规模按此执行即可。若将来 `debug.yaml` 被 A1 或 A3 改动（目前本轮内二者均未计划改它），A2 的测试阈值要相应复核。

### A.6 本附录的职责

附录只是事实对账，目的是减少 A2 在真正开工时的摸索成本：

- 让 A2 知道 `evaluation_mask` 已由 runner 提供、可以直接复用；
- 让 A2 明确 reference solver 对 `trainer.residual_computer` 的字段依赖，与 A5 的兼容约束对齐；
- 让 A2 的“最小 runner 接入”有明确的物理落点。

附录**不新增**职责，也**不放宽** §4、§7、§14 中的禁令。如果 A2 在真正开工时代码已与上述描述不一致，应以当时 `int/multi-agent` 内的实际代码为准。
