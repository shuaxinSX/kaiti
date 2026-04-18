# A5_RESIDUAL_PML_AUDIT.md

> Agent A5 专用执行手册。  
> A5 只负责对 `src/physics/residual.py` 中 **PML 下 `Δ̃τ` 的当前混合处理策略**做量化审计，并在**不碰训练接口、不碰评估接口、不碰顶层文档**的前提下，给出一个可切换、可回退、可比较的候选实现。  
> 本文件只定义 A5 的静态职责、改单边界、动作顺序和交接格式。动态状态、最新集成提交、Gate 开闭、是否允许进入最终集成，一律以 `WORKTREE_BOARD.md` 为准。

---

## 1. 身份与协作位置

**Agent ID**：A5  
**分支**：`fix/residual-pml-audit`  
**worktree**：`../wt/phys`  
**启动 Gate**：`Gate 5`（至少要求 A2 已并入并建立 reference baseline；若 A3 已合并，则必须基于 A3 之后的最新 `int/multi-agent` 开工）  
**代码合并顺位**：5 个执行 agent 中最后一个功能分支  
**直接交付对象**：A0（集成）  
**上游依赖对象**：A2（reference solver / comparison 基线）、A0（最新集成状态）  
**文档交接对象**：A4（`README.md` / `PROJECT_STATE.md` / `CHANGELOG.md` / `DEV_PLAN.md`）  
**只读参考对象**：A3（若已合并，只作为最新 trainer 基线读取；A5 不修改 A3 负责的训练接口）

A5 是本轮并行开发里的“**物理正确性审计员**”。A5 不是来“把 residual 代码写得更复杂”，也不是来“凭感觉修一版更正确的公式”；A5 只做四件事：

1. 把当前 `ResidualComputer` 中 **`Δ̃τ` 的 baseline 混合策略**明确冻结成可命名、可比较的 legacy 模式；
2. 在 **不修改 `trainer.py` / `runner.py` / `src/eval/*`** 的前提下，提供一个**更严格但可回退**的候选 `Δ̃τ` 处理模式；
3. 用 **A2 已建立的 reference baseline** 和 **A5 自己补充的 whole-grid / PML-only 审计指标**比较两者；
4. 向 A0 和 A4 交付一份清楚的结论：当前 legacy 是否保留、候选模式是否仅保留为 audit-only、以及结论基于哪些数字。

A5 的产物不是“物理推导看起来更完整了”，而是下面这套可执行结论：

- **legacy baseline 叫什么**；
- **candidate strict mode 叫什么**；
- 两者在 **physical evaluation / whole grid / PML-only** 三种区域上分别表现如何；
- 哪个模式应继续作为默认实现，哪个模式最多先保留为审计/试验开关；
- 结论是否足够强，可以让 A0 选择在集成阶段切换默认值，还是必须保守维持旧默认。

---

## 2. 当前仓库中，A5 必须面对的真实现状

A5 不是从抽象 PDE 开始，而是要在当前仓库已经存在的代码和边界上做定量审计。与 A5 直接相关的真实现状如下。

### 2.1 `ResidualComputer` 的主公式已经冻结，A5 不能借审计之名重写整条残差链

当前 `src/physics/residual.py` 已经明确实现：

```text
R_pde = [Δ̃Â + iω(2∇̃Â·∇̃τ + Â·Δ̃τ) + R_damping − RHS_eq] / ω²
L_pde = mean(loss_mask · |R_pde|²)
```

并且 `compute()` 的返回键已经稳定为：

- `residual_real`
- `residual_imag`
- `loss_pde`

A5 的工作范围只允许聚焦在 **`Â·Δ̃τ` 里的 `Δ̃τ` 取值方式**，不允许顺手改：

- PDE 主公式结构；
- `loss_pde` 的定义；
- `omega**2` 归一化；
- 返回字典的键名；
- dual-channel 输入约定。

### 2.2 PML 系数当前已经是“复数域正确处理”，A5 不要把已修好的部分又打散

当前 `ResidualComputer.__init__()` 已经明确把 PML 系数保留在复数域：

- `self.A_x = pml.A_x_t.cfloat()`
- `self.A_y = pml.A_y_t.cfloat()`
- `self.B_x = pml.B_x_t.cfloat()`
- `self.B_y = pml.B_y_t.cfloat()`

并且 `tests/test_residual.py` 已经有专门断言：

- `rc.A_x.imag.abs().max().item() > 0.0`
- `rc.A_y.imag.abs().max().item() > 0.0`

这意味着 A5 **不是**来修“PML 系数虚部丢失”这种老问题。那一层已经有基线保障了。

### 2.3 输运项里的 `∇̃τ` 已经做了 PML 拉伸，真正未闭合的是 `Δ̃τ`

当前 `ResidualComputer._precompute_pml_lap_tau()` 已经预计算：

- `grad_tau_x_stretched = A_x * grad_tau_x`
- `grad_tau_y_stretched = A_y * grad_tau_y`

随后在 `compute()` 中进入：

```text
dot_grad = grad_tau_x_stretched * ∂xÂ + grad_tau_y_stretched * ∂yÂ
transport = iω(2·dot_grad + Â·lap_tau_c)
```

也就是说：

- **`2∇̃Â·∇̃τ` 这一半已经带拉伸了**；
- **`Â·Δ̃τ` 这一半当前仍使用未拉伸的 `lap_tau_c`**。

A5 的核心问题就出在这里，而不是整个 transport term 都错了。

### 2.4 当前 `lap_tau_c` 明确就是“未拉伸链式法则结果”，而且代码注释已经承认这是混合策略

当前 `ResidualComputer._precompute_pml_lap_tau()` 最后实际落的是：

```python
self.lap_tau_c = tau_d.lap_tau.cfloat()
```

并且注释已经明写：

- 物理区 `A=1, B=0`，所以 `Δ̃τ = Δτ`；
- PML 区严格做法应重新用 PML 系数加权；
- 当前代码采用“未拉伸 `lap_tau` + 已拉伸输运项”的混合策略；
- 采用此策略的理由是：**PML 区 `RHS=0` 且波场应衰减，因此误差贡献可能较小**。

所以 A5 的正确姿势不是“发现了 bug”，而是：

> **把当前 mixed strategy 当成一个已知、可解释、但尚未被定量验证的 baseline。**

### 2.5 `TauDerivatives` 现有缓存足够支撑“更严格候选模式”，但不够支撑跨文件大重构

当前 `src/physics/tau_ops.py` 中 `TauDerivatives` 已缓存：

- `grad_tau_x`
- `grad_tau_y`
- `lap_tau`
- `grad_alpha_x`
- `grad_alpha_y`
- `lap_alpha`

但它**没有**把以下分量全部以结构化形式暴露给 `ResidualComputer`：

- `alpha`
- `tau0`
- `grad_tau0_x`
- `grad_tau0_y`
- `lap_tau0`

这意味着：

- A5 可以基于**已安全重组好的 `grad_tau_x / grad_tau_y`** 构造一个更严格的候选 `Δ̃τ`；
- 但 A5 不应该为了“拿到更完整的解析分量”去改 `TauDerivatives`、`BackgroundField`、`Trainer` 的签名链。

也就是说，A5 的候选严格模式必须建立在**当前 `ResidualComputer` 已经拿到的对象**之上，而不是扩张 upstream 参数表。

### 2.6 PML 区虽然 `RHS_eq = 0`，但**训练损失并没有屏蔽 PML**，所以 A5 不能拿“PML 里不重要”当免责理由

这是当前仓库里最容易被忽略、但对 A5 最重要的事实之一。

当前 `Medium2D` 中，PML 层会退化到背景介质：

```text
PML 内 velocity = 1 / s0
=> slowness = s0
=> perturbation = s² - s0² = 0
```

因此 `compute_rhs()` 在 PML 内自然得到 `RHS_eq = 0`。这也是 residual.py 注释里“PML 区 RHS=0”的来源。

但是，当前 `compute_loss_mask()` 只做了**震源近场穿孔**：

```python
return (source.distance > mask_radius * grid.h).astype(np.float64)
```

它**没有**把 PML 区置零。也就是说当前训练的 `loss_pde` 其实是：

- **去掉震源近场**；
- **仍然包含物理区 + PML 区**。

这会带来一个直接结论：

> A5 审计的 `Δ̃τ` 混合策略，不只是“PML 里一个理论瑕疵”，而是可能真实影响训练 loss 的数值来源。

### 2.7 A2 的 reference baseline 对 A5 极其关键，而且 A5 可以直接复用，不需要改 `src/eval/*`

当前 `src/eval/reference_solver.py` 已提供：

- `assemble_reference_operator(trainer)`
- `solve_reference_scattering(trainer)`

其中 `assemble_reference_operator(trainer)` 直接读取：

- `trainer.residual_computer.A_x / A_y / B_x / B_y`
- `trainer.residual_computer.grad_tau_x_stretched / grad_tau_y_stretched`
- `trainer.residual_computer.lap_tau_c`
- `trainer.residual_computer.damp_coeff`

这意味着只要 A5 在 **不改 A2 代码** 的前提下，给 `trainer.residual_computer` 换一个候选 mode，A2 的 reference operator 就会自动跟着切到那个 mode。

这是 A5 的一个巨大优势：

- 不需要改 `src/eval/*`；
- 不需要改 `scripts/evaluate_run.py`；
- 不需要新建第二套 reference solver；
- 只要保持 `ResidualComputer` 的接口兼容，A2 的求解器天然就是 A5 的审计工具。

### 2.8 当前测试已经能守住“不会炸”，但还守不住“哪种 `Δ̃τ` 更合理”

当前已有测试覆盖的重点是：

`tests/test_residual.py` 已覆盖：

- 前向可跑通；
- `residual_real` / `residual_imag` 无 NaN / Inf；
- homogeneous + zero network 时 `loss_pde` 近零；
- smooth lens 下零网络残差非零；
- PML 系数虚部存在；
- 梯度能回传。

`tests/test_pml.py` 已覆盖：

- 物理区 `gamma = 1`；
- 物理区 `A = 1, B = 0`；
- PML 区 `sigma > 0`；
- 张量 shape 正确；
- 理论反射系数不超过 `R0`。

这些都很重要，但它们只能说明：

- 当前实现**能跑**；
- PML 系数构造**形式正确**；
- 没有明显数值炸裂。

它们还没有回答：

- mixed legacy 与 strict candidate 谁更接近 reference；
- 两者差异主要落在 physical 区、whole grid 还是 PML-only；
- strict candidate 是否只是“看起来更完整”，实际并没有带来更好的数值表现。

### 2.9 A2 的标准 summary 指标区域默认不看 PML，所以 A5 必须额外补充 whole-grid / PML-only 审计视角

A2 已冻结的 comparison 区域是：

```text
evaluation_mask = physical_mask & loss_mask
```

也就是：

- 去掉 PML；
- 去掉震源穿孔。

这很适合做用户可读的 headline metrics，但对 A5 不够，因为 A5 正在审计的正是 **PML 相关 `Δ̃τ` 处理**。

因此 A5 必须明确区分至少三种区域：

1. `evaluation_mask = (~pml_mask) & loss_mask`  
   用来与 A2 的既有基线保持对照；
2. `whole_grid_mask = loss_mask`  
   反映当前 `loss_pde` 真实关心的整体区域；
3. `pml_only_mask = pml_mask`  
   反映 strict candidate 是否只在 PML 内改变了结果。

如果 A5 只看 A2 当前 summary 键，就会出现一种危险假阴性：

- 物理区指标几乎不变；
- 但 PML/whole-grid 行为其实明显改变；
- 最后误以为 A5 这条线“没有差异”。

### 2.10 A5 当前**不能**把 mode 切换暴露到 CLI / YAML 主路径，否则就会越界改 trainer / runner

当前 `Trainer` 构造 `ResidualComputer` 的方式是固定的：

```python
self.residual_computer = ResidualComputer(
    self.grid, self.pml, self.tau_d, self.rhs,
    self.loss_mask, omega, self.diff_ops,
).to(self.device)
```

并且 A5 的边界是不允许改：

- `src/train/trainer.py`
- `src/train/runner.py`
- `scripts/run_train.py`
- `scripts/evaluate_run.py`
- `src/config.py`

所以 A5 不能假装“这轮已经把 strict mode 做成可在 CLI 里切换的正式功能”。

A5 当前最稳妥、最符合边界的落地方式只能是：

- 在 `ResidualComputer` 内部增加一个**默认保持 legacy 的构造参数**；
- 让 **tests / programmatic audit script** 可以显式选择候选 mode；
- 不改现有 `Trainer` 默认路径；
- 不新增任何会误导别人以为“CLI 已支持切换”的公开配置字段。

### 2.11 base 网格上的 reference solve 成本更高，A5 的定量审计必须以 debug 配置为主

根据当前代码结构，reference solve 依赖 `scipy.sparse.linalg.spsolve`，并且 `assemble_reference_operator()` 逐点构造稀疏矩阵。对 `base.yaml`（128x128 + PML）这种规模，求解成本会显著上升。

因此 A5 的 CI / 单测 / 日常审计基线必须固定为：

- `configs/base.yaml + configs/debug.yaml`
- `cfg.medium.velocity_model = "smooth_lens"`
- `device = cpu`

A5 不应在默认验收里要求跑 base 网格 reference 审计。base 网格只能作为后续人工复核或离线实验，不属于这轮分支的“最小完成定义”。

---

## 3. A5 的唯一目标

A5 完成后，仓库必须具备下面这一整套能力，而不是只在 `residual.py` 里多写几行注释：

1. **当前 legacy 行为被显式命名并保持为默认值**  
   A5 不能在没有量化结论的情况下悄悄把项目默认 `Δ̃τ` 行为改掉。当前 mixed strategy 必须成为一个有名字、可比较、可回退的 baseline。

2. **存在一个更严格的候选 mode，但它默认不穿透到 CLI / runner**  
   候选 mode 只需满足：
   - 能在 `ResidualComputer` 内被显式选择；
   - 不破坏现有 `Trainer` 默认路径；
   - 不改 `summary.json` / 产物文件 / CLI；
   - 能被 tests 和 A5 自己的程序化 smoke 直接调用。

3. **A5 用同一套 `Trainer` / `ResidualComputer` / `reference_solver` 基线完成定量比较**  
   至少要比较：
   - `evaluation_mask`
   - `whole_grid_mask`
   - `pml_only_mask`

4. **现有数值稳定性回归不能被打坏**  
   A5 不能为了“更严格”而引入：
   - NaN / Inf；
   - 梯度中断；
   - `tests/test_train.py` 或 `tests/test_runner.py` 回归；
   - `ResidualComputer.compute()` 输出契约变化。

5. **最终结论必须是“量化结论 + 默认建议”，不是开放式讨论**  
   A5 交给 A0 的不是“我实现了一个 strict 版本”；而是：
   - legacy 与 candidate 的 mode 名；
   - 哪些指标更好，哪些没变，哪些更差；
   - 默认值是否继续保持 legacy；
   - 若 candidate 仅建议保留为 audit-only，也必须明确写出。

### 3.1 A5 推荐冻结的 mode 命名

为避免今后沟通混乱，A5 建议把模式名固定为：

- `mixed_legacy`：当前实现  
  - `∇̃τ` 已拉伸；
  - `lap_tau_c = tau_d.lap_tau` 未拉伸；
  - 作为默认 mode 保留。

- `stretched_divergence`：A5 候选严格 mode  
  - 用已安全重组好的 `grad_tau_x / grad_tau_y` 再构造一版更一致的 PML 下 `Δ̃τ`；
  - 目标不是“理论绝对最终版”，而是比当前 mixed legacy 更接近“整项都带 stretch”的实现。

A5 在本轮不要再发明第三、第四种 mode。两种就够：

- 一个稳定 baseline；
- 一个严肃候选。

### 3.2 A5 推荐冻结的候选 strict 公式

在当前边界下，A5 最可行、最不越界的 strict candidate 是：

```text
Δ̃τ_candidate
= A_x · ∂x(∂xτ) − B_x · ∂xτ + A_y · ∂y(∂yτ) − B_y · ∂yτ
```

其中：

- `∂xτ` 直接用 `tau_d.grad_tau_x`
- `∂yτ` 直接用 `tau_d.grad_tau_y`
- `∂x(∂xτ)` 用 `diff_ops.diff_x(grad_tau_x)`
- `∂y(∂yτ)` 用 `diff_ops.diff_y(grad_tau_y)`

这样做的原因是：

1. A5 当前能合法拿到这些量；
2. 不需要改 `TauDerivatives` / `BackgroundField` / `Trainer` 的构造链；
3. 相比 `mixed_legacy`，它至少让 `Â·Δ̃τ` 这一项也进入与 PML 一致的“stretch + B-term”框架；
4. 即使它不是终局解析版，也足以成为一个**值得审计的 stricter candidate**。

注意：

- A5 **不要把这个 candidate 宣称为“严格真解版”**；
- A5 只能把它称为“比 current mixed strategy 更一致的候选实现”；
- 是否最终替代默认值，要由量化结果和 A0 集成结论决定。

---

## 4. 文件所有权与并改禁令

### 4.1 A5 独占文件

A5 未合并前，以下文件只允许 A5 修改：

- `src/physics/residual.py`
- `tests/test_residual.py`
- `tests/test_pml.py`

说明：

- `src/physics/residual.py` 是 A5 的主战场，也是本轮最后一个热点物理文件；
- `tests/test_residual.py` 是 A5 的主要回归承载点；
- `tests/test_pml.py` 只用于补充 PML 相关不变量回归，不是另起一套 solver benchmark 的地方。

### 4.2 A5 明确禁止修改的文件

A5 不得修改以下路径，即使“只是顺手补一下”也不允许：

- `src/train/*`
- `src/eval/*`
- `scripts/*`
- `src/models/*`
- `src/core/*`
- `src/config.py`
- `src/physics/pml.py`
- `src/physics/tau_ops.py`
- `src/physics/background.py`
- `src/physics/rhs.py`
- `pyproject.toml`
- `requirements.txt`
- `pytest.ini`
- `configs/base.yaml`
- `configs/debug.yaml`
- `tests/test_runner.py`
- `tests/test_reference_solver.py`
- `README.md`
- `CHANGELOG.md`
- `PROJECT_STATE.md`
- `DEV_PLAN.md`
- `WORKTREE_BOARD.md`

关键说明：

- `src/physics/pml.py` 当前已有完整单测，A5 这轮不改 sigma/gamma/A/B 的生成逻辑；
- `src/physics/tau_ops.py` 虽然和 `Δ̃τ` 紧密相关，但它不是 A5 这轮允许碰的文件；
- `src/eval/*` 归 A2；A5 只能**消费** A2 的 reference baseline，不能改 A2 的指标和 summary 契约；
- 所有顶层文档都归 A4；A5 只能 handoff 结论，不能自己入文档。

### 4.3 A5 的三条硬兼容规则

A5 在 `residual.py` 中必须死守下面三条兼容规则：

1. **`ResidualComputer.compute()` 返回键不变**  
   仍然只返回：
   - `residual_real`
   - `residual_imag`
   - `loss_pde`

2. **`self.lap_tau_c` 这个属性名必须继续存在**  
   因为 A2 的 `assemble_reference_operator()` 直接读取它。A5 可以增加：
   - `self.lap_tau_legacy_c`
   - `self.lap_tau_candidate_c`
   但最后激活态仍必须通过 `self.lap_tau_c` 暴露出去。

3. **默认构造行为必须保持 legacy**  
   也就是说：
   - `Trainer` 现在怎么构造 `ResidualComputer`，A5 合并后仍然得到 legacy baseline；
   - strict candidate 只能通过显式参数启用；
   - 不能让默认训练 / runner / CLI 路径无意中切到 candidate。

---

## 5. 开工前同步协议

A5 每次**开始一个新的物理补丁会话之前**，都必须重新同步。因为 A5 处在最后一个热点文件窗口，任何基线漂移都很危险。

### 5.1 A5 只认 `int/multi-agent`，不追别人的 feature branch

A5 的事实来源只有两个：

1. `WORKTREE_BOARD.md` 中由 A0 写入的当前 Gate 与集成状态；
2. `int/multi-agent` 中已经被合并的代码。

A5 不得：

- 直接追 A2 的 feature branch；
- 直接追 A3 的 feature branch；
- 根据别人 handoff 的“计划变更”就开始改 `residual.py`。

A5 只承认：

> **已经进了 `int/multi-agent` 的东西，才算当前项目最新状态。**

### 5.2 开工前必须确认 Gate 5 已开

A5 开工前先看 `WORKTREE_BOARD.md`，确认：

- A2 已并入；
- `reference_envelope.npy` 的格式已经冻结；
- `tests/test_reference_solver.py`（若已存在）处于稳定可跑状态；
- `src/train/trainer.py` / `src/train/runner.py` 没有处在 freeze / 回退 / conflict 状态；
- 若 A3 已合并，则 A5 必须以 A3 之后的 `int/multi-agent` 为最新基线。

### 5.3 同步步骤

进入 `../wt/phys` 后，顺序执行：

```bash
git status --short
git fetch --all --prune
```

要求：

- `git status --short` 必须为空；
- 若工作树脏，先收尾或 stash；
- 然后读取 `WORKTREE_BOARD.md`，记下最新集成提交与 Gate 状态；
- 最后同步最新集成基线：

```bash
git merge --ff-only int/multi-agent
```

若 `int/multi-agent` 尚未领先于 `int/base-current`，也必须以看板记录为准，不得自己猜测“应该从哪个分支合”。

### 5.4 A5 的硬规则

只要 `int/multi-agent` 的最新 commit 变了，A5 就**不能在旧基线上开启新的 `residual.py` 修改批次**。  
已经进行到一半、且只差极小收尾的补丁可以先结束，但结束后必须立即同步，再继续下一批更改。

---

## 6. A5 的执行顺序

A5 必须按下面顺序推进。A5 的最大风险不是改动量，而是如果先动了热点文件、后补指标，很容易最后只剩“感觉更对”。

### 第 1 步：先冻结审计问题与比较口径，不先动代码

A5 开工时，先把下面这些口径写死，后续所有比较都按这套口径走。

#### 6.1 mode 名称先冻结

A5 统一只比较两种 mode：

- `mixed_legacy`
- `stretched_divergence`

不要在中途追加第三种 mode。否则审计维度会爆炸，最后没有结论。

#### 6.2 三种 mask 口径先冻结

A5 统一使用：

- `evaluation_mask = (~grid.pml_mask()) & loss_mask.astype(bool)`
- `whole_grid_mask = loss_mask.astype(bool)`
- `pml_only_mask = grid.pml_mask()`

说明：

- `evaluation_mask` 对齐 A2 的标准 comparison 口径；
- `whole_grid_mask` 对齐当前 `loss_pde` 的真实关注区域；
- `pml_only_mask` 用来回答“差异是不是主要集中在 PML”。

#### 6.3 A5 最少记录的指标先冻结

A5 至少记录下面 6 个数字：

1. `rel_l2_baseline_vs_candidate_eval`
2. `rel_l2_baseline_vs_candidate_whole`
3. `rel_l2_baseline_vs_candidate_pml`
4. `self_loss_baseline`
5. `self_loss_candidate`
6. `self_loss_crosscheck`（可选，但强烈建议）

推荐含义：

- `rel_l2_*`：baseline reference 解与 candidate reference 解的相对 L2 差异；
- `self_loss_baseline`：baseline reference 解送回 baseline operator 后的 `loss_pde`；
- `self_loss_candidate`：candidate reference 解送回 candidate operator 后的 `loss_pde`；
- `self_loss_crosscheck`：把 baseline 解丢给 candidate operator，或把 candidate 解丢给 baseline operator，看是否存在强烈 operator mismatch。

A5 不需要把这些都塞进仓库产物或 summary；但 handoff 里必须给 A0。

### 第 2 步：先跑 baseline 定量测量，再动 `residual.py`

A5 不能先写 strict mode 再说“以后再看看有没有改善”。  
在任何代码改动之前，先在当前 legacy baseline 上跑一次最小审计，用来确定后续比较的零点。

推荐基线场景：

- `load_config("configs/base.yaml", "configs/debug.yaml")`
- `cfg.medium.velocity_model = "smooth_lens"`
- `device = "cpu"`

推荐动作：

1. 构造 `Trainer(cfg, device="cpu")`
2. 直接调用 `solve_reference_scattering(trainer)` 得到 baseline reference envelope
3. 记录：
   - baseline reference 的 `loss_pde`
   - baseline reference 在 evaluation / whole / pml 三个 mask 上的残差统计（若方便）
4. 把这些数字作为 A5 的 baseline 参考点

目的不是追求漂亮数值，而是：

- 后面 candidate 有数字可对照；
- 不会出现“candidate 数字变了，但我们根本不知道 legacy 原来是多少”。

### 第 3 步：在 `ResidualComputer` 内引入 candidate mode，但默认值继续是 legacy

A5 在 `src/physics/residual.py` 中应优先做一个**局部、兼容、可回退**的改法，而不是直接替换现有逻辑。

#### 6.4 建议的接口改法

建议把构造器改成：

```python
class ResidualComputer:
    def __init__(..., diff_ops, lap_tau_mode="mixed_legacy"):
        ...
```

要求：

- `lap_tau_mode` 放在末尾，并给默认值；
- 默认值必须是 `mixed_legacy`；
- 未传该参数的旧调用路径完全不受影响。

#### 6.5 建议的内部缓存方式

A5 推荐在 `ResidualComputer` 内同时缓存：

- `self.lap_tau_legacy_c`
- `self.lap_tau_candidate_c`
- `self.lap_tau_c`（当前激活态 alias）

这样做的好处是：

- A2 的 `assemble_reference_operator()` 仍继续读 `self.lap_tau_c`，无需改代码；
- A5 自己也能在 tests 中直接比较两个缓存；
- 默认 / candidate 切换不会破坏外部接口。

#### 6.6 candidate 的推荐实现方式

对 `stretched_divergence`，A5 推荐按下式构造：

```text
lap_tau_candidate
= A_x * ∂x(grad_tau_x) - B_x * grad_tau_x
+ A_y * ∂y(grad_tau_y) - B_y * grad_tau_y
```

实现时注意：

- `grad_tau_x` / `grad_tau_y` 当前是实数 tensor；
- `diff_ops.diff_x(...)` / `diff_ops.diff_y(...)` 返回的也是实数；
- 与 `A_x / B_x / A_y / B_y` 组合时需要转为 complex；
- 最终 `self.lap_tau_candidate_c` 应为 complex tensor，device/dtype 与其他缓存一致。

#### 6.7 A5 不要动 `compute()` 的其他部分

在 `compute()` 中，A5 只允许把：

```python
A_lap_tau = A_c * self.lap_tau_c
```

对应的 `self.lap_tau_c` 切到 active mode。除此之外，不要动：

- `lap_A` 的构造；
- `dot_grad` 的构造；
- `R_damp` 的构造；
- `R_pde` 的归一化；
- `loss_pde` 的定义。

### 第 4 步：补测试时，先守兼容，再看 candidate

A5 的测试顺序必须是：

1. 先证明 legacy 默认路径没坏；
2. 再证明 candidate 路径不炸；
3. 最后做 baseline vs candidate 的定量比较。

不要一上来就写“candidate 更好”的断言。

### 第 5 步：默认值是否切换，由 A0 决定；A5 只交付证据，不擅自翻默认

这是 A5 本轮最重要的收口规则之一：

> **A5 不在自己的分支里把项目默认 mode 从 legacy 改成 candidate。**

A5 只做：

- 新增 candidate mode；
- 保持 legacy 为默认；
- 用数字说明 candidate 值不值得以后成为默认；
- 把这个选择权交给 A0。

这样可以避免：

- A5 结论稍有摇摆，就把整个训练默认路径一起改掉；
- `tests/test_runner.py` / 历史运行结果 / 文档说明被无声扭转。

---

## 7. 分文件改动规则

### 7.1 `src/physics/residual.py`

这是 A5 的主战场。A5 在这个文件中允许做的事情只有下面这些。

#### 7.1.1 允许做的改动

1. 给 `ResidualComputer.__init__()` 增加末尾可选参数：
   - `lap_tau_mode="mixed_legacy"`
2. 增加 mode 验证逻辑，例如：
   - 只允许 `mixed_legacy` / `stretched_divergence`
   - 其他值直接 `ValueError`
3. 把当前 `tau_d.lap_tau.cfloat()` 明确缓存成 `self.lap_tau_legacy_c`
4. 新增一个 candidate 缓存构造逻辑，得到 `self.lap_tau_candidate_c`
5. 用 `self.lap_tau_c` 作为当前激活态 alias
6. 更新 `to()`，把新缓存也移到 device 上
7. 必要时补充极短、事实性注释，说明：
   - legacy 是当前 mixed strategy；
   - candidate 是 A5 的 audit mode；
   - 默认仍是 legacy。

#### 7.1.2 必须保持不变的部分

A5 必须保持以下内容不变：

- `compute()` 的输入仍是 `[B, 2, H, W]` dual-channel tensor
- `compute()` 返回键不变
- `loss_pde` 公式不变
- `loss_mask` 使用方式不变
- `omega**2` 归一化不变
- `A_x / A_y / B_x / B_y / damp_coeff` 的生成方式不变
- `grad_tau_x_stretched / grad_tau_y_stretched` 的构造方式不变

#### 7.1.3 明确禁止做的改动

A5 在这个文件里也**不得**做下面这些事情：

- 不改 batch 语义；
- 不把 `compute()` 改成返回更多字段；
- 不把 `loss_pde` 改成同时返回分区域 loss；
- 不加入日志打印每步指标；
- 不引入新的 public API 给 runner/CLI；
- 不偷偷把 default mode 切到 candidate。

#### 7.1.4 一个非常重要的实现细节：`to()` 必须一起搬动新缓存

如果 A5 增加了：

- `lap_tau_legacy_c`
- `lap_tau_candidate_c`
- `lap_tau_c`

那么 `to()` 里必须同步搬移它们。否则很容易出现：

- `self.A_x` 在 GPU / CPU 上；
- `self.lap_tau_c` 还停在另一个 device；
- `compute()` 里才报 device mismatch。

A5 不要等这个问题出现在 smoke 里才补。设计时就写完整。

### 7.2 `tests/test_residual.py`

A5 的大部分新测试都应该落在这里。

#### 7.2.1 必补的测试类型

A5 至少补下面四类测试：

1. **default legacy backward-compat test**  
   证明不传 `lap_tau_mode` 时，行为仍等价于当前 legacy 路径。

2. **candidate forward safety test**  
   在 `smooth_lens` + PML 激活下：
   - `residual_real` / `residual_imag` 不含 NaN / Inf；
   - `loss_pde` 有限；
   - 梯度可回传。

3. **mode comparison test on physical slice**  
   在物理区（尤其远离源点）比较 legacy 与 candidate 的 `lap_tau_c` 或 residual 行为，要求：
   - 不做 bitwise equality；
   - 只验证“candidate 没把物理区搞成完全不同的数量级”。

4. **reference-based audit test**  
   用 debug + smooth_lens 场景：
   - baseline trainer + baseline residual 生成 `Â_ref_legacy`
   - baseline trainer + candidate residual 生成 `Â_ref_candidate`
   - 比较 `evaluation_mask / whole_grid_mask / pml_only_mask` 三个区域上的相对差异
   - 至少断言所有统计量有限，并把最关键的数值输出到测试失败信息或本地审计脚本中。

#### 7.2.2 推荐的 helper 写法

A5 在 `tests/test_residual.py` 里可以写两个很小的 helper：

- `build_trainer(cfg, mode)`：构建 trainer，并在需要时替换 `trainer.residual_computer`
- `complex_to_dual(arr)`：把 complex `[H, W]` 转成 `[1, 2, H, W]`

但不要把这里膨胀成一套新的 benchmark harness。

#### 7.2.3 reference-based audit 的推荐比较方式

A5 推荐至少比较这三组：

1. `rel_l2(Â_ref_candidate, Â_ref_legacy, evaluation_mask)`
2. `rel_l2(Â_ref_candidate, Â_ref_legacy, whole_grid_mask)`
3. `rel_l2(Â_ref_candidate, Â_ref_legacy, pml_only_mask)`

必要时再加：

4. `loss_pde_baseline(Â_ref_legacy)`
5. `loss_pde_candidate(Â_ref_candidate)`
6. `loss_pde_candidate(Â_ref_legacy)`
7. `loss_pde_baseline(Â_ref_candidate)`

第 6 和第 7 不是必须，但它们很适合判断：

- candidate 与 legacy 只是“给出不同但各自自洽的离散系统”；
- 还是 candidate 引入了明显更差的 operator mismatch。

### 7.3 `tests/test_pml.py`

A5 只在这个文件里做**很小的** PML 侧回归，不要把 heavy audit 全塞进来。

#### 7.3.1 可以做的补充

A5 若有需要，可以在这里补充：

- 物理区 `A=1, B=0` 的已有断言继续有效；
- `grid.pml_mask()` 的 PML 区域存在且非空；
- `pml_only_mask` 与 `physical_slice` 的逻辑关系未被破坏。

#### 7.3.2 不应该做的事情

A5 不要在 `tests/test_pml.py` 里：

- 调用 reference solver；
- 跑完整 trainer 审计；
- 写过长的数值比较；
- 引入依赖 `src/eval/*` 的复杂 helper。

`tests/test_pml.py` 仍应保持“PML 张量不变量测试”的定位。

### 7.4 `configs/experiments/pml_audit.yaml`

本轮 **默认不建议新建** 这个文件。

原因很简单：

- A5 这轮不改 `trainer.py` / `runner.py`；
- 即使写了一个新的 overlay，当前 CLI 也不能靠它切 `lap_tau_mode`；
- 这会制造一种错觉：仓库已经支持用 YAML 切换 strict mode，但实际上没有。

因此 A5 的推荐策略是：

- **不要在首轮 patch 里新建 `configs/experiments/pml_audit.yaml`**；
- 若 A0 后续决定开放一个 config-to-residual 的桥接层，再单独开收口 patch；
- A5 这轮只保留 **programmatic opt-in**。

---

## 8. 提交粒度要求

A5 不允许把 mode 实现、测试、注释全堆成一个大提交。最少拆成下面 2 类提交：

1. **physics implementation 提交**  
   只处理 `src/physics/residual.py`：
   - mode 参数
   - active alias
   - candidate cache
   - `to()` 同步搬运

2. **audit test 提交**  
   只处理：
   - `tests/test_residual.py`
   - `tests/test_pml.py`（若确有必要）

推荐提交顺序：

```text
feat(physics): add lap_tau audit mode while keeping mixed legacy as default

test(physics): add residual and pml audit coverage for legacy vs candidate modes
```

如果实现里顺手补了少量注释，不必单独拆第三个提交；但不要把 `tests` 和 `src` 改动糊成一个大杂烩 patch。

---

## 9. 验证命令

A5 自测必须至少跑完下面这组命令，并把结果交给 A0。A5 虽然不碰 trainer / runner 文件，但 residual 属于训练主链核心，必须做跨层回归。

### 9.1 物理侧单测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q tests/test_residual.py tests/test_pml.py
```

这是 A5 的最低要求。

### 9.2 reference baseline 守门

前提：A2 已并入并提供 `tests/test_reference_solver.py`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q tests/test_reference_solver.py
```

目的不是测 A5 写了什么，而是确认你没有通过改 `self.lap_tau_c` / `ResidualComputer` 接口把 A2 的 reference baseline 打坏。

### 9.3 训练链兼容守门

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q tests/test_train.py tests/test_runner.py
```

目的：

- 确认 `Trainer` 默认路径仍用 legacy；
- 确认 A5 没有通过 residual 改动导致训练主链回归；
- 确认 runner 产物契约仍无变化。

### 9.4 A5 专用 programmatic audit smoke

这是 A5 最重要的 smoke。它不依赖 CLI 扩展，只依赖当前项目已有对象。推荐脚本如下：

```bash
python - <<'PY'
import numpy as np
import torch
from pathlib import Path
from src.config import load_config
from src.train.trainer import Trainer
from src.physics.residual import ResidualComputer
from src.eval.reference_solver import solve_reference_scattering


def make_cfg():
    cfg = load_config(Path('configs/base.yaml'), Path('configs/debug.yaml'))
    cfg.medium.velocity_model = 'smooth_lens'
    return cfg


def make_trainer(mode):
    trainer = Trainer(make_cfg(), device='cpu')
    if mode != 'mixed_legacy':
        trainer.residual_computer = ResidualComputer(
            trainer.grid,
            trainer.pml,
            trainer.tau_d,
            trainer.rhs,
            trainer.loss_mask,
            trainer.omega,
            trainer.diff_ops,
            lap_tau_mode=mode,
        ).to(trainer.device)
    return trainer


def rel_l2(a, b, mask):
    da = (a - b)[mask]
    db = b[mask]
    num = np.linalg.norm(da.ravel())
    den = max(np.linalg.norm(db.ravel()), 1.0e-12)
    return float(num / den)


def complex_to_dual(arr):
    stacked = np.stack([arr.real, arr.imag], axis=0).astype(np.float32)
    return torch.from_numpy(stacked).unsqueeze(0)


def loss_on_solution(trainer, env):
    dual = complex_to_dual(env).to(trainer.device)
    out = trainer.residual_computer.compute(dual)
    return float(out['loss_pde'].item())


legacy = make_trainer('mixed_legacy')
candidate = make_trainer('stretched_divergence')

ref_legacy = solve_reference_scattering(legacy)
ref_candidate = solve_reference_scattering(candidate)

pml_mask = legacy.grid.pml_mask()
source_mask = legacy.loss_mask.astype(bool)
eval_mask = (~pml_mask) & source_mask
whole_mask = source_mask
pml_only_mask = pml_mask

print('rel_l2_eval', rel_l2(ref_candidate, ref_legacy, eval_mask))
print('rel_l2_whole', rel_l2(ref_candidate, ref_legacy, whole_mask))
print('rel_l2_pml', rel_l2(ref_candidate, ref_legacy, pml_only_mask))
print('self_loss_legacy', loss_on_solution(legacy, ref_legacy))
print('self_loss_candidate', loss_on_solution(candidate, ref_candidate))
print('cross_loss_candidate_on_legacy', loss_on_solution(candidate, ref_legacy))
print('cross_loss_legacy_on_candidate', loss_on_solution(legacy, ref_candidate))
PY
```

要求：

- 所有输出必须有限；
- `ref_legacy` 与 `ref_candidate` 的 shape 一致；
- 不能出现 NaN / Inf；
- 结果无需“更小才算成功”，但必须能支持 A5 做结论。

### 9.5 A5 的一条底线

只要出现下面任一情况，A5 就不能宣称 candidate 可以进入默认路径：

- `tests/test_train.py` 或 `tests/test_runner.py` 回归；
- candidate 自己的 `loss_pde` 评估出现 NaN / Inf；
- A2 的 reference tests 因 A5 接口变化而失效；
- `ResidualComputer` 默认构造不再是 legacy。

---

## 10. 与其他 agent 的协作规则

### 10.1 与 A2 的协作

A5 与 A2 的关系非常明确：

- A2 负责 reference baseline 的求解与标准 comparison 口径；
- A5 负责利用这套 baseline 审计 `Δ̃τ` mode；
- A5 不改 A2 的 API，不改 A2 的 summary 键，不改 `src/eval/*`。

A5 可以依赖 A2 的内容包括：

- `solve_reference_scattering(trainer)`
- `assemble_reference_operator(trainer)`
- A2 已冻结的 `evaluation_mask` 定义

A5 不得向 A2 要求：

- 为 A5 单独加一套 PML-only summary 键；
- 修改 comparison 区域；
- 让 A2 为 A5 暴露新的 CLI 入口。

A5 自己需要额外的 whole-grid / PML-only 审计指标，就在**A5 的 test / smoke / handoff**里完成，不污染 A2 的正式接口。

### 10.2 与 A0 的协作

A5 必须在 handoff 中明确告诉 A0：

- 当前默认 mode 仍是哪个；
- candidate mode 名字是什么；
- 是否改了任何 public output / summary / CLI（预期答案：没有）；
- 哪些指标支持“保留 legacy 为默认”；
- 哪些指标支持“candidate 值得未来进入默认”；
- A5 的最终建议是什么。

A0 需要的不是“strict 实现代码”，而是**带数字的默认建议**。

### 10.3 与 A4 的协作

A5 不直接改文档，但必须把下列信息 handoff 给 A4：

- 当前 residual 的 `Δ̃τ` mixed strategy 是否仍是已知技术债；
- 是否已经有 candidate audit mode；
- candidate 是否仅作为 audit-only 保留；
- A2 的标准 summary 键为什么不足以完整反映 A5 结论；
- A5 最终建议一句话写法。

A4 需要把这些写进：

- `PROJECT_STATE.md` 的“已知技术债 / 已审计结论”；
- `CHANGELOG.md` 的一句话变更记录；
- 必要时，在 README 中只提一句“PML `Δ̃τ` 已做定量审计，结论见 PROJECT_STATE”。

### 10.4 与 A3 的协作

A5 不依赖 A3 才能做 operator 审计，但如果 A3 已合并：

- A5 必须基于 A3 之后的最新 `int/multi-agent` 同步；
- A5 不得改 `Trainer.train()`；
- A5 的所有结论都应说明：
  - 这次没有改 hybrid loss；
  - 这次没有改 supervision；
  - 这次只改 residual 内部 `lap_tau_mode` 选择。

### 10.5 与 A1 的协作

A5 与 A1 没有直接文件接口，但 A5 受益于 A1 统一后的环境与 entrypoint。A5 只需遵守：

- 不回退 A1 已整理好的 packaging / test 环境；
- 验证命令优先使用 A1 合并后的统一环境安装方式；
- 不为了自己的审计去改 packaging。

---

## 11. Handoff 模板

### 11.1 A5 -> A0

```text
[A5 -> A0 HANDOFF]
branch: fix/residual-pml-audit
base_commit: <A5 开工时基线 commit>
final_commit: <A5 最后提交 commit>

files_changed:
- src/physics/residual.py
- tests/test_residual.py
- tests/test_pml.py   # 若未改则省略

default_mode_after_patch:
- mixed_legacy

candidate_mode_added:
- stretched_divergence

public_contract_changed:
- runner outputs changed: no
- summary keys changed: no
- CLI flags changed: no
- Trainer default behavior changed: no
- ResidualComputer.compute() return keys changed: no

reference_consumer_compatibility:
- self.lap_tau_c still exists: yes/no
- solve_reference_scattering works without eval-side edits: yes/no

audit_metrics:
- rel_l2_baseline_vs_candidate_eval: <number>
- rel_l2_baseline_vs_candidate_whole: <number>
- rel_l2_baseline_vs_candidate_pml: <number>
- self_loss_legacy: <number>
- self_loss_candidate: <number>
- cross_loss_candidate_on_legacy: <number or n/a>
- cross_loss_legacy_on_candidate: <number or n/a>

commands_run:
- pytest -q tests/test_residual.py tests/test_pml.py
- pytest -q tests/test_reference_solver.py
- pytest -q tests/test_train.py tests/test_runner.py
- programmatic audit smoke (debug + smooth_lens)

recommendation:
- keep legacy as default / promote candidate later / candidate audit-only

why:
- <2-5 句话，基于数字说明>

needs_follow_up_from_A0:
- <若未来要把 candidate 接到 config/CLI，这里说明，但本轮不做>
```

### 11.2 A5 -> A4 DOC INPUT

```text
[A5 -> A4 DOC INPUT]
agent: A5
accepted_by_A0: yes/no
final_commit: <commit>

one_sentence_conclusion:
- <例如：对 residual 中 PML Δ̃τ 做了定量审计；保留 mixed_legacy 为默认，新增 stretched_divergence 作为 audit-only 候选模式。>

what_changed_user_visibly:
- <通常应为：默认行为无变化，仅内部新增审计候选模式与测试覆盖>

what_changed_technically:
- ResidualComputer now has explicit lap_tau modes
- default remains mixed_legacy
- candidate mode is stretched_divergence

what_did_not_change:
- runner outputs
- summary.json keys
- CLI entrypoints
- training config surface

key_numbers_to_document:
- rel_l2_eval: <number>
- rel_l2_whole: <number>
- rel_l2_pml: <number>
- self_loss_legacy: <number>
- self_loss_candidate: <number>

documentation_note:
- A2 standard reference summary keys still use evaluation_mask and do not fully cover PML-only audit conclusions.
```

---

## 12. Blocked 条件与处理办法

### 12.1 A2 尚未并入，或 reference solver 还不稳定

这是 A5 的真正 blocker。因为没有 A2，A5 只能做：

- NaN / Inf 守门；
- 物理区 / PML 区局部数值比较；
- baseline vs candidate 的内部差值对比。

但这还不构成“定量审计闭环”。

处理方式：

- 可以先做 mode 框架和基础安全测试；
- 但不能最终声称 candidate 更好；
- 标记为 `BLOCKED: waiting for A2 integrated reference baseline`；
- 最终 handoff 必须等 A2 基线可用后再提交。

### 12.2 想把 mode 暴露到 YAML / CLI，但 trainer / runner 都不在 A5 范围内

这不是 A5 应该硬闯的方向。处理方式是：

- 本轮只做 `ResidualComputer(..., lap_tau_mode=...)` 的程序化切换；
- 在 handoff 中告诉 A0：如果以后要开放给 CLI，需要额外桥接层；
- A5 自己不要越界去改 `trainer.py` / `runner.py`。

### 12.3 candidate 与 legacy 的物理区 headline metrics 接近，但 PML-only 指标明显变化

这**不是** blocker。恰恰说明 A5 的审计有价值。

处理方式：

- 不要因为 A2 的标准 summary 键看起来没变，就说“没有差异”；
- 把 whole-grid / PML-only 结论写进 handoff；
- 由 A4 在 `PROJECT_STATE.md` 中记录“标准 summary 不足以覆盖本次审计维度”。

### 12.4 candidate 不是严格改进，甚至更差

这也不是 blocker。A5 的职责是审计，不是必须成功替换默认值。

处理方式：

- 保留 legacy 为默认；
- candidate 若仍有研究价值，可以保留为 audit-only mode；
- 若 candidate 没有任何价值，也可以在 A5 分支中直接回退，不进入最终集成；
- 关键是留下数字，而不是留下一个“更复杂但没人敢删”的模式。

### 12.5 要实现真正更解析的 `Δ̃τ`，却发现必须改 `tau_ops.py` / `background.py`

这意味着超出了 A5 当前边界。处理方式：

- 本轮停在 `stretched_divergence` 这种基于现有缓存的 candidate；
- 在 handoff 中明确写出：若要进一步追求 fully analytic strict Laplacian，需要 reopen upstream scope；
- 不要在 A5 分支里擅自扩大文件边界。

### 12.6 base 网格上 reference solve 太慢或内存太大

这不构成 blocker。处理方式：

- 单测和默认 smoke 一律用 debug 网格；
- 若需要 base 网格复核，可作为人工离线验证，不写入 A5 的最小验收要求；
- 不要为了跑 base 网格去削弱测试内容。

---

## 13. 完成定义（Definition of Done）

A5 只有在下面所有条件都成立时才算完成：

- [ ] `ResidualComputer` 明确支持 `lap_tau_mode`，且默认值仍为 `mixed_legacy`
- [ ] `self.lap_tau_c` 仍保留为当前激活态 alias，A2 无需改 eval 代码即可继续工作
- [ ] `stretched_divergence` candidate mode 已可通过程序化方式显式启用
- [ ] `tests/test_residual.py` 已覆盖 legacy 兼容、candidate 安全、reference-based audit
- [ ] `tests/test_pml.py` 若有改动，也只是在补不变量守门，而不是引入复杂 benchmark
- [ ] `tests/test_reference_solver.py`、`tests/test_train.py`、`tests/test_runner.py` 对 A5 改动无回归
- [ ] A5 已提供 evaluation / whole-grid / PML-only 三种区域的量化比较结果
- [ ] A5 已明确向 A0 给出默认建议（不是开放式描述）
- [ ] A5 没有改任何文档、runner、trainer、eval、config 主路径

只满足“代码能跑”和“候选 mode 已存在”，**不算完成**。  
A5 的完成标准必须包含：

- 兼容性；
- 定量结论；
- 默认建议。

---

## 14. 本轮明确不做

A5 本轮明确不做以下事情：

1. 不改 `src/physics/pml.py` 的 `sigma_max` / `gamma` / `A/B` 生成逻辑；
2. 不改 `src/physics/tau_ops.py` 去重构整套 `tau` 分量缓存；
3. 不改 `src/train/trainer.py` / `src/train/runner.py` 去开放 YAML/CLI 切 mode；
4. 不改 `src/eval/*` 去新增一套 A5 专属 summary 键；
5. 不改 `loss_mask` 去屏蔽 PML；
6. 不把 candidate mode 直接设为默认；
7. 不在 CI / 单测里强制跑 base 网格 reference solve；
8. 不把“更严格”误写成“已经证明更好”；
9. 不把 A5 的结论直接写进顶层文档，由 A4 统一收口。

A5 的边界一句话概括就是：

> **只在 `ResidualComputer` 内做可比较、可回退的 `Δ̃τ` 审计，不碰训练/评估/文档主接口，不拿“感觉更正确”代替数字。**

---

## 附录 A. 源码核查与补充说明

本附录记录将本手册落盘前对仓库代码所做的关键交叉核查结果，目的是让 A5 在真正动手前对当前 `ResidualComputer` 的实现事实、PML 相关不变量、`loss_mask` 覆盖范围等关键点有可信依据。附录不改变正文中的职责边界。

### A.1 `ResidualComputer.__init__` 签名与字段（逐项核验）

核查文件：`src/physics/residual.py`。当前构造函数签名与缓存字段如下（行号可能随仓库演化微调，但结构稳定）：

```python
def __init__(self, grid, pml, tau_d, rhs, loss_mask, omega, diff_ops):
    ...
    self.A_x = pml.A_x_t.cfloat()
    self.A_y = pml.A_y_t.cfloat()
    self.B_x = pml.B_x_t.cfloat()
    self.B_y = pml.B_y_t.cfloat()
    self.grad_tau_x = tau_d.grad_tau_x.float()
    self.grad_tau_y = tau_d.grad_tau_y.float()
    ...
    self._precompute_pml_lap_tau(tau_d)
    ...
```

本文件 §2.2、§6.4 中要求 A5 新增 `lap_tau_mode` 参数时把它放在末尾并保持默认 `"mixed_legacy"`，推荐签名：

```python
def __init__(self, grid, pml, tau_d, rhs, loss_mask, omega, diff_ops,
             lap_tau_mode="mixed_legacy"):
```

这一放位直接对应 `Trainer` 当前的构造调用：

```python
self.residual_computer = ResidualComputer(
    self.grid, self.pml, self.tau_d, self.rhs,
    self.loss_mask, omega, self.diff_ops,
).to(self.device)
```

不改动 `Trainer` 即可保持默认走 `mixed_legacy` 路径。

### A.2 `_precompute_pml_lap_tau` 当前实现事实

核查 `src/physics/residual.py` 中 `_precompute_pml_lap_tau(self, tau_d)` 现有主体：

- 计算 `grad_tau_x_stretched = self.A_x * tau_d.grad_tau_x.cfloat()`；
- 计算 `grad_tau_y_stretched = self.A_y * tau_d.grad_tau_y.cfloat()`；
- 最终 `self.lap_tau_c = tau_d.lap_tau.cfloat()`（**未拉伸** 的链式法则结果）。

这恰好是本文件 §2.3–§2.4 所描述的“transport 项一半已拉伸、`Â·Δ̃τ` 一半未拉伸”的混合策略。A5 在 §3.2 提出的 `stretched_divergence` 候选公式正是在同一份缓存上扩展，不需要改 `TauDerivatives`。

### A.3 `compute_loss_mask` 不屏蔽 PML —— 关键事实验证

核查文件：`src/physics/rhs.py`。当前实现精确如下：

```python
def compute_loss_mask(grid, source, cfg):
    mask_radius = cfg.loss.source_mask_radius
    return (source.distance > mask_radius * grid.h).astype(np.float64)
```

也就是说：

- `compute_loss_mask` 只做**震源近场穿孔**；
- 它**没有**把 PML 层置零；
- 当前训练主循环使用 `loss_mask * |R_pde|²` 作为 `loss_pde`，意味着 **PML 区的 residual 确实被纳入了训练目标**。

这就锁死了本文件 §2.6 的结论：A5 审计的 `Δ̃τ` 混合策略不是“PML 里一个理论瑕疵”，而是会真实影响训练 loss 的数值来源。因此 §6.2 要求的 `whole_grid_mask = loss_mask.astype(bool)` 审计口径不可省略。

### A.4 `src/eval/reference_solver.py` 对 `self.lap_tau_c` 的读取依赖

核查事实：`assemble_reference_operator(trainer)` 在组装稀疏矩阵时通过 `trainer.residual_computer.lap_tau_c` 直接取 `Δ̃τ`，而不是重新从 `TauDerivatives` 计算。这意味着：

- 只要 A5 在候选模式下把 `self.lap_tau_c` 切到 candidate 缓存，A2 的 reference solver **自动**切到 candidate 离散，无需改 `src/eval/*`；
- 一旦 A5 在重命名 / 删除 `self.lap_tau_c`，A2 侧会立刻失效。

A5 §4.3 已把“`self.lap_tau_c` 必须继续存在”写成硬规则，与此事实一致。

### A.5 `tests/test_residual.py` 已有断言（A5 不改不破坏）

核查当前断言范围（不同测试函数里的关键断言）：

- 前向可跑通、返回字典包含 `residual_real`、`residual_imag`、`loss_pde`；
- homogeneous + 零网络时 `loss_pde` 接近 0；
- smooth_lens + 零网络时 `loss_pde` 严格大于 0；
- `rc.A_x.imag.abs().max().item() > 0.0` 与 `rc.A_y.imag.abs().max().item() > 0.0`（PML 系数复数域保真）；
- 梯度能回传（至少一次 `loss_pde.backward()` 不抛）。

A5 本轮所有补测试必须保证这些断言继续通过；特别是：

- 新增的 `lap_tau_mode` 默认值 `"mixed_legacy"` 不得让 `residual_real` 数值发生变动；
- `to(device)` 后 `self.lap_tau_legacy_c` / `self.lap_tau_candidate_c` / `self.lap_tau_c` 必须位于同一 device 上。

### A.6 `Trainer` 构造 `ResidualComputer` 的调用位置

核查事实：`src/train/trainer.py` 中当前只有**一处**构造 `ResidualComputer`，位于 `Trainer.__init__` 中：

```python
self.residual_computer = ResidualComputer(
    self.grid, self.pml, self.tau_d, self.rhs,
    self.loss_mask, omega, self.diff_ops,
).to(self.device)
```

`evaluate_saved_run()` 中的 `Trainer` 也经同一路径构造。这对 A5 的重要含义是：

- 只要 A5 给 `ResidualComputer.__init__` 的新参数提供默认值，**所有现有调用者都保持 legacy 行为**；
- 无需在 `Trainer` / `runner.py` / `scripts/*.py` 做一行改动；
- A5 §2.10 所述“不暴露到 CLI / YAML 主路径”的约束与现有事实完全兼容。

### A.7 附录的职责

附录只是事实对账，目的是：

- 让 A5 在动 `residual.py` 前对当前混合策略、`loss_mask` 真实覆盖面、A2 依赖点有精确认知；
- 让 A5 §3.2 的 candidate 公式有可直接对应的代码落点；
- 让 A5 §9.4 的 programmatic audit smoke 中对三种 mask 的取值方式有事实支撑。

附录**不新增** A5 职责，也**不放宽** §4、§7、§14 中的禁令。特别地，附录**不**鼓励 A5 在本轮把 candidate 设为默认，也**不**鼓励 A5 把 `compute_loss_mask` 改成屏蔽 PML——这属于超出 A5 审计边界的修改方向，应由 A0 / A4 在集成阶段另行评估。
