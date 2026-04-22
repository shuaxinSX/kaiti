# A3_HYBRID_LOSS.md

> Agent A3 专用执行手册。  
> A3 只负责把**现有 `loss_data()` / `lambda_data` 真正接进训练主循环**，让仓库具备 PDE-only、data-only、PDE+data 三种训练模式；不负责 reference 求解、不负责 runner 产物扩展、不负责物理离散修正、不负责顶层文档。  
> 本文件只定义 A3 的静态职责、改单边界、动作顺序和交接格式。动态状态、最新集成提交、Gate 开闭、共享文件是否允许开工，一律以 `WORKTREE_BOARD.md` 为准。

---

## 1. 身份与协作位置

**Agent ID**：A3  
**分支**：`feat/hybrid-loss`  
**worktree**：`../wt/train`  
**启动 Gate**：`Gate 3` 已开（A2 已合并，reference 产物契约已冻结）  
**代码合并顺位**：A2 之后的第 3 个功能分支  
**直接交付对象**：A0（集成）  
**上游依赖对象**：A2（reference 产物格式与路径契约）  
**文档交接对象**：A4（README / PROJECT_STATE / CHANGELOG / DEV_PLAN）  
**下游受益对象**：A5（以后可以比较 PDE-only 与 hybrid 训练结果，但 A5 不依赖 A3 改动才能开工）

A3 是本轮并行开发里的“**训练闭环打通者**”。A3 不是来重写训练器，也不是来扩展评估器；A3 只做四件事：

1. 把 `src/train/losses.py` 中已经存在、但尚未真正生效的 `loss_data()` / `lambda_data` 接进训练主循环；
2. 让 `Trainer` 能在**有参考标签**时计算监督损失，在**没标签**时完全保持当前 PDE-only 行为；
3. 在不改 `runner.py` 产物契约的前提下，记录并暴露 `loss_pde` / `loss_data` / `loss_total` 三条历史；
4. 给 A0 和 A4 提供一套稳定的 hybrid 训练配置约定与使用说明。

A3 的交付不是“训练代码更复杂了”，而是让项目终于能明确区分下面三类模式：

- **PDE-only**：只优化 `loss_pde`，保持当前默认行为；
- **data-only**：只优化 `loss_data`，验证监督项通路本身是通的；
- **hybrid**：同时优化 `loss_pde` 与 `loss_data`，给后续实验留出接口。

---

## 2. 当前仓库中，A3 必须面对的真实现状

A3 不是从空白页开始，而是要在**当前代码已经给出的结构和限制**上收口。与 A3 直接相关的真实现状如下。

### 2.1 `loss_data()` 和 `lambda_data` 已存在，但还没有真正进入训练闭环

当前 `src/train/losses.py` 已经实现：

- `loss_data(A_pred, A_true)`
- `loss_total(loss_pde, loss_data_val=None, lambda_pde=1.0, lambda_data=0.0)`

其数学形式已经明确：

```text
L_total = λ_pde * L_pde + λ_data * L_data
```

其中：

- `L_pde` 由 `ResidualComputer.compute()` 返回；
- `L_data` 当前定义为 dual-channel 张量上的均方误差：`mean((A_pred - A_true)^2)`；
- `loss_total()` 本身已经支持传入 `loss_data_val`。

也就是说：**A3 不需要重新发明损失函数公式**。当前缺的不是数学定义，而是训练主循环没有把它用起来。

### 2.2 `Trainer.train()` 当前只用 PDE 损失，`lambda_data` 形同虚设

当前 `src/train/trainer.py` 的 `Trainer.train()` 每一步大致做的是：

1. `A_scat = self.model(self.net_input)`
2. `result = self.residual_computer.compute(A_scat)`
3. `total = loss_total(result['loss_pde'], lambda_pde=lambda_pde, lambda_data=lambda_data)`
4. 反向传播 `total.backward()`

注意第 3 步：**没有把 `loss_data_val` 传进去**。这意味着：

- 即使配置里把 `training.lambda_data` 设成非零；
- 甚至即使未来加了监督标签路径；
- 当前训练结果依然只会由 `loss_pde` 决定。

因此，A3 的主战场不在 `losses.py`，而在 `trainer.py`。

### 2.3 当前 `Trainer` 没有任何监督数据接口

当前 `Trainer.__init__()` 会构建：

- `Grid2D`
- `Medium2D`
- `PointSource`
- `BackgroundField`
- `EikonalSolver`
- `TauDerivatives`
- `PMLTensors`
- `ResidualComputer`
- `self.net_input`
- `self.model`
- `self.optimizer`

但它**没有**：

- `self.reference_target`
- `self.supervision_enabled`
- `self.loss_history_pde`
- `self.loss_history_data`
- 任何 reference 文件读取逻辑

因此 A3 要接监督项，必须先解决三个问题：

1. 怎么从配置安全地拿到 supervision 开关与路径；
2. 怎么把磁盘上的 reference 标签变成 `[1, 2, H, W]` 的 torch 张量；
3. 怎么在不改返回值契约的前提下，保存三类 loss 历史。

### 2.4 当前配置系统有两个现实限制：只有单 overlay；`Config` 也没有 `.get()`

当前 `src/config.py` 的 `load_config(base_path, overlay_path)` 只支持：

- `base + 单个 overlay`

它**不支持**：

- `base + debug + hybrid` 这种多 overlay 叠加。

并且当前 `Config` 容器只支持：

- 属性访问，如 `cfg.training.lr`
- `__contains__`，如 `'training' in cfg`

它**没有**标准 dict 的：

- `.get()`

所以 A3 在 `trainer.py` 里不能写出这种代码：

```python
cfg.training.get("supervision", {})
```

这种写法在当前项目里会直接出错。A3 必须用：

- `hasattr(cfg.training, "supervision")`
- 或 `'supervision' in cfg.training`

来做向后兼容判断。

### 2.5 当前 `base.yaml` 里没有 supervision 结构，默认行为必须完全向后兼容

当前 `configs/base.yaml` 只有：

```yaml
training:
  lr: 1.0e-3
  epochs: 1000
  batch_size: 1
  lambda_pde: 1.0
  lambda_data: 0.0
```

并没有：

- `training.supervision.enabled`
- `training.supervision.reference_path`

这意味着 A3 不能假设 `cfg.training.supervision` 一定存在。A3 的代码必须满足：

- **老配置不报错**；
- **`lambda_data=0` 的默认路径和当前行为一致**；
- 只有在显式启用 supervision 且 `lambda_data > 0` 时，才走数据损失路径。

### 2.6 当前 `runner.py` 和 `tests/test_runner.py` 已锁定了训练返回值与产物契约

当前 `src/train/runner.py` 的关键约定是：

- `Trainer.train()` 返回的是**一维 `loss` 历史列表**；
- `save_losses(losses, output_dir)` 只会把这条一维历史写成 `losses.npy` / `losses.csv` / `loss_tail.csv`；
- `summary.json` 中的 `final_loss` 目前就是总 loss；
- `tests/test_runner.py` 会检查现有产物文件名和 `evaluate_saved_run()` 的返回结构。

这意味着一个非常关键的硬边界：

> **A3 不能把 `Trainer.train()` 改成返回 dict、tuple 或多数组对象。**

否则 `runner.py` 会直接被打断，`tests/test_runner.py` 也会一起坏掉。

A3 的正确策略是：

- `train()` 仍返回一维 `loss_total` 历史；
- 额外的 `loss_pde` / `loss_data` 历史存在 `Trainer` 实例属性上；
- 是否把它们写入磁盘、写入 summary，由 A0 在集成阶段决定；A3 本轮不碰 `runner.py`。

### 2.7 A2 的 reference 产物契约必须被 A3 当成外部事实，而不是重写一遍

根据当前 `WORKTREE_BOARD.md` 中已经冻结的 A2 约定，A2 会输出：

- `reference_envelope.npy`
- `reference_wavefield.npy`
- `reference_metrics.json`
- `reference_comparison.csv`
- `reference_error_heatmaps.png`

其中 A3 真正要消费的只有：

- `reference_envelope.npy`

并且其语义应固定为：

- **复数数组**；
- 形状 `[H, W]`；
- 对应**全网格**（含 PML）；
- 网格顺序与 `trainer.grid.ny_total, trainer.grid.nx_total` 一致；
- 表示散射包络 `Â_ref`，不是总波场。

A3 不应：

- 去读取 `reference_wavefield.npy` 再反推 envelope；
- 去调用 `reference_solver.py` 现场求标签；
- 在 `trainer.py` 里再复制一份 reference 生成逻辑。

### 2.8 当前已有测试已经给了 A3 很好的回归锚点

当前 `tests/test_train.py` 已经锁定了几件重要事情：

- `loss_data()` 数学结果正确；
- `loss_total()` 在 PDE-only 与带数据项时数值正确；
- homogeneous 场景下当前训练 loss 为 0；
- `smooth_lens` 场景下当前 loss 能下降；
- `Trainer` 的设备选择、网络输入 shape 和 NaN/Inf 行为都已稳定。

这对 A3 很有价值，因为它说明：

- A3 不需要重写现有测试；
- 只需要在 `tests/test_train.py` 上追加 supervision 路径覆盖；
- 只要默认路径不变，现有测试就是最好的回归守门员。

### 2.9 A3 当前最容易犯的错，就是越界到 runner / eval / physics

A3 很容易被诱导去做这些错误动作：

- 为了保存三条 loss 历史，直接改 `src/train/runner.py`
- 为了更方便拿 reference，直接 import A2 的 CLI 脚本
- 为了让 hybrid 看起来“更强”，顺手改 `src/physics/residual.py`
- 为了让 CLI 更方便，新增一堆 `run_train.py` 参数
- 为了让 summary 更完整，顺手改 `tests/test_runner.py`

这些都属于越界。A3 的正确策略是：

- 训练逻辑只收在 `src/train/*`；
- reference 只当外部标签文件读取；
- `runner.py` 仍由 A0 统一收口；
- 现有输出文件名一个都不改。

---

## 3. A3 的唯一目标

A3 完成后，仓库必须具备下面这一整套能力，而不是只完成其中一部分：

1. **默认 PDE-only 路径完全不变**：
   - `configs/base.yaml` 不需要新增字段也能训练；
   - `lambda_data=0.0` 时行为与当前一致；
   - 现有 `tests/test_train.py` 与 `tests/test_runner.py` 不因返回值变化而破坏。
2. **显式 supervision 路径可用**：
   - 当 `training.lambda_data > 0` 且 `training.supervision.enabled = true` 且 `reference_path` 有效时，训练会真正计算并使用 `loss_data`。
3. **三种模式可区分**：
   - PDE-only：`lambda_pde > 0, lambda_data = 0`
   - data-only：`lambda_pde = 0, lambda_data > 0`
   - hybrid：`lambda_pde > 0, lambda_data > 0`
4. **训练主返回值契约不变**：
   - `Trainer.train()` 继续返回一维总 loss 历史；
   - 同时在 `Trainer` 对象上保存 `loss_pde` / `loss_data` / `loss_total` 历史。
5. **标签读取契约明确**：
   - A3 只认 `reference_envelope.npy`；
   - 标签 shape、dtype、路径错误时会在训练开始前 fail fast；
   - 不会把错误拖到 epoch 中途才爆。
6. **A4 能据此写文档，A0 能据此做集成**：
   - A3 会提供新的配置字段定义、错误行为说明、训练模式真值表、最小使用方法。

A3 的目标不是“多加一个 loss”，而是把现有训练器从**只有 PDE 残差的一条路**，升级成**支持监督混合训练的稳定接口**。

---

## 4. 文件所有权与并改禁令

### 4.1 A3 独占文件

A3 未合并前，以下文件或目录只允许 A3 修改：

- `src/train/losses.py`
- `src/train/trainer.py`
- `tests/test_train.py`
- `configs/experiments/hybrid.yaml`（新建）
- `src/train/supervision.py`（可选新建；仅在确有必要时）

说明：

- `src/train/trainer.py` 是 A3 的主战场，也是热点文件；本轮只允许 A3 主写。
- 如果为了把标签读取逻辑从 `trainer.py` 中抽出来，A3 可以新建一个极小的 `src/train/supervision.py`，但前提是它只做“配置解析 + reference 文件加载 + 形状检查”，不能膨胀成第二套训练框架。

### 4.2 A3 明确禁止修改的文件

A3 不得修改以下文件，即使“只是顺手补一下”：

- `src/train/runner.py`
- `src/eval/*`
- `scripts/solve_reference.py`
- `scripts/run_train.py`
- `scripts/evaluate_run.py`
- `src/physics/*`
- `src/models/*`
- `src/core/*`
- `src/config.py`
- `pyproject.toml`
- `requirements.txt`
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

- `runner.py` 虽然与训练强相关，但它已是热点文件，本轮由 A0 做最终收口；A3 不得把 loss 历史写盘、summary 扩展、CLI 参数扩展塞进去。
- `src/eval/*` 与 `scripts/solve_reference.py` 全归 A2；A3 只消费标签文件，不生产标签。
- `tests/test_runner.py` 是集成产物契约，A3 不得为了让自己的改动更容易过测试而去改断言。

---

## 5. 开工前同步协议

A3 每次**开始一个新的修改会话之前**，都必须重新做一次同步检查。A3 不能把“上次同步时 A2 还没合并”当成继续改训练器的理由。

### 5.1 A3 只认 `int/multi-agent`，不追别人的 feature branch

A3 的上游只有两个：

1. `int/base-current`：理论冻结基线
2. `int/multi-agent`：实际最新集成状态

A3 不得：

- 直接追 A2 的 feature branch；
- 直接追 A4 / A5 的 feature branch；
- 以“我看过别人本地文件”为依据开始修改热点文件。

A3 只承认：

> **已经被 A0 合并进 `int/multi-agent` 的内容，才叫当前项目最新状态。**

### 5.2 启动前必须确认 Gate 3 已开

A3 开工前先看 `WORKTREE_BOARD.md`，确认以下事实成立：

- Gate 3 已开；
- A2 的 reference 产物契约已经冻结；
- `reference_envelope.npy` 的语义没有处于“待讨论”状态；
- A0 没有在看板上标注 `trainer.py` 冻结或回退中。

只要 Gate 3 未开，A3 就不能最终化标签读取格式；最多只允许准备不依赖 A2 的脚手架。

### 5.3 同步检查步骤

进入 `../wt/train` 后，顺序执行：

```bash
git status --short
git fetch --all --prune
```

要求：

- `git status --short` 必须为空；若工作树脏，先收尾或 stash，禁止在脏树上同步。
- 然后查看 `WORKTREE_BOARD.md`，确认最新集成提交和 Gate 状态。

之后按下面规则同步：

- 如果 `int/multi-agent` 已领先，A3 必须先同步 `int/multi-agent`；
- 否则可从 `int/base-current` 快进同步。

推荐命令：

```bash
git merge --ff-only int/multi-agent
# 若 int/multi-agent 尚未领先，可退而使用：
git merge --ff-only int/base-current
```

### 5.4 硬规则

只要 `int/multi-agent` 的基线更新了，A3 就**不能在旧基线上开启新的文件修改**。  
已经进行到一半、且只差极小收尾的补丁可以先结束，但结束后必须立即同步，再继续下一批文件。

---

## 6. A3 的执行顺序

A3 必须按下面顺序推进。A3 的风险不在于代码量，而在于一旦先动错地方，就会把 runner 契约和 reference 契约一起拖乱。

### 第 1 步：先冻结 supervision 配置契约与标签契约

在写代码前，A3 先把下面两组事实固定住，后面所有实现都按这套契约走。

#### 6.1 标准 supervision 配置结构

A3 统一约定以下配置结构：

```yaml
training:
  lambda_pde: 1.0
  lambda_data: 0.0
  supervision:
    enabled: false
    reference_path: null
```

含义固定为：

- `lambda_pde`：PDE loss 权重
- `lambda_data`：监督 data loss 权重
- `supervision.enabled`：是否允许尝试读取监督标签
- `supervision.reference_path`：指向 A2 输出的 `reference_envelope.npy`

A3 本轮**不再扩展更多层级字段**，例如：

- `reference_format`
- `phase_weight`
- `amp_weight`
- `normalization`

这些都不在本轮范围内。

#### 6.2 训练模式真值表

A3 必须按下面的真值表实现，而不是自由发挥：

| 条件 | 行为 |
| --- | --- |
| `lambda_data == 0` 且 `supervision` 缺失/关闭 | PDE-only；行为与当前完全一致 |
| `lambda_data == 0` 且 `supervision.enabled == true` | 仍视作 PDE-only；可记录 warning，但不得报错 |
| `lambda_data > 0` 且 `supervision.enabled == false` | **配置错误，训练前直接报错** |
| `lambda_data > 0` 且 `supervision.enabled == true` 且 `reference_path` 有效 | 计算 `loss_data`，进入 data-only 或 hybrid |
| `lambda_data > 0` 且 `reference_path` 缺失/无效 | **配置错误，训练前直接报错** |

这个真值表有两个目的：

1. 默认行为必须兼容旧配置；
2. 用户一旦明确想用 `lambda_data`，就不能悄悄退回无监督路径。

#### 6.3 标签文件契约

A3 只认一个标签文件：

- `reference_envelope.npy`

并且要求：

- 文件内容是 complex ndarray（`complex64` 或 `complex128` 都可接受）；
- 形状严格等于 `[trainer.grid.ny_total, trainer.grid.nx_total]`；
- 表示全网格（含 PML）的散射包络；
- 坐标顺序与 `Trainer` 当前网格一致；
- 不是裁剪后的 physical 区，也不是 wavefield。

A3 必须在训练开始前做这些检查，而不是训练了半天才因为 shape 对不上才炸。

### 第 2 步：先决定“历史如何记录”，再去改训练循环

A3 进入代码前，必须先冻结这条向后兼容原则：

> **`Trainer.train()` 仍返回 total loss 的一维 Python list，不改变 `runner.py` 现有调用方式。**

在此基础上，A3 再引入额外历史。推荐最小方案：

- `self.loss_history_total`
- `self.loss_history_pde`
- `self.loss_history_data`
- `self.supervision_enabled`
- `self.reference_path`
- `self.last_step_metrics`

其中：

- `loss_history_total` 与当前 `train()` 返回值保持一致；
- `loss_history_pde` / `loss_history_data` 只存在内存里，不在 A3 分支中落盘；
- `last_step_metrics` 可以用于测试与日志；
- A0 若未来要写入 `summary.json`，可以在集成阶段消费这些字段。

### 第 3 步：把 supervision 解析与标签加载做成“初始化阶段一次性动作”

A3 不应在每个 epoch 都重新读取 reference 文件。正确顺序应是：

1. `Trainer.__init__()` 解析 supervision 配置；
2. 若当前模式需要 data loss，则在初始化阶段读取 `reference_envelope.npy`；
3. 立刻校验 dtype、shape、路径；
4. 转成模型需要的 dual-channel tensor；
5. 缓存在 `self.reference_target` 上并搬到 `self.device`；
6. 训练循环里直接复用，不再做文件 IO。

推荐转换规则：

```text
complex [H, W] -> float32 dual-channel [1, 2, H, W]
```

即：

- 第 0 通道为实部
- 第 1 通道为虚部
- batch 维固定为 1

这样能与当前模型输出 `A_scat` 的 shape 完全对齐。

#### 6.4 配置解析的向后兼容写法

因为 `Config` 不是 dict，A3 推荐用如下思路：

```python
training_cfg = cfg.training
has_supervision = hasattr(training_cfg, "supervision")
```

然后再安全读取：

- `enabled`
- `reference_path`

而不是写 `.get()`。

#### 6.5 推荐的 fail-fast 错误

A3 在以下场景应在训练开始前直接失败：

1. `lambda_data > 0`，但 `supervision.enabled = false`
2. `lambda_data > 0`，但 `reference_path is None`
3. `reference_path` 指向的文件不存在
4. `reference_envelope.npy` 不是复数数组
5. 标签 shape 与当前网格 shape 不一致

推荐错误类型：

- 路径不存在：`FileNotFoundError`
- 配置组合非法或 shape/dtype 不符：`ValueError`

### 第 4 步：在 `trainer.py` 中最小接入 data loss，不改现有训练主干

A3 正确的接法是在当前训练主循环里做最小闭环，而不是重新设计训练器。

推荐步骤：

1. 保留当前 `A_scat = self.model(self.net_input)`
2. 保留当前 `result = self.residual_computer.compute(A_scat)`
3. 从 `result` 取 `loss_pde = result['loss_pde']`
4. 如果当前 supervision 生效，则：
   - `loss_data_val = loss_data(A_scat, self.reference_target)`
5. 否则：
   - `loss_data_val = None`
6. 调用：
   - `total = loss_total(loss_pde, loss_data_val, lambda_pde=lambda_pde, lambda_data=lambda_data)`
7. 反向传播与优化器步骤保持不变
8. 每步把三类 loss 记进各自历史
9. 返回 `loss_history_total`

关键点：

- `loss_total()` 的签名本来就支持 `loss_data_val`，A3 只需要把它真正传进去；
- `loss_data()` 数学形式本轮不改；
- `Trainer.train()` 的主返回值仍然是总 loss 历史。

### 第 5 步：本轮不做“data-only 性能优化”，优先保证行为正确

当 `lambda_pde = 0` 且 `lambda_data > 0` 时，理论上可以考虑跳过 PDE residual 计算。但 A3 本轮**不做这个优化**，原因如下：

1. 当前 `ResidualComputer.compute()` 不只是为了 total loss，也能提供一致的 `loss_pde` 日志；
2. 一旦分出“data-only 走另一套分支”，训练器复杂度会显著升高；
3. A3 当前任务是把通路打通，不是做性能微调。

因此，A3 本轮的推荐策略是：

- 即使 data-only，也仍然计算 `loss_pde`；
- 只是让 `lambda_pde = 0` 时它不参与 total loss。

这样最稳，也最不容易引入分支级 bug。

### 第 6 步：把 `configs/experiments/hybrid.yaml` 做成“单 overlay、可表达 schema 的完整样例”

A3 必须正视当前 CLI 与配置系统的现实：

- `load_config()` 目前只收一个 overlay；
- `run_training()` / `runner.py` 也只接一个 `overlay_config`；
- 本轮 A3 又禁止修改 `runner.py`。

因此 A3 **不能依赖**：

- `base + debug + hybrid` 三层叠加。

更稳的做法是：

> **把 `configs/experiments/hybrid.yaml` 做成一个“完整的 debug-sized overlay + supervision schema 示例”。**

也就是把 `debug.yaml` 里真正影响 smoke 成本的关键覆盖项直接带进去，再补 supervision 字段。

推荐最小内容：

```yaml
physics:
  omega: 10.0

grid:
  nx: 32
  ny: 32

pml:
  width: 5

model:
  nsno_blocks: 2
  nsno_channels: 16
  fno_modes: 8

training:
  epochs: 50
  lr: 1.0e-3
  lambda_pde: 1.0
  lambda_data: 0.0
  supervision:
    enabled: false
    reference_path: null

logging:
  level: "DEBUG"
```

注意：

- `reference_path` 必须保持 `null`，不能提交任何机器本地绝对路径；
- 这个文件的首要目的是**把 schema 固化进仓库**，而不是保证开箱即跑 hybrid；
- 真正的 reference 路径在测试里写临时文件，在人工实验时由用户填入。

### 第 7 步：测试优先锁“通路正确”，不在 A3 单测里重复验证 A2 数值正确性

A3 的测试重点不是“reference solver 算得对不对”，那已经是 A2 的职责。A3 要在 `tests/test_train.py` 里锁定的是：

1. PDE-only 默认路径不变；
2. 有效 supervision 配置时，`loss_data` 真正生效；
3. data-only 与 hybrid 路径都能走通；
4. 标签 shape / dtype / 路径错误时会 fail fast；
5. `Trainer.train()` 的返回值仍是 list，不破坏 runner。

#### 6.6 推荐测试策略：用测试内构造的 synthetic complex label 文件

A3 的单测不必依赖 A2 solver 实时求 reference。更稳的做法是：

- 在测试里根据当前 trainer 的网格 shape，构造一个 synthetic complex ndarray；
- 写入临时目录的 `reference_envelope.npy`；
- 把 `cfg.training.supervision.reference_path` 指向它；
- 验证训练路径和 loss 组合逻辑。

这样能带来两个好处：

1. 测试速度快，不依赖 `spsolve`；
2. A3 只验证“标签消费接口”，不与 A2 的数值正确性测试重复。

#### 6.7 A3 至少要补的测试

至少新增下面 4 类覆盖：

**(1) 默认 PDE-only 行为不变**

- 使用当前 `cfg_homogeneous` 或 `cfg_lens`
- 不新增 supervision 字段
- 断言 `Trainer.train()` 返回 list
- 断言 `loss_history_data` 全为 0 或空
- 断言已有测试行为不变

**(2) data-only 路径可用**

- 构造 synthetic complex label 文件
- 设 `lambda_pde = 0.0`
- 设 `lambda_data > 0`
- `supervision.enabled = true`
- 训练几个 epoch
- 断言 `loss_history_data[0] > 0`
- 断言 `loss_history_total[-1] <= loss_history_total[0]` 或至少能计算且为有限值

**(3) hybrid 路径真正把两项都纳入 total**

- 构造 synthetic label
- 设 `lambda_pde > 0` 且 `lambda_data > 0`
- 训练 1 个 epoch
- 断言某一步满足：

```text
loss_total == lambda_pde * loss_pde + lambda_data * loss_data
```

允许使用 `pytest.approx` 做数值比较。

**(4) 配置/文件错误会 fail fast**

至少覆盖：

- `lambda_data > 0` 但 `supervision.enabled = false`
- `reference_path` 不存在
- reference 文件 shape 不对
- reference 文件不是复数数组

A3 的测试目标是把“默默不生效”的问题变成“要么正确工作，要么一开始就明确报错”。

### 第 8 步：通过日志暴露三类 loss，不通过 runner 新增产物

A3 本轮不改 runner，所以无法把三类 loss 正式写入 `summary.json` / `losses.csv`。  
但 A3 仍需在 `trainer.py` 的日志中清楚区分：

- `loss_total`
- `loss_pde`
- `loss_data`

推荐日志格式：

```text
Epoch 10/50, loss_total=..., loss_pde=..., loss_data=...
```

当 supervision 未启用时：

- `loss_data` 打成 `0.0` 即可；
- 不要省略字段，避免日志格式忽左忽右。

### 第 9 步：把 runner 相关需求写成 handoff，交给 A0，而不是自己越界实现

如果 A3 最终发现下面这些需求很合理：

- 想把 `loss_pde` / `loss_data` / `loss_total` 三条曲线写盘；
- 想把 `summary.json` 区分 `final_loss_total` 和 `final_loss_pde`；
- 想让 CLI 直接传 `--reference-path`；

A3 的做法应该是：

- **在 handoff 中清楚提出**；
- 交给 A0 评估是否在集成阶段统一落地；
- A3 自己不改 `runner.py`。

---

## 7. 分文件改动规则

### 7.1 `src/train/losses.py`

A3 对这个文件应采取**最小修改策略**。

允许做：

- 补类型注释或注释说明；
- 补 docstring，写明 `loss_data()` 作用在 dual-channel envelope 上；
- 如确有必要，补极小的辅助函数。

不建议做：

- 改 `loss_data()` 的数学定义；
- 改 `loss_total()` 的组合方式；
- 新增 amplitude loss / phase loss / 复杂正则项。

A3 这轮的任务不是设计新损失，而是把现有接口接通。

### 7.2 `src/train/trainer.py`

这是 A3 的主文件。A3 的主要改动应集中在这里。

推荐新增或修改的职责包括：

- 解析 supervision 配置；
- 一次性加载 `reference_envelope.npy`；
- 把 complex label 转成 dual-channel tensor；
- 在 `train()` 中计算 `loss_data_val`；
- 记录 `loss_history_total` / `loss_history_pde` / `loss_history_data`；
- 暴露 `last_step_metrics`；
- 保持 `reconstruct_wavefield()`、`build_network_input()`、设备解析逻辑不变。

A3 不要在这里做：

- 调 reference solver
- 写 CSV / JSON / PNG
- 引入 dataset / dataloader 体系
- 改动 NSNO 模型结构
- 更改 residual 公式

#### 7.2.1 推荐的内部属性名

为了让 A0 / 测试 / 文档都能对齐，推荐统一下列名字：

- `self.supervision_enabled`
- `self.reference_path`
- `self.reference_target`
- `self.loss_history_total`
- `self.loss_history_pde`
- `self.loss_history_data`
- `self.last_step_metrics`

这些名字一旦落地，后续就不要在集成阶段再改第二次。

#### 7.2.2 `train()` 返回值硬规则

`train()` 必须继续返回：

- `list[float]`，代表 total loss 历史

不允许改成：

- dict
- tuple
- 自定义对象
- numpy 多列矩阵

因为这会直接破坏 `runner.py` 和 `tests/test_runner.py`。

### 7.3 `src/train/supervision.py`（可选新建）

只有当 `trainer.py` 因为 supervision 逻辑明显变得过长时，A3 才允许新建此文件。

适合放进去的内容只有：

- `resolve_supervision_config(cfg)`
- `load_reference_target(reference_path, expected_shape, device)`
- `complex_array_to_dual_channel_tensor(array)`

不适合放进去的内容：

- 训练循环
- 优化器逻辑
- reference 求解逻辑
- runner 输出逻辑

如果 `trainer.py` 仍能保持清晰，A3 完全可以不新建这个文件。

### 7.4 `tests/test_train.py`

这是 A3 的主测试文件。A3 只追加测试，不重写原有测试结构。

要求：

- 保留现有 `TestLossFunctions` 和 `TestTrainer` 的已存在断言；
- 新增 supervision/hybrid 相关 fixture 或测试函数；
- synthetic reference 文件尽量用 `tmp_path` 动态生成，不提交静态测试数据文件；
- 保持 CPU 可跑、debug 网格可跑。

A3 不应在这里做：

- 跑大网格
- 跑长 epoch
- 调 A2 的 CLI
- 写依赖本地绝对路径的测试

### 7.5 `configs/experiments/hybrid.yaml`

这个文件必须承担两个角色：

1. 把 supervision schema 固定入库；
2. 作为一个**单 overlay** 的 debug-size 示例。

它不应承担的角色：

- 机器本地实验路径缓存文件
- 多个实验的全集合配置池
- 运行时由脚本自动改写的临时文件

关键规则：

- `reference_path` 必须是 `null`；
- 不能写死 `C:\...` 或 `/home/...`；
- 不能依赖 `debug.yaml` 再叠一层；
- 不改 `configs/base.yaml`。

### 7.6 `src/train/runner.py`

A3 明确禁止修改。

即便你很想把以下功能加进去：

- `loss_data_curve.csv`
- `summary.json` 增加 `final_loss_data`
- CLI 新增 `--reference-path`

也只能记入 handoff，不得在 A3 分支里落地。

### 7.7 `tests/test_runner.py`

A3 明确禁止修改。

如果 A3 改了 `Trainer.train()` 却导致 `tests/test_runner.py` 失败，正确做法是：

- 回头修正 A3 的向后兼容实现；
- 而不是去放宽 `tests/test_runner.py` 的断言。

---

## 8. 提交粒度要求

A3 不允许把 supervision 解析、训练闭环接入、测试、配置样例全堆成一个大提交。最少拆成下面 4 类提交：

1. **supervision config / loader 提交**  
   处理 `trainer.py`（或可选 `supervision.py`）中的配置解析、reference 标签读取、shape/dtype 校验
2. **train loop integration 提交**  
   把 `loss_data_val` 真正接进 `loss_total()`，并记录三类 loss 历史
3. **config example 提交**  
   新建 `configs/experiments/hybrid.yaml`
4. **test 提交**  
   在 `tests/test_train.py` 中补 supervision / data-only / hybrid / fail-fast 覆盖

推荐提交顺序：

```text
feat(train): add supervision config parsing and reference target loading
feat(train): wire data loss into Trainer.train and keep total-loss return contract
chore(config): add hybrid training overlay example
test(train): add supervision and hybrid regression coverage
```

如果 `losses.py` 只是补注释或轻微整理，可并入第 2 个提交；不要为了“看起来干净”单独切一个空洞提交。

---

## 9. 验证命令

A3 自测必须至少跑完下面这组命令，并把结果交给 A0。由于 A3 不碰 runner，这里的重点是训练行为与回归兼容性，而不是新产物文件。

### 9.1 训练单测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q tests/test_train.py
```

### 9.2 runner 回归守门

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q tests/test_runner.py
```

目的不是测试 A3 新功能，而是确认你没有把 `Trainer.train()` 的返回值契约搞坏。

### 9.3 PDE-only smoke（现有路径不变）

```bash
rm -rf ../runs/train/pde_only_smoke 2>/dev/null

python scripts/run_train.py \
  --config configs/base.yaml \
  --overlay configs/debug.yaml \
  --device cpu \
  --epochs 1 \
  --velocity-model smooth_lens \
  --output-dir ../runs/train/pde_only_smoke
```

要求：

- 命令可跑通；
- 现有训练产物仍按原样生成；
- 不需要任何 supervision 字段也能正常执行。

### 9.4 programmatic data-only / hybrid smoke

由于 A3 不改 `runner.py` / CLI 参数，本轮最稳的 hybrid smoke 应该通过一小段 Python 脚本完成。推荐流程：

```bash
python - <<'PY'
from pathlib import Path
import tempfile
import numpy as np
from src.config import load_config
from src.train.trainer import Trainer

base = Path("configs/base.yaml")
hybrid = Path("configs/experiments/hybrid.yaml")

# 先构造一个 PDE-only trainer 只为拿到网格 shape
cfg_probe = load_config(base, hybrid)
cfg_probe.medium.velocity_model = "smooth_lens"
probe = Trainer(cfg_probe, device="cpu")
shape = (probe.grid.ny_total, probe.grid.nx_total)

with tempfile.TemporaryDirectory() as d:
    ref_path = Path(d) / "reference_envelope.npy"
    ref = np.zeros(shape, dtype=np.complex128)
    ref[shape[0] // 2, shape[1] // 2] = 1.0 + 0.5j
    np.save(ref_path, ref)

    cfg = load_config(base, hybrid)
    cfg.medium.velocity_model = "smooth_lens"
    cfg.training.epochs = 2
    cfg.training.lambda_pde = 1.0
    cfg.training.lambda_data = 0.5
    cfg.training.supervision.enabled = True
    cfg.training.supervision.reference_path = str(ref_path)

    trainer = Trainer(cfg, device="cpu")
    losses = trainer.train(epochs=2)

    assert isinstance(losses, list)
    assert len(losses) == 2
    assert len(trainer.loss_history_pde) == 2
    assert len(trainer.loss_history_data) == 2
    assert all(np.isfinite(v) for v in trainer.loss_history_total)
    assert all(np.isfinite(v) for v in trainer.loss_history_pde)
    assert all(np.isfinite(v) for v in trainer.loss_history_data)

print("hybrid smoke ok")
PY
```

这个 smoke 的意义是：

- 验证标签加载通路；
- 验证三类 loss 历史都存在；
- 验证 `train()` 返回值仍是 list；
- 不依赖 A2 solver 的实时计算。

### 9.5 fail-fast smoke

至少再做两个失败路径检查：

1. `lambda_data > 0` 但 `supervision.enabled = false`
2. `reference_path` 指向不存在文件

这两类情况必须在训练开始前明确报错，而不是静默退回 PDE-only。

---

## 10. 与其他 agent 的协作规则

### 10.1 与 A2 的协作

A3 与 A2 的接口只有一个核心：

- `reference_envelope.npy`

A3 需要从 A2 那里确认并锁死的只有这些事实：

- 文件名就是 `reference_envelope.npy`
- 内容是 complex ndarray
- 形状是全网格 `[H, W]`
- 表示 envelope，不是 wavefield

A3 不需要向 A2 索取：

- `reference_metrics.json`
- `reference_comparison.csv`
- CLI 参数细节

A3 也不得要求 A2 为自己改 eval 逻辑；A3 只消费 A2 已经交付的文件契约。

### 10.2 与 A0 的协作

A3 必须明确告诉 A0：

- 这次没有改 `runner.py`
- `Trainer.train()` 返回值保持不变
- 新增了哪些 `Trainer` 实例属性可供后续集成
- 如果未来要把 `loss_pde` / `loss_data` 写进 `summary.json` 或 CSV，需要 A0 在集成阶段决定是否扩展 runner

A3 对 A0 的 handoff 应该包含：

- 新配置字段定义
- fail-fast 规则
- synthetic test 覆盖范围
- 是否新增了 `src/train/supervision.py`

### 10.3 与 A4 的协作

A3 不直接改文档，但必须把以下文案交给 A4：

- 新增 `training.supervision.enabled`
- 新增 `training.supervision.reference_path`
- `reference_path` 必须指向 `reference_envelope.npy`
- `lambda_data=0` 仍是默认路径
- 目前 CLI 不支持直接传 `reference_path`；要么改 YAML，要么程序内设置配置

### 10.4 与 A5 的协作

A5 不直接依赖 A3 才能工作，但 A3 完成后，A5 可以拿 hybrid 训练结果与 PDE-only 做对比。A3 只需要向 A5 说明：

- A3 没有改 residual 离散；
- A3 没有改模型结构；
- A3 唯一变化是训练目标中是否加入监督项。

这样 A5 后续做比较时，变量才清晰。

---

## 11. Handoff 模板

A3 完成后，必须同时给 A0 和 A4 交接。下面模板可直接填写。

### 11.1 给 A0 的 handoff

```md
[A3 -> A0 handoff]

1. 分支与基线
- branch: feat/hybrid-loss
- based on: <commit>
- requires Gate 3: yes/no

2. 本次改动文件
- src/train/trainer.py
- src/train/losses.py   # 若无实质改动请写“仅注释/无改动”
- src/train/supervision.py   # 若未新建请写“none”
- tests/test_train.py
- configs/experiments/hybrid.yaml

3. 新增/固定的配置字段
- training.supervision.enabled: bool
- training.supervision.reference_path: str | null
- lambda_data > 0 时必须 enabled=true 且 reference_path 有效

4. 新增的 Trainer 属性
- supervision_enabled
- reference_path
- reference_target
- loss_history_total
- loss_history_pde
- loss_history_data
- last_step_metrics

5. 向后兼容保证
- Trainer.train() 仍返回 list[float] total loss history
- runner.py 未修改
- tests/test_runner.py 无需改动

6. fail-fast 规则
- lambda_data > 0 且 supervision disabled -> ValueError
- missing reference_path -> FileNotFoundError / ValueError
- non-complex reference npy -> ValueError
- shape mismatch -> ValueError

7. 测试结果
- pytest -q tests/test_train.py : PASS/FAIL
- pytest -q tests/test_runner.py : PASS/FAIL
- pde-only smoke : PASS/FAIL
- hybrid smoke : PASS/FAIL

8. 建议集成项（未在本分支实现）
- 是否把 loss_pde / loss_data / loss_total 写入 summary.json
- 是否新增 loss component CSV / plots
- 是否需要 CLI 直接接收 reference_path
```

### 11.2 给 A4 的 handoff

```md
[A3 -> A4 handoff]

1. 新增配置字段
- training.supervision.enabled
- training.supervision.reference_path

2. 使用说明
- 默认 lambda_data=0 时仍是 PDE-only
- 只有 lambda_data>0 且 supervision.enabled=true 时才会读取 reference_envelope.npy
- reference_path 指向 A2 产出的 reference_envelope.npy
- 当前 CLI 没有单独的 --reference-path 参数

3. 模式说明
- PDE-only: lambda_pde>0, lambda_data=0
- data-only: lambda_pde=0, lambda_data>0
- hybrid: lambda_pde>0, lambda_data>0

4. 错误行为说明
- 配置非法时训练前直接报错，不会静默回退

5. 示例文件
- configs/experiments/hybrid.yaml
```

---

## 12. Blocked 条件与处理办法

### 12.1 Gate 3 未开

如果 A2 尚未合并、`reference_envelope.npy` 契约仍未冻结，A3 不应最终化标签读取语义。  
此时最多只做：

- supervision 配置解析脚手架
- 向后兼容逻辑
- synthetic label 单测脚手架

一旦涉及真实 A2 产物语义，就必须等待 Gate 3。

### 12.2 A0 要求 A3 直接改 runner

A3 的默认答复应是：

- 记录需求；
- 通过 handoff 交给 A0；
- 自己不改 `runner.py`。

只有 `WORKTREE_BOARD.md` 明确改了所有权，A3 才能越过这条边界。

### 12.3 A1 未提供多 overlay 支持

这**不是** A3 的 blocker。A3 的应对方式已经在本文件里固定：

- `configs/experiments/hybrid.yaml` 作为完整单 overlay 示例；
- 不依赖 `base + debug + hybrid` 三层叠加。

### 12.4 reference 文件路径无法通过 CLI 注入

这也**不是** A3 的 blocker。A3 当前分支的策略是：

- 通过 YAML 或程序内修改 `cfg.training.supervision.reference_path` 使用；
- CLI 注入能力属于未来可选集成项，不是 A3 本轮必做。

---

## 13. 完成定义（Definition of Done）

只有同时满足下面全部条件，A3 才算完成：

1. `Trainer.train()` 已真正把 `loss_data_val` 传入 `loss_total()`；
2. 默认旧配置（无 supervision 字段、`lambda_data=0`）训练行为不变；
3. `lambda_data > 0` 且配置正确时，监督项确实参与 total loss；
4. data-only 与 hybrid 两种路径至少有一条单测覆盖；
5. 配置/路径/shape/dtype 错误会在训练开始前 fail fast；
6. `Trainer.train()` 返回值仍是 list[float]；
7. `tests/test_train.py` 通过；
8. `tests/test_runner.py` 不需要改动且能通过；
9. `configs/experiments/hybrid.yaml` 已入库且不含任何机器本地路径；
10. 已向 A0 / A4 提交完整 handoff。

只做到其中几项，例如“能算 loss_data 但返回值契约变了”或“hybrid 能跑但默认路径被破坏了”，都不算完成。

---

## 14. 本轮明确不做

A3 本轮明确不做以下内容：

- 不改 `src/train/runner.py`
- 不新增 `summary.json` 的 loss 组件字段
- 不新增 loss component CSV / PNG 产物
- 不新增 CLI 参数 `--reference-path`
- 不改 `src/eval/*`
- 不调用或复制 reference solver
- 不改 residual / PML / background / eikonal / tau 离散
- 不改 NSNO 模型结构
- 不引入 dataset / dataloader / batch training 体系
- 不设计 amplitude/phase 专用监督损失
- 不改 `configs/base.yaml`
- 不改顶层文档

A3 的成功标准是：**在最小边界内，把现有 data loss 真正接通，并且不破坏当前训练与运行契约。**

---

## 附录 A. 源码核查与补充说明

本附录记录将本手册落盘前对仓库代码所做的关键交叉核查，目的是让 A3 在真正开工时能直接对准已有接口，不必重新摸排。附录不改变正文中的职责边界。

### A.1 `Trainer.train()` 的“line 166 缺陷”在代码中已确认

核查文件：`src/train/trainer.py`。

当前训练主循环的 total 组装逻辑（相关行号随仓库演化可能小幅偏移，但结构稳定）：

```python
result = self.residual_computer.compute(a_scat)
total = loss_total(
    result['loss_pde'],
    lambda_pde=lambda_pde,
    lambda_data=lambda_data,
)
```

此处**未向 `loss_total()` 传入 `loss_data_val`**。这正是本文件 §2.2 所指的缺陷。A3 在接通监督项时的最小改动就是：

- 当 supervision 激活时，在此处计算 `loss_data_val = loss_data(a_scat, self.reference_target)`；
- 调用改为 `loss_total(loss_pde, loss_data_val, lambda_pde=..., lambda_data=...)`；
- 当 supervision 未激活时，仍传 `None`（`loss_total` 内部会按 `loss_data_val is None` 处理）。

### A.2 `Trainer.train()` 的返回值事实

核查事实：当前 `train(self, epochs=None)` 最后返回 `losses`，类型为 Python `list[float]`。`runner.py` 中 `save_losses(losses, output_dir)` 直接据此生成 `losses.npy`、`losses.csv`、`loss_tail.csv`。因此本文件 §2.6 与 §7.2.2 的硬规则完全与代码一致：

> **`Trainer.train()` 必须继续返回 `list[float]`，不得改为 dict / tuple / ndarray。**

A3 在添加三类 loss 历史时，应把它们以 `Trainer` 实例属性形式保存（`self.loss_history_pde` / `self.loss_history_data` / `self.loss_history_total`），让 `train()` 仍 `return self.loss_history_total`（等价旧 `losses`）。

### A.3 `torch.manual_seed(0)` 与 supervision 加载的顺序问题

核查事实：`Trainer.__init__()` 中在构造 `NSNO2D` 之前执行了：

```python
rng_state = torch.get_rng_state()
torch.manual_seed(0)
self.model = NSNO2D(...).to(self.device)
torch.set_rng_state(rng_state)
```

这段代码通过“临时固定种子 + 恢复原 RNG 状态”实现 **模型参数初始化的跨配置确定性**。A3 在 `__init__` 中接入 supervision 时必须遵守下面两条硬约束，才能不破坏这一确定性：

1. Reference 标签加载（`np.load(...)`、`torch.from_numpy(...)`）**不要使用任何随机性 API**（`torch.randn`、`numpy.random.*`、`torch.randint` 等）。A3 的标签转换公式只是“实/虚分通道 + cast + to(device)”，本来就不需要 RNG，但若未来想加扰动请另走一条独立路径。
2. Supervision 相关代码**必须放在 `self.model = NSNO2D(...)` 构造之后**执行，否则可能在 `torch.manual_seed(0)` 的作用窗口内意外消耗 RNG。最安全的放法是放在 `__init__` 末尾（`self.residual_computer` 构造之后）。

这两条规则在本文件正文中未显式写出，但直接来自 `trainer.py` 的现有实现事实，A3 在改 `__init__` 时必须守住。

### A.4 `Config` 访问语义现状

本文件 §2.4 已指出 `Config` 没有 `.get()`。此处补充当前实现允许的安全访问方式，供 A3 §6.4 的兼容写法参照：

```python
training_cfg = cfg.training

# 优先：检查结构是否存在
if "supervision" in training_cfg:
    sup = training_cfg.supervision
    enabled = getattr(sup, "enabled", False)
    reference_path = getattr(sup, "reference_path", None)
else:
    enabled = False
    reference_path = None
```

两点提醒：

- `in training_cfg` 走的是 `Config.__contains__`，当前实现会正常返回 bool。
- `getattr(sup, "enabled", False)` 是合法的，因为 `Config` 支持属性访问；但 **`sup.get("enabled", False)` 会抛 AttributeError**，不要写。

### A.5 `configs/experiments/hybrid.yaml` 必须是单 overlay 完整配置

核查事实：`load_config(base_path, overlay_path=None)` 只支持一个 overlay（见 A1 附录 A.3）。因此本文件 §6 第 6 步的“完整 debug-sized overlay + supervision schema”要求不是美学选择，而是硬约束。

A3 在写这个 yaml 时要同时满足：

1. 直接覆盖 debug 规模（`grid.nx/ny = 32`、`pml.width = 5`、`training.epochs = 50` 等），使 `base + hybrid` 就能跑通 smoke；
2. `supervision.enabled: false`、`supervision.reference_path: null`，保证仓库入库版本不包含本地机器路径；
3. 不要在该文件里重复 `base.yaml` 中已有且不需要覆盖的字段，防止未来 base 字段更新时产生影子。

### A.6 `tests/test_runner.py` 的锁定范围（A3 守门用）

本文件 §2.6 提到该测试会检查 `evaluate_saved_run()` 的输出契约。核查事实：

- 至少锁定 `summary["output_dir"]` 和 `summary["residual_max_evaluation"] >= 0.0` 两个键；
- 未锁定任何 `loss_pde` / `loss_data` / `loss_total` 分量字段。

因此 A3 即使临时把三类 loss 历史保存到 `Trainer` 实例属性，只要 `train()` 返回值类型不变、`summary` 已有键不改，`tests/test_runner.py` 不应回归。A3 的 §9.2 把这条回归守门命令单独列出，是必要的。

### A.7 附录的职责

附录只是事实对账，目的是让 A3：

- 明确“line 166 修法”直接针对现有实现；
- 明确 `torch.manual_seed(0)` 与 supervision 加载顺序的硬约束；
- 明确 `Config` 访问语义不可退化到 dict-like；
- 明确 hybrid overlay 单层的真实原因。

附录**不新增** A3 职责，也**不放宽** §4、§7、§14 中的禁令。如仓库在 A3 开工时与本附录描述已经不一致，应以当时 `int/multi-agent` 内的实际代码为准。
