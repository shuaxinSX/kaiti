# 高波数 2D Helmholtz 神经算子实验矩阵总结报告

- 数据来源：`/Users/ccf/Desktop/开题汇总/开题X/kaiti-highk-results-ep10000`
- 汇总位置：`/Users/ccf/Desktop/开题汇总/开题/reports/highk`
- 研究主线：这批实验服务于**高波数 2D Helmholtz 方程求解**，核心关注相位剥离、NSNO 表达能力、PML residual 一致性、reference floor、监督信号与高频稳定性。
- 统计口径：1413 planned active rows；1412 completed summaries；1207 completed train runs；205 completed reference-only runs；最终缺失 1 个 run。
- 统一训练覆盖：manifest `epochs_override=10000`。因此 C12 的 `ep01000/ep40000` 等标签不能当成真实 epoch-budget 轴。
- Provenance：manifest 记录 `git_sha=nogit`、`spec_branch=local-configs`，弱于旧 A6 的 committed-SHA 结果；本报告按落盘配置和 summary 解释。

---

## 1. 实验目标

这次矩阵不是普通调参，而是在回答：当前 `WKBJ/Eikonal phase stripping + scattering envelope + NSNO + strict PML residual` 路线，是否能向高波数 Helmholtz 问题扩展。

判读时必须同时看四件事：

1. reference solver 自身是否可信，即 reference residual floor 是否足够低。
2. 神经解是否满足方程，即 `residual_rmse_evaluation`。
3. 神经解是否接近 reference，即 `rel_l2_to_reference`、`amp_mae_to_reference`、`phase_mae_to_reference`。
4. PML 和源点位置是否制造了指标偏置，尤其是 interface / PML residual、near-boundary source、thick-PML proxy。

核心结论先放前面：**这批实验已经覆盖高波数主线，能证明 reference floor 可信、strict PML residual 整体优于 legacy，并且 hybrid/data supervision 能显著改善相位和 reference error；但 PDE residual 与波场相位之间仍存在明显冲突，curriculum 和 PML loss weighting 还没有形成稳定收益，不能把 residual 最低的 run 直接写成“高波数已解决”。**

---

## 2. 完整性与批次构成

| batch | runs | train | ref-only | completed | missing | median train rmse | median rel_l2 | median phase | main role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 108 | 0 | 108 | 108 | 0 |  |  |  | reference gate |
| C1 | 144 | 144 | 0 | 144 | 0 | 1.034e-03 | 0.2320 | 0.3944 | high-k scaling |
| C2 | 28 | 28 | 0 | 28 | 0 | 2.739e-03 | 0.1371 | 0.1682 | ppw law |
| C3 | 114 | 24 | 90 | 114 | 0 | 6.429e-04 | 0.6230 | 0.7512 | grid/pollution budget |
| C4 | 100 | 100 | 0 | 100 | 0 | 7.779e-04 | 0.4577 | 0.6784 | PML stability |
| C5 | 108 | 108 | 0 | 107 | 1 | 4.791e-04 | 0.5118 | 0.6578 | strict lapTau audit |
| C6 | 54 | 54 | 0 | 54 | 0 | 4.583e-04 | 0.4389 | 0.5400 | medium complexity |
| C7 | 36 | 36 | 0 | 36 | 0 | 2.280e-03 | 0.0621 | 0.0630 | source robustness |
| C8 | 48 | 48 | 0 | 48 | 0 | 7.683e-04 | 0.4687 | 0.6460 | NSNO depth |
| C9 | 28 | 28 | 0 | 28 | 0 | 7.600e-04 | 0.4449 | 0.7272 | FNO modes |
| C10 | 24 | 24 | 0 | 24 | 0 | 8.686e-04 | 0.4679 | 0.7345 | channels |
| C11 | 42 | 42 | 0 | 42 | 0 | 1.583e-03 | 0.1465 | 0.1776 | supervision/loss |
| C12 | 72 | 72 | 0 | 72 | 0 | 3.574e-03 | 0.1342 | 0.1826 | optimization labels/lr |
| C13 | 28 | 28 | 0 | 28 | 0 | 3.905e-03 | 0.1477 | 0.1993 | curriculum |
| C14 | 36 | 36 | 0 | 36 | 0 | 1.495e-03 | 0.1484 | 0.2239 | phase pathology |
| C15 | 31 | 24 | 7 | 31 | 0 | 1.529e-03 | 0.3018 | 0.4657 | benchmark cards |
| C16 | 86 | 86 | 0 | 86 | 0 | 3.221e-03 | 0.0924 | 0.1384 | neural PML window |
| C17 | 72 | 72 | 0 | 72 | 0 | 3.997e-03 | 0.1492 | 0.1861 | PML interface consistency |
| C18 | 84 | 84 | 0 | 84 | 0 | 4.463e-04 | 0.4614 | 0.6755 | aux-field PML audit |
| C19 | 80 | 80 | 0 | 80 | 0 | 4.782e-03 | 0.1294 | 0.1666 | PML loss weighting |
| C20 | 72 | 72 | 0 | 72 | 0 | 3.570e-03 | 0.1019 | 0.1587 | PML antidote architectures |
| C21 | 18 | 18 | 0 | 18 | 0 | 3.538e-03 | 0.1089 | 0.1592 | radiation baseline proxy |

最终缺失 run：

- `C5_laptau_strictness/layered__omega180__N459__pml24__center__stretched_divergence`

`_meta/failures.jsonl` 有 481 条记录，但这是恢复/重试过程的 ledger。最终完整性应以 `summary.json` 为准：1413 行中 1412 行完成。

---

## 3. Reference Floor

C0 reference gate 共 108 个 reference-only runs，覆盖 `homogeneous/smooth_lens/layered × omega 30/60/90/120 × N 64/128/192 × pml 8/16/24`。

关键事实：

- homogeneous reference residual 全部为 0。
- smooth_lens / layered 在 C0 中最大 `reference_residual_rmse` 为 `2.816e-07`。
- C3/C15 额外 reference-only 结果把更高频/更大 N 的 reference floor 扩展到约 `1e-6` 以内；C15 的 omega300 reference-only cardD 为 `1.379e-08`。
- 因此本矩阵的 reference floor 可信，神经解 `1e-3` 到 `1e-2` 的 residual 不是 reference solver 噪声造成的。

C0 最差 reference floor 仍然很低，集中在 omega30、N192 的 smooth/layered：`2.6e-07` 到 `2.8e-07`。

---

## 4. 高频 Scaling 与 Phase-Residual 冲突

C1 是高波数 scaling 主批次，覆盖 omega 30/60/90/120/180/240/300、ppw、base/deep、PDE-only 和 hybrid `lambda_data=10`。

主要结论：

- PDE-only 通常给最低 residual，例如 smooth_lens omega120 PDE residual 在 `4e-4` 量级，但 rel/phase 常在 `0.6-0.8` 量级。
- hybrid `lambda_data=10` 显著改善 rel/phase，例如 smooth_lens omega120 hybrid rel 约 `0.10-0.12`、phase 约 `0.11-0.14`，但 residual 升到 `3e-3` 到 `4e-3`。
- omega300 已有训练结果：smooth_lens hybrid residual 约 `1.0e-3`，rel 约 `0.16`；layered hybrid residual 约 `2.5e-3`，rel 约 `0.21`。这说明 pipeline 能跑到高波数，但相位/残差 tradeoff 仍明显。

这直接支撑报告主线：高波数 Helmholtz 的关键不是单纯把 residual 降低，而是同时控制 phase error。

---

## 5. ppw 与网格污染

C2/C3 回答 fixed ppw 和 fixed N 的问题。

C2 显示 ppw 增大并不单调改善结果：

- omega90 smooth_lens 在 ppw8/12 附近 rel/phase 较好，ppw24 residual 反而升高。
- omega180 smooth_lens 在 ppw8-16 之间 rel/phase 差别不大，残差没有随 ppw 增大单调下降。
- omega300 smooth_lens 在 ppw6/8/10 上 residual 均约 `1.2e-3`，layered 约 `2.6e-3` 到 `2.8e-3`。

解释：固定 10000 epochs 和固定容量下，大 N 并不等于更好训练。ppw 是必要数值尺度，但本矩阵表明高波数误差也受优化预算、capacity、PML 和 supervision 强烈控制。

C3 的 fixed-N train subset 进一步说明：PDE residual 可以维持在 `1e-4` 到 `1e-3`，但 rel/phase 仍可能很高。因此 fixed-N 高频 residual 不能单独作为“高波数求解成功”的证据。

---

## 6. Strict PML Residual

C5/C17/C18 是本矩阵对 `stretched_divergence` strict PML residual 的核心证据。

C5 paired strict vs legacy：

- residual：35 paired comparisons，strict 更好 30/35，median strict/legacy ratio = 0.8123
- rel_l2：strict 更好 33/35，median ratio = 0.7317
- phase：strict 更好 34/35，median ratio = 0.7849

C17 PML interface consistency gives the stronger high-k interface evidence:

- residual：36 pairs，strict 更好 33/36，median ratio = 0.7013
- interface residual：strict 更好 32/36，median ratio = 0.7416
- PML residual：strict 更好 29/36，median ratio = 0.7678

C18 更细：strict 对 rel/phase 多数更好，但个别 smooth/high-frequency width 组合下 legacy residual 更低。因此最终写法应是：**strict 是当前主方法默认，有 paired evidence 支撑；但 PML window 仍需结合 interface 和 phase 指标，不能只按一个 residual 排序。**

---

## 7. 模型容量

C6/C8/C9/C10 拆开了介质复杂度、NSNO depth、FNO modes 和 channels。

结论：容量不是简单单调轴。

- homogeneous 在 C6 中全部为 0，说明链路 sanity 通过。
- smooth_lens/layered 中，depth 增大不总是改善；C8 的 best residual 和 best rel_l2 常落在不同 depth。
- C9 的 FNO modes 也不是越大越好；例如 smooth_lens omega180 best residual 在 modes=32，但 best rel_l2 在 modes=4。
- C10 的 channels 同样分裂：smooth_lens omega120 best residual 在 channels=128，但 best rel_l2 在 channels=16。

这说明当前主要瓶颈不是“把网络无脑加大”，而是 phase/residual 目标冲突、PML 一致性和监督策略。

---

## 8. Supervision 与 Phase Pathology

C11/C14 是解释 phase 问题的关键。

C11 监督矩阵显示：

- PDE-only：residual 最低，但 rel/phase 差。例如 smooth_lens omega120 PDE-only residual `4.937e-04`，rel `0.7329`，phase `0.6923`。
- data-only：rel/phase 很好，但 residual 大幅变差。例如 smooth_lens omega120 data-only residual `1.44e-02`，rel `0.0719`，phase `0.0790`。
- hybrid：位于两者之间。`lambda_data=10` 对 low/mid frequency phase 很有效，但在 omega120/180 上常把 residual 推到 `1e-3` 到 `4e-3`。

C14 phase pathology 复现同一机制：base/deep 不是核心差别，监督类型才是核心差别。PDE-only 的相位误差常在 `0.5-0.8`，data/hybrid 能降到 `0.01-0.18`，但 residual 代价明显。

因此报告应把“相位对齐”列为下一阶段高波数求解的第一瓶颈，而不是把模型容量列为第一瓶颈。

---

## 9. Source Robustness

C7 覆盖中心、偏移、近边界源点。

注意：source 位置会明显影响指标，尤其 smooth_lens 中 near-corner/near-left 往往 residual 和 rel/phase 都更低。例如 smooth_lens omega120 near_corner residual `5.451e-04`、rel `0.0182`、phase `0.0307`，明显优于 center 的 residual `4.314e-03`、rel `0.0918`、phase `0.0993`。

这类结果不能简单解释成“近边界物理问题更容易被求解”，因为 PML/能量吸收和评估域分布会改变指标。源点鲁棒性结论必须用 paired source-position comparison 表达。

---

## 10. Optimization 与 Curriculum

C12 名义上是 optimization budget，但全局 `epochs_override=10000` 覆盖了所有 epoch 标签：

- C12 generated configs 的 effective epochs 唯一值：[10000]
- 每个 lr 下 `ep01000/ep03000/ep05000/ep10000/ep20000/ep40000` 结果完全重复。

所以 C12 只能作为 lr sweep 解读，不能作为 epoch-budget 结论。lr 层面上，`3e-4` 对 omega120 通常较好；omega180 smooth_lens 中 `1e-3` residual 更低但 rel/phase略差，说明 lr 也存在 residual/reference tradeoff。

C13 curriculum 经过 recovery 后最终完成，但收益不稳定：

- smooth_lens omega120 best residual 是 `no_restart`，best rel 是 `octave_2`。
- smooth_lens omega180 best residual 是 `no_restart`，best rel 是 `direct`。
- layered omega180 best residual 是 `dense`，best rel 是 `octave_3`。

因此 curriculum 目前不能写成稳定优于 direct training，只能写成“已修复可运行，但尚未形成稳健收益”。

---

## 11. PML Window、Loss Weighting 与 Antidote 架构

C16/C21 表明 PML width/R0/power 的影响存在，但没有简单单调规律：

- 加厚 PML 不总是改善 rel/phase；C21 中 thick PML w48/w64 常使 rel/phase 比 w24 更差。
- C16 中 width/R0/power 的 best residual 和 best rel_l2 经常不一致。

C19 PML loss weighting 没有显示稳定收益：

- best residual 往往在小 `lambda_pml` 和小 `pml_fraction`。
- 大 `lambda_pml=100` 有时降低 rel_l2，但 residual 明显变坏，不能作为默认策略。

C20 架构/antidote 试验发现明确异常：

- 3 个 quadratic curriculum run 产生 NaN。
- 1 个 layered omega180 `R0=1e-4` quadratic curriculum residual 爆到 `2.211e+11`。
- quadratic pointwise 在 omega90/120 有时改善 rel/phase，但 high-k curriculum 组合不稳定。

这说明 PML 问题不是靠简单加权或 quadratic curriculum 就能解决；需要更受控的 architecture audit。

---

## 12. Global Best / Worst

Best non-homogeneous residual runs:

- C5 `smooth_lens__omega030__N077__pml08__near_boundary__stretched_divergence`: omega=30.0000, N=77, medium=smooth_lens, rmse=1.923e-04, rel=0.5704, phase=0.7339
- C5 `smooth_lens__omega030__N077__pml16__near_boundary__stretched_divergence`: omega=30.0000, N=77, medium=smooth_lens, rmse=2.223e-04, rel=0.1610, phase=0.3707
- C1 `smooth_lens__omega030__ppw08__N039__base__pde`: omega=30.0000, N=39, medium=smooth_lens, rmse=3.008e-04, rel=2.2744, phase=1.1407
- C6 `smooth_lens__omega180__ppw12__N344__L12`: omega=180.0000, N=344, medium=smooth_lens, rmse=3.056e-04, rel=0.7966, phase=0.7447
- C18 `smooth_lens__omega240__ppw12__N459__w24__mixed_legacy`: omega=240.0000, N=459, medium=smooth_lens, rmse=3.076e-04, rel=0.4602, phase=0.5157
- C5 `smooth_lens__omega030__N077__pml08__near_boundary__mixed_legacy`: omega=30.0000, N=77, medium=smooth_lens, rmse=3.097e-04, rel=1.1104, phase=0.9632
- C3 `smooth_lens__omega060__N128__train_subset`: omega=60.0000, N=128, medium=smooth_lens, rmse=3.100e-04, rel=0.7213, phase=0.5535
- C5 `smooth_lens__omega030__N077__pml16__center__stretched_divergence`: omega=30.0000, N=77, medium=smooth_lens, rmse=3.108e-04, rel=1.7016, phase=1.0420
- C18 `smooth_lens__omega180__ppw12__N344__w16__mixed_legacy`: omega=180.0000, N=344, medium=smooth_lens, rmse=3.116e-04, rel=0.8896, phase=0.8419
- C18 `smooth_lens__omega180__ppw12__N344__w24__stretched_divergence`: omega=180.0000, N=344, medium=smooth_lens, rmse=3.128e-04, rel=0.8221, phase=0.7513

Best non-homogeneous rel_l2 runs:

- C1 `smooth_lens__omega030__ppw16__N077__deep__hybrid1x10`: omega=30.0000, N=77, medium=smooth_lens, rmse=6.530e-04, rel=1.209e-03, phase=1.544e-03
- C1 `smooth_lens__omega030__ppw08__N039__deep__hybrid1x10`: omega=30.0000, N=39, medium=smooth_lens, rmse=5.667e-04, rel=1.757e-03, phase=2.458e-03
- C1 `layered__omega030__ppw16__N077__base__hybrid1x10`: omega=30.0000, N=77, medium=layered, rmse=1.062e-03, rel=2.438e-03, phase=2.093e-03
- C1 `layered__omega030__ppw08__N039__base__hybrid1x10`: omega=30.0000, N=39, medium=layered, rmse=6.506e-04, rel=2.619e-03, phase=3.306e-03
- C1 `smooth_lens__omega030__ppw12__N058__deep__hybrid1x10`: omega=30.0000, N=58, medium=smooth_lens, rmse=7.447e-04, rel=2.810e-03, phase=2.772e-03
- C1 `layered__omega030__ppw16__N077__deep__hybrid1x10`: omega=30.0000, N=77, medium=layered, rmse=1.135e-03, rel=2.839e-03, phase=2.916e-03
- C1 `smooth_lens__omega030__ppw16__N077__base__hybrid1x10`: omega=30.0000, N=77, medium=smooth_lens, rmse=7.452e-04, rel=2.990e-03, phase=3.600e-03
- C1 `smooth_lens__omega030__ppw08__N039__base__hybrid1x10`: omega=30.0000, N=39, medium=smooth_lens, rmse=6.705e-04, rel=3.069e-03, phase=3.372e-03
- C15 `cardA__best_hybrid_proxy`: omega=30.0000, N=256, medium=smooth_lens, rmse=1.551e-03, rel=4.165e-03, phase=4.294e-03
- C11 `smooth_lens__omega060__N153__hybrid_data10`: omega=60.0000, N=153, medium=smooth_lens, rmse=1.043e-03, rel=4.316e-03, phase=5.024e-03

Worst finite residual runs:

- C20 `layered__omega180__N459__R0_0p0001__quadratic_curriculum`: omega=180.0000, N=459, medium=layered, rmse=2.211e+11, rel=2.649e+13, phase=1.8635
- C20 `smooth_lens__omega180__N459__R0_1em08__quadratic_curriculum`: omega=180.0000, N=459, medium=smooth_lens, rmse=0.0528, rel=1.1798, phase=1.0241
- C20 `smooth_lens__omega180__N459__R0_1em06__quadratic_curriculum`: omega=180.0000, N=459, medium=smooth_lens, rmse=0.0378, rel=0.7951, phase=0.7289
- C20 `layered__omega120__N306__R0_0p0001__quadratic_curriculum`: omega=120.0000, N=306, medium=layered, rmse=0.0286, rel=0.8257, phase=1.0363
- C20 `smooth_lens__omega180__N459__R0_0p0001__quadratic_curriculum`: omega=180.0000, N=459, medium=smooth_lens, rmse=0.0164, rel=0.3482, phase=0.4093
- C11 `smooth_lens__omega120__N306__data_only`: omega=120.0000, N=306, medium=smooth_lens, rmse=0.0144, rel=0.0719, phase=0.0790
- C14 `smooth_lens__omega120__N306__data_only__deep`: omega=120.0000, N=306, medium=smooth_lens, rmse=0.0144, rel=0.0719, phase=0.0790
- C14 `smooth_lens__omega120__N306__data_only__base`: omega=120.0000, N=306, medium=smooth_lens, rmse=0.0142, rel=0.0841, phase=0.0936

Non-finite runs:

- C20 `layered__omega120__N306__R0_1em08__quadratic_curriculum`
- C20 `layered__omega180__N459__R0_1em06__quadratic_curriculum`
- C20 `layered__omega180__N459__R0_1em08__quadratic_curriculum`

Large residual > 0.02:

- C20 `layered__omega120__N306__R0_0p0001__quadratic_curriculum`: rmse=0.0286, rel=0.8257, phase=1.0363
- C20 `smooth_lens__omega180__N459__R0_1em06__quadratic_curriculum`: rmse=0.0378, rel=0.7951, phase=0.7289
- C20 `smooth_lens__omega180__N459__R0_1em08__quadratic_curriculum`: rmse=0.0528, rel=1.1798, phase=1.0241
- C20 `layered__omega180__N459__R0_0p0001__quadratic_curriculum`: rmse=2.211e+11, rel=2.649e+13, phase=1.8635

---

## 13. 与旧 A6 矩阵的关系

旧 A6 (`开题2/outputs/matrix_from_branch_ep5000_full`) 是低/中频 baseline：102 runs、0 final failure、commit `03c409a`、epochs=5000。它证明了旧 pipeline 的可运行性、reference floor、lr/source/PML 初始 baseline。

本次 high-k ep10000 矩阵是更新、更大、更接近论文主线的 campaign：

- 覆盖 C0-C21，共 1413 planned rows。
- 频率扩展到 omega300。
- 包含 strict/legacy paired PML 证据。
- 包含 ppw、PML window、PML interface、PML loss weighting、curriculum、architecture antidote 等高波数机制轴。
- 不是 0 失败矩阵：最终 1 个缺失 run，且 recovery 过程中有大量 transient failure ledger。

后续写正式开题/论文时，A6 应作为 baseline 和方法演进背景；本次 high-k 矩阵才是高波数主结果来源。

---

## 14. 不能重跑/需要避免误读的实验

已完成，不应默认重跑：

- C0-C4、C6-C21 均有完整 final summaries。
- C5 只缺 1 个 strict run；其余 paired comparison 已足够支撑 strict 的总体结论。
- C12 已经跑完，但 epoch 轴被覆盖，不能用来证明 1000/3000/40000 epoch 差异。
- C13 curriculum 已完成，不能说“curriculum 没跑”；只能说收益不稳定。
- C20 已经暴露 quadratic curriculum 数值异常，不应把这些异常当成缺失结果。

需要后续若要补的，不是“重跑整个矩阵”，而是很窄的 follow-up：

1. 如果要完整 C5 paired table，可以只补缺失的 `layered__omega180__N459__pml24__center__stretched_divergence`。
2. 如果要真实 optimization budget，需要去掉全局 `epochs_override=10000` 后重做一个小 C12，不要重跑全矩阵。
3. 如果要论文级 curriculum 结论，需要设计新的 controlled curriculum batch，而不是继续解释当前 C13 为正结果。
4. 如果要架构 antidote，需要先隔离 C20 quadratic curriculum 的 NaN/爆炸机制。

---

## 15. 当前结论

本次实验矩阵已经足够支撑一个阶段性判断：当前相位剥离 + NSNO + strict PML residual 路线能扩展到 high-k benchmark，并且 reference floor 与 strict PML 证据扎实；但高波数 Helmholtz 的主要瓶颈已经从“能否跑通”转移到“如何同时满足 PDE residual 和 reference phase/rel_l2”。

最稳的叙事不是“某个模型已经解决高波数 Helmholtz”，而是：

- reference floor 可信；
- strict PML residual 是必要默认；
- PDE-only 残差好但相位错；
- data/hybrid 相位好但 residual 变差；
- 容量增大不是单调解法；
- PML loss weighting 和 curriculum 当前没有稳定收益；
- 下一步应围绕 phase-residual coupling、strict PML/interface metrics、受控 supervision schedule 做小矩阵，而不是重跑整个 C0-C21。
