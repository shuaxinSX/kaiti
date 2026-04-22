# 多 agent 并行开发归档 — 2026-04

本目录保存 2026-04 那一轮 `int/multi-agent` 并行开发的**历史**文档，**不再更新**。

## 背景

基于 `feat/m0-bootstrap` 基线（M0-M9 核心管线完成），通过 `git worktree` 拉出 5 个隔离工作区并行推进：

| Agent | 分支 | 内容 | 合并 commit |
|-------|------|------|-------------|
| A1 | `feat/packaging-ci` | Packaging / 最小 CI | `9b488aa` / `5904d04` |
| A2 | `feat/reference-eval` | 参考解求解器 + 评估 | `6f5ee94` |
| A3 | `feat/hybrid-loss` | 混合损失训练 | `9d08776` / `b80dc88` |
| A5 | `fix/residual-pml-audit` | Residual / PML 审计 | `8bd4162` / `03c409a` |
| A6 | `exp/benchmark-matrix-sync` | B0-B15 实验矩阵 | `700afb6` |

A4 (docs) 初始规划为 TODO，实际由根目录 `README.md` / `CHANGELOG.md` 统一在本轮收尾时更新。

## 内容

- `WORKTREE_BOARD.md` — 并行开发看板（含状态区 §20、热点文件清单、收尾规则）
- `A1_PACKAGING_CI.md` … `A5_RESIDUAL_PML_AUDIT.md` — 每个 agent 的专用执行手册
- `handoffs/` — agent 之间的交接通信快照

## 状态节（§20）在归档时的最终快照

- A0 Integrator：`INTEGRATING`（已完成）
- A1：`MERGED`（2026-04-18）
- A2：`MERGED`（2026-04-18）
- A3：`MERGED`（2026-04-19）
- A4：初始 `TODO` — 由收尾期根目录文档重写覆盖，不单独合并
- A5：`MERGED`（2026-04-19）
- A6：初始 `READY` — 后由 `exp/benchmark-matrix-sync` 实际完成并合入（2026-04-22）

变更记录见根目录 [`CHANGELOG.md`](../../../CHANGELOG.md)。
