# A6 Experiment Matrix

This directory documents the committed A6 matrix specification and report layout
in the integrated main worktree.

Tracked specs live under `configs/experiments/B*.yaml`.

Runtime-generated artifacts are written under `outputs/matrix/` by default:
- `outputs/matrix/<batch>/<run-id>/...`: per-run artifacts from existing training/reference scripts
- `outputs/matrix/_meta/manifest.tsv`: launch manifest emitted by `scripts/run_matrix.py`
- `outputs/matrix/_meta/failures.jsonl`: failure ledger from the launcher
- `outputs/matrix/_reports/matrix_report.csv`: consolidated run table
- `outputs/matrix/_reports/matrix_report.md`: markdown summary
- `outputs/matrix/_reports/matrix_figures/`: generated plots

Current A6 rules:
- Do not modify `src/**`, `tests/**`, or existing runner/reference scripts to add new axes.
- Per-run configs are materialized from `configs/base.yaml` plus one batch file and one run override.
- Summary scripts only read persisted artifacts. Missing fields stay blank instead of being recomputed from hidden state.
- `loss_pde` / `loss_data` are currently available as final-step decompositions in the report layer; full per-epoch histories remain blocked until A7 extends upstream loss persistence beyond the legacy two-column `losses.csv`.
- Reference-label caches are keyed by a stable hash of the fully materialized reference config so changed label-generation overrides do not silently reuse stale outputs.

Blocked axes in the current integrated baseline:
- `training.seed`: not exposed in `configs/base.yaml` or CLI, so seed sweeps stay blocked.
- `residual.lap_tau_mode`: implemented inside `ResidualComputer` but not wired from config into training.
- `loss.omega2_normalize` and `loss.pml_rhs_zero`: present in `base.yaml` but not consumed by the current residual path.
- `eikonal.precision`: present in `base.yaml` but not consumed by the current eikonal path.

Primary entrypoints:
- `scripts/run_matrix.py`: local full-matrix launcher using `configs/experiments/B*.yaml`
- `scripts/run_matrix_sx.cmd`: this machine's `D:\anaconda3\envs\SX\python.exe` wrapper
- `scripts/summarize_matrix.py`: report generator
- `scripts/summarize_matrix_sx.cmd`: this machine's SX wrapper for the report generator
- `scripts/run_benchmark.py`: smaller A6 subset runner kept for quick probes only

Recommended usage on this machine:

```bat
scripts\run_matrix_sx.cmd --dry-run --include-blocked
scripts\run_matrix_sx.cmd --device cuda:0 --epochs-override 5000 --continue-on-error
scripts\summarize_matrix_sx.cmd --output-root outputs\matrix
```
