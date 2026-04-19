# A6 Experiment Matrix

This directory tracks the committed A6 matrix specification and report schema.

Tracked files:
- `README.md`: scope, runtime layout, and blocked axes.

Runtime-generated artifacts are written under `outputs/matrix/` so they stay out of git:
- `outputs/matrix/<batch>/<run-id>/...`: per-run artifacts from existing training/reference scripts
- `outputs/matrix/_meta/manifest.tsv`: launch manifest emitted by `scripts/run_benchmark.py`
- `outputs/matrix/_meta/failures.csv`: failure ledger
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
- `scripts/launch_matrix.sh`
- `scripts/run_benchmark.py`
- `scripts/summarize_matrix.py`

Recommended usage:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

./scripts/launch_matrix.sh --all-batches --dry-run
./scripts/launch_matrix.sh configs/experiments/B0_smoke.yaml --device cpu
python scripts/summarize_matrix.py --output-root outputs/matrix
```
