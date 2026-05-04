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
- Hybrid data supervision must consume the offline phase-stripped `reference_envelope.npy` artifact with `training.supervision.target_kind=scattering_envelope`; full-wavefield artifacts are not valid training labels.
- Run summaries now include A3 reconstruction sensitivity budgets: `phase_tau_error_budget_h2` for smooth O(h^2) travel-time error and `phase_tau_error_budget_h` for order-loss O(h) regions, plus wavefield-scaled budget estimates.
- Run summaries also include A4 capacity diagnostics: `full_wave_nyquist_mode_floor`, `fno_mode_to_full_wave_nyquist`, `scattering_strength_proxy`, `neumann_proxy_convergent`, and `neumann_depth_tail_proxy`.

Runtime support status after the strict PML residual wiring:
- `residual.lap_tau_mode`: now read from config by `Trainer`; default is `stretched_divergence`, with `mixed_legacy` retained for explicit audit runs.

Still-blocked axes in the current integrated baseline:
- `training.seed`: not exposed in `configs/base.yaml` or CLI, so seed sweeps stay blocked.
- `loss.omega2_normalize` and `loss.pml_rhs_zero`: present in `base.yaml` but not consumed by the current residual path.
- `eikonal.precision`: present in `base.yaml` but not consumed by the current eikonal path.

Note: older A6 reports and the historical B9 batch file may still describe
`residual.lap_tau_mode` as blocked because they were generated before this
runtime wiring change. Reactivating B9 as a fresh matrix batch should be a
separate spec update.

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
