# High-Wavenumber Code Backup Notes

Source code snapshot:

- `/Users/ccf/Desktop/开题汇总/开题X/kaiti-main`

Target project:

- `/Users/ccf/Desktop/开题汇总/开题`

Imported for GitHub backup:

- `configs/experiments/C0_*.yaml` through `configs/experiments/C21_*.yaml`
- `scripts/generate_highk_matrix.py`
- C-batch support in `scripts/run_matrix.py` and `scripts/summarize_matrix.py`
- `src/core/region_masks.py` plus the export in `src/core/__init__.py`
- PML/interface diagnostics and region-weighted residual support in
  `src/physics/residual.py`, `src/train/trainer.py`, `src/train/runner.py`, and
  `src/eval/reference_eval.py`
- Curriculum warm-start execution support in `src/train/runner.py`
- C20 architecture probes in `src/models/nsno.py`
- Resume incident documentation:
  `reports/next_phase/HIGH_WAVENUMBER_RESUME_INCIDENT_2026-05-14.md`

Deliberately not imported:

- The full result root `kaiti-highk-results-ep10000/`
- Smoke/output directories
- `__pycache__/`
- `.vendor_sx/`
- `_yaml_vendor/`
- Virtual environments and installed package caches

Local compatibility adjustments:

- `src/core/region_masks.py` was changed from Python 3.10 union/generic syntax to
  Python 3.9-compatible typing, matching this repository's declared Python range.
- `scripts/run_matrix.py` and `scripts/summarize_matrix.py` had stale A6/B-only
  descriptions updated to B/C matrix wording.

Verification performed locally:

- Static file/diff inspection only.
- `git diff --check` completed with no whitespace errors.
- The untracked-file list contains only C-series configs, high-k report files,
  the incident report, `scripts/generate_highk_matrix.py`, and
  `src/core/region_masks.py`.
- No local training, pytest run, or experiment execution was performed during
  this code-backup step.
