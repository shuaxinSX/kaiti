# High-Wavenumber Matrix Reports

This directory contains derived reports from the copied experiment result root:

- Source result root: `/Users/ccf/Desktop/开题汇总/开题X/kaiti-highk-results-ep10000`
- Target project: `/Users/ccf/Desktop/开题汇总/开题`
- Scope: result summarization plus a scoped code/config backup from the server-run
  snapshot under `/Users/ccf/Desktop/开题汇总/开题X/kaiti-main`.

Files:

- `EXPERIMENT_REPORT_2026-05-14_highk_ep10000.md`: main high-wavenumber experiment report.
- `MATRIX_PROVENANCE.md`: provenance, completeness, and recovery notes.
- `CODE_BACKUP_NOTES.md`: code/config files imported for GitHub backup and files
  deliberately left out of the repository.
- `matrix_report.csv`: consolidated row-level table for all manifest rows.
- `matrix_batch_summary.csv`: per-batch counts and median/min/max metrics.
- `matrix_reference_only.csv`: reference-only floor table.
- `matrix_missing_artifacts.csv`: final missing artifact table.
- `matrix_failure_ledger_summary.csv`: transient failure ledger grouped by run.

Interpretation rule: this campaign is a high-wavenumber Helmholtz benchmark matrix, not a generic hyperparameter sweep. The main question is whether the current WKBJ/envelope + NSNO + PML residual pipeline can solve high-wavenumber 2D Helmholtz problems with credible reference floors, stable PML behavior, and controlled phase error.
