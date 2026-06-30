# High-Wavenumber Matrix Provenance

## Source

- Copied result root analyzed: `/Users/ccf/Desktop/开题汇总/开题X/kaiti-highk-results-ep10000`
- Original Windows result root recorded in manifest: `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000`
- Main manifest: `/Users/ccf/Desktop/开题汇总/开题X/kaiti-highk-results-ep10000/_meta/manifest.tsv`
- Recovery incident note in copied code snapshot: `/Users/ccf/Desktop/开题汇总/开题X/kaiti-main/reports/next_phase/HIGH_WAVENUMBER_RESUME_INCIDENT_2026-05-14.md`

## Manifest

- Manifest rows: 1413
- Status counts: {'active': 1413}
- Entrypoint counts: {'reference_only': 205, 'train': 1208}
- Completed summaries: 1412 / 1413
- Training runs with `losses.csv`: 1207 / 1208
- Reference-only completed summaries: 205 / 205
- Final missing summary: 1
- Manifest `epochs_override`: ['10000']
- Manifest `git_sha`: ['nogit']
- Manifest `spec_branch`: ['local-configs']
- Manifest `launched_at`: ['2026-05-14T23:21:42']

## Important Provenance Caveats

1. The manifest records `git_sha=nogit` and `spec_branch=local-configs`; this is weaker provenance than the old A6 matrix with a committed SHA.
2. The campaign was interrupted and resumed. The recovery note records an initial interrupted state, dependency fixes in the `SX` environment, and a curriculum reference-label fix.
3. `_meta/failures.jsonl` has 481 failure-ledger rows, but most were transient resume/retry failures. The final completeness criterion is the presence of `summary.json`, not the failure ledger alone.
4. The final missing run is `C5_laptau_strictness/layered__omega180__N459__pml24__center__stretched_divergence`.
5. All generated C12 configs have `training.epochs=10000`; C12 run IDs like `ep01000` and `ep40000` are labels after the global override, not valid epoch-budget comparisons.
6. This report did not copy source code from `开题X/kaiti-main` into the main project.

## Runtime Accounting

- Sum of completed training-run `runtime_sec`: 2339343.9 s = 649.82 h = 27.08 days.
- The final resume log records a separate launcher segment runtime and should not be interpreted as the full campaign compute time.
