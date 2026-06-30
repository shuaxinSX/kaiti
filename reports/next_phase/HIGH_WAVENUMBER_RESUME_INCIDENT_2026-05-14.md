# High-Wavenumber Resume Incident Record

Date: 2026-05-14
Project: `C:\Users\Lenovo\Desktop\kaiti-main`
Result root: `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000`

## 1. Purpose

This document records the full recovery process after the high-wavenumber
experiment matrix was interrupted by a machine restart, including:

- what failed
- how the failure was diagnosed
- which handling steps were wrong or incomplete
- which code/runtime fixes were applied
- what the current run state is after recovery

This document is intentionally explicit about failed attempts. It is a recovery
log, not a polished summary.

## 2. Executive Summary

The campaign did not finish before the machine restart. At first inspection on
2026-05-14, the run had stopped with:

- `421 / 1413` active runs completed
- `992` active runs remaining
- `34` recorded failures
- `35` interrupted or partial output directories without `summary.json`

Recovery exposed four separate problems:

1. The machine restart killed the campaign and there was no auto-resume.
2. `scripts/run_matrix.py` treats `output_dir.exists()` as "already handled",
   so interrupted directories would be skipped on resume even if they were not
   actually finished.
3. The `SX` environment was not self-contained for all required packages.
   `PyYAML`, `SciPy`, and `Matplotlib` were not reliably available from
   `D:\anaconda3\envs\SX\Lib\site-packages`.
4. The curriculum support path in `src/train/runner.py` reused the final-stage
   `training.supervision.reference_path` for all stages. For stage 1 of
   curriculum runs this caused label shape mismatches such as:
   `expected (109, 109), got (168, 168)` and
   `expected (109, 109), got (523, 523)`.

The environment issue has been corrected by installing the missing packages into
the `SX` environment itself. The curriculum shape-mismatch issue has been
corrected in `src/train/runner.py` by materializing a stage-specific reference
label for each curriculum stage.

At the end of this recovery round, the resumed campaign was running again from a
dedicated terminal window and GPU utilization had returned to approximately
`99%`.

## 3. Impact

### 3.1 What was affected

- The full `C0-C21` matrix under
  `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000`
- Resume reliability after restart
- Curriculum batches such as `C13` and `C15`
- Environment reproducibility for `SX`

### 3.2 What was not corrupted

- Completed runs that already had `summary.json` were preserved.
- No completed run directories were overwritten during recovery.
- Partial directories were moved aside before relaunch.

### 3.3 Observed user-facing symptom

The most visible symptom after recovery attempts was:

- the user observed that GPU utilization was no longer saturated

This turned out not to mean "the code now trains more slowly". The actual issue
was that some curriculum jobs were failing almost immediately, so the launcher
was spending time in fast failure / task-switch cycles instead of sustained
training.

## 4. Timeline

| Time | Event |
|---|---|
| `2026-05-14 00:01:28` | Original campaign log `campaign_stdout.log` stopped updating. Last visible progress was around `[456/1413]`. |
| `2026-05-14 09:09` | After reboot, inspection confirmed the campaign had not completed and no `SX` training process was active on GPU. |
| `2026-05-14 09:09` | Completed-count check showed `421 / 1413` active runs finished and `34` failures recorded. |
| `2026-05-14 09:12-09:18` | First resume attempts failed because `scripts/run_matrix.py` could not import `yaml` in `SX`. |
| `2026-05-14 09:18-09:30` | Multiple environment recovery attempts were tried, including user-site install, direct install to `SX`, project-local `.vendor_sx`, and `_yaml_vendor`. These were not all correct; details are recorded in Section 6. |
| `2026-05-14 09:28-09:32` | Elevated package installation logs were written to `_meta/sx_pip_install_20260514_0930.log` and `_meta/sx_pip_install_force_20260514_0932.log`. |
| `2026-05-14 09:31-09:33` | A visible dedicated terminal was used to resume the matrix. GPU utilization returned to `~99%`, confirming real training was happening. |
| `2026-05-14 09:35` | Curriculum runs in `C13` and `C15` were found to fail with `reference_envelope.npy shape mismatch`. |
| `2026-05-14 09:36-09:45` | `src/train/runner.py` was inspected and patched so each curriculum stage generates and uses its own stage-specific reference label. |
| `2026-05-14 09:45` | CPU smoke test on a previously failing curriculum config (`layered__omega180__N459__no_restart`) progressed beyond the original mismatch and entered real stage training. |
| `2026-05-14 09:46` | Old partial outputs were moved to a second backup root and the matrix was resumed again with the curriculum fix applied. |
| `2026-05-14 09:48` | New visible resume session was confirmed. Top-level log reached `[247/1413]` and GPU utilization was back at `99%`. |

## 5. Evidence Collected During Recovery

### 5.1 Stop point before recovery

- Main historical log:
  `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\campaign_stdout.log`
- Last write time observed:
  `2026-05-14 00:01:28`
- Last visible lines included:
  `[456/1413] C17 / layered__omega120__N306__w24__p3__stretched_divergence`

### 5.2 State at first inspection

- Active total: `1413`
- Completed total: `421`
- Remaining total: `992`
- Failure ledger count at that point: `34`

### 5.3 Evidence of curriculum failure

Representative failure log:

`C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\logs\C13\layered__omega180__N459__no_restart__train.log`

Representative error:

```text
ValueError: reference_envelope.npy shape mismatch: expected (109, 109), got (168, 168).
```

Another representative failure:

`C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\logs\C15\cardG__best_curriculum_proxy__train.log`

Representative error:

```text
ValueError: reference_envelope.npy shape mismatch: expected (109, 109), got (523, 523).
```

### 5.4 Evidence that GPU saturation returned

Sample GPU reading after corrected relaunch:

```text
99 %, 4068 MiB, 282.36 W, 70C
```

## 6. Recovery Attempts, Including Incorrect Ones

This section records the actual order of attempts. Several were wrong or only
partially correct.

### 6.1 First resume failure: `yaml` import missing

The first blocked point was:

```text
ModuleNotFoundError: No module named 'yaml'
```

The immediate cause was that `SX` could not reliably import `PyYAML`.

### 6.2 Incorrect / incomplete handling during dependency recovery

The following attempts happened before the final correct state was reached:

1. `pip install PyYAML` was run without forcing installation into the `SX`
   environment root.
   Result:
   package presence became tied to a user-site path rather than guaranteed
   inside `D:\anaconda3\envs\SX\Lib\site-packages`.

2. Direct `pip --target D:\anaconda3\envs\SX\Lib\site-packages` failed due
   permissions on the environment's `site-packages` root.

3. A project-local fallback using `.vendor_sx` was attempted.
   Result:
   the resulting `yaml` package was not usable for final recovery and was not
   used in the final runtime path.

4. A pure-Python copy fallback under `_yaml_vendor` was attempted.
   Result:
   this was also abandoned and is not the final runtime solution.

The user's complaint that the package had not actually been installed into `SX`
was correct.

### 6.3 Final correct dependency state

The final stable approach was:

- install `PyYAML`, `SciPy`, `Matplotlib`, and `tqdm` into the `SX`
  environment itself
- verify import paths using `D:\anaconda3\envs\SX\python.exe`
- launch resumed runs with `PYTHONNOUSERSITE=1` to ensure only `SX` packages
  are consumed

Verified import paths included:

- `D:\anaconda3\envs\SX\lib\site-packages\yaml\__init__.py`
- `D:\anaconda3\envs\SX\lib\site-packages\scipy\__init__.py`
- `D:\anaconda3\envs\SX\lib\site-packages\matplotlib\__init__.py`
- `D:\anaconda3\envs\SX\lib\site-packages\tqdm\__init__.py`

## 7. Root Cause Analysis

### 7.1 Root cause A: restart with no resume supervisor

The machine restart terminated the campaign and there was no process supervisor,
scheduled task, or watchdog to bring it back automatically.

### 7.2 Root cause B: launcher resume semantics are too weak

`scripts/run_matrix.py` currently treats an existing output directory as
"already handled":

```python
if output_dir.exists():
    if row["entrypoint"] == "reference_only":
        backfill_reference_only_summary(row, device)
    return
```

This means:

- a directory with only `config_merged.yaml`
- a directory with stage scaffolding only
- a directory interrupted before `summary.json`

will all be skipped on resume even though the run is not actually complete.

This behavior is why partial directories had to be moved aside manually before
relaunch.

### 7.3 Root cause C: `SX` environment was not hermetic enough

The restart exposed that package availability had been depending on more than
just `D:\anaconda3\envs\SX\Lib\site-packages`. After reboot, imports that had
previously seemed available were not consistently available to resumed runs.

The final recovery path therefore enforced:

- packages must exist in `SX` itself
- launch must set `PYTHONNOUSERSITE=1`

### 7.4 Root cause D: curriculum reference path bug in code

This was the main code defect found during recovery.

Curriculum stage configs were being built by overlaying stage-specific
`physics/grid/pml/model/training` values onto a base config, but the base config
still carried a single `training.supervision.reference_path` that pointed to the
final-stage label.

For example:

- final run config could target `omega=180`, `N=459`
- stage 1 could target `omega=30`, `N=77`
- the label path still pointed to a reference envelope generated for the final
  stage instead of the stage-1 grid

That directly produced shape mismatches such as:

- stage expects `109 x 109`
- label file provides `168 x 168` or `523 x 523`

This defect was introduced by the curriculum support path and was not a GPU
throughput problem.

## 8. Corrective Actions Applied

### 8.1 Output protection and resume hygiene

Partial run directories without `summary.json` were moved aside instead of being
deleted, so the launcher would not incorrectly skip them.

Backup roots created during recovery:

- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\resume_backups\20260514_091500_flat`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\resume_backups\20260514_0947_after_curriculum_fix`

### 8.2 Runtime dependency repair

Missing runtime packages were installed into `SX` itself:

- `PyYAML`
- `SciPy`
- `Matplotlib`
- `tqdm`

Support logs:

- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\sx_pip_install_20260514_0930.log`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\sx_pip_install_force_20260514_0932.log`

### 8.3 Curriculum reference fix in code

`src/train/runner.py` was updated so that curriculum stages do not reuse the
final-stage label blindly.

The fix added:

- `curriculum_stage_uses_supervision(...)`
- `build_reference_only_cfg(...)`
- `ensure_curriculum_stage_reference(...)`

Behavior after the fix:

- before each curriculum stage with data supervision
- a stage-specific reference label is generated from that stage's own config
- the stage's `training.supervision.reference_path` is rewritten to the new
  stage-specific label

Important scope note:

- this fix changes label routing only
- it does not change the model architecture
- it does not change the loss definition
- it does not change the PDE residual formula
- it does not change optimizer math

### 8.4 Validation

A CPU smoke test was run on a previously failing curriculum config:

- config:
  `...C13_curriculum_warm_start\layered__omega180__N459__no_restart.yaml`
- output:
  `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_smoke\curriculum_fix_cpu`

The smoke test progressed past the previous shape-mismatch point and entered
real training:

```text
Curriculum stage 1/4: omega030_L12
...
Epoch 125/1250, ...
```

That confirmed the original curriculum failure mode was fixed.

## 9. Current Run State At Time of Writing

At the time this document was written:

- active runs total: `1413`
- completed runs: `421`
- remaining runs: `992`
- current visible resume log:
  `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\campaign_resume_20260514_0948.log`
- top-level log tail had reached:
  `[247/1413] C13 / smooth_lens__omega120__N306__direct`

Current count may still remain `421` for a while even when the GPU is busy,
because the launcher only writes a new `summary.json` after a run finishes. Long
training runs can keep the GPU fully occupied without immediately increasing the
completed-count.

At confirmation time, GPU utilization had returned to approximately `99%`.

## 10. Files and Artifacts Touched During the Incident

### 10.1 Source file changed

- `src/train/runner.py`

### 10.2 Recovery support artifacts created

- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\resume_matrix_sx.cmd`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\campaign_resume_20260514_0940.log`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\campaign_resume_20260514_0948.log`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\resume_backups\...`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\sx_pip_install_20260514_0930.log`
- `C:\Users\Lenovo\Desktop\kaiti-highk-results-ep10000\_meta\sx_pip_install_force_20260514_0932.log`

### 10.3 Temporary / abandoned workaround locations

These were created during failed or partial recovery attempts and are not the
final runtime solution:

- `.vendor_sx`
- `_yaml_vendor`

## 11. Lessons Learned

### 11.1 What should be done differently next time

1. Do not assume "it imported before" means the package is actually installed
   into the target environment.

2. For this project, resumed runs should launch with:
   `PYTHONNOUSERSITE=1`
   so hidden user-site dependencies cannot mask environment drift.

3. The launcher should treat `summary.json` as the real completion sentinel,
   not just directory existence.

4. Every new curriculum path should be smoke-tested with at least one
   supervised multi-stage run before launching a full matrix.

5. Visible dedicated resume terminals are more trustworthy on this machine than
   hidden background launch attempts when debugging environment or path issues.

### 11.2 Remaining structural risk

The current recovery restored the campaign, but one structural weakness remains:

- `scripts/run_matrix.py` resume behavior still skips any existing output
  directory, even if the run never produced `summary.json`

This was worked around operationally by moving incomplete directories aside.
The launcher itself still deserves a proper resume-safe fix in a later change.

## 12. Final Assessment

The incident was not a single bug. It was a chain:

- machine restart
- non-resume-safe launcher semantics
- non-hermetic runtime environment
- curriculum stage label routing bug

The most serious code defect discovered during this recovery was the curriculum
reference-path bug in `src/train/runner.py`. The most serious operational defect
was assuming that an existing output directory meant a run was complete.

As of the end of this record:

- completed results remained intact
- interrupted results were backed up
- required dependencies were installed into `SX`
- curriculum stage supervision was repaired
- the matrix had been relaunched and GPU utilization had returned to a saturated
  training state
