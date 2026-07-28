# Difficulty coverage diagnostic

**Decision status: diagnosis only. No remedy is selected or implemented.**

This report uses only the frozen TRAIN and VALID parquets and two existing seed-42 LightGBM binaries. TEST remained `closed_not_read`; the recorded 44.7% TEST figure is cited only as inherited context. The diagnostic trained and tuned no model; neither frozen run was retrained, promoted, or rewired, and no dataset/default/source artifact was changed.

## 1. Preconditions and frozen artifacts

- Initial `git status --short`: **clean** (empty output).
- Initial `git log -3 --oneline`:

```text
653e7f1 R2 five-seed confirmation: CONFIRMED for baseline_41, NOT CONFIRMED for concurrent_43
235a1db Pre-register the R2 five-seed confirmation analysis rule
0914e8f Regularization screening, seed 42: R2 (num_leaves 31) is the only candidate
```

- M1 run: `D:\AI\Real projects\Academic_Advisor\models\runs\2026-07-26_1551__baseline-41-gpa-trend-control`
- M1 binary: `D:\AI\Real projects\Academic_Advisor\models\runs\2026-07-26_1551__baseline-41-gpa-trend-control\m1_pass_model.lgbm`
- M2 run: `D:\AI\Real projects\Academic_Advisor\models\runs\2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag`
- M2 binary: `D:\AI\Real projects\Academic_Advisor\models\runs\2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag\m2_grade_model.lgbm`
- Both binaries existed before analysis. Their saved metadata records seed 42, the requested feature contract, this dataset version, and `test_policy = closed_not_read`.
- Full suite: `.venv\Scripts\python.exe -m unittest discover -s tests -t .` — 117 tests, OK in 36.433s.

## 2. Definitions pinned before interpretation

### Exact difficulty computation

`src/course_difficulty.py` is the current implementation. It fits all statistics from TRAIN history only. TRAIN is processed semester by semester, so semester `t` sees strictly earlier TRAIN semesters (`build_temporal_train`, lines 604-646). VALID receives a frozen state fit on complete TRAIN (`fit_difficulty_state`, lines 410-429).

The raw pass statistic is:

```python
work["pass_value"] = (
    (frame["final_mark"] >= 50) & frame["final_mark"].notna()
).astype("int64")
support_count=("mark_present", "sum")
sum_pass=("pass_value", "sum")
```

Each level is empirical-Bayes smoothed toward its parent/global value with `k = 20`:

```python
table[output] = (local_sum + k * parent_value) / (n + k)
```

The fallback hierarchy is Level 1 degree+course, Level 2 course across degrees, Level 3 degree+requirement type+rounded credits, Level 4 faculty+requirement type+rounded credits, Level 5 requirement type+rounded credits, then Level 6 global TRAIN history. The concurrent difficulty scalar is exactly `d = 1.0 - course_pass_rate_historical` (`src/concurrent_group_features.py:29,208-209`).

### What coverage means

Two related definitions must not be conflated:

1. **Inherited Level-1 coverage** is `difficulty_fallback_level == 1`. It means the exact degree-course key was found. It has no minimum-support requirement.
2. **Model-facing confident coverage**, used for all covered/uncovered splits below, is `course_difficulty_missing == 0`. Current source sets the missing flag when `course_is_new == 1` or `course_low_support == 1`; low support means `0 < course_history_count < 20`. Therefore covered means a Level-1 or Level-2 known course with at least 20 historical rows. The statistics are TRAIN-only.

Exact current code (`src/course_difficulty.py:578-590`):

```python
course_is_new = (~supports[1].notna() & ~supports[2].notna())
course_low_support = (
    (course_history > 0) & (course_history < state.config.min_support)
)
feature_values["course_difficulty_missing"] = (
    (course_is_new == 1) | (course_low_support == 1)
)
```

### Fallback/imputation and its indicator

There is not one universal imputed course value. If an exact course degree pairing (Level 1) is unavailable, the code first tries the same course across degrees (Level 2). If the course has no history at either level, it uses the first available Level 3-5 group estimate, each smoothed toward its parent/global TRAIN statistic. Only when no group exists does Level 6 use the global TRAIN values. On this TRAIN those are pass rate 0.841315 (difficulty 0.158685), mean mark 65.582094, and retake rate 0.160623.

Both frozen feature contracts contain the model feature `course_difficulty_missing`, so the models are told that a value is weak/imputed rather than confidently observed. `course_history_count` is also a model feature. `difficulty_fallback_level`, `course_is_new`, and `course_low_support` are audit-only and do not enter either model.

### What 0.186 and 0.134 are

- **0.186:** recorded approximate TRAIN mean of `peer_difficulty_mean` in `scripts/build_concurrent_group_features.py:141-145`.
- **0.134:** recorded approximate VALID mean of `peer_difficulty_mean` in the same constant block.

They are split-level approximate means of the concurrent peer-difficulty feature, not accuracy metrics. Their difference is -0.052 (VALID minus TRAIN); the inherited note calls this a shift, but this diagnostic does not treat the note's causal wording as proof.

### Recomputed coverage

| Split | Rows | Level-1 rows | Recomputed Level-1 | Inherited Level-1 | Difference (pp) | Model-facing confident |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 450,465 | 425,121 | 94.37% | 93.60% | +0.77 | 89.61% |
| VALID | 156,097 | 120,858 | 77.42% | 76.20% | +1.22 | 82.78% |

The inherited Level-1 figures do **not** reproduce on the specified parquets: TRAIN is +0.77 percentage points and VALID is +1.22 points higher. The recorded 44.7% TEST value was not recomputed because TEST was never read.

## 3. VALID uncovered-row decomposition

Under the model-facing definition, VALID has 26,882 uncovered rows.

| Cause | Rows | % uncovered | % VALID | Meaning |
| --- | --- | --- | --- | --- |
| never_in_train | 25,627 | 95.33% | 16.42% | course_id does not appear anywhere in TRAIN |
| thin_history | 1,255 | 4.67% | 0.80% | course_id appears in TRAIN but course_history_count < 20 |
| other | 0 | 0.00% | 0.00% | uncovered row not explained by course absence or support below threshold |

`thin_history` is potentially recoverable from more historical observations. `never_in_train` is the current ceiling for what no mere re-cut of the existing TRAIN history can fix.

### First appearance of never-in-TRAIN courses

| First semester | Distinct courses | % never-in-TRAIN courses | All VALID rows for those courses | Rows in debut semester |
| --- | --- | --- | --- | --- |
| 20221 | 59 | 32.42% | 17,526 | 3,340 |
| 20222 | 23 | 12.64% | 2,894 | 745 |
| 20223 | 11 | 6.04% | 520 | 93 |
| 20231 | 37 | 20.33% | 3,748 | 1,722 |
| 20232 | 36 | 19.78% | 868 | 736 |
| 20233 | 16 | 8.79% | 71 | 71 |

## 4. Coverage over time

| Semester | Split | Rows | Confident coverage | Level-1 coverage |
| --- | --- | --- | --- | --- |
| 20051 | TRAIN | 55 | 0.00% | 0.00% |
| 20052 | TRAIN | 55 | 0.00% | 18.18% |
| 20053 | TRAIN | 4 | 0.00% | 100.00% |
| 20061 | TRAIN | 41 | 0.00% | 73.17% |
| 20062 | TRAIN | 28 | 0.00% | 57.14% |
| 20063 | TRAIN | 5 | 0.00% | 100.00% |
| 20071 | TRAIN | 78 | 0.00% | 84.62% |
| 20072 | TRAIN | 65 | 6.15% | 73.85% |
| 20073 | TRAIN | 6 | 0.00% | 100.00% |
| 20081 | TRAIN | 485 | 51.13% | 28.45% |
| 20082 | TRAIN | 445 | 0.00% | 32.81% |
| 20083 | TRAIN | 48 | 25.00% | 87.50% |
| 20091 | TRAIN | 1,159 | 2.16% | 73.77% |
| 20092 | TRAIN | 1,032 | 3.10% | 76.74% |
| 20093 | TRAIN | 144 | 68.75% | 97.92% |
| 20101 | TRAIN | 1,152 | 12.76% | 76.82% |
| 20102 | TRAIN | 1,062 | 17.80% | 77.31% |
| 20103 | TRAIN | 141 | 70.92% | 99.29% |
| 20111 | TRAIN | 6,301 | 12.76% | 22.52% |
| 20112 | TRAIN | 6,741 | 12.68% | 26.97% |
| 20113 | TRAIN | 642 | 92.52% | 98.91% |
| 20121 | TRAIN | 5,481 | 28.43% | 62.74% |
| 20122 | TRAIN | 4,778 | 33.53% | 62.35% |
| 20123 | TRAIN | 647 | 91.96% | 99.85% |
| 20131 | TRAIN | 8,148 | 55.79% | 79.28% |
| 20132 | TRAIN | 7,878 | 58.66% | 77.56% |
| 20133 | TRAIN | 2,240 | 70.76% | 97.28% |
| 20134 | TRAIN | 1,288 | 95.26% | 99.92% |
| 20141 | TRAIN | 11,258 | 75.32% | 88.18% |
| 20142 | TRAIN | 11,256 | 74.64% | 88.01% |
| 20143 | TRAIN | 3,580 | 90.28% | 97.37% |
| 20144 | TRAIN | 2,639 | 97.99% | 100.00% |
| 20151 | TRAIN | 14,183 | 87.15% | 91.49% |
| 20152 | TRAIN | 13,542 | 86.83% | 92.84% |
| 20153 | TRAIN | 4,113 | 94.36% | 99.08% |
| 20154 | TRAIN | 4,391 | 98.82% | 100.00% |
| 20161 | TRAIN | 17,598 | 97.16% | 99.15% |
| 20162 | TRAIN | 17,169 | 97.09% | 98.95% |
| 20163 | TRAIN | 4,996 | 98.54% | 100.00% |
| 20164 | TRAIN | 5,166 | 98.99% | 99.26% |
| 20171 | TRAIN | 21,605 | 97.19% | 98.22% |
| 20172 | TRAIN | 20,722 | 98.25% | 99.25% |
| 20173 | TRAIN | 5,601 | 98.86% | 99.95% |
| 20181 | TRAIN | 20,913 | 98.73% | 99.94% |
| 20182 | TRAIN | 21,217 | 98.77% | 99.54% |
| 20183 | TRAIN | 6,841 | 98.70% | 99.46% |
| 20191 | TRAIN | 24,676 | 98.65% | 99.72% |
| 20192 | TRAIN | 23,479 | 99.08% | 99.81% |
| 20193 | TRAIN | 7,908 | 99.48% | 100.00% |
| 20201 | TRAIN | 28,426 | 98.22% | 99.84% |
| 20202 | TRAIN | 26,679 | 98.53% | 99.79% |
| 20203 | TRAIN | 9,093 | 98.99% | 99.98% |
| 20211 | TRAIN | 32,311 | 98.09% | 99.91% |
| 20212 | TRAIN | 30,843 | 98.45% | 99.94% |
| 20213 | TRAIN | 10,111 | 98.63% | 99.99% |
| 20221 | VALID | 33,920 | 89.35% | 88.30% |
| 20222 | VALID | 31,976 | 87.86% | 86.51% |
| 20223 | VALID | 11,045 | 89.90% | 88.14% |
| 20231 | VALID | 35,546 | 77.87% | 68.97% |
| 20232 | VALID | 33,082 | 75.43% | 65.69% |
| 20233 | VALID | 10,528 | 78.36% | 69.00% |

Early TRAIN coverage ramps up from a history-free first semester; by the last six TRAIN semesters it is consistently above 98%. The TRAIN/VALID boundary is between 20213 and 20221, where coverage changes by -9.28 percentage points. The decline is not gradual: after the boundary drop it contains another cliff between VALID semesters 20223 and 20231 (-12.04 percentage points).

## 5. Accuracy on covered versus uncovered VALID rows

These predictions were produced by loading the existing frozen binaries listed in section 1. No model was retrained, re-tuned, or threshold-tuned. The complete-VALID re-score exactly reproduces the saved run metrics (M1 unrounded; M2 at the saved four-decimal precision). Gaps below are **uncovered minus covered**.

### M1 — frozen seed-42 `baseline_41` control

| Group | n | Fail rate | ROC AUC | Fail AP | Brier | Fail P @.80 | Fail R @.80 | Fail F1 @.80 | CM (TN,FP,FN,TP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| covered | 129,215 | 9.54% | 0.816886 | 0.319622 | 0.074945 | 0.332319 | 0.407558 | 0.366113 | 5026,7306,10098,106785 |
| uncovered | 26,882 | 14.19% | 0.764525 | 0.331319 | 0.108818 | 0.326237 | 0.457929 | 0.381025 | 1747,2068,3608,19459 |

| Gap metric | Uncovered - covered |
| --- | --- |
| fail_rate | +0.046479 |
| roc_auc | -0.052361 |
| fail_average_precision | +0.011697 |
| brier | +0.033873 |
| fail_precision | -0.006082 |
| fail_recall | +0.050372 |
| fail_f1 | +0.014912 |
| confusion_count_tn | -3279 |
| confusion_count_fp | -5238 |
| confusion_count_fn | -6490 |
| confusion_count_tp | -87326 |

### M2 — frozen seed-42 `concurrent_43` run

| Group | n | Mean final mark | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| covered | 129,215 | 69.772743 | 9.210392 | 12.410075 | 0.370335 |
| uncovered | 26,882 | 68.319507 | 11.347178 | 14.844096 | 0.273161 |

| Gap metric | Uncovered - covered |
| --- | --- |
| mean_final_mark | -1.453236 |
| mae | +2.136785 |
| rmse | +2.434020 |
| r2 | -0.097175 |

The base-rate columns are part of the evidence: uncovered rows are a different population, so group metric gaps must not automatically be attributed solely to imputation.

## 6. Student-semester prevalence

The exact plan grain is `university_id, student_id, degree_id, part_id` (`src/feature_engineering.py:28-35`). VALID contains 34,293 student-semesters; 8,076 (23.55%) contain at least one uncovered course.

| Uncovered courses | Student-semesters | % all | % affected |
| --- | --- | --- | --- |
| 0 | 26,217 | 76.45% | — |
| 1 | 1,397 | 4.07% | 17.30% |
| 2 | 1,684 | 4.91% | 20.85% |
| 3 | 1,160 | 3.38% | 14.36% |
| 4 | 1,386 | 4.04% | 17.16% |
| 5 | 1,645 | 4.80% | 20.37% |
| 6 | 760 | 2.22% | 9.41% |
| 7 | 44 | 0.13% | 0.54% |

Among affected student-semesters, 5,932 (73.45%) have a majority of courses uncovered.

**If the system recommended a plan today, 23.55% of cases would contain at least one course carrying an imputed/weak difficulty.**

## 7. Counterfactual cutoff movement

This is an in-memory estimate only; no split was changed. For each cutoff, the admitted prefix leaves VALID. Only later rows that change from currently uncovered to covered are counted as “newly covered.”

| Move | Semesters admitted | Uncovered rows absorbed | Uncovered remaining | Newly covered | Newly covered: never / thin | % original uncovered | % remaining uncovered | Remaining VALID rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 semester(s) | 20221 | 3,613 | 23,269 | 8,939 | 8,650 / 289 | 33.25% | 38.42% | 122,177 |
| 2 semester(s) | 20221, 20222 | 7,495 | 19,387 | 9,715 | 9,401 / 314 | 36.14% | 50.11% | 90,201 |
| 3 semester(s) | 20221, 20222, 20223 | 8,610 | 18,272 | 9,384 | 9,068 / 316 | 34.91% | 51.36% | 79,156 |

Structural ceiling requested by the task: 25,627 current uncovered rows (95.33% of all uncovered rows) belong to courses absent from current TRAIN. No re-cut of the existing pre-VALID history can give those rows an observed course prior. A forward cutoff can consume an earlier VALID appearance into TRAIN and thereby cover later repetitions; the table counts that later-row effect but does not call the absorbed rows covered.

Cost statement (not a trade-off evaluation): moving the TRAIN cutoff forward makes every existing run, dataset hash, and `models/runs/NOISE_BAND.md` non-comparable, and VALID shrinks.

## 8. PART B — identical-segments defect (diagnosis only)

The two masks in `src/model_training.py` are:

```python
"first_semester": df["is_first_active_semester"] == 1,
"cold_start_gpa": df["no_previous_progress"] == 1,
```

Computed independently on VALID:

| Mask | Rows |
| --- | --- |
| first_semester | 14,732 |
| cold_start_gpa | 14,732 |

Mask mismatch rows: 0. The masks name different columns, but the current preprocessing assigns both columns from the identical no_previous_progress boolean series. Their equality is structural in this prepared data, not an accidental equality of two independently computed populations. The upstream assignments are quoted in `src/feature_engineering.py:402-403`:

```python
semester_df["no_previous_progress"] = no_previous_progress.astype(int)
semester_df["is_first_active_semester"] = no_previous_progress.astype(int)
```

No proposal or fix is made in this section.

## 9. Candidate remedies — options only

- Extend historical TRAIN backward: can help thin-history courses without consuming current VALID semesters, but requires older reliable records and a new dataset/model comparability baseline.
- Move the cutoff forward: the quantified effects are in section 7; VALID shrinks and all existing run/hash/noise-band comparisons break.
- Keep the current fallback hierarchy and missing indicator: preserves comparability, but accepts the prevalence and group accuracy observed above.
- Add non-outcome course/catalog priors for genuinely new offerings: can address `never_in_train`, but introduces new data contracts and requires separate validation.

These are unranked options. This report recommends none.

## 10. Scope confirmations

- TEST dataset: **never read**. Policy remained `closed_not_read`.
- Models: existing frozen binaries only; **no retraining or retuning**.
- Test-suite nuance: existing unit tests train synthetic toy models in the OS temporary directory. They did not retrain either frozen run, read the project TEST split, or write a project model/dataset artifact.
- Data: no dataset file was written or modified.
- Source/defaults/wiring: no file under `src/`, no default, `CURRENT_VERSION.txt`, promotion marker, or inference/recommendation wiring was changed.
- Push: not performed.
