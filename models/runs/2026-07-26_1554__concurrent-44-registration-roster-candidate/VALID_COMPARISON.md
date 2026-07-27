# VALID-only comparison — baseline_41 vs concurrent_44

- Baseline run : `2026-07-26_1551__baseline-41-gpa-trend-control` (41 features)
- Candidate run: `2026-07-26_1554__concurrent-44-registration-roster-candidate` (44 features)
- Reporting threshold (locked, both arms): **0.8**
- TEST policy: **closed_not_read** (both arms)
- Dataset version: `2026-07-26_batched_fixes__registration_roster_concurrent`

## M1 classifier — threshold-independent

| Metric | baseline_41 | concurrent_44 | delta |
|---|---:|---:|---:|
| **TRAIN** | | | |
| train ROC AUC | 0.864572 | 0.866625 | +0.002053 |
| train pass-class AP | 0.970091 | 0.970666 | +0.000575 |
| train fail-class AP | 0.567996 | 0.570696 | +0.002700 |
| train Brier | 0.096130 | 0.095735 | -0.000395 |
| **VALID** | | | |
| valid ROC AUC | 0.809189 | 0.809775 | +0.000587 |
| valid pass-class AP | 0.972612 | 0.972734 | +0.000122 |
| valid fail-class AP | 0.321983 | 0.322845 | +0.000862 |
| valid Brier | 0.080778 | 0.080698 | -0.000081 |
| train-valid AUC gap | 0.055384 | 0.056850 | +0.001466 |
| M1 best iteration | 137 | 143 | +6 |

## M1 at the fixed reporting threshold (0.8) — VALID

| Metric | baseline_41 | concurrent_44 | delta |
|---|---:|---:|---:|
| fail precision | 0.3307 | 0.3298 | -0.0009 |
| fail recall | 0.4195 | 0.4224 | +0.0029 |
| fail F1 | 0.3698 | 0.3704 | +0.0006 |
| pass precision | 0.9309 | 0.9311 | +0.0002 |
| pass recall | 0.9021 | 0.9010 | -0.0011 |
| pass F1 | 0.9162 | 0.9158 | -0.0004 |

Confusion matrix (rows = actual, cols = predicted):

| Run | Act-FAIL/Pred-FAIL (tn) | Act-FAIL/Pred-PASS (fp) | Act-PASS/Pred-FAIL (fn) | Act-PASS/Pred-PASS (tp) |
|---|---:|---:|---:|---:|
| baseline_41 | 6,773 | 9,374 | 13,706 | 126,244 |
| concurrent_44 | 6,821 | 9,326 | 13,861 | 126,089 |

## VALID segments (existing definitions only)

| Segment | n | baseline AUC | concurrent AUC | delta |
|---|---:|---:|---:|---:|
| cold_start_gpa | 14,732 | 0.732931 | 0.741121 | +0.008190 |
| first_semester | 14,732 | 0.732931 | 0.741121 | +0.008190 |
| low_difficulty_support | 25,627 | 0.764405 | 0.771110 | +0.006705 |
| retake_attempt | 17,958 | 0.676827 | 0.676463 | -0.000364 |

> **FLAG:** `first_semester` and `cold_start_gpa` have IDENTICAL population (n=14,732) and IDENTICAL AUC. They are not independent evidence; treat as one segment.

Note: `retake_attempt` and `low_difficulty_support` are pre-existing segment definitions in `model_training._segment_masks`; no segment was added for this experiment.

## Diagnostic: Level-1 difficulty coverage slice (VALID)

`difficulty_fallback_level == 1`. Diagnostic only — this column is loaded for segment reporting and is in the dropped-feature guard, so it is not a model input.

| Slice | n | baseline AUC | concurrent AUC | delta |
|---|---:|---:|---:|---:|
| difficulty_fallback_level == 1 | 120,858 | 0.820962 | 0.820423 | -0.000538 |
| (whole VALID, re-scored check) | — | 0.809189 | 0.809775 | +0.000587 |

## M2 regressor

| Metric | baseline_41 | concurrent_44 | delta |
|---|---:|---:|---:|
| train MAE | 8.7566 | 8.5848 | -0.1718 |
| train RMSE | 12.4661 | 12.2955 | -0.1706 |
| train R2 | 0.4911 | 0.5050 | +0.0139 |
| valid MAE | 9.5667 | 9.5293 | -0.0374 |
| valid RMSE | 12.8549 | 12.8140 | -0.0409 |
| valid R2 | 0.3519 | 0.3560 | +0.0041 |
| M2 best iteration | 438 | 559 | +121 |

## Feature evidence — the three concurrent features (concurrent_44, M1)

Total M1 features: 44. Total gain: 986,242.77

| Feature | gain | gain rank | % of total gain | split | split rank | unused? |
|---|---:|---:|---:|---:|---:|---|
| concurrent_peer_difficulty_mean | 12,808.22 | 16 / 44 | 1.299% | 692 | 10 / 44 | no |
| concurrent_peer_difficulty_max | 15,370.89 | 14 / 44 | 1.559% | 778 | 7 / 44 | no |
| concurrent_peer_difficulty_missing | 0.00 | 40 / 44 | 0.000% | 0 | 40 / 44 | **YES — zero split** |

Combined share of total M1 gain: **2.857%**

> Feature importance alone is NOT a keep/drop criterion; it is corroborating evidence only.

## Controlled-delta proof

| Controlled property | identical? | value |
|---|---|---|
| same train file | YES | `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet` |
| same valid file | YES | `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet` |
| same train sha256 | YES | `8aaff32aeac5b375…` |
| same valid sha256 | YES | `228719fa492da84b…` |
| same M1 target | YES | `(final_mark >= 50).astype(int)` |
| same M2 target | YES | `final_mark` |
| same categorical features | YES | `['requirement_type_id', 'diploma_type_bucket']` |
| same categorical levels | YES | `identical` |
| same unknown category code | YES | `-1` |
| same LightGBM params | YES | `identical` |
| same random seed | YES | `42` |
| same reporting threshold | YES | `0.8` |
| same test policy | YES | `closed_not_read` |
| same dropped-feature guard | YES | `identical` |
| same derived sources | YES | `identical` |
| same git commit | YES | `0291dd26ec27` |
| shared features identical & in order | YES | 41 features |
| only delta | — | `['concurrent_peer_difficulty_max', 'concurrent_peer_difficulty_mean', 'concurrent_peer_difficulty_missing']` |

All controlled properties identical: **True**

