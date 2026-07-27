# concurrent-44-registration-roster-candidate

**Run ID:** 2026-07-26_1554__concurrent-44-registration-roster-candidate
**Date:** 2026-07-26T15:54:09+03:00
**Features:** 44
**Compared with:** 2026-07-26_1551__baseline-41-gpa-trend-control

## What changed

- Candidate: baseline_41 + the 3 concurrent peer-difficulty features; identical data/seed/params/threshold; TEST closed.

## Why

Record this isolated training experiment for reproducible comparison.

## Main result

Selection is based on VALID only. Higher is better for fail-class AP/AUC;
lower is better for Brier and the train-valid AUC gap. TEST is descriptive only.

- M1 valid fail-class Average Precision: 0.3220 -> 0.3228 (+0.0009)
- M1 valid AUC: 0.8092 -> 0.8098 (+0.0006)
- M1 valid Brier: 0.0808 -> 0.0807 (-0.0001)
- M1 train-valid AUC gap: 0.0554 -> 0.0568 (+0.0015)

## M2 regressor

- M2 valid MAE: 9.5667 -> 9.5293 (-0.0374)
- M2 valid RMSE: 12.8549 -> 12.8140 (-0.0409)
- M2 valid R2: 0.3519 -> 0.3560 (+0.0041)

## Fail-class at the locked reporting threshold (VALID)

- M1 valid fail precision: 0.3307 -> 0.3298 (-0.0009)
- M1 valid fail recall: 0.4195 -> 0.4224 (+0.0029)
- M1 valid fail F1: 0.3698 -> 0.3704 (+0.0006)

## Segment result (VALID; existing segment definitions only)

- cold_start_gpa valid AUC (n=14732): 0.7329 -> 0.7411 (+0.0082)
- first_semester valid AUC (n=14732): 0.7329 -> 0.7411 (+0.0082)
- low_difficulty_support valid AUC (n=25627): 0.7644 -> 0.7711 (+0.0067)
- retake_attempt valid AUC (n=17958): 0.6768 -> 0.6765 (-0.0003)
- **NOTE:** `first_semester` and `cold_start_gpa` have identical population size (n=14732) and identical AUC — treat them as one segment until the definitions are separated.

## Run settings

- dataset_version: `2026-07-26_batched_fixes__registration_roster_concurrent`
- feature_contract: `baseline_41` -> `concurrent_44`  **DIFFERS FROM BASELINE**
- feature_count: `41` -> `44`  **DIFFERS FROM BASELINE**
- num_threads: `4`
- random_seed: `42`
- reporting_threshold: `0.8`
- test_policy: `closed_not_read`
- train_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_train_final.parquet`
- valid_path: `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/df_valid_final.parquet`

- Intended delta: ['feature_contract', 'feature_count']; unintended differences: none

## Important flags

- diploma_gpa has no dedicated missing-value indicator; source-level median fill remains unchanged and unmatched diploma records may be null.
